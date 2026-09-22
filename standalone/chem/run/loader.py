# chem/loader.py
# %%
import os
import json
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader 
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parents[1]))
from chem.utils import load_pickle, save_pickle
from chem.splitter import (
    DataSplit, 
    generate_scaffold,
    label_balanced_scaffold_split_df,
    scaffold_split_df,
    random_scaffold_split_df,
)
from chem.featurizer import (
    featurize_atom,
    featurize_bond,
    BOND_FEATURE_DIM,
)

_VALID_SPLITS = {"scaffold", "label_balanced_scaffold", "random_scaffold", "random"}
_GRAPH_CACHE: Dict[str, object] = {}
_PLACEHOLDER_GRAPH = None  # single-carbon fallback for invalid SMILES


def smiles_to_pyg(smiles: str):
    """Cache-aware SMILES → PyG Data; falls back to a single-carbon graph for invalid/empty SMILES."""
    global _PLACEHOLDER_GRAPH
    if smiles not in _GRAPH_CACHE:
        g = smiles_to_graph(smiles)
        if g is None:
            if _PLACEHOLDER_GRAPH is None:
                _PLACEHOLDER_GRAPH = smiles_to_graph("C")
            g = _PLACEHOLDER_GRAPH
        _GRAPH_CACHE[smiles] = g
    return _GRAPH_CACHE[smiles]


def safe_smiles(s) -> str:
    """Normalise SMILES to a plain string regardless of input type."""
    if isinstance(s, str):           return s
    if isinstance(s, (list, tuple)): return s[0]
    return str(s)


def smiles_to_graph(smiles: str, label: Optional[float] = None) -> Optional[Data]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None for unparseable SMILES (caller must filter).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([featurize_atom(a) for a in mol.GetAtoms()], dtype=torch.long)

    src, dst, edge_attr = [], [], []
    for bond in mol.GetBonds():
        i, j   = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat   = featurize_bond(bond)
        src   += [i, j];  dst  += [j, i]
        edge_attr += [feat, feat]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.long)

    data        = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


class ChemDataset(TorchDataset):
    """
    Lightweight SMILES dataset for HuggingFace transformer fine-tuning / probing.
    Inherits from torch.utils.data.Dataset (not PyG) since it returns raw
    SMILES strings, not graph objects.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        task_cols: Union[str, List[str], None] = None,
    ):
        df = df.copy()
        self.has_labels = task_cols is not None
        if self.has_labels:
            task_cols = [task_cols] if isinstance(task_cols, str) else list(task_cols)
            # df = df[df[task_cols].notna().all(axis=1)].reset_index(drop=True)
            df = df.reset_index(drop=True)
            self.labels = df[task_cols].astype(float).values.tolist()
        else:
            task_cols   = []
            self.labels = None
        self.smiles = df[smiles_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        item = {"smiles": self.smiles[idx]}
        if self.has_labels:
            item["labels"] = self.labels[idx]
        return item


class ChemCollator:
    """
    Collate SMILES strings into tokenised tensors for transformer training.

    task_mode follows the project-wide convention:
    "binary_classification" | "multilabel_classification" | "regression"
    All three produce float label tensors — the distinction is used by the
    loss function, not here.
    """

    _VALID_TASK_MODES = {"binary_classification", "multilabel_classification", "regression"}

    def __init__(self, tokenizer, max_length: int = 128, task_mode: str = "binary_classification"):
        if task_mode not in self._VALID_TASK_MODES:
            raise ValueError(
                f"task_mode must be one of {self._VALID_TASK_MODES}, got {task_mode!r}"
            )
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.task_mode  = task_mode

    def __call__(self, batch: List[dict]) -> dict:
        enc = self.tokenizer(
            [x["smiles"] for x in batch],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "labels" in batch[0]:
            enc["labels"] = torch.tensor(
                [x["labels"] for x in batch], dtype=torch.float
            )
        return enc
    

class ChemGraphDataset(InMemoryDataset):
    """
    In-memory PyG dataset built from a list of Data objects.

    Uses InMemoryDataset (not bare Dataset) so PyG's internal
    collation, slicing, and index operators all work correctly.
    """

    def __init__(
        self,
        data_list: Optional[List[Data]] = None,
        data: Optional[Data] = None,
        slices: Optional[dict] = None,
    ):
        super().__init__()
        if data is not None and slices is not None:
            self.data, self.slices = data, slices
        else:
            valid = [d for d in data_list if d is not None]
            self.data, self.slices = self.collate(valid)

    def len(self) -> int:
        return self.slices["x"].numel() - 1

    def get(self, idx: int) -> Data:
        return super().get(idx)


def _compute_pos_weight_ds(dataset, task_idx: int) -> torch.Tensor:
    y = torch.stack([
        dataset[i].y.view(-1)[task_idx]
        for i in range(len(dataset))
    ])

    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()

    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"[loader] pos_weight: n_pos={n_pos}  n_neg={n_neg}  "
          f"pos_weight={pos_weight.item():.3f}")
    
    return pos_weight


def _compute_pos_weight_indices(dataset, indices: List[int], task_idx: int) -> torch.Tensor:
    y = torch.stack([
        dataset[i].y.view(-1)[task_idx]
        for i in indices
    ])

    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()

    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"[loader] pos_weight: n_pos={n_pos}  n_neg={n_neg}  "
          f"pos_weight={pos_weight.item():.3f}")

    return pos_weight


def _make_graph_split(dataset, split_indices, batch_size: int) -> DataSplit:
    train_idx, valid_idx, test_idx = split_indices
    return DataSplit(
        train=PyGDataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True),
        valid=PyGDataLoader(Subset(dataset, valid_idx), batch_size=batch_size, shuffle=False),
        test=PyGDataLoader(Subset(dataset, test_idx),  batch_size=batch_size, shuffle=False),
    )


def _compute_pos_weight_df(train_df: pd.DataFrame, task_col: str) -> torch.Tensor:
    """
    Compute BCEWithLogitsLoss pos_weight = n_neg / n_pos from a training DataFrame.
    Falls back to 1.0 if there are no positive labels.
    Only meaningful for binary classification tasks.
    """
    labels = train_df[task_col].dropna().values
    n_pos  = (labels == 1).sum()
    n_neg  = (labels == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"[loader] pos_weight: n_pos={n_pos}  n_neg={n_neg}  "
          f"pos_weight={pos_weight.item():.3f}")
    
    return pos_weight


def _ik_first_local(ik):
    return ik.split("-")[0] if isinstance(ik, str) and "-" in ik else ik


def _downsample_with_protected_iks(
    df: pd.DataFrame,
    ik_col: Optional[str],
    sample_frac: Optional[float],
    protected_iks: Optional[set],
    seed: int,
) -> pd.DataFrame:
    if sample_frac is None:
        return df

    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError(f"downsample_frac must be in (0, 1], got {sample_frac}")
    if sample_frac >= 1:
        return df

    if protected_iks and ik_col is not None and ik_col in df.columns:
        protected_mask = df[ik_col].apply(_ik_first_local).isin(protected_iks)
    else:
        protected_mask = pd.Series(False, index=df.index)

    protected_df = df[protected_mask]
    candidate_df = df[~protected_mask]
    sampled_df = candidate_df.sample(frac=sample_frac, random_state=seed)
    out = pd.concat([protected_df, sampled_df], axis=0).sort_index().reset_index(drop=True)
    print(
        f"[loader] downsample: protected={len(protected_df)}  "
        f"sampled_nonprotected={len(sampled_df)}/{len(candidate_df)}  "
        f"total={len(df)} → {len(out)}"
    )
    return out


def _normalise_excluded_scaffolds(exclude_scaffolds) -> set:
    if not exclude_scaffolds:
        return set()
    return {str(sc) for sc in exclude_scaffolds if sc is not None and str(sc) != ""}


def _exclude_scaffolds_hash(exclude_scaffolds) -> str:
    values = sorted(_normalise_excluded_scaffolds(exclude_scaffolds))
    if not values:
        return ""
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()


def _exclude_scaffolds_df(
    df: pd.DataFrame,
    smiles_col: str,
    exclude_scaffolds=None,
    context: str = "dataset",
) -> pd.DataFrame:
    excluded = _normalise_excluded_scaffolds(exclude_scaffolds)
    if not excluded:
        return df
    scaffolds = df[smiles_col].astype(str).map(
        lambda s: generate_scaffold(s, include_chirality=True)
    )
    keep = ~scaffolds.isin(excluded)
    filtered = df.loc[keep].reset_index(drop=True)
    print(
        f"[loader] scaffold exclusion ({context}): {len(df)} → {len(filtered)} molecules "
        f"after removing {len(excluded)} held-out scaffolds"
    )
    return filtered


def _exclude_iks_hash(exclude_iks) -> str:
    values = sorted(str(ik) for ik in exclude_iks if ik is not None and str(ik) != "")
    if not values:
        return ""
    return hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()


def _exclude_iks_df(
    df: pd.DataFrame,
    ik_col: Optional[str],
    exclude_iks=None,
    context: str = "dataset",
) -> pd.DataFrame:
    if not exclude_iks or ik_col is None or ik_col not in df.columns:
        return df
    excluded = {_ik_first_local(str(ik)) for ik in exclude_iks if ik is not None and str(ik) != ""}
    before = len(df)
    keep = ~df[ik_col].apply(_ik_first_local).isin(excluded)
    filtered = df.loc[keep].reset_index(drop=True)
    print(
        f"[loader] IK exclusion ({context}): {before} → {len(filtered)} molecules "
        f"after removing {len(excluded)} matched InChIKeys"
    )
    return filtered


def load_chem_graph_dataset(
    task_path: str,
    smiles_col: str,
    task_cols: Union[str, List[str]],
    split: str = "scaffold",
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
    batch_size: int = 64,
    return_index: bool = True,
    filter_iks: Optional[set] = None,
    ik_col: Optional[str] = None,
    downsample_frac: Optional[float] = None,
    protected_iks: Optional[set] = None,
    exclude_scaffolds: Optional[set] = None,
    exclude_iks: Optional[set] = None,
    force_train_iks: Optional[set] = None,
    force_test_scaffolds: Optional[set] = None,
    label_balance_trials: int = 1000,
    min_valid_labeled_per_task: int = 10,
    min_test_labeled_per_task: int = 10,
    min_valid_positive_per_task: int = 1,
    min_test_positive_per_task: int = 1,
    min_valid_negative_per_task: int = 1,
    min_test_negative_per_task: int = 1,
    ) -> Union[
        Tuple[DataSplit, torch.Tensor],
        Tuple[DataSplit, torch.Tensor, tuple],
    ]:
    """
    Load a CSV, featurize molecules, and split into train/val/test DataLoaders.

    Split strategies
    ────────────────
    scaffold        — deterministic Bemis-Murcko scaffold split
        label_balanced_scaffold — scaffold-disjoint split selected to preserve
                        endpoint label/class support in validation/test
        random_scaffold — scaffold-grouped, order randomised (reproducible via seed)
    random          — sklearn train_test_split without stratification

    Returns
    ───────
    (DataSplit, pos_weight)                        if return_index=False
    (DataSplit, pos_weight, (train_idx, valid_idx, test_idx)) if return_index=True
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    task_cols = [task_cols] if isinstance(task_cols, str) else list(task_cols)
    import csv as _csv
    with open(task_path, newline="") as _f:
        sep = _csv.Sniffer().sniff(_f.read(4096)).delimiter
    df = pd.read_csv(task_path, sep=sep)
    need_ik_col = (filter_iks is not None or exclude_iks is not None or protected_iks is not None or force_train_iks is not None) and ik_col is not None
    select_cols = [smiles_col] + task_cols + ([ik_col] if need_ik_col and ik_col not in [smiles_col] + task_cols else [])
    df = df[select_cols].dropna(subset=[smiles_col]).reset_index(drop=True)
    if len(task_cols) == 1:
        before = len(df)
        df = df.dropna(subset=task_cols).reset_index(drop=True)
        print(f"[loader] task label filter ({task_cols[0]}): {before} → {len(df)} molecules")
    if filter_iks is not None and ik_col is not None:
        before = len(df)
        df = df[df[ik_col].apply(_ik_first_local).isin(filter_iks)].reset_index(drop=True)
        print(f"[loader] IK pre-filter: {before} → {len(df)} molecules")
    df = _exclude_scaffolds_df(
        df,
        smiles_col,
        exclude_scaffolds=exclude_scaffolds,
        context="graph split",
    )
    df = _exclude_iks_df(df, ik_col, exclude_iks=exclude_iks, context="graph split")
    df = _downsample_with_protected_iks(df, ik_col, downsample_frac, protected_iks, seed)

    # Pre-split separation:
    # force_train_iks  → entire scaffold groups containing a forced IK go to train.
    #   Handled inside scaffold_split() via ik_list; only force_test_scaffolds
    #   compounds are pre-allocated here (they have no splitter-native path).
    # force_test_scaffolds → compounds routed to test before scaffold split runs.
    _track_iks = force_train_iks is not None and ik_col is not None and ik_col in df.columns
    _raw_iks = df[ik_col].apply(_ik_first_local).tolist() if _track_iks else None
    _force_train_set = {_ik_first_local(str(ik)) for ik in force_train_iks if ik is not None} if _track_iks else set()
    _force_test_set = _normalise_excluded_scaffolds(force_test_scaffolds) if force_test_scaffolds else set()

    force_test_graphs, force_test_smiles, force_test_labels = [], [], []
    all_graphs, all_smiles_list, all_labels, all_iks = [], [], [], []

    for idx, (smiles, label) in enumerate(zip(df[smiles_col].astype(str).tolist(), df[task_cols].values.tolist())):
        graph = smiles_to_graph(smiles, label)
        if graph is None:
            continue
        sc = generate_scaffold(smiles, include_chirality=True) if _force_test_set else None
        if _force_test_set and sc in _force_test_set:
            force_test_graphs.append(graph)
            force_test_smiles.append(smiles)
            force_test_labels.append(label)
        else:
            all_graphs.append(graph)
            all_smiles_list.append(smiles)
            all_labels.append(label)
            all_iks.append(_raw_iks[idx] if _track_iks else None)

    if _force_train_set:
        print(f"[loader] force_train_iks: {len(_force_train_set)} IKs — scaffold groups assigned to train by splitter")
    if _force_test_set:
        print(f"[loader] force_test_scaffolds: {len(force_test_graphs)} compounds pre-allocated to test "
              f"({len(_force_test_set)} unique scaffolds)")

    # Layout: force-train [0..n_ft-1] | split pool [n_ft..n_ft+n_sp-1] | force-test [n_ft+n_sp..]
    n_force_test = len(force_test_graphs)
    graph_list   = all_graphs + force_test_graphs
    valid_smiles = all_smiles_list + force_test_smiles
    valid_labels = all_labels + force_test_labels
    n_pool = len(all_graphs)

    if not graph_list:
        raise ValueError(f"No valid molecular graphs could be built from {task_path}")

    dataset = ChemGraphDataset(graph_list)
    if len(dataset) != len(valid_smiles):
        raise RuntimeError(
            f"Graph dataset / SMILES length mismatch: {len(dataset)} vs {len(valid_smiles)}"
        )
    frac_test = round(1.0 - frac_train - frac_val, 10)

    ik_list_for_split = all_iks if _force_train_set else None

    if split in ("scaffold", "label_balanced_scaffold", "random_scaffold"):
        kw = dict(frac_train=frac_train, frac_valid=frac_val, frac_test=frac_test)
        if split == "scaffold":
            s_train, s_val, s_test = scaffold_split_df(
                all_smiles_list, **kw,
                force_train_iks=_force_train_set or None,
                ik_list=ik_list_for_split,
            )
        elif split == "label_balanced_scaffold":
            s_train, s_val, s_test = label_balanced_scaffold_split_df(
                all_smiles_list,
                np.asarray(all_labels, dtype=float),
                seed=seed,
                n_trials=label_balance_trials,
                min_valid_labeled_per_task=min_valid_labeled_per_task,
                min_test_labeled_per_task=min_test_labeled_per_task,
                min_valid_positive_per_task=min_valid_positive_per_task,
                min_test_positive_per_task=min_test_positive_per_task,
                min_valid_negative_per_task=min_valid_negative_per_task,
                min_test_negative_per_task=min_test_negative_per_task,
                force_train_iks=_force_train_set or None,
                ik_list=ik_list_for_split,
                **kw,
            )
        else:
            s_train, s_val, s_test = random_scaffold_split_df(all_smiles_list, seed=seed, **kw)
        train_idx = s_train
        valid_idx = s_val
        test_idx  = s_test + list(range(n_pool, n_pool + n_force_test))
        split_indices = (train_idx, valid_idx, test_idx)

    else:
        pool = list(range(n_pool))
        tmp_idx, test_idx_split = train_test_split(pool, test_size=frac_test, random_state=seed)
        train_idx, valid_idx = train_test_split(
            tmp_idx, test_size=frac_val / (frac_train + frac_val), random_state=seed,
        )
        test_idx = list(test_idx_split) + list(range(n_pool, n_pool + n_force_test))
        split_indices = (train_idx, valid_idx, test_idx)

    if len(task_cols) == 1:
        pos_weight = _compute_pos_weight_indices(dataset, train_idx, 0)
    else:
        pos_weight = torch.stack([
            _compute_pos_weight_indices(dataset, train_idx, i)
            for i in range(len(task_cols))
        ]).squeeze(-1)

    pw_str = f"{pos_weight.tolist()}" if pos_weight.numel() > 1 else f"{pos_weight.item():.3f}"
    print(
        f"[loader] {os.path.basename(task_path)}  "
        f"total={len(dataset)}  train={len(train_idx)}  "
        f"val={len(valid_idx)}  test={len(test_idx)}"
        + (f" (incl. {n_force_test} force-test)" if n_force_test else "")
        + f"  pos_weight={pw_str}"
    )

    split = _make_graph_split(dataset, split_indices, batch_size)

    if return_index:
        return split, pos_weight, split_indices
    
    return split, pos_weight


def _load_chem_dataset_from_path(save_path):
    dataset = load_pickle(save_path)

    return dataset


def _save_chem_dataset_from_path(dataset, save_path):
    save_pickle(dataset, save_path)


def _expected_task_dim(config) -> int:
    return int(config.get("task_dim") or (len(config["task_cols"]) if isinstance(config["task_cols"], list) else 1))


def _is_single_output_binary(config) -> bool:
    return config.get("task_mode") == "binary_classification" and _expected_task_dim(config) == 1


def _cache_meta_path(save_path: str) -> str:
    return f"{save_path}.meta.json"


def _cache_meta_matches(save_path: str, config: dict, cache_kind: Optional[str] = None) -> bool:
    meta_path = _cache_meta_path(save_path)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    if cache_kind is not None and meta.get("cache_kind") != cache_kind:
        return False
    return (
        meta.get("task_mode") == config.get("task_mode")
        and meta.get("task_dim") == _expected_task_dim(config)
        and meta.get("task_cols") == list(config["task_cols"])
        and meta.get("split") == config.get("split")
        and meta.get("frac_train") == config.get("frac_train")
        and meta.get("frac_val") == config.get("frac_val")
        and meta.get("label_balance_trials") == config.get("label_balance_trials")
        and meta.get("min_valid_labeled_per_task") == config.get("min_valid_labeled_per_task")
        and meta.get("min_test_labeled_per_task") == config.get("min_test_labeled_per_task")
        and meta.get("min_valid_positive_per_task") == config.get("min_valid_positive_per_task")
        and meta.get("min_test_positive_per_task") == config.get("min_test_positive_per_task")
        and meta.get("min_valid_negative_per_task") == config.get("min_valid_negative_per_task")
        and meta.get("min_test_negative_per_task") == config.get("min_test_negative_per_task")
        and meta.get("downsample_frac") == config.get("downsample_frac")
        and meta.get("protected_iks_count") == len(config.get("protected_iks") or [])
        and meta.get("exclude_scaffolds_count", 0) == len(config.get("exclude_scaffolds") or [])
        and meta.get("exclude_scaffolds_hash", "") == _exclude_scaffolds_hash(config.get("exclude_scaffolds"))
        and meta.get("exclude_iks_count", 0) == len(config.get("exclude_iks") or [])
        and meta.get("exclude_iks_hash", "") == _exclude_iks_hash(config.get("exclude_iks") or [])
        and meta.get("force_train_iks_count", 0) == len(config.get("force_train_iks") or [])
        and meta.get("force_train_iks_hash", "") == _exclude_iks_hash(config.get("force_train_iks") or [])
        and meta.get("force_test_scaffolds_count", 0) == len(config.get("force_test_scaffolds") or [])
        and meta.get("force_test_scaffolds_hash", "") == _exclude_scaffolds_hash(config.get("force_test_scaffolds") or [])
    )


def _write_cache_meta(save_path: str, config: dict, cache_kind: Optional[str] = None) -> None:
    meta = {
        "task_mode": config.get("task_mode"),
        "task_dim": _expected_task_dim(config),
        "task_cols": list(config["task_cols"]),
        "split": config.get("split"),
        "frac_train": config.get("frac_train"),
        "frac_val": config.get("frac_val"),
        "label_balance_trials": config.get("label_balance_trials"),
        "min_valid_labeled_per_task": config.get("min_valid_labeled_per_task"),
        "min_test_labeled_per_task": config.get("min_test_labeled_per_task"),
        "min_valid_positive_per_task": config.get("min_valid_positive_per_task"),
        "min_test_positive_per_task": config.get("min_test_positive_per_task"),
        "min_valid_negative_per_task": config.get("min_valid_negative_per_task"),
        "min_test_negative_per_task": config.get("min_test_negative_per_task"),
        "downsample_frac": config.get("downsample_frac"),
        "protected_iks_count": len(config.get("protected_iks") or []),
        "exclude_scaffolds_count": len(config.get("exclude_scaffolds") or []),
        "exclude_scaffolds_hash": _exclude_scaffolds_hash(config.get("exclude_scaffolds")),
        "exclude_iks_count": len(config.get("exclude_iks") or []),
        "exclude_iks_hash": _exclude_iks_hash(config.get("exclude_iks") or []),
        "force_train_iks_count": len(config.get("force_train_iks") or []),
        "force_train_iks_hash": _exclude_iks_hash(config.get("force_train_iks") or []),
        "force_test_scaffolds_count": len(config.get("force_test_scaffolds") or []),
        "force_test_scaffolds_hash": _exclude_scaffolds_hash(config.get("force_test_scaffolds") or []),
    }
    if cache_kind is not None:
        meta["cache_kind"] = cache_kind
    with open(_cache_meta_path(save_path), "w") as f:
        json.dump(meta, f, indent=2)


def _should_rebuild_without_loading(
    save_path: str,
    config: dict,
    cache_kind: Optional[str] = None,
) -> bool:
    if not os.path.exists(save_path):
        return True
    if _cache_meta_matches(save_path, config, cache_kind=cache_kind):
        return False
    if cache_kind is not None:
        print(f"[loader] Existing split cache metadata does not match {cache_kind}. Rebuilding.")
        return True
    meta_path = _cache_meta_path(save_path)
    if not os.path.exists(meta_path) and (
        config.get("exclude_iks") or config.get("exclude_scaffolds") or config.get("force_train_iks")
    ):
        print(
            "[loader] Cache file has no metadata but exclusion/force-train params are set. Rebuilding."
        )
        return True
    if config.get("downsample_frac") is not None:
        print("[loader] Existing split cache has no matching downsample metadata. Rebuilding.")
        return True
    if _is_single_output_binary(config):
        size_gb = os.path.getsize(save_path) / (1024 ** 3)
        if size_gb > float(config.get("max_split_cache_load_gb", 5)):
            print(
                f"[loader] Existing split cache is {size_gb:.1f} GiB and has no matching metadata. "
                "Rebuilding without loading it."
            )
            return True
    return False


def _build_graph_cache_payload(dataset, split_indices, pos_weight_train):
    return {
        "cache_kind": "graph_dataset_v3",
        "data": dataset._data,
        "slices": dataset.slices,
        "split_indices": split_indices,
        "pos_weight": pos_weight_train,
    }


def _load_graph_cache_payload(save_path: str, batch_size: int):
    payload = _load_chem_dataset_from_path(save_path)
    if isinstance(payload, dict) and payload.get("cache_kind") == "graph_dataset_v3":
        dataset = ChemGraphDataset(data=payload["data"], slices=payload["slices"])
        split_indices = payload["split_indices"]
        split = _make_graph_split(dataset, split_indices, batch_size)
        return split, payload["pos_weight"], split_indices

    if isinstance(payload, dict) and payload.get("cache_kind") == "graph_dataset_v2":
        dataset = payload["dataset"]
        if hasattr(dataset, "_data_list"):
            dataset._data_list = None
        split_indices = payload["split_indices"]
        split = _make_graph_split(dataset, split_indices, batch_size)
        return split, payload["pos_weight"], split_indices

    # Legacy cache format: (DataSplit(loaders), pos_weight, split_indices).
    split, pos_weight_train, split_indices = payload
    return split, pos_weight_train, split_indices


def load_chem_graph_dataset_from_config(
    config,
    return_index=False,
    print_sample_size=True,
    filter_iks=None,
    cache_only: bool = False,
):
    expected_task_dim = _expected_task_dim(config)
    needs_rebuild = _should_rebuild_without_loading(
        config["split_graph_path"],
        config,
        cache_kind="graph_dataset_v3",
    )
    if cache_only and not needs_rebuild:
        return None

    if not needs_rebuild:
        split, pos_weight_train, split_indices = _load_graph_cache_payload(
            config["split_graph_path"],
            config.get("gnn_batch_size", config.get("batch_size", 64)),
        )
        cached_task_dim = split.train.dataset[0].y.shape[-1]
        if cached_task_dim != expected_task_dim:
            print(f"[loader] Cached graph split y dim {cached_task_dim} != expected {expected_task_dim}. Rebuilding.")
            needs_rebuild = True

    if needs_rebuild:
        graph_batch_size = config.get("gnn_batch_size", config.get("batch_size", 64))
        split, pos_weight_train, split_indices = load_chem_graph_dataset(
            task_path=config["task_path"],
            smiles_col=config["smiles_col"],
            task_cols=config["task_cols"],
            frac_train=config["frac_train"],
            frac_val=config["frac_val"],
            seed=42,
            split=config["split"],
            batch_size=graph_batch_size,
            return_index=True,
            filter_iks=filter_iks,
            ik_col=config.get("inchikey_col"),
            downsample_frac=config.get("downsample_frac"),
            protected_iks=config.get("protected_iks"),
            exclude_scaffolds=config.get("exclude_scaffolds"),
            exclude_iks=config.get("exclude_iks"),
            force_train_iks=config.get("force_train_iks"),
            force_test_scaffolds=config.get("force_test_scaffolds"),
            label_balance_trials=int(config.get("label_balance_trials", 1000)),
            min_valid_labeled_per_task=int(config.get("min_valid_labeled_per_task", 10)),
            min_test_labeled_per_task=int(config.get("min_test_labeled_per_task", 10)),
            min_valid_positive_per_task=int(config.get("min_valid_positive_per_task", 1)),
            min_test_positive_per_task=int(config.get("min_test_positive_per_task", 1)),
            min_valid_negative_per_task=int(config.get("min_valid_negative_per_task", 1)),
            min_test_negative_per_task=int(config.get("min_test_negative_per_task", 1)),
        )
        dataset = split.train.dataset.dataset
        _save_chem_dataset_from_path(
            _build_graph_cache_payload(dataset, split_indices, pos_weight_train),
            config["split_graph_path"],
        )
        _write_cache_meta(config["split_graph_path"], config, cache_kind="graph_dataset_v3")
        if cache_only:
            del split
            del dataset
            return None
    
    if print_sample_size:
        print(len(split.train.dataset))
        print(len(split.valid.dataset))
        print(len(split.test.dataset))
    
    if return_index:
        return split, pos_weight_train, split_indices
    else:
        return split, pos_weight_train


def load_chem_dataset(
    task_path: str,
    smiles_col: str,
    task_cols: Union[str, List[str]],
    split: str = "scaffold",
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    seed: int = 42,
    return_index: bool = True,
    filter_iks: Optional[set] = None,
    ik_col: Optional[str] = None,
    downsample_frac: Optional[float] = None,
    protected_iks: Optional[set] = None,
    exclude_scaffolds: Optional[set] = None,
    exclude_iks: Optional[set] = None,
    force_train_iks: Optional[set] = None,
    force_test_scaffolds: Optional[set] = None,
    label_balance_trials: int = 1000,
    min_valid_labeled_per_task: int = 10,
    min_test_labeled_per_task: int = 10,
    min_valid_positive_per_task: int = 1,
    min_test_positive_per_task: int = 1,
    min_valid_negative_per_task: int = 1,
    min_test_negative_per_task: int = 1,
    ) -> Tuple[DataSplit, torch.Tensor]:
    """
    Load a CSV and split into train/val/test DataFrames for Transformer training.

    Splitting delegates entirely to splitter.py:
      scaffold        → scaffold_split_df()
      label_balanced_scaffold → scaffold-disjoint split selected to preserve
                        endpoint label/class support in validation/test
      random_scaffold → random_scaffold_split_df()
      random          → sklearn train_test_split (no stratification —
                        multi-label targets make stratification ill-defined)

    Returns
    ───────
    (DataSplit, pos_weight)
    DataSplit.train / .valid / .test are DataFrames.
    pos_weight is computed from the training split only, using the first
    task column (single-task) or each column independently (multi-task).
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

    task_cols   = [task_cols] if isinstance(task_cols, str) else list(task_cols)
    import csv as _csv
    with open(task_path, newline="") as _f:
        sep = _csv.Sniffer().sniff(_f.read(4096)).delimiter
    df = pd.read_csv(task_path, sep=sep)
    need_ik_col = (
        filter_iks is not None
        or exclude_iks is not None
        or protected_iks is not None
        or force_train_iks is not None
    ) and ik_col is not None
    select_cols = [smiles_col] + task_cols + ([ik_col] if need_ik_col and ik_col not in [smiles_col] + task_cols else [])
    df = df[select_cols].dropna(subset=[smiles_col]).reset_index(drop=True)
    if len(task_cols) == 1:
        before = len(df)
        df = df.dropna(subset=task_cols).reset_index(drop=True)
        print(f"[loader] task label filter ({task_cols[0]}): {before} → {len(df)} molecules")
    if filter_iks is not None and ik_col is not None:
        before = len(df)
        df = df[df[ik_col].apply(_ik_first_local).isin(filter_iks)].reset_index(drop=True)
        print(f"[loader] IK pre-filter: {before} → {len(df)} molecules")
    df = _exclude_scaffolds_df(
        df,
        smiles_col,
        exclude_scaffolds=exclude_scaffolds,
        context="df split",
    )
    df = _exclude_iks_df(df, ik_col, exclude_iks=exclude_iks, context="df split")
    df = _downsample_with_protected_iks(df, ik_col, downsample_frac, protected_iks, seed)
    frac_test   = round(1.0 - frac_train - frac_val, 10)

    _force_train_set = (
        {_ik_first_local(str(ik)) for ik in force_train_iks if ik is not None}
        if force_train_iks is not None and ik_col is not None and ik_col in df.columns
        else set()
    )
    _force_test_set = _normalise_excluded_scaffolds(force_test_scaffolds) if force_test_scaffolds else set()

    # force_test_scaffolds: pre-allocate to test before the split (these compounds
    # are not part of the scaffold-group logic in the splitter).
    if _force_test_set:
        all_scaffolds = df[smiles_col].astype(str).map(
            lambda s: generate_scaffold(s, include_chirality=True)
        )
        force_test_mask = all_scaffolds.isin(_force_test_set)
        force_test_df = df.loc[force_test_mask].copy()
        df = df.loc[~force_test_mask].reset_index(drop=True)
        print(
            f"[loader] force_test_scaffolds: {len(force_test_df)} compounds "
            f"pre-allocated to test ({len(_force_test_set)} unique scaffolds)"
        )
    else:
        force_test_df = df.iloc[0:0].copy()

    # Pass force_train_iks and the per-row IK list directly into the splitter so
    # that any scaffold group containing a forced IK is assigned to train in its
    # entirety — no pre-allocation or post-hoc patching required.
    smiles_list = df[smiles_col].astype(str).tolist()
    ik_list = (
        df[ik_col].apply(_ik_first_local).tolist()
        if _force_train_set and ik_col and ik_col in df.columns
        else None
    )

    if split == "scaffold":
        train_idx, valid_idx, test_idx = scaffold_split_df(
            smiles_list, frac_train, frac_val, frac_test,
            force_train_iks=_force_train_set or None,
            ik_list=ik_list,
        )
    elif split == "label_balanced_scaffold":
        train_idx, valid_idx, test_idx = label_balanced_scaffold_split_df(
            smiles_list,
            df[task_cols].to_numpy(dtype=float),
            frac_train=frac_train,
            frac_valid=frac_val,
            frac_test=frac_test,
            seed=seed,
            n_trials=label_balance_trials,
            min_valid_labeled_per_task=min_valid_labeled_per_task,
            min_test_labeled_per_task=min_test_labeled_per_task,
            min_valid_positive_per_task=min_valid_positive_per_task,
            min_test_positive_per_task=min_test_positive_per_task,
            min_valid_negative_per_task=min_valid_negative_per_task,
            min_test_negative_per_task=min_test_negative_per_task,
            force_train_iks=_force_train_set or None,
            ik_list=ik_list,
        )
    elif split == "random_scaffold":
        train_idx, valid_idx, test_idx = random_scaffold_split_df(
            smiles_list, frac_train, frac_val, frac_test, seed=seed,
        )
    else:  # random
        tmp_idx, test_idx = train_test_split(
            list(range(len(df))), test_size=frac_test, random_state=seed,
        )
        train_idx, valid_idx = train_test_split(
            tmp_idx, test_size=frac_val / (frac_train + frac_val), random_state=seed,
        )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df  = df.iloc[valid_idx].reset_index(drop=True)
    test_df   = pd.concat(
        [df.iloc[test_idx], force_test_df], axis=0
    ).reset_index(drop=True)

    split_indices = (
        list(range(len(train_df))),
        list(range(len(train_df), len(train_df) + len(valid_df))),
        list(range(len(train_df) + len(valid_df), len(train_df) + len(valid_df) + len(test_df))),
    )

    # pos_weight per task column — stack into a tensor for multi-task use
    if len(task_cols) == 1:
        pos_weight_train = _compute_pos_weight_df(train_df, task_cols[0])
    else:
        pos_weight_train = torch.stack([
            _compute_pos_weight_df(train_df, col) for col in task_cols
        ]).squeeze(-1)

    print(
        f"[loader] {os.path.basename(task_path)}: {len(df)} valid molecules\n"
        f"train={len(train_df)} | val={len(valid_df)} | test={len(test_df)}"
    )

    split = DataSplit(train=train_df, valid=valid_df, test=test_df)

    if return_index:
        return split, pos_weight_train, split_indices
    else:
        return split, pos_weight_train


def expand_biochem_split_to_full_chem(
    biochem_split: DataSplit,
    task_path: str,
    smiles_col: str,
    task_cols: List[str],
    ik_col: Optional[str] = None,
    split_df_path: Optional[str] = None,
) -> Tuple["DataSplit", torch.Tensor]:
    """
    Expand a bio+chem scaffold split (LINCS×Tox21 intersection) to the full
    Tox21 chemical dataset by scaffold-consistent fold assignment.

    Each full-Tox21 compound is assigned to the fold whose scaffold group it
    belongs to in the bio+chem split.  Compounds with a scaffold not seen in
    the bio+chem split are placed in train (the majority fold).

    Parameters
    ----------
    biochem_split : DataSplit whose .train/.valid/.test are DataFrames
    task_path     : path to the full Tox21 CSV
    smiles_col    : SMILES column name
    task_cols     : task label column names
    ik_col        : InChIKey column (kept in output DataFrames if present)
    split_df_path : if given, save the resulting DataSplit tuple here
    """
    import csv as _csv

    # Return cached split if it already exists on disk.
    if split_df_path and os.path.exists(split_df_path):
        print(f"[loader] expand_biochem_split_to_full_chem: loading cached split from {split_df_path}")
        cached_split, cached_pos_weight, _ = load_pickle(split_df_path)
        return cached_split, cached_pos_weight

    # 1. Build scaffold → fold from bio+chem split.
    # Process test first so test scaffolds take priority over train scaffolds
    # when the same Murcko scaffold appears in both (scaffold-disjoint splits
    # guarantee this never happens, but be safe).
    scaffold_to_fold: dict = {}
    for fold_name, fold_df in (("test",  biochem_split.test),
                                ("valid", biochem_split.valid),
                                ("train", biochem_split.train)):
        for smi in fold_df[smiles_col].astype(str):
            sc = generate_scaffold(smi, include_chirality=True)
            if sc is not None and sc not in scaffold_to_fold:
                scaffold_to_fold[sc] = fold_name

    # 2. Load and filter full Tox21 CSV
    with open(task_path, newline="") as _f:
        sep = _csv.Sniffer().sniff(_f.read(4096)).delimiter
    df_full = pd.read_csv(task_path, sep=sep)
    keep_cols = [smiles_col] + task_cols
    if ik_col and ik_col in df_full.columns and ik_col not in keep_cols:
        keep_cols.append(ik_col)
    df_full = df_full[[c for c in keep_cols if c in df_full.columns]]
    df_full = df_full.dropna(subset=[smiles_col]).reset_index(drop=True)

    # 3. Assign each compound to a fold
    folds: List[str] = []
    for smi in df_full[smiles_col].astype(str):
        sc = generate_scaffold(smi, include_chirality=True)
        folds.append(scaffold_to_fold.get(sc, "train") if sc is not None else "train")
    df_full["_fold"] = folds

    train_df = df_full[df_full["_fold"] == "train"].drop(columns="_fold").reset_index(drop=True)
    valid_df = df_full[df_full["_fold"] == "valid"].drop(columns="_fold").reset_index(drop=True)
    test_df  = df_full[df_full["_fold"] == "test"].drop(columns="_fold").reset_index(drop=True)

    print(
        f"[loader] expand_biochem_split_to_full_chem: {len(df_full)} total\n"
        f"  scaffold_to_fold: {len(scaffold_to_fold)} scaffolds from bio+chem split\n"
        f"  train={len(train_df)} | val={len(valid_df)} | test={len(test_df)}"
    )

    if len(task_cols) == 1:
        pos_weight = _compute_pos_weight_df(train_df, task_cols[0])
    else:
        pos_weight = torch.stack([
            _compute_pos_weight_df(train_df, col) for col in task_cols
        ]).squeeze(-1)

    split = DataSplit(train=train_df, valid=valid_df, test=test_df)
    if split_df_path:
        os.makedirs(os.path.dirname(split_df_path), exist_ok=True)
        save_pickle((split, pos_weight, None), split_df_path)
        print(f"[loader] saved expanded split → {split_df_path}")

    return split, pos_weight


def filter_chem_graph_split_by_inchikey(
    split: DataSplit,
    bio_iks: set,
    task_cols: List[str],
    ik_attr: str = "ik",
) -> tuple:
    """
    Restrict each split partition to molecules whose InChIKey
    (first block, connectivity only) appears in bio_iks.
 
    Parameters
    ----------
    split       : DataSplit with PyGDataLoader train/valid/test
    bio_iks     : set of first-block InChIKeys from the LINCS adata
    task_cols   : list of task column names (for pos_weight computation)
    ik_attr     : attribute name on each PyG Data object holding the IK
                  (default: "ik" — set to match your graph dataset)
    """
    def _ik_first(ik: str) -> str:
        if isinstance(ik, str) and "-" in ik:
            return ik.split("-")[0]
        return ik
 
    def _filter(loader, shuffle: bool) -> PyGDataLoader:
        ds       = loader.dataset
        filtered = [
            ds[i] for i in range(len(ds))
            if _ik_first(getattr(ds[i], ik_attr, "")) in bio_iks
        ]
        new_ds = ChemGraphDataset(filtered)
        print(f"[loader] graph IK filter: {len(ds)} → {len(new_ds)}")
        return PyGDataLoader(new_ds, batch_size=loader.batch_size, shuffle=shuffle)
 
    if len(task_cols) == 1:
        pos_weight_train = _compute_pos_weight_ds(split.train.dataset, 0)
    else:
        pos_weight_train = torch.stack([
            _compute_pos_weight_ds(split.train.dataset, i)
            for i in range(len(task_cols))
        ]).squeeze(-1)
 
    split = DataSplit(
        train=_filter(split.train, shuffle=True),
        valid=_filter(split.valid, shuffle=False),
        test=_filter(split.test,  shuffle=False),
    )
    return split, pos_weight_train
 
 
def filter_chem_df_split_by_inchikey(
    split: DataSplit,
    bio_iks: set,
    ik_col: str,
    task_cols: List[str],
) -> tuple:
    """
    Restrict each split partition (DataFrames) to rows whose
    InChIKey (first block) appears in bio_iks.
 
    Parameters
    ----------
    split     : DataSplit with pd.DataFrame train/valid/test
    bio_iks   : set of first-block InChIKeys from the LINCS adata
    ik_col    : column name in the DataFrame holding the InChIKey
    task_cols : list of task column names (for pos_weight computation)
    """
    def _ik_first(ik):
        if isinstance(ik, str) and "-" in ik:
            return ik.split("-")[0]
        return ik
 
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        mask     = df[ik_col].apply(_ik_first).isin(bio_iks)
        filtered = df[mask].reset_index(drop=True)
        print(f"[loader] df IK filter: {len(df)} → {len(filtered)}")
        return filtered
 
    if len(task_cols) == 1:
        pos_weight_train = _compute_pos_weight_df(split.train, task_cols[0])
    else:
        pos_weight_train = torch.stack([
            _compute_pos_weight_df(split.train, col) for col in task_cols
        ]).squeeze(-1)
 
    split = DataSplit(
        train=_filter(split.train),
        valid=_filter(split.valid),
        test=_filter(split.test),
    )
    return split, pos_weight_train


def filter_chem_graph_split_by_smiles(split: DataSplit, bio_smiles: set, task_cols: List[str]) -> DataSplit:
    """Restrict each split partition to molecules whose SMILES appear in bio_smiles."""

    def _filter(loader, shuffle: bool) -> PyGDataLoader:
        ds = loader.dataset
        filtered = [ds[i] for i in range(len(ds)) if ds[i].smiles in bio_smiles]
        new_ds = ChemGraphDataset(filtered)
        print(f"[loader] graph filter: {len(ds)} → {len(new_ds)}")
        return PyGDataLoader(new_ds, batch_size=loader.batch_size, shuffle=shuffle)
    
    if len(task_cols) == 1:
        pos_weight_train = _compute_pos_weight_ds(
            split.train.dataset,
            0,
        )
    else:
        pos_weight_train = torch.stack([
            _compute_pos_weight_ds(
                split.train.dataset,
                i,
            )
            for i in range(len(task_cols))
        ]).squeeze(-1)

    split = DataSplit(
        train=_filter(split.train, shuffle=True),
        valid=_filter(split.valid, shuffle=False),
        test=_filter(split.test, shuffle=False),
    )

    return split, pos_weight_train


def filter_chem_df_split_by_smiles(split: DataSplit, bio_smiles: set, smiles_col: str, task_cols: List[str]) -> DataSplit:
    """Restrict each split partition to rows whose SMILES appear in bio_smiles."""

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[df[smiles_col].isin(bio_smiles)].reset_index(drop=True)
        print(f"[loader] df filter: {len(df)} → {len(filtered)}")
        return filtered

    # pos_weight per task column — stack into a tensor for multi-task use
    if len(task_cols) == 1:
        pos_weight_train = _compute_pos_weight_df(split.train, task_cols[0])
    else:
        pos_weight_train = torch.stack([
            _compute_pos_weight_df(split.train, col) for col in task_cols
        ]).squeeze(-1)

    split = DataSplit(
        train=_filter(split.train),
        valid=_filter(split.valid),
        test=_filter(split.test),
    )

    return split, pos_weight_train

def load_chem_dataset_from_config(config: dict, return_index=False, print_sample_size=True, filter_iks=None) -> Tuple[DataSplit, torch.Tensor]:
    """
    Config-dict wrapper around load_chem_dataset().

    Expected config keys
    ────────────────────
    task_path, smiles_col, task_cols, split, frac_train, frac_val, seed
    """
    expected_cols = set([config["smiles_col"]] + list(config["task_cols"]))
    needs_rebuild = _should_rebuild_without_loading(config["split_df_path"], config)
    if not needs_rebuild:
        split, pos_weight_train, split_indices = _load_chem_dataset_from_path(config["split_df_path"])
        cached_cols = set(split.train.columns)
        if not expected_cols.issubset(cached_cols):
            print(f"[loader] Cached df split columns {sorted(cached_cols)} do not match "
                  f"expected {sorted(expected_cols)}. Rebuilding.")
            needs_rebuild = True

    if needs_rebuild:
        split, pos_weight_train, split_indices = load_chem_dataset(
            task_path    = config["task_path"],
            smiles_col   = config["smiles_col"],
            task_cols    = config["task_cols"],
            split        = config["split"],
            frac_train   = config["frac_train"],
            frac_val     = config["frac_val"],
            seed         = 42,
            return_index = True,
            filter_iks   = filter_iks,
            ik_col       = config.get("inchikey_col"),
            downsample_frac = config.get("downsample_frac"),
            protected_iks   = config.get("protected_iks"),
            exclude_scaffolds = config.get("exclude_scaffolds"),
            exclude_iks  = config.get("exclude_iks"),
            force_train_iks = config.get("force_train_iks"),
            force_test_scaffolds = config.get("force_test_scaffolds"),
            label_balance_trials = int(config.get("label_balance_trials", 1000)),
            min_valid_labeled_per_task = int(config.get("min_valid_labeled_per_task", 10)),
            min_test_labeled_per_task = int(config.get("min_test_labeled_per_task", 10)),
            min_valid_positive_per_task = int(config.get("min_valid_positive_per_task", 1)),
            min_test_positive_per_task = int(config.get("min_test_positive_per_task", 1)),
            min_valid_negative_per_task = int(config.get("min_valid_negative_per_task", 1)),
            min_test_negative_per_task = int(config.get("min_test_negative_per_task", 1)),
        )
        _save_chem_dataset_from_path((split, pos_weight_train, split_indices), config["split_df_path"])
        _write_cache_meta(config["split_df_path"], config)
    
    if print_sample_size:
        print(len(split.train))
        print(len(split.valid))
        print(len(split.test))
    
    if return_index:
        return split, pos_weight_train, split_indices
    else:
        return split, pos_weight_train
