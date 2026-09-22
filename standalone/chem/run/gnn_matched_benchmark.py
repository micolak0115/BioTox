# chem/gnn_matched_benchmark.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).parents[1]))

from chem.benchmark import (
    _load_cached_metrics,
    _load_cached_seeds,
    _metrics_cache_path,
    _run_seed_jobs,
    _save_cached_metrics,
)
from chem.featurizer import ATOM_FEATURE_DIM
from chem.loader import smiles_to_graph
from chem.model import GAT, GCN, GIN, GraphSAGE
from chem.splitter import DataSplit
from chem.trainer import compute_pos_weight
from chem.utils import seed_all
from publication.utils import load_matched_split

GNN_CLASSES = {"GCN": GCN, "GAT": GAT, "GIN": GIN, "GraphSAGE": GraphSAGE}
DEFAULT_EXPECTED_SPLIT_SIZES = {"train": 412, "valid": 137, "test": 138}


def _assert_split_sizes(matched_df, expected: dict) -> None:
    sizes = matched_df["split"].value_counts().to_dict()
    for split_name, n in expected.items():
        actual = sizes.get(split_name, 0)
        if actual != n:
            raise ValueError(
                f"matched split size mismatch: expected {split_name}={n}, got {actual}. "
                "Wrong split file -- must point at the linear_publication "
                "412/137/138 scaffold split."
            )


def _build_loader(df, task_cols: list, smiles_col: str, batch_size: int, shuffle: bool):
    from torch_geometric.loader import DataLoader as PyGDataLoader

    data_list = []
    y = df[task_cols].to_numpy(dtype=np.float64)
    for smi, y_row in zip(df[smiles_col].tolist(), y):
        g = smiles_to_graph(smi)
        if g is None:
            g = smiles_to_graph("C")
        # unsqueeze to (1, n_tasks): PyG's default collation concatenates
        # graph-level tensor attributes along dim 0, so an unbatched (12,)
        # y would flatten across the batch into (batch_size*12,) instead of
        # stacking into (batch_size, 12) -- confirmed by a shape-mismatch
        # crash in BCEWithLogitsLoss without this.
        g.y = torch.tensor(y_row, dtype=torch.float).unsqueeze(0)
        data_list.append(g)
    return PyGDataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def _run_gnn_matched_seed(model_name: str, gnn_cls, seed: int, config: dict, matched_df) -> tuple:
    tag = "gnn_matched"
    cache_path = _metrics_cache_path(config["save_dir"], model_name, tag, seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[GNN-matched][SKIP] {model_name} seed={seed}: cached")
        return seed, cached

    task_cols = config["task_cols"]
    batch_size = config.get("gnn_batch_size", 16)

    train_df = matched_df[matched_df["split"] == "train"]
    valid_df = matched_df[matched_df["split"] == "valid"]
    test_df = matched_df[matched_df["split"] == "test"]

    try:
        seed_all(seed)
        train_loader = _build_loader(train_df, task_cols, "smiles_canon", batch_size, shuffle=True)
        valid_loader = _build_loader(valid_df, task_cols, "smiles_canon", batch_size, shuffle=False)
        test_loader = _build_loader(test_df, task_cols, "smiles_canon", batch_size, shuffle=False)
        split = DataSplit(train=train_loader, valid=valid_loader, test=test_loader)

        pos_weight = compute_pos_weight(
            train_df[task_cols].to_numpy(dtype=np.float64), "multilabel_classification"
        )

        model = gnn_cls(
            input_dim=ATOM_FEATURE_DIM,
            hidden_dim=config.get("gnn_hidden_dim", 300),
            output_dim=len(task_cols),
            n_layers=config.get("gnn_num_layers", 5),
            dropout=config.get("gnn_dropout", 0.5),
            graph_pooling=config.get("gnn_graph_pooling", "mean"),
            JK=config.get("gnn_JK", "last"),
            task_cols=task_cols,
            task_family_heads=config.get("task_family_heads", True),
            task_family_head_type=config.get("task_family_head_type", "mlp"),
            task_family_hidden_dim=config.get("task_family_hidden_dim", 256),
            task_family_dropout=config.get("task_family_dropout", 0.1),
            task_mode="multilabel_classification",
            lr=config.get("gnn_lr", 1e-3),
            epochs=config.get("gnn_epochs", 100),
            seed=seed,
            save_path=None,
            patience=config.get("gnn_patience", 20),
            min_delta=config.get("gnn_min_delta", 0.0),
            device=config.get("device"),
        )
        model.fit(split, pos_weight=pos_weight)
        metrics = model.predict(split)

        _save_cached_metrics(cache_path, metrics)
        print(f"[GNN-matched][DONE] {model_name} seed={seed}: auprc={metrics.get('auprc')} auroc={metrics.get('auroc')}")
        return seed, metrics
    except Exception as e:
        print(f"[GNN-matched][FAIL] {model_name} seed={seed}: {e}")
        import traceback

        traceback.print_exc()
        return seed, {"error": str(e)}


def gnn_matched_benchmark(config: dict, matched_split_path: str, seeds: list, models: Optional[list] = None) -> dict:
    matched_df = load_matched_split(matched_split_path)
    expected = config.get("expected_split_sizes", DEFAULT_EXPECTED_SPLIT_SIZES)
    _assert_split_sizes(matched_df, expected)

    allowed = set(models or config.get("gnn_models") or [])
    variants = {name: cls for name, cls in GNN_CLASSES.items() if not allowed or name in allowed}

    results = {}
    for model_name, gnn_cls in variants.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "gnn_matched", seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(f"\n[GNN-matched] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs = config.get("n_jobs", 1)
            jobs = [
                (lambda seed=seed: _run_gnn_matched_seed(model_name, gnn_cls, seed, config, matched_df))
                for seed in remaining
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results
