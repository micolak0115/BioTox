# chem/splitter.py
# %%
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from collections import defaultdict
from itertools import compress

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Subset


@dataclass
class DataSplit:
    train: object
    valid: object
    test: object


def generate_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
    """
    Return the Bemis-Murcko scaffold SMILES for a molecule.
    Returns None for invalid or empty SMILES.
    Example:
    smiles = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
    scaffold = generate_scaffold(smiles)
    scaffold == 'c1ccc(Oc2ccccn2)cc1'
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return None


def filter_by_task(
    smiles_list: List[str],
    dataset,
    task_idx: Optional[int],
    null_value: float,
    ) -> List[Tuple[int, str]]:
    """
    Return (original_index, smiles) pairs whose label is not null_value.
    """

    indexed_smiles = list(enumerate(smiles_list))

    if task_idx is None:
        return indexed_smiles

    mask = np.array([
        d.y[task_idx].item() for d in dataset
    ]) != null_value

    return [
        (i, s)
        for (i, s), keep in zip(indexed_smiles, mask)
        if keep
    ]


def print_split_info(train_idx, valid_idx, test_idx, n_scaffolds):
    print(
        f"[Split] valid_molecules={len(train_idx) + len(valid_idx) + len(test_idx)}  "
        f"unique_scaffolds={n_scaffolds}  "
        f"train={len(train_idx)}  valid={len(valid_idx)}  test={len(test_idx)}"
    )


def build_scaffold_dict(indexed_smiles: List[Tuple[int, str]]) -> dict:
    """Group original indices by their Murcko scaffold SMILES."""
    scaffold_dict: dict = defaultdict(list)
    for i, smiles in indexed_smiles:
        sc = generate_scaffold(smiles, include_chirality=True)
        if sc is not None:
            scaffold_dict[sc].append(i)
    return scaffold_dict


def split_dataset(
    dataset,
    train_idx: List[int],
    valid_idx: List[int],
    test_idx:  List[int],
    n_scaffolds: int,
    return_index: bool = True,
    print_info: bool = True,
    ):

    assert not set(train_idx) & set(valid_idx), "Train/valid overlap"
    assert not set(train_idx) & set(test_idx),  "Train/test overlap"
    assert not set(valid_idx) & set(test_idx),  "Valid/test overlap"

    train_ds = Subset(dataset, train_idx)
    valid_ds = Subset(dataset, valid_idx)
    test_ds  = Subset(dataset, test_idx)
    
    if print_info:
        print_split_info(train_idx, valid_idx, test_idx, n_scaffolds)

    if return_index:
        return (train_ds, valid_ds, test_ds), (train_idx, valid_idx, test_idx)
    
    return train_ds, valid_ds, test_ds


def scaffold_split(
    dataset,
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test:  float = 0.1,
    task_idx:   Optional[int] = None,
    null_value: float = 0.0,
    return_index: bool = False,
    ):
    """
    Deterministic Bemis-Murcko scaffold split for a PyG dataset.
    Scaffold groups are sorted largest → smallest and greedily assigned to
    train first, then valid, then test. This maximises scaffold diversity in
    the test set and is the standard MoleculeNet benchmark protocol
    (Wu et al., 2018).

    Parameters
    ----------
    dataset     : PyG dataset object
    smiles_list : SMILES strings corresponding to dataset indices
    task_idx    : column index of data.y to filter on (None = no filter)
    null_value  : label value treated as missing when task_idx is set
    return_index: if True, also return (train_idx, valid_idx, test_idx)
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    indexed_smiles = filter_by_task(smiles_list, dataset, task_idx, null_value)
    scaffold_dict  = build_scaffold_dict(indexed_smiles)
    n_valid        = sum(len(v) for v in scaffold_dict.values())

    # Largest scaffold groups assigned to train first
    scaffold_sets = sorted(
        scaffold_dict.values(), key=lambda g: (len(g), g[0]), reverse=True
    )

    train_cutoff = frac_train * n_valid
    valid_cutoff = (frac_train + frac_valid) * n_valid

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(valid_idx) + len(group) <= valid_cutoff:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return split_dataset(
        dataset, train_idx, valid_idx, test_idx, len(scaffold_dict), return_index   
    )


def random_scaffold_split(
    dataset,
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test:  float = 0.1,
    seed: int = 42,
    task_idx:   Optional[int] = None,
    null_value: float = 0.0,
    return_index: bool = False,
    ):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Scaffold split with scaffold-group order randomised (reproducible via seed).
    Valid/test sizes are capped at floor(frac * len(dataset)) — computed
    against the *total* (unfiltered) dataset size, matching the original
    chainer-chemistry implementation. Groups that exceed the cap overflow
    to train. Scaffold grouping is preserved: no molecule from the same
    scaffold appears in two splits.

    Parameters
    ----------
    dataset     : PyG dataset object
    smiles_list : SMILES strings corresponding to dataset indices
    seed        : random seed for scaffold-group shuffle
    task_idx    : column index of data.y to filter on (None = no filter)
    null_value  : label value treated as missing when task_idx is set
    return_index: if True, also return (train_idx, valid_idx, test_idx)
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    indexed_smiles = filter_by_task(smiles_list, dataset, task_idx, null_value)
    scaffold_dict  = build_scaffold_dict(indexed_smiles)

    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_dict.values())
    scaffold_sets = [scaffold_groups[i] for i in rng.permutation(len(scaffold_groups))]

    # Quotas computed against total dataset size — intentional, matches original
    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test  = int(np.floor(frac_test  * len(dataset)))

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(valid_idx) + len(group) <= n_total_valid:
            valid_idx.extend(group)
        elif len(test_idx) + len(group) <= n_total_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return split_dataset(
        dataset, train_idx, valid_idx, test_idx, len(scaffold_dict), return_index
    )


def scaffold_split_df(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test:  float = 0.1,
    force_train_iks: Optional[set] = None,
    ik_list: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Deterministic scaffold split returning raw index lists for DataFrame.iloc[].

    If force_train_iks and ik_list are provided, any scaffold group containing
    at least one forced IK is assigned to train in its entirety before the greedy
    quota assignment runs on the remaining free groups.  This guarantees full
    scaffold disjointness between folds without any post-split patching.
    Quotas are computed over all compounds so that the stated frac_* ratios are
    preserved in the final split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    scaffold_dict = build_scaffold_dict(list(enumerate(smiles_list)))
    n_total = sum(len(v) for v in scaffold_dict.values())

    # Separate scaffold groups that contain at least one forced-train compound.
    forced_train_idx: List[int] = []
    free_groups: List[List[int]] = []
    if force_train_iks and ik_list:
        force_set = set(force_train_iks)
        for indices in scaffold_dict.values():
            if any(ik_list[i] in force_set for i in indices):
                forced_train_idx.extend(indices)
            else:
                free_groups.append(indices)
    else:
        free_groups = list(scaffold_dict.values())

    # Greedy assignment on free groups.  Quotas target the stated frac_* ratios
    # across all n_total compounds; the forced-train slots are already consumed.
    free_sorted = sorted(free_groups, key=lambda g: (len(g), g[0]), reverse=True)
    target_free_train = frac_train  * n_total - len(forced_train_idx)
    target_free_val   = frac_valid  * n_total

    free_train: List[int] = []
    valid_idx:  List[int] = []
    test_idx:   List[int] = []
    for group in free_sorted:
        if len(free_train) + len(group) <= target_free_train:
            free_train.extend(group)
        elif len(valid_idx) + len(group) <= target_free_val:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return list(forced_train_idx) + free_train, valid_idx, test_idx


def random_scaffold_split_df(
    smiles_list: List[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test:  float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Randomised scaffold split returning raw index lists for DataFrame.iloc[].
    Quotas use total smiles_list length as denominator (consistent with
    random_scaffold_split's original convention).
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    scaffold_dict = build_scaffold_dict(list(enumerate(smiles_list)))
    n_total       = len(smiles_list)

    rng = np.random.RandomState(seed)
    scaffold_groups = list(scaffold_dict.values())
    scaffold_sets = [scaffold_groups[i] for i in rng.permutation(len(scaffold_groups))]

    n_total_valid = int(np.floor(frac_valid * n_total))
    n_total_test  = int(np.floor(frac_test  * n_total))

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(valid_idx) + len(group) <= n_total_valid:
            valid_idx.extend(group)
        elif len(test_idx) + len(group) <= n_total_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return train_idx, valid_idx, test_idx


def _counts_for_indices(labels: np.ndarray, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(indices) == 0:
        n_tasks = labels.shape[1]
        z = np.zeros(n_tasks, dtype=int)
        return z, z, z
    subset = labels[np.asarray(indices, dtype=int)]
    observed = np.isfinite(subset)
    positives = observed & (subset == 1)
    negatives = observed & (subset == 0)
    return observed.sum(axis=0), positives.sum(axis=0), negatives.sum(axis=0)


def _balanced_split_score(
    split_indices: Tuple[List[int], List[int], List[int]],
    labels: np.ndarray,
    min_valid_labeled: int,
    min_test_labeled: int,
    min_valid_positive: int,
    min_test_positive: int,
    min_valid_negative: int,
    min_test_negative: int,
    frac_train: float,
    frac_valid: float,
    frac_test: float,
) -> Tuple[float, dict]:
    train_idx, valid_idx, test_idx = split_indices
    total_labeled, total_pos, total_neg = _counts_for_indices(
        labels, list(range(labels.shape[0]))
    )

    # Requirements are capped to what is feasible while preserving at least one
    # training example of that class whenever possible.
    valid_labeled_req = np.minimum(min_valid_labeled, np.floor(total_labeled * frac_valid).astype(int))
    test_labeled_req = np.minimum(min_test_labeled, np.floor(total_labeled * frac_test).astype(int))
    test_pos_req = np.minimum(min_test_positive, np.maximum(0, total_pos - 1))
    valid_pos_req = np.minimum(min_valid_positive, np.maximum(0, total_pos - 1 - test_pos_req))
    test_neg_req = np.minimum(min_test_negative, np.maximum(0, total_neg - 1))
    valid_neg_req = np.minimum(min_valid_negative, np.maximum(0, total_neg - 1 - test_neg_req))

    valid_labeled, valid_pos, valid_neg = _counts_for_indices(labels, valid_idx)
    test_labeled, test_pos, test_neg = _counts_for_indices(labels, test_idx)

    deficits = {
        "valid_labeled": np.maximum(0, valid_labeled_req - valid_labeled),
        "test_labeled": np.maximum(0, test_labeled_req - test_labeled),
        "valid_positive": np.maximum(0, valid_pos_req - valid_pos),
        "test_positive": np.maximum(0, test_pos_req - test_pos),
        "valid_negative": np.maximum(0, valid_neg_req - valid_neg),
        "test_negative": np.maximum(0, test_neg_req - test_neg),
    }
    # Positive/negative deficits dominate; labeled-count deficits are secondary.
    class_deficit = (
        deficits["valid_positive"].sum()
        + deficits["test_positive"].sum()
        + deficits["valid_negative"].sum()
        + deficits["test_negative"].sum()
    )
    labeled_deficit = deficits["valid_labeled"].sum() + deficits["test_labeled"].sum()
    n = labels.shape[0]
    size_penalty = (
        abs(len(train_idx) / n - frac_train)
        + abs(len(valid_idx) / n - frac_valid)
        + abs(len(test_idx) / n - frac_test)
    )
    score = float(1000 * class_deficit + 10 * labeled_deficit + size_penalty)
    return score, deficits


def label_balanced_scaffold_split_df(
    smiles_list: List[str],
    labels: Union[np.ndarray, List[List[float]]],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
    n_trials: int = 1000,
    min_valid_labeled_per_task: int = 10,
    min_test_labeled_per_task: int = 10,
    min_valid_positive_per_task: int = 1,
    min_test_positive_per_task: int = 1,
    min_valid_negative_per_task: int = 1,
    min_test_negative_per_task: int = 1,
    force_train_iks: Optional[set] = None,
    ik_list: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Scaffold-disjoint split selected over random scaffold assignments to retain
    enough observed labels and class examples per endpoint in valid/test folds.

    If force_train_iks and ik_list are provided, scaffold groups containing any
    forced IK are assigned to train unconditionally before the random search.
    Quotas target the stated frac_* ratios across all compounds.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    labels = np.asarray(labels, dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.shape[0] != len(smiles_list):
        raise ValueError(
            f"labels rows ({labels.shape[0]}) must match smiles_list length ({len(smiles_list)})"
        )

    scaffold_dict = build_scaffold_dict(list(enumerate(smiles_list)))
    if not scaffold_dict:
        return [], [], []

    # Separate forced-train groups from the free pool before any assignment.
    forced_train_idx: List[int] = []
    free_groups: List[List[int]] = []
    if force_train_iks and ik_list:
        force_set = set(force_train_iks)
        for indices in scaffold_dict.values():
            if any(ik_list[i] in force_set for i in indices):
                forced_train_idx.extend(indices)
            else:
                free_groups.append(indices)
    else:
        free_groups = list(scaffold_dict.values())

    n_total = len(smiles_list)
    target_free_val  = int(round(frac_valid * n_total))
    target_free_test = int(round(frac_test  * n_total))
    if target_free_val + target_free_test > n_total - len(forced_train_idx):
        overflow = target_free_val + target_free_test - (n_total - len(forced_train_idx))
        target_free_test = max(0, target_free_test - overflow)

    def _assign(scaffold_sets):
        valid_idx, test_idx, train_idx = [], [], []
        for group in scaffold_sets:
            if len(test_idx) + len(group) <= target_free_test:
                test_idx.extend(group)
            elif len(valid_idx) + len(group) <= target_free_val:
                valid_idx.extend(group)
            else:
                train_idx.extend(group)
        return list(forced_train_idx) + train_idx, valid_idx, test_idx

    deterministic_groups = sorted(
        free_groups, key=lambda g: (len(g), g[0]), reverse=True
    )
    best_split = _assign(deterministic_groups)
    best_score, best_deficits = _balanced_split_score(
        best_split,
        labels,
        min_valid_labeled_per_task,
        min_test_labeled_per_task,
        min_valid_positive_per_task,
        min_test_positive_per_task,
        min_valid_negative_per_task,
        min_test_negative_per_task,
        frac_train,
        frac_valid,
        frac_test,
    )
    if best_score == 0:
        return best_split

    rng = np.random.RandomState(seed)
    for _ in range(int(n_trials)):
        order = rng.permutation(len(free_groups))
        candidate = _assign([free_groups[i] for i in order])
        score, deficits = _balanced_split_score(
            candidate,
            labels,
            min_valid_labeled_per_task,
            min_test_labeled_per_task,
            min_valid_positive_per_task,
            min_test_positive_per_task,
            min_valid_negative_per_task,
            min_test_negative_per_task,
            frac_train,
            frac_valid,
            frac_test,
        )
        if score < best_score:
            best_split, best_score, best_deficits = candidate, score, deficits
            if best_score == 0:
                break

    residual_deficit = sum(int(np.asarray(value).sum()) for value in best_deficits.values())
    if residual_deficit > 0:
        deficit_summary = {
            key: int(np.asarray(value).sum())
            for key, value in best_deficits.items()
        }
        print(
            "[splitter][WARN] label-balanced scaffold split kept residual "
            f"deficits after {n_trials} trials: {deficit_summary}"
        )
    return best_split
