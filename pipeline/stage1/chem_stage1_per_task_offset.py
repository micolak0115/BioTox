# publication/chem_stage1_per_task_offset.py
"""Fit the fixed-seed Stage-1 chemical candidate library.

Design:
  1. PER TASK, independently (no cross-task sharing -- NR/SR heterogeneity
     means the best (fingerprint x classifier) combo can differ by
     endpoint; a single global architecture would risk negative transfer).
  2. 12 candidates per task: MACCS/Morgan/RDKit x LogReg/XGBoost (6) +
     GCN/GAT/GIN/GraphSAGE-pretrained (4) + ChemBERTa-pretrained (1) +
     Chemprop/D-MPNN (1, subprocess into the dedicated BioTox_chemprop
     -- chemprop==2.2.2 pins torch/lightning versions incompatible with
     this repo's main BioTox; see stage1/chemprop_stage1_candidate.py).
  3. ONE fixed random seed and ONE frozen scaffold-disjoint
     train/valid/test partition of the
     1,732-compound fine-tuning pool (scaffold-disjoint from every cell's
     cohort -- Phase 1), built once in build_matched_cohort.py -- NOT
     cross-validated, NOT cross-fitted. Each candidate is trained once
     with the same fixed seed on the train partition and early-stopped on
     valid.
  4. **Three partitions, three roles**: train fits candidate parameters,
     valid performs within-candidate tuning, and the Stage-1 test partition
     ranks candidates for the task-specific K=4 development choice.
  5. The final frozen prior is deployed separately by
     `deploy_stage1_k4_mean.py` as the arithmetic mean of the four
     development-selected candidates. The matched Stage-2 cohort is never
     used for this selection.
  6. Implementation efficiency: the 12 independent candidate fits are
     launched concurrently. Each fit's predict pass is run on a
     concatenation of [test, cohort_union] simultaneously (valid is never
     a prediction target, only an early-stopping monitor set), so no
     separate refit is needed to deploy the frozen mean ensemble on the
     cohort union. Total chemical-model fits: 12 candidates per task, NOT
     retrained per (cell, fold) -- structurally
     unnecessary, since the fine-tuning pool never overlaps (at the
     scaffold level) with any cohort compound in any cell in any fold, so
     inference on any cohort compound is automatically out-of-sample.

Each candidate is fit as a genuinely SINGLE-task model (task_dim=1), not
the multi-task (12-head) architecture stage1/build_chem_offset_ensemble.py
uses -- matching "per-task independent" (no cross-task parameter sharing
at all, the strongest form of avoiding negative transfer).
"""
from __future__ import annotations

import functools
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1.benchmark import _resolve_pretrained_gnn_checkpoint, _load_pretrained_gnn_backbone, _run_seed_jobs  # noqa: E402
from stage1 import classical_baseline  # noqa: E402
from stage1.classical_baseline import CLASSICAL_VARIANTS, _fit_predict_one_task, featurize  # noqa: E402
from stage1.gnn_matched_benchmark import _build_loader  # noqa: E402
from stage1.model import ChemBerta, GAT, GCN, GIN, GraphSAGE  # noqa: E402
from stage1.splitter import DataSplit  # noqa: E402
from stage1.trainer import compute_pos_weight  # noqa: E402
from stage1.utils import seed_all  # noqa: E402
from stage1.build_matched_cohort import ik_skeleton  # noqa: E402
from stage2.utils import CELL_IDS, TOX21_TASKS  # noqa: E402

GNN_CLASSES = {"GCN": GCN, "GAT": GAT, "GIN": GIN, "GraphSAGE": GraphSAGE}
GNN_ARCHS = ["GCN", "GAT", "GIN", "GraphSAGE"]

CANDIDATE_NAMES = list(CLASSICAL_VARIANTS.keys()) + [f"{a}_Pretrained" for a in GNN_ARCHS] + ["ChemBERTa_Pretrained", "Chemprop"]
ENSEMBLE_RULE = "mean"
FIXED_SEED = 1
EPS = 1e-6
MODEL_SETTINGS = {}


def configure_model_settings(candidates=None, settings=None):
    """Apply config-driven candidate and training settings before ``main``."""
    global CANDIDATE_NAMES, MODEL_SETTINGS
    if candidates is not None:
        unknown = sorted(set(candidates) - set(CLASSICAL_VARIANTS) -
                         {f"{a}_Pretrained" for a in GNN_ARCHS} -
                         {"ChemBERTa_Pretrained", "Chemprop"})
        if unknown:
            raise ValueError(f"Unknown candidate model(s): {unknown}")
        CANDIDATE_NAMES = list(candidates)
    MODEL_SETTINGS = dict(settings or {})
    classical_baseline.configure_settings(MODEL_SETTINGS.get("classical", {}))


def _setting(group, key, default):
    return MODEL_SETTINGS.get(group, {}).get(key, default)

# Parallelism: the 12 candidate models are independent fits and run
# concurrently via joblib (stage1.benchmark._run_seed_jobs, loky backend --
# separate processes, CUDA-safe). Classical candidates are CPU-only and
# internally single-threaded. GNN/ChemBERTa/Chemprop candidates round-robin
# across GPU_IDS.
# Restricted to the two RTX 3090s (24GB each) -- GPU 2 (GTX 1080 Ti) is
# intentionally excluded per explicit direction, not auto-detected.
GPU_IDS = [0, 1] if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else (
    [0] if torch.cuda.is_available() else [-1]
)
CANDIDATE_N_JOBS = len(CANDIDATE_NAMES)

COHORT_DIR = Path(__file__).parent / "_run_output" / "matched_split_rebuilt"
DEFAULT_OUT_DIR = (
    Path(__file__).parent / "_run_output" / "chem_stage1_candidates_fresh_v1"
)


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# single-task fit/predict per candidate kind
# ---------------------------------------------------------------------------
def fit_predict_classical(fp_kind: str, clf_kind: str, fit_df, predict_df, task: str, seed: int,
                           es_df=None) -> np.ndarray:
    """es_df (valid), when given, is used strictly within-candidate -- XGBoost
    early stopping via eval_set, LogisticRegression's C selected by valid
    AUPRC -- matching the same train=fit/valid=tune-only contract the
    GNN/ChemBERTa/Chemprop candidates already follow. Never used for
    cross-candidate comparison (see stage1/classical_baseline.py::_fit_predict_one_task)."""
    X_fit = featurize(fit_df["smiles_canon"].tolist(), fp_kind)
    X_pred = featurize(predict_df["smiles_canon"].tolist(), fp_kind)
    y_fit = fit_df[task].to_numpy(dtype=np.float64)
    X_valid, y_valid = None, None
    if es_df is not None and len(es_df):
        X_valid = featurize(es_df["smiles_canon"].tolist(), fp_kind)
        y_valid = es_df[task].to_numpy(dtype=np.float64)
    return _fit_predict_one_task(clf_kind, X_fit, y_fit, X_pred, seed, X_valid=X_valid, y_valid_col=y_valid)


def fit_predict_gnn_pretrained(arch: str, fit_df, es_df, predict_df, task: str, seed: int, gpu_id: int = -1) -> np.ndarray:
    from stage1.featurizer import ATOM_FEATURE_DIM

    device = f"cuda:{gpu_id}" if gpu_id is not None and gpu_id >= 0 else "cpu"
    seed_all(seed)
    batch_size = int(_setting("gnn", "batch_size", 32))
    fit_loader = _build_loader(fit_df, [task], "smiles_canon", batch_size, shuffle=True)
    es_loader = _build_loader(es_df, [task], "smiles_canon", batch_size, shuffle=False)
    pred_loader = _build_loader(predict_df, [task], "smiles_canon", batch_size, shuffle=False)

    # task_mode="binary_classification" (NOT "multilabel_classification") --
    # this is a genuinely single-task model (output_dim=1). trainer.py's
    # compute_metrics asserts 2D (N, n_tasks) preds for "multilabel_classification"
    # and crashes on this model's 1D output; "binary_classification" handles
    # 1D preds/labels correctly (and squeezes (N,1)->(N,) automatically).
    pos_weight = compute_pos_weight(fit_df[task].to_numpy(dtype=np.float64), "binary_classification")
    model = GNN_CLASSES[arch](
        input_dim=ATOM_FEATURE_DIM, hidden_dim=int(_setting("gnn", "hidden_dim", 300)), output_dim=1,
        n_layers=int(_setting("gnn", "n_layers", 5)),
        dropout=float(_setting("gnn", "dropout", 0.5)),
        graph_pooling=_setting("gnn", "graph_pooling", "mean"),
        JK=_setting("gnn", "JK", "last"), task_cols=[task],
        task_family_heads=False, task_mode="binary_classification",
        lr=float(_setting("gnn", "lr", 1e-4)),
        epochs=int(_setting("gnn", "epochs", 100)), seed=seed, save_path=None,
        patience=int(_setting("gnn", "patience", 20)),
        min_delta=float(_setting("gnn", "min_delta", 0.0)), device=device,
    )
    checkpoint_path = _resolve_pretrained_gnn_checkpoint(
        {
            "pretrain_gnn_root": os.environ.get("BIOTOX_PRETRAIN_GNN_ROOT"),
            "gnn_pretrained_variant": os.environ.get(
                "BIOTOX_GNN_PRETRAINED_VARIANT", "supervised_contextpred"
            ),
        },
        arch,
    )
    _load_pretrained_gnn_backbone(model, checkpoint_path)
    fit_split = DataSplit(train=fit_loader, valid=es_loader, test=es_loader)
    model.fit(fit_split, pos_weight=pos_weight)

    model.eval()
    dev = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for batch in pred_loader:
            batch = batch.to(dev)
            preds.append(torch.sigmoid(model(batch)).cpu())
    return torch.cat(preds).numpy().reshape(-1)


def fit_predict_chemberta_pretrained(fit_df, es_df, predict_df, task: str, seed: int, gpu_id: int = -1) -> np.ndarray:
    from stage1.finetune import predict_smiles

    device = f"cuda:{gpu_id}" if gpu_id is not None and gpu_id >= 0 else "cpu"
    fit_df = fit_df.reset_index(drop=True)
    es_df = es_df.reset_index(drop=True)

    seed_all(seed)
    pos_weight = compute_pos_weight(fit_df[task].to_numpy(dtype=np.float64), "binary_classification")
    model = ChemBerta(
        model_name=os.environ.get("BIOTOX_CHEMBERTA_MODEL", ChemBerta.DEFAULT_MODEL), task_mode="binary_classification", num_labels=1,
        max_length=int(_setting("chemberta", "max_length", 128)), frozen=False,
        smiles_col="smiles_canon", task_cols=[task], task_family_heads=False,
        seed=seed, lr=float(_setting("chemberta", "lr", 1e-4)),
        epochs=int(_setting("chemberta", "epochs", 20)),
        patience=int(_setting("chemberta", "patience", 5)),
        batch_size=int(_setting("chemberta", "batch_size", 32)), save_path=None,
        trust_remote_code=bool(_setting("chemberta", "trust_remote_code", True)),
        weight_decay=float(_setting("chemberta", "weight_decay", 0.0)),
        warmup_ratio=float(_setting("chemberta", "warmup_ratio", 0.06)),
        device=device,
    )
    fit_split = DataSplit(train=fit_df, valid=es_df, test=es_df)
    model = model.fit(fit_split, pos_weight=pos_weight)
    result = predict_smiles(
        model, predict_df["smiles_canon"].tolist(),
        batch_size=int(_setting("chemberta", "batch_size", 32)),
        max_length=int(_setting("chemberta", "max_length", 128)),
    )
    return np.asarray(result["probs"]).reshape(-1)


CHEMPROP_ENV_PYTHON = os.environ.get(
    "BIOTOX_CHEMPROP_PYTHON",
    "/data/kyungan/miniconda3/envs/BioTox_chemprop/bin/python",
)
CHEMPROP_SCRIPT = os.environ.get(
    "BIOTOX_CHEMPROP_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "stage1" / "chemprop_stage1_candidate.py"),
)


def fit_predict_chemprop(fit_df, es_df, predict_df, task: str, seed: int, gpu_id: int = -1) -> np.ndarray:
    """Chemprop (D-MPNN) runs in a SEPARATE conda env (`BioTox_chemprop`
    -- chemprop==2.2.2 pins torch==2.6.0/lightning==2.6.0, incompatible
    with this repo's main `BioTox` used for every other candidate here),
    invoked as a subprocess. Serializes fit/es/predict to temp CSVs
    (stage1/chemprop_stage1_candidate.py's interface), reads predictions
    back -- functionally the same fit_df/es_df/predict_df/task/seed/gpu_id
    contract as every other fit_predict_* candidate, just crossing a
    process boundary to get there. Single-task (`--task-col`), matching
    every other candidate's "no cross-task parameter sharing" design."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"chemprop_{task}_seed{seed}_") as tmpdir:
        fit_path = f"{tmpdir}/fit.csv"
        es_path = f"{tmpdir}/es.csv"
        predict_path = f"{tmpdir}/predict.csv"
        out_path = f"{tmpdir}/preds.csv"
        fit_df[["smiles_canon", task]].to_csv(fit_path, index=False)
        es_df[["smiles_canon", task]].to_csv(es_path, index=False)
        predict_df[["smiles_canon"]].to_csv(predict_path, index=False)

        cmd = [
            CHEMPROP_ENV_PYTHON, CHEMPROP_SCRIPT,
            "--fit-csv", fit_path, "--es-csv", es_path, "--predict-csv", predict_path,
            "--task-col", task, "--seed", str(seed), "--out-csv", out_path,
            "--gpu-id", str(gpu_id),
            "--epochs", str(int(_setting("chemprop", "epochs", 100))),
            "--patience", str(int(_setting("chemprop", "patience", 20))),
            "--batch-size", str(int(_setting("chemprop", "batch_size", 32))),
            "--hidden-dim", str(int(_setting("chemprop", "hidden_dim", 300))),
            "--depth", str(int(_setting("chemprop", "depth", 3))),
            "--dropout", str(float(_setting("chemprop", "dropout", 0.0))),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(
                f"chemprop_stage1_candidate.py failed (task={task}, seed={seed}, gpu_id={gpu_id}): "
                f"{result.stderr[-4000:]}"
            )
        preds = pd.read_csv(out_path)["pred"].to_numpy()
        if len(preds) != len(predict_df):
            raise RuntimeError(
                f"chemprop_stage1_candidate.py returned {len(preds)} predictions, expected {len(predict_df)} "
                f"(task={task}, seed={seed})"
            )
        return preds


def _is_gpu_bound(name: str) -> bool:
    return (name.endswith("_Pretrained") and name.replace("_Pretrained", "") in GNN_CLASSES) or name in (
        "ChemBERTa_Pretrained", "Chemprop",
    )


def _fit_predict_one_seed(name: str, fit_df, es_df, predict_df, task: str, seed: int, gpu_id: int) -> np.ndarray:
    if name in CLASSICAL_VARIANTS:
        fp_kind, clf_kind = CLASSICAL_VARIANTS[name]
        return fit_predict_classical(fp_kind, clf_kind, fit_df, predict_df, task, seed, es_df=es_df)
    elif name.endswith("_Pretrained") and name.replace("_Pretrained", "") in GNN_CLASSES:
        arch = name.replace("_Pretrained", "")
        return fit_predict_gnn_pretrained(arch, fit_df, es_df, predict_df, task, seed, gpu_id)
    elif name == "ChemBERTa_Pretrained":
        return fit_predict_chemberta_pretrained(fit_df, es_df, predict_df, task, seed, gpu_id)
    elif name == "Chemprop":
        return fit_predict_chemprop(fit_df, es_df, predict_df, task, seed, gpu_id)
    else:
        raise ValueError(f"unknown candidate {name!r}")


# ---------------------------------------------------------------------------
# pre-specified equal-weight ensemble
# ---------------------------------------------------------------------------
def mean_ensemble_predictions(preds_by_candidate: dict, candidate_names: list = None) -> np.ndarray:
    """Return the fixed equal-weight mean over every pre-specified candidate.

    Missing candidates or non-finite predictions are hard errors. Silently
    averaging a smaller subset would change the pre-specified ensemble and
    reintroduce data-dependent weighting.
    """
    candidate_names = list(candidate_names or CANDIDATE_NAMES)
    missing = [name for name in candidate_names if name not in preds_by_candidate]
    if missing:
        raise ValueError(f"mean ensemble is missing candidate predictions: {missing}")

    matrix = np.column_stack([preds_by_candidate[name] for name in candidate_names])
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise ValueError(f"mean ensemble contains {bad} non-finite candidate predictions")
    return matrix.mean(axis=1)


def _fit_single_seed_candidate_job(name: str, fit_df, es_df, predict_df,
                                   task: str, seed: int, gpu_id: int):
    """Fit one candidate once with the study-wide fixed seed."""
    prediction = _fit_predict_one_seed(
        name, fit_df, es_df, predict_df, task, seed, gpu_id
    )
    return name, np.asarray(prediction, dtype=float).reshape(-1)


def _fit_task_candidates(task: str, finetune_train_df, finetune_valid_df, finetune_test_df, cohort_union_df,
                          candidates: list, seed: int = FIXED_SEED, gpu_ids: list = None,
                          candidate_n_jobs: int = CANDIDATE_N_JOBS) -> dict:
    """Fit exactly one fixed-seed instance of every requested candidate.

    Candidate models are independent and run concurrently. Classical
    candidates use CPU; GPU candidates round-robin across ``gpu_ids``.
    """
    from sklearn.metrics import average_precision_score

    if candidate_n_jobs < 1:
        raise ValueError("candidate_n_jobs must be at least 1")
    gpu_ids = gpu_ids or GPU_IDS
    ft_train = finetune_train_df[finetune_train_df[task].notna()].reset_index(drop=True)
    ft_valid = finetune_valid_df[finetune_valid_df[task].notna()].reset_index(drop=True)
    ft_test = finetune_test_df[finetune_test_df[task].notna()].reset_index(drop=True)
    y_test = ft_test[task].to_numpy(dtype=float)

    n_test = len(ft_test)
    combined_predict_df = pd.concat([ft_test, cohort_union_df], axis=0, ignore_index=True)

    print(f"[stage1-per-task] {task}: fitting {len(candidates)} candidate(s) {candidates} "
          f"(single frozen split, fixed_seed={seed}, "
          f"candidate_n_jobs={candidate_n_jobs}, gpu_ids={gpu_ids}) "
          f"on train={len(ft_train)} / "
          f"valid(early-stop only)={len(ft_valid)} / "
          f"development-selection={n_test}")

    gpu_candidate_index = 0
    jobs = []
    for name in candidates:
        gpu_id = -1
        if _is_gpu_bound(name):
            gpu_id = gpu_ids[gpu_candidate_index % len(gpu_ids)]
            gpu_candidate_index += 1
        jobs.append(
            functools.partial(
                _fit_single_seed_candidate_job,
                name,
                ft_train,
                ft_valid,
                combined_predict_df,
                task,
                seed,
                gpu_id,
            )
        )
    candidate_results = _run_seed_jobs(
        jobs, min(int(candidate_n_jobs), len(jobs))
    )

    test_preds_by_candidate = {}
    cohort_preds_by_candidate = {}
    candidate_rows = []
    for name, prediction in candidate_results:
        test_pred = prediction[:n_test]
        cohort_pred = prediction[n_test:]
        test_preds_by_candidate[name] = test_pred
        cohort_preds_by_candidate[name] = cohort_pred

        mask = np.isfinite(test_pred)
        test_auprc = (
            float(average_precision_score(y_test[mask], test_pred[mask]))
            if mask.sum() >= 2 and len(np.unique(y_test[mask])) >= 2
            else float("nan")
        )
        candidate_rows.append({
            "task": task,
            "candidate": name,
            "seed": int(seed),
            "test_auprc": test_auprc,
        })
        print(
            f"[stage1-per-task] {task}/{name}: fixed-seed Stage-1 "
            f"development-selection AUPRC={test_auprc:.4f} (seed={seed})"
        )

    return {
        "test_preds_by_candidate": test_preds_by_candidate,
        "cohort_preds_by_candidate": cohort_preds_by_candidate,
        "candidate_rows": candidate_rows,
        "y_test": y_test,
        "seed": int(seed),
    }


def _build_task_result(task: str, test_preds_by_candidate: dict, cohort_preds_by_candidate: dict,
                       candidate_rows: list, y_test: np.ndarray,
                       candidate_names: list = None, seed: int = FIXED_SEED) -> dict:
    """Build the compact mean-only cache from fitted candidate predictions."""
    from sklearn.metrics import average_precision_score

    candidate_names = list(candidate_names or CANDIDATE_NAMES)
    test_ensemble_pred = mean_ensemble_predictions(test_preds_by_candidate, candidate_names)
    test_auprc = (
        float(average_precision_score(y_test, test_ensemble_pred))
        if len(np.unique(y_test)) >= 2 else float("nan")
    )
    print(
        f"[stage1-per-task] {task}: fixed equal-weight mean over "
        f"{len(candidate_names)} candidates, Stage-1 development-selection "
        f"AUPRC={test_auprc:.4f}"
    )

    return {
        "cache_schema_version": 3,
        "task": task,
        "ensemble_rule": ENSEMBLE_RULE,
        "candidate_names": candidate_names,
        "candidate_order": candidate_names,
        "seed": int(seed),
        "test_preds_by_candidate": test_preds_by_candidate,
        "cohort_preds_by_candidate": cohort_preds_by_candidate,
        "y_test": y_test,
        "candidate_rows": list(candidate_rows),
        "test_ensemble_auprc": test_auprc,
    }


def benchmark_task(task: str, finetune_train_df, finetune_valid_df, finetune_test_df, cohort_union_df,
                    seed: int = FIXED_SEED, gpu_ids: list = None,
                    candidate_n_jobs: int = CANDIDATE_N_JOBS,
                    candidates: list = None) -> dict:
    """Fit the pre-specified candidates and build the fixed mean ensemble.

    Three partitions, three distinct roles -- do not conflate them:
      - train: each candidate's own parameter fitting (once).
      - valid: WITHIN-candidate tuning only -- early stopping / checkpoint
        selection for the GNN/ChemBERTa candidates.
      - development-selection test: ranks candidates for the later
        task-specific K=4 arithmetic mean. It does not change candidate
        training, the equal-weight rule, or the fixed model seed.

    Each candidate's fixed-seed fit predicts on [test, cohort_union] in one
    combined pass (valid is consumed internally as the early-stopping
    monitor set during fitting). The same globally frozen mean ensemble is
    Candidate predictions are cached for later K=4 selection and deployment
    to every Stage-2 cohort.
    """
    candidates = candidates or CANDIDATE_NAMES
    fit_result = _fit_task_candidates(
        task, finetune_train_df, finetune_valid_df, finetune_test_df, cohort_union_df,
        candidates, seed=seed, gpu_ids=gpu_ids,
        candidate_n_jobs=candidate_n_jobs,
    )
    return _build_task_result(
        task, fit_result["test_preds_by_candidate"], fit_result["cohort_preds_by_candidate"],
        fit_result["candidate_rows"], fit_result["y_test"],
        candidate_names=candidates, seed=fit_result["seed"],
    )


def cached_candidate_names(cached: dict) -> set:
    return set(cached.get("candidate_names", cached.get("candidate_order", [])))


def cached_seed_value(cached: dict):
    if "seed" in cached:
        return int(cached["seed"])
    seeds = list(cached.get("seeds", []))
    return int(seeds[0]) if len(seeds) == 1 else None


def normalize_cached_task_result(cached: dict) -> dict:
    """Normalize a fixed-single-seed cache to the current schema."""
    seed = cached_seed_value(cached)
    if seed is None:
        raise ValueError("multi-seed caches cannot be reused by the fixed-seed run")
    return _build_task_result(
        cached["task"],
        cached["test_preds_by_candidate"],
        cached["cohort_preds_by_candidate"],
        cached["candidate_rows"],
        cached["y_test"],
        candidate_names=CANDIDATE_NAMES,
        seed=seed,
    )


def merge_task_result(cached: dict, new_fit: dict, seed: int) -> dict:
    """Merge newly fitted candidates into a legacy/current task cache."""
    task = cached["task"]
    test_preds_by_candidate = {**cached["test_preds_by_candidate"], **new_fit["test_preds_by_candidate"]}
    cohort_preds_by_candidate = {**cached["cohort_preds_by_candidate"], **new_fit["cohort_preds_by_candidate"]}
    # y_test is identical between cached and new (same frozen split, same
    # task) by construction -- not re-derived, just carried over from cache.
    y_test = cached["y_test"]
    merged_by_candidate = {r["candidate"]: r for r in cached["candidate_rows"]}
    merged_by_candidate.update({r["candidate"]: r for r in new_fit["candidate_rows"]})
    candidate_rows = list(merged_by_candidate.values())

    return _build_task_result(
        task, test_preds_by_candidate, cohort_preds_by_candidate,
        candidate_rows, y_test, candidate_names=CANDIDATE_NAMES, seed=seed,
    )


def load_finetune_split(cohort_dir: Path):
    path = cohort_dir / "tox21_scaffold_df_split_chem_finetune_pool.pkl"
    with open(path, "rb") as f:
        split: DataSplit = pickle.load(f)
    ordered = []
    for frame in (split.train, split.valid, split.test):
        if "ik" not in frame.columns:
            if "inchikey" not in frame.columns:
                raise ValueError(f"{path} has neither 'ik' nor 'inchikey'")
            frame = frame.copy()
            frame["ik"] = frame["inchikey"].map(ik_skeleton)
        ordered.append(
            frame.sort_values("ik", kind="stable").reset_index(drop=True)
        )
    return tuple(ordered)


def load_cohort_union(cohort_dir: Path = COHORT_DIR) -> pd.DataFrame:
    frames = []
    for cell in CELL_IDS:
        path = cohort_dir / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
        with open(path, "rb") as f:
            split: DataSplit = pickle.load(f)
        for df in (split.train, split.valid, split.test):
            if len(df):
                frames.append(df[["ik", "smiles_canon"] + TOX21_TASKS])
    return (
        pd.concat(frames, axis=0, ignore_index=True)
        .drop_duplicates("ik")
        .sort_values("ik", kind="stable")
        .reset_index(drop=True)
    )


def validate_cached_cohort_ids(
    cached: dict,
    expected_ids: np.ndarray,
    source_path: Path,
) -> None:
    """Require an explicit, correctly ordered compound axis in reused caches."""
    if "cohort_ik" not in cached:
        raise ValueError(
            f"{source_path} is a legacy cache without cohort_ik. Prediction "
            "vectors cannot be mapped safely after cohort reconstruction; "
            "fit this task into the new output directory instead."
        )
    cached_ids = np.asarray(cached["cohort_ik"]).astype(str)
    if not np.array_equal(cached_ids, expected_ids):
        raise ValueError(
            f"{source_path} cohort_ik order does not match the requested "
            "deterministic cohort union."
        )


def main(tasks=None, seed: int = FIXED_SEED, gpu_ids: list = None,
         candidate_n_jobs: int = CANDIDATE_N_JOBS,
         out_dir: Path = DEFAULT_OUT_DIR,
         cohort_dir: Path = COHORT_DIR,
         cache_source_dir: Path | None = None):
    tasks = tasks or TOX21_TASKS
    seed = int(seed)
    gpu_ids = gpu_ids or GPU_IDS
    out_dir = Path(out_dir)
    cohort_dir = Path(cohort_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Stage-1 output: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "chem_offset_final.csv"
    test_performance_path = out_dir / "chem_prior_deployment_report.csv"
    candidate_report_path = out_dir / "chem_candidate_test_performance.csv"
    task_cache_dir = out_dir / "task_benchmark_cache"
    task_cache_dir.mkdir(parents=True, exist_ok=False)
    source_cache_dir = (
        Path(cache_source_dir).resolve() / "task_benchmark_cache"
        if cache_source_dir is not None
        else None
    )
    finetune_train, finetune_valid, finetune_test = load_finetune_split(
        cohort_dir
    )
    cohort_union = load_cohort_union(cohort_dir)
    print(f"[stage1-per-task] fine-tune-train={len(finetune_train)}, fine-tune-valid={len(finetune_valid)}, "
          f"fine-tune-test={len(finetune_test)}, cohort union={len(cohort_union)}, "
          f"fixed_seed={seed}, candidate_n_jobs={candidate_n_jobs}, gpu_ids={gpu_ids}, "
          f"out_dir={out_dir}, cache_source_dir={source_cache_dir}")

    # ---- Phase A: benchmark every task and cache candidate predictions ----
    # Each task's full benchmark result is pickled as soon as it completes.
    # A cache is reusable only when it contains exactly the same fixed seed;
    # legacy multi-seed caches are intentionally rejected. This phase is the
    # expensive one (144 fits across a full 12-task, 12-candidate run), so a
    # crash partway through must not lose
    # already-benchmarked tasks. Cache is CANDIDATE-SET-AWARE: adding a
    # candidate to CANDIDATE_NAMES (e.g. Chemprop) does not invalidate an
    # existing task's cache wholesale -- only the new candidate is fit,
    # then merged into the cached result (merge_task_result), reusing the
    # already-fit candidates' predictions rather than a blind, wasteful
    # full rebuild. A cache whose candidate set is NOT a clean subset of
    # the current one (e.g. a candidate was removed) falls back to a full
    # rebuild for that task, since there's no valid partial reuse in that
    # direction.
    current_candidates = set(CANDIDATE_NAMES)
    cohort_ids = cohort_union["ik"].astype(str).to_numpy()
    task_results = {}
    all_candidate_rows = []
    for task in tasks:
        cache_path = task_cache_dir / f"{task}.pkl"
        cached = None
        source_cache_path = (
            source_cache_dir / f"{task}.pkl"
            if source_cache_dir is not None
            else None
        )
        if source_cache_path is not None and source_cache_path.exists():
            with open(source_cache_path, "rb") as f:
                cached = pickle.load(f)
            validate_cached_cohort_ids(cached, cohort_ids, source_cache_path)

        cached_candidates = cached_candidate_names(cached) if cached is not None else set()
        cached_seed = cached_seed_value(cached) if cached is not None else None
        seed_matches = cached_seed == seed
        if cached is not None and cached_candidates == current_candidates and seed_matches:
            task_results[task] = normalize_cached_task_result(cached)
            print(
                f"[stage1-per-task] {task}: loaded {len(current_candidates)} candidates from cache "
                f"and normalized to fixed mean-only schema"
            )
        elif cached is not None and cached_candidates < current_candidates and seed_matches:
            missing = sorted(current_candidates - cached_candidates)
            print(f"[stage1-per-task] {task}: cache covers {len(cached_candidates)}/{len(current_candidates)} "
                  f"candidates -- fitting {len(missing)} new: {missing}, merging into cache")
            new_fit = _fit_task_candidates(
                task, finetune_train, finetune_valid, finetune_test, cohort_union,
                missing, seed=seed, gpu_ids=gpu_ids,
                candidate_n_jobs=candidate_n_jobs,
            )
            task_results[task] = merge_task_result(cached, new_fit, seed)
            print(f"[stage1-per-task] {task}: merged, now {len(task_results[task]['candidate_order'])} candidates, "
                  f"cached -> {cache_path}")
        else:
            if cached is not None:
                print(
                    f"[stage1-per-task] {task}: cache mismatch "
                    f"(cached_candidates={sorted(cached_candidates)}, "
                    f"cached_seed={cached_seed}, requested_seed={seed}) "
                    f"-- rebuilding fully in {out_dir}"
                )
            task_results[task] = benchmark_task(
                task, finetune_train, finetune_valid, finetune_test, cohort_union,
                seed=seed, gpu_ids=gpu_ids,
                candidate_n_jobs=candidate_n_jobs,
            )
            print(f"[stage1-per-task] {task}: benchmarked fixed mean over "
                  f"{len(task_results[task]['candidate_names'])} candidates, "
                  f"cached -> {cache_path}")
        task_results[task]["cohort_ik"] = cohort_ids.tolist()
        with open(cache_path, "xb") as f:
            pickle.dump(task_results[task], f)
        all_candidate_rows.extend(task_results[task]["candidate_rows"])

    # ---- Phase B: evaluate and deploy the pre-specified frozen mean ----
    result = pd.DataFrame({"ik": cohort_union["ik"]})
    deployment_report_rows = []
    for task in tasks:
        tr = task_results[task]
        candidate_names = tr["candidate_names"]
        cohort_probs = mean_ensemble_predictions(tr["cohort_preds_by_candidate"], candidate_names)
        result[f"{task}_chem_logit"] = _to_logit(cohort_probs)
        deployment_report_rows.append({
            "task": task,
            "rule": ENSEMBLE_RULE,
            "k": len(candidate_names),
            "ensemble_candidates": ";".join(candidate_names),
            "test_auprc": tr["test_ensemble_auprc"],
        })
        print(
            f"[stage1-per-task] {task}: prepared fixed equal-weight candidate "
            f"mean (K={len(candidate_names)})"
        )

    pd.DataFrame(all_candidate_rows).to_csv(candidate_report_path, index=False)
    result.to_csv(out_path, index=False)
    pd.DataFrame(deployment_report_rows).to_csv(
        test_performance_path, index=False
    )
    print(f"[stage1-per-task] complete -> {out_path} ({len(result)} compounds x {len(tasks)} tasks), "
          f"deployment report -> {test_performance_path}, "
          f"candidate test report -> {candidate_report_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fit the fixed-seed Stage-1 candidate library on the external "
            "scaffold-disjoint chemical pool."
        )
    )
    parser.add_argument("--task", default=None)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=None,
                        help="GPU device ids to round-robin across GPU candidate models")
    parser.add_argument(
        "--candidate-n-jobs",
        type=int,
        default=CANDIDATE_N_JOBS,
        help="number of candidate models to launch concurrently for the fixed seed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="New output directory. Existing paths are never overwritten.",
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=COHORT_DIR,
        help="Directory produced by build_matched_cohort.py.",
    )
    parser.add_argument(
        "--cache-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior Stage-1 output read only as a cache source. "
            "All results are still written to the new --out-dir."
        ),
    )
    args = parser.parse_args()
    tasks = [args.task] if args.task else None
    main(
        tasks=tasks,
        seed=args.seed,
        gpu_ids=args.gpu_ids,
        candidate_n_jobs=args.candidate_n_jobs,
        out_dir=args.out_dir,
        cohort_dir=args.cohort_dir,
        cache_source_dir=args.cache_source_dir,
    )
