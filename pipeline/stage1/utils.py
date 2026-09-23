import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import json
import pickle
import subprocess
from pathlib import Path
from string import Template


def find_per_seed_csv(directory, summary_file_tag=None):
    tag = str(summary_file_tag or "").strip()
    pattern = f"*_{tag}_per_seed.csv" if tag else "*_per_seed.csv"
    candidates = sorted(Path(directory).glob(pattern))
    if not candidates and tag:
        candidates = sorted(Path(directory).glob("*_per_seed.csv"))
    return candidates[0] if candidates else None


def load_from_per_seed_csv(csv_path):
    df_raw    = pd.read_csv(csv_path)
    non_rand  = df_raw[df_raw["model"].astype(str).str.lower() != "random"]
    metric_cols = [
        c for c in non_rand.columns
        if c not in {"model", "seed"}
        and pd.api.types.is_numeric_dtype(non_rand[c])
    ]
    df_mean   = non_rand.groupby("model")[metric_cols].mean()
    df_std    = non_rand.groupby("model")[metric_cols].std().fillna(0)
    rand_rows = df_raw[df_raw["model"].astype(str).str.lower() == "random"]
    prevalence = (
        float(rand_rows.iloc[0]["auprc"])
        if len(rand_rows) > 0 and not pd.isna(rand_rows.iloc[0]["auprc"])
        else None
    )
    return df_mean, df_std, prevalence


def as_list(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def ik_first(ik):
    if isinstance(ik, str) and "-" in ik:
        return ik.split("-")[0]
    return ik


def resolve_device(device=None, gpu_id=None, fallback: int = 1) -> torch.device:
    """Resolve an explicit config device before falling back to auto GPU choice.

    Accepted config forms:
      - {"device": "cuda:0"}
      - {"device": "cpu"}
      - {"gpu_id": 0}
    """
    if isinstance(device, torch.device):
        return device
    if device is not None:
        text = str(device).strip()
        if text:
            resolved = torch.device(text)
            if resolved.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            print(f"[resolve_device] Using configured device={resolved}")
            return resolved
    if gpu_id is not None:
        try:
            idx = int(gpu_id)
        except (TypeError, ValueError):
            idx = -1
        if idx >= 0:
            if not torch.cuda.is_available():
                return torch.device("cpu")
            resolved = torch.device(f"cuda:{idx}")
            print(f"[resolve_device] Using configured gpu_id={idx} ({resolved})")
            return resolved
    return pick_gpu(fallback=fallback)


def pick_gpu(fallback: int = 1) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # When CUDA_VISIBLE_DEVICES restricts the worker to a single GPU, honour it
    # directly — querying nvidia-smi would see *all* physical GPUs and might
    # pick a different one than the one assigned to this worker process.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and visible not in ("-1", "NoDevFiles"):
        return torch.device("cuda:0")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            best_idx, best_free = 0, -1
            for line in result.stdout.strip().splitlines():
                idx_str, free_str = line.split(",")
                idx, free = int(idx_str.strip()), int(free_str.strip())
                if idx > 1:
                    continue
                if free > best_free:
                    best_free, best_idx = free, idx
            device = torch.device(f"cuda:{best_idx}")
            print(f"[pick_gpu] Selected cuda:{best_idx} ({best_free} MiB free)")
            return device
    except Exception:
        pass

    n = torch.cuda.device_count()
    idx = fallback if n > fallback else 0
    print(f"[pick_gpu] nvidia-smi unavailable; using cuda:{idx}")
    return torch.device(f"cuda:{idx}")


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_config(config_file):
    with open(config_file, mode="rb") as fr:
        config = json.load(fr)

    def _substitute(cfg):
        scalar_cfg = {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}
        return {
            k: Template(v).safe_substitute(scalar_cfg) if isinstance(v, str) else v
            for k, v in cfg.items()
        }

    config = _substitute(_substitute(config))
    return config


def load_pickle(load_path):
    with open(load_path, "rb") as fr:
        dataset = pickle.load(fr)
    
    return dataset


def save_pickle(dataset, save_path):
    with open(save_path, "wb") as fw:
        pickle.dump(dataset, fw)


def get_prevalence_positive_baseline(
    test_loader,
    task_cols=None,
    reduce: bool = True,
    ):
    """
    Compute positive-label prevalence baseline.

    Parameters
    ----------
    reduce :
        True  -> return scalar mean prevalence
        False -> return per-task prevalence vector
    """

    if isinstance(test_loader, pd.DataFrame):

        if task_cols is None:
            raise ValueError(
                "task_cols is required when test_loader is a DataFrame"
            )

        cols = (
            [task_cols]
            if isinstance(task_cols, str)
            else list(task_cols)
        )

        labels = test_loader[cols].values.astype(float)

        prevalences = []

        for i in range(labels.shape[1]):

            y = labels[:, i]
            y = y[~np.isnan(y)]

            n_pos = (y == 1).sum()
            n_total = len(y)

            prevalence = n_pos / n_total if n_total > 0 else 0.0
            prevalences.append(prevalence)

        prevalences = np.array(prevalences, dtype=float)

        return (float(prevalences.mean()) if reduce else prevalences)

    else:

        all_y = []

        for batch in test_loader:

            if isinstance(batch, dict):
                y = batch.get("y", batch.get("labels"))
            else:
                y = batch.y

            y = y.float()

            if y.ndim == 1:
                y = y.unsqueeze(-1)

            all_y.append(y)

        all_y = torch.cat(all_y, dim=0).cpu().numpy()

        prevalences = []

        for i in range(all_y.shape[1]):

            y = all_y[:, i]

            n_pos = (y == 1).sum()
            n_total = len(y)

            prevalence = n_pos / n_total if n_total > 0 else 0.0

            prevalences.append(prevalence)

        prevalences = np.array(prevalences, dtype=float)

        return (float(prevalences.mean()) if reduce else prevalences)
    

def average_binary_results(all_results: list) -> dict:
    avg = copy.deepcopy(all_results[0])
    n   = len(all_results)
    for grp, model_dict in avg.items():
        for model, seed_dict in model_dict.items():
            for seed, metrics in seed_dict.items():
                if "error" in metrics:
                    continue
                for other in all_results[1:]:
                    other_m = other.get(grp, {}).get(model, {}).get(seed, {})
                    if "error" not in other_m:
                        for k in metrics:
                            if not isinstance(metrics[k], list):
                                metrics[k] = metrics.get(k, 0) + other_m.get(k, 0)
                for k in metrics:
                    if not isinstance(metrics[k], list):
                        metrics[k] /= n
    return avg
