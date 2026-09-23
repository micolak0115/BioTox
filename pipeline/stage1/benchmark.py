# stage1/benchmark.py
# %%
import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parents[1]))
from stage1.utils import seed_all, pick_gpu, load_pickle
from stage1.featurizer import ATOM_FEATURE_DIM
from stage1.loader import _load_graph_cache_payload
from stage1.model import CHEM_ENCODERS, Fingerprint, UndirectedMPNN, Transformer


DEFAULT_CHEMBERTA_FEATURE_CACHE_DIR = "/data/kyungan/cache/biotox_chemberta_features"


def chemberta_feature_cache_dir(config: dict) -> str:
    return (
        config.get("chem_feature_cache_dir")
        or os.environ.get("BIOTOX_CHEMBERTA_FEATURE_CACHE_DIR")
        or DEFAULT_CHEMBERTA_FEATURE_CACHE_DIR
    )


def configured_device(config: dict):
    if config.get("device") is not None:
        return config.get("device")
    if config.get("gpu_id") is not None:
        return f"cuda:{int(config['gpu_id'])}"
    return None


def configured_visible_devices(config: dict) -> list[str]:
    devices = config.get("cuda_visible_devices")
    if devices:
        return [str(d).split(":", 1)[-1] for d in devices]
    devices = config.get("devices")
    if devices:
        out = []
        for dev in devices:
            text = str(dev)
            if text.lower() == "cpu":
                continue
            out.append(text.split(":", 1)[-1] if text.lower().startswith("cuda:") else text)
        return out
    if config.get("gpu_id") is not None:
        return [str(int(config["gpu_id"]))]
    return []


def _job_gpu_ids(config: dict, n_seeds: int, max_jobs: int = None):
    n_jobs, auto_gpu_ids = _get_parallel_config(n_seeds, max_jobs=max_jobs)
    explicit = configured_visible_devices(config)
    if explicit:
        return n_jobs, [explicit[i % len(explicit)] for i in range(n_seeds)]
    return n_jobs, auto_gpu_ids


def _get_parallel_config(n_seeds: int, max_jobs: int = None):
    """
    Returns (n_jobs, gpu_ids) for running n_seeds jobs in parallel.

    max_jobs caps concurrency (use 1 for sequential). On a k-GPU machine
    seeds are assigned round-robin; on CPU-only machines gpu_id=-1 is used.
    """
    n_jobs = n_seeds if max_jobs is None else min(n_seeds, max_jobs)
    if not torch.cuda.is_available():
        return n_jobs, [-1] * n_seeds
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return n_jobs, [-1] * n_seeds
    gpu_ids = [i % n_gpus for i in range(n_seeds)]
    return n_jobs, gpu_ids


def _run_seed_jobs(callables, n_jobs: int):
    if n_jobs == 1:
        return [fn() for fn in callables]
    return Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fn)() for fn in callables
    )


def _metrics_cache_path(save_dir: str, model_name: str, tag: str, seed: int) -> str:
    return os.path.join(save_dir, f"{model_name.lower()}_{tag}_seed{seed}_metrics.json")


def _load_cached_metrics(path: str):
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        if "error" not in m:
            return m
    return None


def load_group_results_from_cache(save_dir: str, model_names: list, tag: str, seeds: list) -> dict:
    """Rebuild {model_name: {seed: metrics}} from per-seed JSON files without running any benchmark."""
    results = {}
    for model_name in model_names:
        seed_dict = {}
        for seed in seeds:
            m = _load_cached_metrics(_metrics_cache_path(save_dir, model_name, tag, seed))
            if m is not None:
                seed_dict[seed] = m
        if seed_dict:
            results[model_name] = seed_dict
    return results


def _save_cached_metrics(path: str, metrics: dict):
    with open(path, "w") as f:
        json.dump(metrics, f)


def _pick_gpu_dynamic() -> int:
    """Return physical GPU index with most free memory, or -1 on CPU."""
    if not torch.cuda.is_available():
        return -1
    env_orig = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        device = pick_gpu()
        idx = device.index if device.index is not None else 0
    except Exception:
        idx = 0
    finally:
        if env_orig is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_orig
    return idx


def _worker_gpu_id(gpu_id):
    if gpu_id is None:
        return None
    try:
        idx = int(str(gpu_id).split(":", 1)[-1])
    except (TypeError, ValueError):
        return None
    return idx if idx >= 0 else None


def _set_worker_cuda(gpu_id):
    idx = _worker_gpu_id(gpu_id)
    if idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    return idx


def _run_gnn_seed(model_name, gnn_cls, seed, config, split_graph_path, gpu_id: int = -1) -> tuple:
    """Train + evaluate one GNN model for one seed. Returns (seed, metrics)."""
    cache_path = _metrics_cache_path(config["save_dir"], model_name, "gnn", seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[GNN][SKIP] {model_name} seed={seed}: cached")
        return seed, cached
    checkpoint_path = os.path.join(config["save_dir"], f"{model_name.lower()}_seed{seed}.pt")

    split, gnn_pos_weight, _ = _load_graph_cache_payload(
        split_graph_path,
        config.get("gnn_batch_size", config.get("batch_size", 64)),
    )

    def _attempt(gid):
        _set_worker_cuda(gid)
        seed_all(seed)
        mpnn = gnn_cls(
            input_dim  = ATOM_FEATURE_DIM,
            hidden_dim = config["gnn_hidden_dim"],
            output_dim = config["task_dim"],
            n_layers   = config["gnn_num_layers"],
            dropout    = config.get("gnn_dropout", 0.5),
            graph_pooling = config.get("gnn_graph_pooling", config.get("graph_pooling", "mean")),
            JK         = config.get("gnn_JK", config.get("JK", "last")),
            task_cols  = config.get("task_cols"),
            task_family_heads = config.get("task_family_heads", False),
            task_family_head_type = config.get("task_family_head_type", "linear"),
            task_family_hidden_dim = config.get("task_family_hidden_dim", 256),
            task_family_dropout = config.get("task_family_dropout", 0.1),
            task_mode  = config["task_mode"],
            lr         = config["gnn_lr"],
            epochs     = config["gnn_epochs"],
            seed       = seed,
            save_path  = checkpoint_path,
            patience   = config["gnn_patience"],
            min_delta  = config["gnn_min_delta"],
            device     = configured_device(config),
        )
        mpnn.fit(split, pos_weight=gnn_pos_weight)
        return mpnn.predict(split)

    try:
        try:
            metrics = _attempt(gpu_id)
        except RuntimeError as oom:
            if "out of memory" not in str(oom).lower() or _worker_gpu_id(gpu_id) is None:
                raise
            torch.cuda.empty_cache()
            alt = _pick_gpu_dynamic()
            print(f"[GNN][OOM] {model_name} seed={seed}: retrying on cuda:{alt}")
            metrics = _attempt(alt)

        _save_cached_metrics(cache_path, metrics)
        print(f"[GNN][DONE] {model_name} seed={seed}: {metrics}")
        return seed, metrics
    except Exception as e:
        print(f"[GNN][FAIL] {model_name} seed={seed}: {e}")
        traceback.print_exc()
        return seed, {"error": str(e)}


def _run_tf_seed(model_name, transformer_cls, seed, config, split_df_path, mode: str, gpu_id: int = -1) -> tuple:
    """
    Train + evaluate one Transformer model for one seed.

    mode : "probe" | "finetune" | "finetuned_probe"
    Returns (seed, metrics).
    """
    cache_path = _metrics_cache_path(config["save_dir"], model_name, mode, seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[TF-{mode}][SKIP] {model_name} seed={seed}: cached")
        return seed, cached

    split, tf_pos_weight, _ = load_pickle(split_df_path)
    feature_cache_dir = chemberta_feature_cache_dir(config)
    probe_backbone_dir = None

    if mode == "finetune":
        ckp_dir = os.path.join(config["save_dir"], f"{model_name.lower()}_{mode}_seed{seed}")
        if os.path.exists(os.path.join(ckp_dir, "checkpoint.pt")):
            print(f"[TF-{mode}][CKPT] {model_name} seed={seed}: checkpoint found, running predict-only")
            try:
                _set_worker_cuda(gpu_id)
                seed_all(seed)
                model = transformer_cls(**_transformer_kwargs(
                    config,
                    model_name,
                    seed,
                    "finetune",
                    model_name,
                    ckp_dir,
                    feature_cache_dir,
                ))
                metrics = model.predict(split.test)
                _save_cached_metrics(cache_path, metrics)
                print(f"[TF-{mode}][DONE] {model_name} seed={seed} (from checkpoint): {metrics}")
                return seed, metrics
            except Exception as e:
                print(f"[TF-{mode}][WARN] {model_name} seed={seed}: checkpoint predict failed ({e}), re-training")
    if mode in {"probe", "finetuned_probe"}:
        probe_dir = os.path.join(config["save_dir"], f"{model_name.lower()}_{mode}_seed{seed}")
        probe_backbone_dir = _cached_finetune_backbone_dir(config, model_name, seed)
        if mode == "probe" and not bool(config.get("tf_probe_use_finetuned_backbone", False)):
            probe_backbone_dir = None
        if mode == "finetuned_probe" and not probe_backbone_dir:
            msg = (
                "fine-tuned ChemBERTa checkpoint not found; set "
                "chem_finetune_cache_dir to the pretraining-stage save_dir"
            )
            print(f"[TF-{mode}][SKIP] {model_name} seed={seed}: {msg}")
            return seed, {"error": msg}
        if os.path.exists(os.path.join(probe_dir, "probe_head.pt")):
            try:
                _set_worker_cuda(gpu_id)
                seed_all(seed)
                model = transformer_cls(
                    num_labels        = config["task_dim"],
                    task_mode         = config["task_mode"],
                    trust_remote_code = config["trust_remote_code"],
                    smiles_col        = config["smiles_col"],
                    task_cols         = config["task_cols"],
                    task_family_heads = config.get("task_family_heads", False),
                    task_family_head_type = config.get("head_type", config.get("task_family_head_type", config.get("tf_probe_type", "mlp"))),
                    task_family_hidden_dim = config.get("head_hidden_dim", config.get("task_family_hidden_dim", config.get("tf_probe_hidden_dim", 512))),
                    task_family_dropout = config.get("dropout", config.get("task_family_dropout", config.get("tf_probe_dropout", 0.1))),
                    batch_size        = config["tf_batch_size"],
                    max_length        = config["tf_max_length"],
                    seed              = seed,
                    save_path         = probe_dir,
                    patience          = config["tf_patience"],
                    min_delta         = config["tf_min_delta"],
                    lr                = config["tf_probe_lr"],
                    epochs            = config["tf_epochs"],
                    max_grad_norm     = config["tf_max_grad_norm"],
                    use_amp           = config["tf_use_amp"],
                    feature_cache_dir = feature_cache_dir,
                    device            = configured_device(config),
                )
                from stage1.probe import load_probe
                load_probe(model, probe_dir)
                metrics = model.predict_probe(split.test)
                _save_cached_metrics(cache_path, metrics)
                print(f"[TF-{mode}][DONE] {model_name} seed={seed} (from probe checkpoint): {metrics}")
                return seed, metrics
            except Exception as e:
                print(f"[TF-{mode}][WARN] {model_name} seed={seed}: probe checkpoint predict failed ({e}), re-training")

    def _attempt(gid):
        _set_worker_cuda(gid)
        seed_all(seed)
        source_name = probe_backbone_dir
        if mode in {"probe", "finetuned_probe"} and probe_backbone_dir:
            print(f"[TF-{mode}] {model_name} seed={seed}: warm-starting from {probe_backbone_dir}")
        model = transformer_cls(**_transformer_kwargs(
            config,
            model_name,
            seed,
            mode,
            source_name,
            source_name if mode in {"probe", "finetuned_probe"} and probe_backbone_dir else None,
            feature_cache_dir,
        ))
        if mode in {"probe", "finetuned_probe"}:
            model   = model.fit_probe(
                split,
                pos_weight    = tf_pos_weight,
                probe_type    = config.get("head_type", config.get("task_family_head_type", config.get("tf_probe_type", "mlp"))),
                hidden_dim    = config.get("head_hidden_dim", config.get("task_family_hidden_dim", config.get("tf_probe_hidden_dim", 512))),
                dropout       = config.get("dropout", config.get("task_family_dropout", config.get("tf_probe_dropout", 0.1))),
                chem_proj_dim = config.get("chem_proj_dim"),
            )
            return model.predict_probe(split.test)
        else:
            model   = model.fit(split, pos_weight=tf_pos_weight)
            return model.predict(split.test)

    try:
        try:
            metrics = _attempt(gpu_id)
        except RuntimeError as oom:
            if "out of memory" not in str(oom).lower() or _worker_gpu_id(gpu_id) is None:
                raise
            torch.cuda.empty_cache()
            alt = _pick_gpu_dynamic()
            print(f"[TF-{mode}][OOM] {model_name} seed={seed}: retrying on cuda:{alt}")
            metrics = _attempt(alt)

        _save_cached_metrics(cache_path, metrics)
        print(f"[TF-{mode}][DONE] {model_name} seed={seed}: {metrics}")
        return seed, metrics

    except Exception as e:
        print(f"[TF-{mode}][FAIL] {model_name} seed={seed}: {e}")
        traceback.print_exc()
        return seed, {"error": str(e)}


def _run_fp_seed(model_name, fingerprint_cls, seed, config, split_df_path, gpu_id: int = -1) -> tuple:
    """Train + evaluate one fingerprint model for one seed. Returns (seed, metrics)."""
    cache_path = _metrics_cache_path(config["save_dir"], model_name, "fp_probe", seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[FP-probe][SKIP] {model_name} seed={seed}: cached")
        return seed, cached
    checkpoint_path = os.path.join(config["save_dir"], f"{model_name.lower()}_probe_seed{seed}")

    split, fp_pos_weight, _ = load_pickle(split_df_path)

    def _attempt(gid):
        _set_worker_cuda(gid)
        seed_all(seed)
        model = fingerprint_cls(
            fp_bits=config.get("fp_bits", 2048),
            radius=config.get("fp_radius", 2),
            task_mode=config["task_mode"],
            num_labels=config["task_dim"],
            task_cols=config["task_cols"],
            task_family_heads=config.get("task_family_heads", False),
            task_family_head_type=config.get("task_family_head_type", "linear"),
            task_family_hidden_dim=config.get("task_family_hidden_dim", 256),
            task_family_dropout=config.get("task_family_dropout", 0.1),
            seed=seed,
            save_path=checkpoint_path,
            device=configured_device(config),
        )
        model = model.fit_probe(
            split,
            pos_weight=fp_pos_weight,
            probe_type=config.get("task_family_head_type", config.get("fp_probe_type", "linear")),
            hidden_dim=config.get("task_family_hidden_dim", config.get("fp_probe_hidden_dim", 256)),
            dropout=config.get("task_family_dropout", config.get("fp_probe_dropout", 0.1)),
        )
        return model.predict_probe(split.test)

    try:
        try:
            metrics = _attempt(gpu_id)
        except RuntimeError as oom:
            if "out of memory" not in str(oom).lower() or _worker_gpu_id(gpu_id) is None:
                raise
            torch.cuda.empty_cache()
            alt = _pick_gpu_dynamic()
            print(f"[FP-probe][OOM] {model_name} seed={seed}: retrying on cuda:{alt}")
            metrics = _attempt(alt)

        _save_cached_metrics(cache_path, metrics)
        print(f"[FP-probe][DONE] {model_name} seed={seed}: {metrics}")
        return seed, metrics
    except Exception as e:
        print(f"[FP-probe][FAIL] {model_name} seed={seed}: {e}")
        traceback.print_exc()
        return seed, {"error": str(e)}


def _load_cached_seeds(save_dir: str, model_name: str, tag: str, seeds: list) -> dict:
    """Return {seed: metrics} for seeds that already have a valid cache entry."""
    cached = {}
    for seed in seeds:
        m = _load_cached_metrics(_metrics_cache_path(save_dir, model_name, tag, seed))
        if m is not None:
            print(f"[{tag.upper()}][SKIP] {model_name} seed={seed}: cached")
            cached[seed] = m
    return cached


def _cached_finetune_backbone_dir(config: dict, model_name: str, seed: int) -> Optional[str]:
    """
    Return the cached finetuned backbone checkpoint directory for this seed.

    This is used to warm-start probe runs from the cached full chemical-only
    finetuned model when available.
    """
    cache_root = config.get("chem_finetune_cache_dir")
    task_name = config.get("task_name")
    if not cache_root or not task_name:
        return None
    task_dir = cache_root if task_name == "all_tasks" else os.path.join(cache_root, task_name)
    ckpt_dir = os.path.join(task_dir, f"{model_name.lower()}_finetune_seed{seed}")
    if os.path.exists(os.path.join(ckpt_dir, "checkpoint.pt")):
        return ckpt_dir
    return None


def _resolve_pretrained_gnn_checkpoint(config: dict, model_name: str) -> str:
    """Return the external Pretrain-GNN checkpoint for a GNN encoder."""
    custom_map = config.get("gnn_pretrained_checkpoint_map") or {}
    if model_name in custom_map:
        path = os.path.expanduser(str(custom_map[model_name]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configured pretrained checkpoint for {model_name} not found: {path}")
        return path

    root = os.path.expanduser(str(config.get("snap_chem_path") or config.get("pretrain_gnn_root") or "/data/kyungan/pretrain-gnns/chem"))
    variant = str(config.get("gnn_pretrained_variant", "supervised_contextpred"))
    lower = model_name.lower()
    if lower == "gin":
        path = os.path.join(root, "model_gin", f"{variant}.pth")
    else:
        path = os.path.join(root, "model_architecture", f"{lower}_{variant}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained GNN checkpoint for {model_name} not found: {path}. "
            "Set gnn_pretrained_checkpoint_map or gnn_pretrained_variant."
        )
    return path


def _load_pretrained_gnn_backbone(model: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load a Pretrain-GNN backbone into the local BioTox GNN wrapper.

    External checkpoints contain only backbone parameters. They use three
    chirality embedding rows, while the local benchmark featurizer reserves an
    additional CHI_OTHER row. Matching rows are copied and the extra row remains
    locally initialized; prediction heads are trained by the probe.
    """
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported checkpoint format for {checkpoint_path}")

    current = model.state_dict()
    loaded, partial, skipped = [], [], []
    for key, value in raw.items():
        if key not in current:
            skipped.append(key)
            continue
        target = current[key]
        if target.shape == value.shape:
            current[key] = value
            loaded.append(key)
            continue
        same_tail = target.ndim == value.ndim and target.shape[1:] == value.shape[1:]
        if same_tail and target.shape[0] >= value.shape[0]:
            copied = target.clone()
            copied[: value.shape[0]] = value
            current[key] = copied
            partial.append(key)
            continue
        skipped.append(key)

    model.load_state_dict(current, strict=True)
    print(
        f"[GNN-PRETRAIN] loaded {len(loaded)} tensors from {checkpoint_path}; "
        f"partial={partial or 'none'} skipped={skipped or 'none'}"
    )


def _transformer_kwargs(
    config: dict,
    model_name: str,
    seed: int,
    mode: str,
    source_name: str,
    checkpoint_path: Optional[str],
    feature_cache_dir: str,
) -> dict:
    """Build the shared constructor kwargs for ChemBERTa probe/finetune runs."""
    if source_name == model_name and checkpoint_path is None:
        source_name = None
    head_type = config.get("task_family_head_type", config.get("tf_probe_type", "linear"))
    hidden_dim = config.get("task_family_hidden_dim", config.get("tf_probe_hidden_dim", 256))
    dropout = config.get("task_family_dropout", config.get("tf_probe_dropout", 0.1))
    lr = config["tf_probe_lr"] if mode in {"probe", "finetuned_probe"} else config["tf_finetune_lr"]
    return dict(
        num_labels=config["task_dim"],
        task_mode=config["task_mode"],
        trust_remote_code=config["trust_remote_code"],
        smiles_col=config["smiles_col"],
        task_cols=config["task_cols"],
        task_family_heads=config.get("task_family_heads", False),
        task_family_head_type=head_type,
        task_family_hidden_dim=hidden_dim,
        task_family_dropout=dropout,
        batch_size=config["tf_batch_size"],
        max_length=config["tf_max_length"],
        seed=seed,
        save_path=os.path.join(config["save_dir"], f"{model_name.lower()}_{mode}_seed{seed}"),
        patience=config["tf_patience"],
        min_delta=config["tf_min_delta"],
        lr=lr,
        epochs=config["tf_epochs"],
        max_grad_norm=config["tf_max_grad_norm"],
        use_amp=config["tf_use_amp"],
        feature_cache_dir=feature_cache_dir,
        device=configured_device(config),
        model_name=source_name,
        checkpoint_path=checkpoint_path,
        local_files_only=bool(checkpoint_path) or config.get("local_files_only", False),
    )


def _run_gnn_probe_seed(
    model_name: str,
    gnn_cls,
    seed: int,
    config: dict,
    probe_df_split_path: str,
    gpu_id: int = -1,
    cache_tag: str = "gnn_probe",
    checkpoint_path: Optional[str] = None,
) -> tuple:
    """
    Load a pre-trained GNN checkpoint, freeze backbone, train a probe head
    on the matched bio+chem DataFrame split, evaluate on matched test.
    """
    cache_path = _metrics_cache_path(config["save_dir"], model_name, cache_tag, seed)
    cached = _load_cached_metrics(cache_path)
    if cached is not None:
        print(f"[{cache_tag.upper()}][SKIP] {model_name} seed={seed}: cached")
        return seed, cached

    checkpoint_root = (
        config.get("gnn_checkpoint_cache_dir")
        or config.get("chem_gnn_cache_dir")
        or config["save_dir"]
    )
    checkpoint_path = checkpoint_path or os.path.join(checkpoint_root, f"{model_name.lower()}_seed{seed}.pt")
    if not os.path.exists(checkpoint_path):
        msg = f"no checkpoint at {checkpoint_path}"
        print(f"[{cache_tag.upper()}][SKIP] {model_name} seed={seed}: {msg}")
        return seed, {"error": msg}

    matched_split, pos_weight, _ = load_pickle(probe_df_split_path)

    def _attempt(gid):
        _set_worker_cuda(gid)
        seed_all(seed)
        gnn = gnn_cls(
            input_dim  = ATOM_FEATURE_DIM,
            hidden_dim = config["gnn_hidden_dim"],
            output_dim = config["task_dim"],
            n_layers   = config["gnn_num_layers"],
            dropout    = config.get("gnn_dropout", 0.5),
            graph_pooling = config.get("gnn_graph_pooling", config.get("graph_pooling", "mean")),
            JK         = config.get("gnn_JK", config.get("JK", "last")),
            task_cols  = config.get("task_cols"),
            task_family_heads     = config.get("task_family_heads", False),
            task_family_head_type = config.get("task_family_head_type", "linear"),
            task_family_hidden_dim = config.get("task_family_hidden_dim", 256),
            task_family_dropout   = config.get("task_family_dropout", 0.1),
            task_mode  = config["task_mode"],
            lr         = config["gnn_lr"],
            epochs     = config["gnn_epochs"],
            seed       = seed,
            patience   = config["gnn_patience"],
            min_delta  = config["gnn_min_delta"],
            device     = configured_device(config),
        )
        if cache_tag == "gnn_pretrained_probe":
            _load_pretrained_gnn_backbone(gnn, checkpoint_path)
        else:
            state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            gnn.load_state_dict(state)

        save_name = f"{model_name.lower()}_pretrained_probe_seed{seed}" if cache_tag == "gnn_pretrained_probe" else f"{model_name.lower()}_probe_seed{seed}"
        save_path = os.path.join(config["save_dir"], save_name)
        gnn.fit_probe(
            matched_split,
            smiles_col    = config.get("smiles_col", "smiles_canon"),
            task_cols     = config["task_cols"],
            task_mode     = config["task_mode"],
            probe_type    = config.get("head_type", config.get("task_family_head_type", "mlp")),
            hidden_dim    = config.get("head_hidden_dim", config.get("task_family_hidden_dim", 512)),
            dropout       = config.get("dropout", config.get("task_family_dropout", 0.1)),
            lr            = config.get("gnn_probe_lr", config.get("gnn_lr", 1e-3)),
            epochs        = config.get("gnn_probe_epochs", 20),
            batch_size    = config.get("gnn_probe_batch_size", 256),
            patience      = config.get("gnn_probe_patience", config.get("gnn_patience", 10)),
            min_delta     = config.get("gnn_min_delta", 0.0),
            pos_weight    = pos_weight,
            save_path     = save_path,
            seed          = seed,
            chem_proj_dim = config.get("chem_proj_dim"),
        )
        return gnn.predict_probe(
            matched_split.test,
            smiles_col = config.get("smiles_col", "smiles_canon"),
            task_cols  = config["task_cols"],
            task_mode  = config["task_mode"],
        )

    try:
        try:
            metrics = _attempt(gpu_id)
        except RuntimeError as oom:
            if "out of memory" not in str(oom).lower() or _worker_gpu_id(gpu_id) is None:
                raise
            torch.cuda.empty_cache()
            alt = _pick_gpu_dynamic()
            print(f"[{cache_tag.upper()}][OOM] {model_name} seed={seed}: retrying on cuda:{alt}")
            metrics = _attempt(alt)

        _save_cached_metrics(cache_path, metrics)
        print(f"[{cache_tag.upper()}][DONE] {model_name} seed={seed}: {metrics}")
        return seed, metrics
    except Exception as e:
        print(f"[{cache_tag.upper()}][FAIL] {model_name} seed={seed}: {e}")
        traceback.print_exc()
        return seed, {"error": str(e)}


def gnn_probe_benchmark(
    config: dict,
    probe_df_split_path: str,
    seeds: list,
    models: Optional[list] = None,
) -> dict:
    """
    Benchmark GNN encoders in the freeze-then-probe regime.

    Loads the pre-trained GNN checkpoint from config["save_dir"], freezes the
    backbone, trains a lightweight head on the matched bio+chem DataFrame split,
    and evaluates on the matched test set.
    """
    allowed = set(models or config.get("gnn_models") or [])
    GNN_ENCODERS = {
        k: v for k, v in CHEM_ENCODERS.items()
        if issubclass(v, UndirectedMPNN) and (not allowed or k in allowed)
    }
    results = {}
    for model_name, GNN in GNN_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "gnn_probe", seeds)
        remaining    = [s for s in seeds if s not in cached_pairs]
        print(f"\n[GNN-probe] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_jobs"))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_gnn_probe_seed(
                    model_name, GNN, seed, config, probe_df_split_path, gpu_id
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def gnn_pretrained_probe_benchmark(
    config: dict,
    probe_df_split_path: str,
    seeds: list,
    models: Optional[list] = None,
) -> dict:
    """
    Probe external Pretrain-GNN backbones on the frozen matched BioTox split.

    The GNN message-passing backbone is loaded from /data/kyungan/pretrain-gnns
    or from config["gnn_pretrained_checkpoint_map"], frozen, and only the probe
    head is trained on the matched training fold.
    """
    allowed = set(models or config.get("gnn_models") or [])
    GNN_ENCODERS = {
        k: v for k, v in CHEM_ENCODERS.items()
        if issubclass(v, UndirectedMPNN) and (not allowed or k in allowed)
    }
    results = {}
    for model_name, GNN in GNN_ENCODERS.items():
        try:
            checkpoint_path = _resolve_pretrained_gnn_checkpoint(config, model_name)
        except Exception as exc:
            print(f"\n[GNN-pretrained-probe][SKIP] {model_name}: {exc}")
            results[model_name] = {seed: {"error": str(exc)} for seed in seeds}
            continue

        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "gnn_pretrained_probe", seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(
            f"\n[GNN-pretrained-probe] {model_name} — {len(cached_pairs)} cached, "
            f"{len(remaining)} to run; checkpoint={checkpoint_path}"
        )

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_jobs"))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_gnn_probe_seed(
                    model_name,
                    GNN,
                    seed,
                    config,
                    probe_df_split_path,
                    gpu_id,
                    cache_tag="gnn_pretrained_probe",
                    checkpoint_path=checkpoint_path,
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def gnn_benchmark(config: dict, split_graph_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Benchmark all UndirectedMPNN encoders across seeds in parallel.
    Workers load the split from disk independently to avoid serializing the
    large InMemoryDataset N times via loky.
    """
    allowed = set(models or config.get("gnn_models") or [])
    GNN_ENCODERS = {
        k: v for k, v in CHEM_ENCODERS.items()
        if issubclass(v, UndirectedMPNN) and (not allowed or k in allowed)
    }
    results = {}
    for model_name, GNN in GNN_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "gnn", seeds)
        remaining    = [s for s in seeds if s not in cached_pairs]
        print(f"\n[GNN] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_jobs"))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_gnn_seed(
                    model_name, GNN, seed, config, split_graph_path, gpu_id
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def tf_probe_benchmark(config: dict, split_df_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Benchmark Transformer encoders with linear probing, seeds in parallel.

    models : optional list of model names to run (e.g. ["ChemBerta"]).
             Falls back to config["tf_models"] if set, otherwise runs all.
    Workers load the split from disk independently to avoid serializing it
    N times via loky.
    """
    allowed     = set(models or config.get("tf_models") or [])
    TF_ENCODERS = {k: v for k, v in CHEM_ENCODERS.items()
                   if issubclass(v, Transformer) and (not allowed or k in allowed)}

    results = {}
    for model_name, transformer_cls in TF_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "probe", seeds)
        remaining    = [s for s in seeds if s not in cached_pairs]
        print(f"\n[TF-probe] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_jobs"))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_tf_seed(
                    model_name, transformer_cls, seed, config, split_df_path, "probe", gpu_id
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def tf_finetuned_probe_benchmark(config: dict, split_df_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Evaluate a fine-tuned ChemBERTa backbone on the BioTox OOD split.

    This is the second-stage chemical-only baseline used for BioTox
    comparison: the ChemBERTa backbone must already have been fine-tuned on
    the leakage-controlled chemical pretraining split, then frozen while the
    same task-family MLP head used by the concat models is trained on the
    matched BioTox train fold and evaluated on the matched OOD test fold.
    """
    allowed = set(models or config.get("tf_models") or [])
    TF_ENCODERS = {
        k: v for k, v in CHEM_ENCODERS.items()
        if issubclass(v, Transformer) and (not allowed or k in allowed)
    }

    results = {}
    for model_name, transformer_cls in TF_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "finetuned_probe", seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(
            f"\n[TF-finetuned-probe] {model_name} — {len(cached_pairs)} cached, "
            f"{len(remaining)} to run"
        )

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_tf_jobs", 1))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_tf_seed(
                    model_name,
                    transformer_cls,
                    seed,
                    config,
                    split_df_path,
                    "finetuned_probe",
                    gpu_id,
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def fp_probe_benchmark(config: dict, split_df_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Benchmark fingerprint encoders with trainable family heads, seeds in parallel.
    """
    allowed = set(models or config.get("fp_models") or [])
    FP_ENCODERS = {
        k: v for k, v in CHEM_ENCODERS.items()
        if issubclass(v, Fingerprint) and (not allowed or k in allowed)
    }

    results = {}
    for model_name, fingerprint_cls in FP_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "fp_probe", seeds)
        remaining = [s for s in seeds if s not in cached_pairs]
        print(f"\n[FP-probe] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_jobs"))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_fp_seed(
                    model_name, fingerprint_cls, seed, config, split_df_path, gpu_id
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def tf_finetune_benchmark(config: dict, split_df_path: str, seeds: list, models: Optional[list] = None) -> dict:
    """
    Benchmark Transformer encoders with full fine-tuning, seeds in parallel.

    models : optional list of model names to run (e.g. ["ChemBerta"]).
             Falls back to config["tf_models"] if set, otherwise runs all.
    Workers load the split from disk independently to avoid serializing it
    N times via loky.
    Concurrency is capped by config["n_tf_jobs"] (default 1) to prevent GPU OOM.
    """
    allowed = set(models or config.get("tf_models") or [])
    TF_ENCODERS = {k: v for k, v in CHEM_ENCODERS.items()
                   if issubclass(v, Transformer) and (not allowed or k in allowed)}

    results = {}
    for model_name, transformer_cls in TF_ENCODERS.items():
        cached_pairs = _load_cached_seeds(config["save_dir"], model_name, "finetune", seeds)
        remaining    = [s for s in seeds if s not in cached_pairs]
        print(f"\n[TF-finetune] {model_name} — {len(cached_pairs)} cached, {len(remaining)} to run")

        new_pairs = {}
        if remaining:
            n_jobs, gpu_ids = _job_gpu_ids(config, len(remaining), max_jobs=config.get("n_tf_jobs", 1))
            jobs = [
                (lambda seed=seed, gpu_id=gpu_id: _run_tf_seed(
                    model_name, transformer_cls, seed, config, split_df_path, "finetune", gpu_id
                ))
                for seed, gpu_id in zip(remaining, gpu_ids)
            ]
            new_pairs = dict(_run_seed_jobs(jobs, n_jobs))

        results[model_name] = {**cached_pairs, **new_pairs}

    return results


def _metric_row_from_metrics(metrics: dict, task_cols: Optional[list] = None) -> dict:
    row = {
        k: v for k, v in metrics.items()
        if k != "error" and not isinstance(v, list)
    }
    if not task_cols:
        return row

    indices = metrics.get("per_task_indices")
    if indices is None:
        indices = list(range(len(metrics.get("per_task_auroc", []))))
    for metric in ("auroc", "auprc", "acc", "random_auprc"):
        values = metrics.get(f"per_task_{metric}")
        if not isinstance(values, list):
            continue
        for idx, value in zip(indices, values):
            try:
                task = task_cols[int(idx)]
            except (TypeError, ValueError, IndexError):
                continue
            row[f"task:{task}:{metric}"] = value
    return row


def results_to_df(results: dict, label_suffix: str = "", task_cols: Optional[list] = None) -> pd.DataFrame:
    rows = []
    for name, seed_results in results.items():
        for seed, metrics in seed_results.items():
            row = {"model": f"{name}{label_suffix}", "seed": seed}
            if isinstance(metrics, dict):
                row.update(_metric_row_from_metrics(metrics, task_cols=task_cols))
            rows.append(row)
    return pd.DataFrame(rows)


_RESULT_KEYS = {
    "fp_probe":    "",
    "gnn":         "",
    "gnn_probe":   "\n(probed)",
    "tf_finetuned_probe": "\n(fine-tuned OOD)",
    "tf_probe":    "\n(linear-probed)",
    "tf_finetune": "\n(fine-tuned)",
}


def summarize_results(
    config:            dict,
    dict_results:      dict,
    prevalence_positive: float,
    encoder_registry:  Optional[dict] = None,
) -> tuple:
    """
    Aggregate per-seed results, print mean ± std, and save CSVs.

    Parameters
    ----------
    config            : provides save_dir, task_dataset, run_* toggle flags
    dict_results      : {"gnn": {...}, "tf_probe": {...}, "tf_finetune": {...}}
    prevalence_positive : AUPRC of a random classifier
    encoder_registry  : CHEM_ENCODERS.
                        If provided, only model names present in the registry
                        are included and output follows registry order.
                        If None, all keys in dict_results are included.

    Returns
    ───────
    (df_mean, df_std)  both indexed by model name
    """
    save_dir     = config["save_dir"]
    task_dataset = config["task_dataset"]
    task_mode    = config["task_mode"]

    parts = [pd.DataFrame([{
        "model": "Random", 
        "seed": 0,
        "auroc": 0.5, 
        "auprc": prevalence_positive,
    }])]

    for key, suffix in _RESULT_KEYS.items():
        data = dict_results.get(key, {})
        if not data:
            continue

        if encoder_registry is not None:
            data = {
                name: seed_results
                for name, seed_results in data.items()
                if name in encoder_registry
            }

        if data:
            parts.append(results_to_df(data, label_suffix=suffix, task_cols=config.get("task_cols")))

    df_all = pd.concat(parts, ignore_index=True)
    excluded_cols = {"model", "seed", "pooling_method"}
    metric_cols = []
    for col in df_all.columns:
        if col in excluded_cols:
            continue
        numeric = pd.to_numeric(df_all[col], errors="coerce")
        if numeric.notna().any():
            df_all[col] = numeric
            metric_cols.append(col)
    df_mean = df_all.groupby("model")[metric_cols].mean().round(4)
    df_std = df_all.groupby("model")[metric_cols].std().round(4)
    df_summary  = df_mean.astype(str) + " ± " + df_std.fillna(0).astype(str)

    print("\nMean ± Std\n")
    print(df_summary.to_string())

    os.makedirs(save_dir, exist_ok=True)
    df_all.to_csv(
        os.path.join(save_dir, f"benchmark_chem_{task_dataset}_{task_mode}_per_seed.csv"),
        index=False,
    )
    df_summary.to_csv(
        os.path.join(save_dir, f"benchmark_chem_{task_dataset}_{task_mode}_summary.csv")
    )
    with open(os.path.join(save_dir, f"benchmark_chem_{task_dataset}_{task_mode}.json"), "w") as f:
        json.dump({
            grp: {
                model: {str(seed): metrics for seed, metrics in seed_dict.items()}
                for model, seed_dict in model_dict.items()
            }
            for grp, model_dict in dict_results.items()
        }, f, indent=2)
    print(f"\nSaved: {save_dir}")

    return df_mean, df_std


def load_benchmark_results(config: dict) -> Optional[dict]:
    """Return saved dict_results from JSON, or None if not found."""
    path = os.path.join(
        config["save_dir"],
        f"benchmark_chem_{config['task_dataset']}_{config['task_mode']}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        grp: {
            model: {int(seed): metrics for seed, metrics in seed_dict.items()}
            for model, seed_dict in model_dict.items()
        }
        for grp, model_dict in raw.items()
    }
# %%
