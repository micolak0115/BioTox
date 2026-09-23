"""Publication figures for the Stage-1 Top-K ensemble development analysis.

Candidate models are ranked separately within each endpoint by their
Stage-1 development-test AUPRC. The K=4 arithmetic mean selected here is frozen
before any matched-cohort Stage-2 outcome is evaluated. Stage-1 test results
are model-development evidence, not an unbiased post-selection performance
estimate for the deployed K=4 ensemble.
"""
from __future__ import annotations

import argparse
import atexit
import os
import pickle
import shutil
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


HERE = Path(__file__).resolve().parent
_STANDALONE_ROOT = HERE.parent  # pipeline root; retained name for config compatibility
if str(_STANDALONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_ROOT))


@lru_cache(maxsize=None)
def _default_alias_root() -> Path:
    """Ephemeral per-process alias root: avoids leaving a persistent
    _pkg_alias/ directory in the tracked standalone/ tree; removed
    automatically when the process exits. Set BIOTOX_PACKAGE_ALIAS_ROOT
    to override with a fixed, non-cleaned-up location instead."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="biotox_pkg_alias_"))
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    return tmp_dir


def _register_bundled_package(name: str, *locations: Path) -> None:
    """Make a bundled sibling ``run/`` directory importable under its
    original package name.

    An in-memory ``sys.modules`` entry alone is invisible to spawned
    worker processes (e.g. joblib/loky candidate-fitting workers):
    each starts a fresh interpreter that must resolve
    ``publication.X`` / ``stage1.X`` imports itself. This also creates
    ONE namespace-package directory
    (``_STANDALONE_ROOT/_pkg_alias/<name>``, no ``__init__.py`` of
    its own) containing a flat, relative on-disk symlink to every
    ``.py`` file from every ``location``, and adds its PARENT
    directory to ``sys.path`` and ``PYTHONPATH`` -- ``sys.path``
    because multiprocessing's spawn start method propagates the
    parent's ``sys.path`` to workers rather than re-deriving it from
    ``PYTHONPATH``; ``PYTHONPATH`` for plain ``subprocess.run`` stage
    launches, which DO read it at their own interpreter startup.
    """
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(location) for location in locations]
        sys.modules[name] = module

    alias_root = (
        Path(os.environ["BIOTOX_PACKAGE_ALIAS_ROOT"])
        if os.environ.get("BIOTOX_PACKAGE_ALIAS_ROOT")
        else _default_alias_root()
    )
    alias_dir = alias_root / name
    alias_dir.mkdir(parents=True, exist_ok=True)
    for location in locations:
        for source in sorted(location.glob("*.py")):
            if source.name == "__init__.py":
                continue
            link = alias_dir / source.name
            if not link.exists():
                link.symlink_to(os.path.relpath(source, alias_dir))
    if str(alias_root) not in sys.path:
        sys.path.insert(0, str(alias_root))
    existing_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    merged = os.pathsep.join(
        entry for entry in [str(alias_root)] + existing_entries if entry
    )
    os.environ["PYTHONPATH"] = merged


from stage2.utils import TOX21_TASKS  # noqa: E402


DEFAULT_STAGE1_DIR = _STANDALONE_ROOT / "_run_output" / "chem_offset_valid_tuned_classical"
DEFAULT_OUTPUT_DIR = (
    _STANDALONE_ROOT / "_run_output" / "stage1_topk_summary_fresh_v1"
)
FIXED_SEED = 1
K_REFERENCE = 4
BOOTSTRAP_SEED = 20260723

NR_COLOR = "#3B73A7"
SR_COLOR = "#BF4F52"
K4_COLOR = "#D89B2B"
DEPLOYED_COLOR = "#5641B5"
MACRO_COLOR = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create revised publication figures for Stage-1 Top-K ensembles."
    )
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _metric_pair(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    return (
        float(average_precision_score(y_true, y_score)),
        float(roc_auc_score(y_true, y_score)),
    )


def _bootstrap_metric_sd(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if positive.size == 0 or negative.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    auprc = np.empty(n_bootstrap, dtype=float)
    auroc = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sample = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        auprc[bootstrap_index], auroc[bootstrap_index] = _metric_pair(
            y_true[sample], y_score[sample]
        )
    return float(np.std(auprc, ddof=1)), float(np.std(auroc, ddof=1))


def build_topk_results(
    stage1_dir: Path,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = stage1_dir / "task_benchmark_cache"
    curve_rows = []
    reference_rows = []

    for task_index, task in enumerate(TOX21_TASKS):
        cache_path = cache_dir / f"{task}.pkl"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing Stage-1 cache: {cache_path}")
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)

        if int(cache.get("seed", -1)) != FIXED_SEED:
            raise ValueError(
                f"{cache_path} uses seed={cache.get('seed')}; expected {FIXED_SEED}"
            )
        if cache.get("ensemble_rule") != "mean":
            raise ValueError(f"{cache_path} does not use the fixed mean rule")

        candidate_order = list(cache["candidate_names"])
        y_true = np.asarray(cache["y_test"], dtype=int).reshape(-1)
        predictions = {
            candidate: np.asarray(
                cache["test_preds_by_candidate"][candidate], dtype=float
            ).reshape(-1)
            for candidate in candidate_order
        }
        if any(prediction.size != y_true.size for prediction in predictions.values()):
            raise ValueError(f"{cache_path} contains prediction length mismatch")

        own_auprc = {
            candidate: average_precision_score(y_true, prediction)
            for candidate, prediction in predictions.items()
        }
        original_position = {
            candidate: index for index, candidate in enumerate(candidate_order)
        }
        ranked = sorted(
            candidate_order,
            key=lambda candidate: (
                -own_auprc[candidate],
                original_position[candidate],
            ),
        )

        task_curves = []
        for k in range(1, len(ranked) + 1):
            topk_prediction = np.mean(
                np.column_stack([predictions[name] for name in ranked[:k]]),
                axis=1,
            )
            auprc, auroc = _metric_pair(y_true, topk_prediction)
            row = {
                "family": "NR" if task.startswith("NR-") else "SR",
                "task": task,
                "k": k,
                "auprc": auprc,
                "auroc": auroc,
                "kth_model_added": ranked[k - 1],
                "kth_model_own_auprc": own_auprc[ranked[k - 1]],
                "n_test": y_true.size,
                "n_positive": int(y_true.sum()),
                "prevalence": float(y_true.mean()),
            }
            curve_rows.append(row)
            task_curves.append(row)

        k4_prediction = np.mean(
            np.column_stack(
                [predictions[name] for name in ranked[:K_REFERENCE]]
            ),
            axis=1,
        )
        k4_auprc_sd, k4_auroc_sd = _bootstrap_metric_sd(
            y_true,
            k4_prediction,
            n_bootstrap=n_bootstrap,
            seed=BOOTSTRAP_SEED + task_index,
        )
        task_frame = pd.DataFrame(task_curves)
        peak_index = int(task_frame["auprc"].idxmax())
        peak = task_frame.loc[peak_index]
        k4 = task_frame.loc[task_frame["k"] == K_REFERENCE].iloc[0]
        reference_rows.append(
            {
                "family": "NR" if task.startswith("NR-") else "SR",
                "task": task,
                "peak_k": int(peak["k"]),
                "peak_auprc": float(peak["auprc"]),
                "k4_auprc": float(k4["auprc"]),
                "k4_auprc_bootstrap_sd": k4_auprc_sd,
                "peak_within_k4_auprc_plus_1sd": bool(
                    float(peak["auprc"]) <= float(k4["auprc"]) + k4_auprc_sd
                ),
                "k4_auroc": float(k4["auroc"]),
                "k4_auroc_bootstrap_sd": k4_auroc_sd,
                "ranked_candidates": ";".join(ranked),
            }
        )

    curve = pd.DataFrame(curve_rows)
    reference = pd.DataFrame(reference_rows)
    return curve, reference


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_per_task(
    curve: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    _style()
    figure, axes = plt.subplots(3, 4, figsize=(16.8, 11.2))
    axes = axes.ravel()

    for ax, task in zip(axes, TOX21_TASKS):
        task_curve = curve[curve["task"] == task].sort_values("k")
        task_reference = reference.loc[reference["task"] == task].iloc[0]
        family = task_reference["family"]
        color = NR_COLOR if family == "NR" else SR_COLOR

        k_values = task_curve["k"].to_numpy(dtype=int)
        values = task_curve["auprc"].to_numpy(dtype=float)
        k4_value = float(task_reference["k4_auprc"])
        k4_sd = float(task_reference["k4_auprc_bootstrap_sd"])
        peak_k = int(task_reference["peak_k"])
        peak_value = float(task_reference["peak_auprc"])

        ax.axhspan(
            k4_value - k4_sd,
            k4_value + k4_sd,
            color=K4_COLOR,
            alpha=0.14,
            linewidth=0,
            zorder=0,
        )
        ax.axvline(
            K_REFERENCE,
            color=K4_COLOR,
            linestyle=(0, (2, 3)),
            linewidth=1.0,
            alpha=0.85,
            zorder=1,
        )
        ax.plot(
            k_values,
            values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.2,
            zorder=2,
        )
        ax.scatter(
            [K_REFERENCE],
            [k4_value],
            s=74,
            marker="o",
            facecolor=K4_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        ax.scatter(
            [peak_k],
            [peak_value],
            s=86,
            marker="*",
            facecolor="#E66B2D",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        deployed = task_curve.loc[task_curve["k"] == 12, "auprc"].iloc[0]
        ax.scatter(
            [12],
            [deployed],
            s=52,
            marker="s",
            facecolor=DEPLOYED_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

        padding = max(0.012, 0.11 * (values.max() - values.min()))
        lower = min(values.min(), k4_value - k4_sd) - padding
        upper = max(values.max(), k4_value + k4_sd) + padding
        ax.set_ylim(max(0.0, lower), min(1.0, upper))
        ax.set_xlim(0.6, 12.4)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_title(task, color=color, fontweight="bold", pad=7)
        ax.grid(axis="y", color="#E2E2E2", linewidth=0.7)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for row in range(3):
        axes[row * 4].set_ylabel("Development-selection AUPRC")
    for ax in axes[8:]:
        ax.set_xlabel("Top-K ensemble size")

    coverage = int(reference["peak_within_k4_auprc_plus_1sd"].sum())
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#676767",
            marker="o",
            markerfacecolor="white",
            linewidth=1.8,
            label="Top-K mean ensemble",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markerfacecolor=K4_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="K=4",
        ),
        Patch(
            facecolor=K4_COLOR,
            alpha=0.18,
            edgecolor="none",
            label="K=4 ±1 bootstrap SD",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markerfacecolor="#E66B2D",
            markeredgecolor="white",
            markersize=11,
            label="Observed task peak",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="s",
            markerfacecolor=DEPLOYED_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="K=12 deployed mean",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    figure.text(
        0.5,
        0.025,
        (
            f"The observed task-specific peak lies within the K=4 +1 bootstrap "
            f"SD limit for {coverage}/{len(reference)} endpoints. Bands quantify "
            "development-set compound-resampling variability at K=4; ranking "
            "AUPRC is descriptive and was not used to select the deployed model."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.91,
        bottom=0.105,
        wspace=0.25,
        hspace=0.34,
    )
    _save(
        figure,
        output_dir / "stage1_topk_ensemble_per_task_revised",
        dpi=dpi,
    )


def _scatter_task_distributions(
    ax: plt.Axes,
    curve: pd.DataFrame,
    metric: str,
    metric_label: str,
) -> None:
    k_values = np.arange(1, 13)
    offsets = np.linspace(-0.24, 0.24, len(TOX21_TASKS))

    for task_index, task in enumerate(TOX21_TASKS):
        task_frame = curve[curve["task"] == task].sort_values("k")
        family = task_frame["family"].iloc[0]
        color = NR_COLOR if family == "NR" else SR_COLOR
        ax.scatter(
            k_values + offsets[task_index],
            task_frame[metric].to_numpy(dtype=float),
            s=33,
            facecolor=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.82,
            zorder=2,
        )

    macro = curve.groupby("k", sort=True)[metric].agg(["mean", "std"])
    k4_mean = float(macro.loc[K_REFERENCE, "mean"])
    k4_sd = float(macro.loc[K_REFERENCE, "std"])
    ax.axhspan(
        k4_mean - k4_sd,
        k4_mean + k4_sd,
        color=K4_COLOR,
        alpha=0.11,
        linewidth=0,
        zorder=0,
    )
    ax.axhline(
        k4_mean,
        color=K4_COLOR,
        linestyle=(0, (2, 3)),
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )
    ax.axvline(
        K_REFERENCE,
        color=K4_COLOR,
        linestyle=(0, (2, 3)),
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )
    ax.errorbar(
        macro.index.to_numpy(dtype=float),
        macro["mean"].to_numpy(dtype=float),
        yerr=macro["std"].to_numpy(dtype=float),
        fmt="D",
        color=MACRO_COLOR,
        markerfacecolor="white",
        markeredgewidth=1.25,
        markersize=5.5,
        elinewidth=1.0,
        capsize=2.4,
        linestyle="none",
        zorder=4,
        label="Macro mean ±1 SD across tasks",
    )
    ax.scatter(
        [K_REFERENCE],
        [k4_mean],
        marker="D",
        s=78,
        facecolor=K4_COLOR,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(k_values)
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_macro_distributions(
    curve: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    _style()
    figure, axes = plt.subplots(2, 1, figsize=(12.8, 9.2), sharex=True)
    _scatter_task_distributions(
        axes[0], curve, "auprc", "Development-selection AUPRC"
    )
    _scatter_task_distributions(axes[1], curve, "auroc", "Held-out test AUROC")
    axes[0].set_title("A  AUPRC across endpoints", loc="left", fontweight="bold")
    axes[1].set_title("B  AUROC across endpoints", loc="left", fontweight="bold")
    axes[1].set_xlabel("Top-K ensemble size")

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=NR_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="NR endpoint",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=SR_COLOR,
            markeredgecolor="white",
            markersize=7,
            label="SR endpoint",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="D",
            markerfacecolor="white",
            markeredgecolor=MACRO_COLOR,
            markersize=6.5,
            label="Macro mean ±1 SD across tasks",
        ),
        Patch(
            facecolor=K4_COLOR,
            alpha=0.18,
            edgecolor="none",
            label="K=4 macro mean ±1 SD",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        columnspacing=1.7,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Each colored point is one Tox21 endpoint at the indicated K. "
            "Black diamonds and error bars show the macro mean ±1 SD across "
            "the 12 endpoints; the gold band is the corresponding K=4 range."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.11,
        hspace=0.27,
    )
    _save(
        figure,
        output_dir / "stage1_topk_macro_task_scatter_revised",
        dpi=dpi,
    )

    return (
        curve.groupby("k", sort=True)
        .agg(
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
        )
        .reset_index()
    )


def _save(figure: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(
            stem.with_suffix(suffix),
            dpi=dpi if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage-1 Top-K output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    curve, reference = build_topk_results(
        stage1_dir=args.stage1_dir,
        n_bootstrap=args.n_bootstrap,
    )
    macro = plot_macro_distributions(curve, args.output_dir, dpi=args.dpi)
    plot_per_task(curve, reference, args.output_dir, dpi=args.dpi)

    curve.to_csv(
        args.output_dir / "stage1_topk_ensemble_curve_revised.csv", index=False
    )
    reference.to_csv(
        args.output_dir / "stage1_topk_k4_reference_by_task.csv", index=False
    )
    macro.to_csv(
        args.output_dir / "stage1_topk_macro_summary_revised.csv", index=False
    )

    coverage = int(reference["peak_within_k4_auprc_plus_1sd"].sum())
    print(
        f"[topk-figure] wrote revised figures for {len(reference)} endpoints; "
        f"observed peak within K=4 +1 bootstrap SD for "
        f"{coverage}/{len(reference)} endpoints"
    )


if __name__ == "__main__":
    main()
