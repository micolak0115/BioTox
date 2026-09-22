# chem/trainer.py
# %%
import os
import copy
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
)

import sys
sys.path.append(str(Path(__file__).parents[1]))
from chem.utils import resolve_device


def init_best_score(task_mode: str) -> float:
    if task_mode in (
        "binary_classification",
        "multilabel_classification",
    ):
        return float("-inf")
    return float("inf")


def is_better(
    current:   float,
    best:      float,
    task_mode: str,
    min_delta: float = 0.0,
) -> bool:
    if task_mode in (
        "binary_classification",
        "multilabel_classification",
    ):
        return current > best + min_delta
    return current < best - min_delta


def primary_metric(metrics: dict, task_mode: str) -> float:
    if task_mode in (
        "binary_classification",
        "multilabel_classification",
    ):
        return metrics.get("auprc", float("-inf"))
    return metrics.get("rmse", float("inf"))


# ------------------------------------------------------------------
# loss
# ------------------------------------------------------------------
def get_loss_fn(
    task_mode:  str,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    binary_classification:
        BCEWithLogitsLoss
        pos_weight: scalar or (1,)

    multilabel_classification:
        MaskedBCELoss — handles NaN labels per task
        pos_weight: (n_tasks,) per-task class weight
                    pos_weight_t = N_negative_t / N_positive_t

    regression:
        MSELoss
    """
    if task_mode == "binary_classification":
        return MaskedBCELoss(pos_weight=pos_weight)

    if task_mode == "multilabel_classification":
        return MaskedBCELoss(pos_weight=pos_weight)

    if task_mode == "regression":
        return nn.MSELoss()

    raise ValueError(
        f"task_mode must be 'binary_classification', "
        f"'multilabel_classification', or 'regression'. Got: {task_mode!r}"
    )


class MaskedBCELoss(nn.Module):
    """
    Masked BCE loss for multilabel binary classification.
    Standard implementation for Tox21-style datasets.

    Handles NaN labels — each task independently masked.

    pos_weight: (n_tasks,)
                pos_weight_t = N_negative_t / N_positive_t
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        # register as buffer so it moves with .to(device)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits:  torch.Tensor,   # (N, n_tasks)
        targets: torch.Tensor,   # (N, n_tasks) — NaN for missing
    ) -> torch.Tensor:

        # mask missing labels
        mask    = ~torch.isnan(targets)
        targets = torch.nan_to_num(targets, nan=0.0)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight = self.pos_weight.to(logits.device)
                         if self.pos_weight is not None else None,
            reduction  = "none",          # (N, n_tasks)
        )

        # average over valid entries only
        masked_loss = (loss * mask.float()).sum()
        n_valid     = mask.float().sum().clamp(min=1.0)

        return masked_loss / n_valid


# ------------------------------------------------------------------
# pos_weight computation
# ------------------------------------------------------------------
def compute_pos_weight(
    labels:    np.ndarray,
    task_mode: str,
) -> torch.Tensor:
    """
    Compute pos_weight for imbalanced classification.

    binary_classification:
        scalar — N_negative / N_positive

    multilabel_classification:
        (n_tasks,) — per task N_negative_t / N_positive_t
    """
    labels = np.array(labels)

    if task_mode == "binary_classification":
        valid = labels[~np.isnan(labels)]
        pos   = (valid == 1).sum()
        neg   = (valid == 0).sum()
        pw    = neg / pos if pos > 0 else 1.0
        return torch.tensor([pw], dtype=torch.float32)

    if task_mode == "multilabel_classification":
        assert labels.ndim == 2, \
            "multilabel labels must be (N, n_tasks)"
        n_tasks = labels.shape[1]
        weights = []
        for t in range(n_tasks):
            col   = labels[:, t]
            valid = col[~np.isnan(col)]
            pos   = (valid == 1).sum()
            neg   = (valid == 0).sum()
            weights.append(neg / pos if pos > 0 else 1.0)
        return torch.tensor(weights, dtype=torch.float32)

    return torch.tensor([1.0], dtype=torch.float32)


# ------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------
def compute_metrics(
    preds:     np.ndarray,
    labels:    np.ndarray,
    task_mode: str,
) -> dict:
    """
    preds:     sigmoid probabilities (NOT logits)
               binary_classification:      (N,)
               multilabel_classification:  (N, n_tasks)
               regression:                 (N,)

    labels:    ground truth
               binary_classification:      (N,)
               multilabel_classification:  (N, n_tasks) — NaN for missing
               regression:                 (N,)

    Tox21 standard evaluation (multilabel_classification):
        mean ROC-AUC across valid tasks  ← primary metric
        mean PR-AUC  across valid tasks  ← secondary metric
        per-task scores returned for detailed analysis
    """
    preds  = np.array(preds)
    labels = np.array(labels)

    # flatten (N,1) → (N,)
    if preds.ndim  == 2 and preds.shape[1]  == 1:
        preds  = preds.squeeze(1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    # ------------------------------------------------------------------
    # binary_classification — single endpoint
    # ------------------------------------------------------------------
    if task_mode == "binary_classification":
        valid = ~np.isnan(labels)
        if valid.sum() == 0:
            return {}
        if len(np.unique(labels[valid])) < 2:
            return {}

        return {
            "auroc":        float(roc_auc_score(labels[valid], preds[valid])),
            "auprc":        float(average_precision_score(labels[valid], preds[valid])),
            "acc":          float(((preds[valid] > 0.5).astype(int) == labels[valid]).mean()),
            "random_auprc": float(labels[valid].mean()),
        }

    # ------------------------------------------------------------------
    # multilabel_classification — Tox21 standard
    # 12 independent binary tasks
    # mean ROC-AUC across valid tasks is the primary metric
    # ------------------------------------------------------------------
    elif task_mode == "multilabel_classification":
        assert preds.ndim  == 2, \
            f"multilabel preds must be (N, n_tasks), got {preds.shape}"
        assert labels.ndim == 2, \
            f"multilabel labels must be (N, n_tasks), got {labels.shape}"
        assert preds.shape[1] == labels.shape[1], \
            f"preds tasks ({preds.shape[1]}) != labels tasks ({labels.shape[1]})"

        n_tasks        = preds.shape[1]
        per_task_auroc = []
        per_task_auprc = []
        per_task_acc   = []
        per_task_random_auprc = []
        per_task_indices = []
        skipped        = []

        for t in range(n_tasks):
            y_true = labels[:, t]
            y_pred = preds[:, t]

            # mask missing labels
            valid = ~np.isnan(y_true)
            if valid.sum() == 0:
                skipped.append(t)
                continue

            y_true_v = y_true[valid]
            y_pred_v = y_pred[valid]

            # need both classes for ROC-AUC
            if len(np.unique(y_true_v)) < 2:
                skipped.append(t)
                continue

            per_task_auroc.append(
                float(roc_auc_score(y_true_v, y_pred_v))
            )
            per_task_auprc.append(
                float(average_precision_score(y_true_v, y_pred_v))
            )
            per_task_acc.append(
                float(((y_pred_v > 0.5).astype(int) == y_true_v).mean())
            )
            per_task_random_auprc.append(float(y_true_v.mean()))
            per_task_indices.append(t)

        if len(per_task_auroc) == 0:
            return {
                "auroc": float("nan"),
                "auprc": float("nan"),
                "acc":   float("nan"),
            }

        if skipped:
            print(
                f"[metrics] skipped {len(skipped)} tasks "
                f"(no valid labels or single class): {skipped}"
            )

        return {
            # primary metrics — means across valid tasks
            "auroc":           float(np.mean(per_task_auroc)),
            "auprc":           float(np.mean(per_task_auprc)),
            "acc":             float(np.mean(per_task_acc)),
            "random_auprc":    float(np.mean(per_task_random_auprc)),
            # per-task breakdown for detailed analysis
            "per_task_auroc":  per_task_auroc,
            "per_task_auprc":  per_task_auprc,
            "per_task_acc":    per_task_acc,
            "per_task_random_auprc": per_task_random_auprc,
            "per_task_indices": per_task_indices,
            "n_valid_tasks":   len(per_task_auroc),
            "n_skipped_tasks": len(skipped),
        }

    # ------------------------------------------------------------------
    # regression
    # ------------------------------------------------------------------
    elif task_mode == "regression":
        valid = ~np.isnan(labels)
        if valid.sum() == 0:
            return {}
        return {
            "rmse": float(np.sqrt(mean_squared_error(labels[valid], preds[valid]))),
            "mae":  float(mean_absolute_error(labels[valid], preds[valid])),
            "r2":   float(
                1 - np.sum((labels[valid] - preds[valid]) ** 2) /
                    np.sum((labels[valid] - labels[valid].mean()) ** 2)
            ),
        }

    else:
        raise ValueError(
            f"Unknown task_mode: {task_mode!r}. "
            "Choose from: binary_classification, multilabel_classification, regression"
        )


# ------------------------------------------------------------------
# UndirectedMPNN trainer
# ------------------------------------------------------------------
class UndirectedMPNN:
    """Training loop for PyG-based GNN models (GCN, GAT, GIN, GraphSAGE)."""

    def __init__(
        self,
        task_mode:  str           = "binary_classification",
        lr:         float         = 1e-3,
        epochs:     int           = 100,
        seed:       int           = 42,
        save_path:  Optional[str] = None,
        patience:   Optional[int] = None,
        min_delta:  float         = 0.0,
        device                    = None,
    ):
        self.task_mode = task_mode
        self.lr        = lr
        self.epochs    = epochs
        self.seed      = seed
        self.save_path = save_path
        self.patience  = patience
        self.min_delta = min_delta
        self.device    = device

    def _train_epoch(self, model, loader, optimizer, criterion, device) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_hat = model(batch)
            y     = batch.y.float()
            if y_hat.ndim < y.ndim:
                y = y.squeeze(-1)
            if self.task_mode == "binary_classification":
                valid = ~torch.isnan(y)
                if valid.sum() == 0:
                    continue
                y_hat, y = y_hat[valid], y[valid]
            loss  = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, model, loader, device) -> dict:
        model.eval()
        preds, labels = [], []
        for batch in loader:
            batch = batch.to(device)
            y_hat = model(batch)
            # sigmoid for both binary and multilabel — NOT softmax
            pred  = torch.sigmoid(y_hat).cpu() \
                    if self.task_mode != "regression" \
                    else y_hat.cpu()
            y = batch.y.float().cpu()
            if pred.ndim < y.ndim:
                y = y.squeeze(-1)
            preds.append(pred)
            labels.append(y)
        return compute_metrics(
            torch.cat(preds).numpy(),
            torch.cat(labels).numpy(),
            self.task_mode,
        )

    def fit(
        self,
        split,
        model:      nn.Module,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        device    = resolve_device(self.device)
        model     = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = get_loss_fn(
            self.task_mode,
            pos_weight = pos_weight.to(device) if pos_weight is not None else None,
        )

        best_score = init_best_score(self.task_mode)
        best_state, best_epoch, no_improve = None, 0, 0

        for epoch in range(1, self.epochs + 1):
            train_loss  = self._train_epoch(model, split.train, optimizer, criterion, device)
            val_metrics = self._eval_epoch(model, split.valid, device)
            current     = primary_metric(val_metrics, self.task_mode)

            if is_better(current, best_score, self.task_mode, self.min_delta):
                best_score = current
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                no_improve = 0
                if self.save_path:
                    os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                    torch.save({
                        "epoch":            epoch,
                        "model_state_dict": best_state,
                        "best_score":       best_score,
                        "task_mode":        self.task_mode,
                    }, self.save_path)
            else:
                no_improve += 1

            if epoch % 10 == 0:
                if self.task_mode != "regression":
                    print(
                        f"epoch {epoch:3d} | loss={train_loss:.4f} | "
                        f"auroc={val_metrics.get('auroc', float('nan')):.4f} | "
                        f"auprc={val_metrics.get('auprc', float('nan')):.4f} | "
                        f"best={best_epoch}"
                    )
                else:
                    print(
                        f"epoch {epoch:3d} | loss={train_loss:.4f} | "
                        f"rmse={val_metrics.get('rmse', float('nan')):.4f} | "
                        f"best={best_epoch}"
                    )

            if self.patience and no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch}).")
                break

        if best_state is None:
            raise RuntimeError("Training produced no improvement.")
        model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def predict(self, model: nn.Module, split) -> dict:
        device = resolve_device(self.device) if self.device is not None else next(model.parameters()).device
        model = model.to(device)
        return self._eval_epoch(model, split.test, device)


# ------------------------------------------------------------------
# BioModel trainer
# ------------------------------------------------------------------
class BioModel:
    """Training loop for BioModel (bio/model.py)."""

    def __init__(
        self,
        task_mode:  str           = "binary_classification",
        lr:         float         = 1e-3,
        epochs:     int           = 100,
        seed:       int           = 42,
        save_path:  Optional[str] = None,
        patience:   Optional[int] = None,
        min_delta:  float         = 0.0,
        device                    = None,
    ):
        self.task_mode = task_mode
        self.lr        = lr
        self.epochs    = epochs
        self.seed      = seed
        self.save_path = save_path
        self.patience  = patience
        self.min_delta = min_delta
        self.device    = device

    def _train_epoch(self, model, loader, optimizer, criterion, device) -> float:
        model.train()
        total_loss = 0.0
        n_samples  = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_hat = model(batch)
            loss  = criterion(y_hat, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.y.size(0)
            n_samples  += batch.y.size(0)
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _eval_epoch(self, model, loader, device) -> dict:
        model.eval()
        preds, labels = [], []
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu() \
                     if self.task_mode != "regression" \
                     else logits.cpu()
            preds.append(probs)
            labels.append(batch.y.float().cpu())
        return compute_metrics(
            torch.cat(preds).numpy(),
            torch.cat(labels).numpy(),
            self.task_mode,
        )

    def fit(self, split, model, pos_weight: Optional[torch.Tensor] = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        device    = resolve_device(self.device)
        model     = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        criterion = get_loss_fn(
            self.task_mode,
            pos_weight = pos_weight.to(device) if pos_weight is not None else None,
        )

        best_score = init_best_score(self.task_mode)
        best_state, best_epoch, no_improve = None, 0, 0

        for epoch in range(1, self.epochs + 1):
            train_loss  = self._train_epoch(model, split.train, optimizer, criterion, device)
            val_metrics = self._eval_epoch(model, split.valid, device)
            current     = primary_metric(val_metrics, self.task_mode)

            if is_better(current, best_score, self.task_mode, self.min_delta):
                best_score = current
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                no_improve = 0
                if self.save_path:
                    os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
                    torch.save({
                        "epoch":            epoch,
                        "model_state_dict": best_state,
                        "best_score":       best_score,
                        "task_mode":        self.task_mode,
                    }, self.save_path)
            else:
                no_improve += 1

            if self.task_mode != "regression":
                print(
                    f"epoch {epoch:3d} | loss={train_loss:.4f} | "
                    f"auroc={val_metrics.get('auroc', float('nan')):.4f} | "
                    f"auprc={val_metrics.get('auprc', float('nan')):.4f} | "
                    f"best={best_epoch}"
                )
            else:
                print(
                    f"epoch {epoch:3d} | loss={train_loss:.4f} | "
                    f"rmse={val_metrics.get('rmse', float('nan')):.4f} | "
                    f"best={best_epoch}"
                )

            if self.patience and no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch}).")
                break

        if best_state is None:
            raise RuntimeError("Training produced no improvement.")
        model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def predict(self, model, split) -> dict:
        device = resolve_device(self.device) if self.device is not None else next(model.parameters()).device
        model = model.to(device)
        return self._eval_epoch(model, split.test, device)
