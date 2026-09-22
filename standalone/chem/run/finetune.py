# chem/finetune.py
# %%
import os
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parents[1]))
from chem.utils import resolve_device
from chem.loader import ChemDataset, ChemCollator
from chem.trainer import (
    get_loss_fn, 
    compute_metrics, 
    init_best_score, 
    is_better, 
    primary_metric
)

def _train_epoch(model, 
                 loader, 
                 optimizer, 
                 scheduler, 
                 criterion, 
                 device, 
                 scaler, 
                 max_grad_norm, 
                 use_amp
    ):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            y_hat = model(input_ids=batch["input_ids"],
                          attention_mask=batch.get("attention_mask")).logits
            y     = batch["labels"].float()
            loss  = criterion(y_hat, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, device, task_mode, use_amp):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(enabled=use_amp):
            logits = model(
                input_ids      = batch["input_ids"],
                attention_mask = batch.get("attention_mask"),
            ).logits
        pred = torch.sigmoid(logits) if task_mode != "regression" else logits
        preds.append(pred.detach().cpu())
        labels.append(batch["labels"].detach().cpu())
    return compute_metrics(
        torch.cat(preds).numpy(),
        torch.cat(labels).numpy(),
        task_mode,
    )


def fit(
    split,
    model_wrapper,
    smiles_col: str,
    task_cols: str,
    task_mode: str,
    seed: int = 42,
    pos_weight: torch.Tensor = None,
    save_path: str = None,
    patience: int = None,
    min_delta: float = 0.0,
    lr: float = 4e-5,
    epochs: int = 10,
    batch_size: int = 32,
    max_length: int = 128,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.06,
    max_grad_norm: float = 1.0,
    num_workers: int = 0,
    use_amp: bool = False,
    device=None,
    ):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device    = resolve_device(device)
    tokenizer = model_wrapper.get_tokenizer()
    model     = model_wrapper.get_model().to(device)

    collator     = ChemCollator(tokenizer=tokenizer, max_length=max_length, task_mode=task_mode)
    pin_memory   = torch.cuda.is_available()
    loader_kwargs = dict(collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(
        ChemDataset(split.train, smiles_col, task_cols),
        batch_size=batch_size, shuffle=True, **loader_kwargs,
    )
    valid_loader = DataLoader(
        ChemDataset(split.valid, smiles_col, task_cols),
        batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps  = len(train_loader) * epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    criterion = get_loss_fn(task_mode, pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    scaler    = GradScaler(enabled=use_amp and torch.cuda.is_available())

    best_score, best_state, best_epoch, no_improve = init_best_score(task_mode), None, 0, 0

    for epoch in range(1, epochs + 1):
        train_loss  = _train_epoch(model, train_loader, optimizer, scheduler, criterion,
                                   device, scaler, max_grad_norm, use_amp and torch.cuda.is_available())
        val_metrics = _eval_epoch(model, valid_loader, device, task_mode, use_amp and torch.cuda.is_available())
        current     = primary_metric(val_metrics, task_mode)

        if is_better(current, best_score, task_mode, min_delta):
            best_score = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        _log_epoch(epoch, train_loss, val_metrics, task_mode, best_epoch)

        if patience and no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}).")
            break

    if best_state is None:
        raise RuntimeError("No improvement was recorded during training.")

    model.load_state_dict(best_state)
    model_wrapper.model = model

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_wrapper.save_pretrained(save_path)
        torch.save({"best_epoch": best_epoch, 
                    "best_score": best_score,
                    "task_mode": task_mode, 
                    "seed": seed, 
                    "model_state_dict": best_state
                    },
                os.path.join(save_path, "checkpoint.pt")
                )
        print(f"Saved: {save_path} (epoch={best_epoch}, score={best_score:.4f})")

    return model_wrapper


@torch.no_grad()
def predict(model_wrapper, 
            test_df, 
            smiles_col,
            task_cols, 
            task_mode,
            batch_size=32,
            max_length=128, 
            num_workers=0, 
            device=None, 
            use_amp=False
    ):
    model     = model_wrapper.get_model()
    device    = device or next(model.parameters()).device
    tokenizer = model_wrapper.get_tokenizer()
    model     = model.to(device)

    loader = DataLoader(
        ChemDataset(test_df, smiles_col, task_cols),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=ChemCollator(tokenizer, max_length, task_mode),
    )
    return _eval_epoch(model, loader, device, task_mode, use_amp and torch.cuda.is_available())


@torch.no_grad()
def predict_smiles(
    model_wrapper,
    smiles_list,
    batch_size=32,
    max_length=128,
    num_workers=0,
    device=None,
    use_amp=False,
    ) -> dict:
    """
    Run inference on a bare list of SMILES strings using the fine-tuned model.

    Returns
    ───────
    classification : {"smiles", "probs", "preds"}
    regression     : {"smiles", "preds"}
    """
    task_mode      = model_wrapper.task_mode
    model     = model_wrapper.get_model()
    device    = device or next(model.parameters()).device
    tokenizer = model_wrapper.get_tokenizer()
    model     = model.to(device).eval()

    infer_df = pd.DataFrame({"smiles": [str(s) for s in smiles_list]})
    loader   = DataLoader(
        ChemDataset(infer_df, smiles_col="smiles", task_cols=None),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ChemCollator(tokenizer, max_length, task_mode),
    )

    all_logits = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(enabled=use_amp and torch.cuda.is_available()):
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits
        all_logits.append(logits.cpu())

    logits_np = torch.cat(all_logits).numpy()

    if task_mode in ("binary_classification", "multilabel_classification"):
        probs = 1 / (1 + np.exp(-logits_np))
        preds = (probs > 0.5).astype(int)
        return {"smiles": smiles_list, "probs": probs, "preds": preds}

    return {"smiles": smiles_list, "preds": logits_np}


def _log_epoch(epoch, 
               train_loss, 
               val_metrics, 
               task_mode, 
               best_epoch
    ):
    base = f"epoch {epoch:3d} | loss={train_loss:.4f}"
    if task_mode != "regression":
        print(f"{base} | auroc={val_metrics.get('auroc', float('nan')):.4f} | auprc={val_metrics.get('auprc', float('nan')):.4f}")
    else:
        print(f"{base} | rmse={val_metrics.get('rmse', float('nan')):.4f} | mae={val_metrics.get('mae', float('nan')):.4f} | best={best_epoch}")
