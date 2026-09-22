# chem/chemprop_stage1_candidate.py
"""
Chemprop (D-MPNN) single-task candidate for publication/chem_stage1_per_task_offset.py.

Runs in the DEDICATED `crisp_chemprop_env` conda env (chemprop==2.2.2 pins
torch==2.6.0/lightning==2.6.0, incompatible with this repo's main
`crisp_env`, pinned to torch==2.2.2 for the rest of the GNN/ChemBERTa
pipeline) -- invoked as a subprocess by the caller running in `crisp_env`,
never imported directly from there.

Deliberately NOT a reuse of chem/chemprop_baseline.py's `_run_chemprop_seed`:
that function is multi-task (fits all 12 Tox21 tasks with one shared MPNN
head) and reads a fixed pickled DataSplit file. This pipeline's Stage 1
design fits every candidate -- including this one -- as a genuinely
single-task model (task_dim=1, no cross-task parameter sharing at all),
matching every other candidate in chem_stage1_per_task_offset.py, and needs
to accept the SAME frozen train/valid/[test+cohort_union] partitions the
rest of Stage 1 uses, not a different split file. Plain-CSV in/out is the
interface across the conda-env subprocess boundary: three input CSVs
(fit/early-stop/predict, each `smiles_canon` [+ the one task column where
present]) and one output CSV (`pred`, same row order as the predict CSV).

Usage (invoked by fit_predict_chemprop() in chem_stage1_per_task_offset.py):
  conda run -n crisp_chemprop_env python chem/chemprop_stage1_candidate.py \
    --fit-csv fit.csv --es-csv es.csv --predict-csv predict.csv \
    --task-col SR-p53 --seed 1 --out-csv preds.csv --gpu-id 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))


def _build_dataset(smiles: list, y: np.ndarray, featurizer):
    from chemprop import data

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y_row)
        for smi, y_row in zip(smiles, y)
    ]
    return data.MoleculeDataset(datapoints, featurizer)


def fit_predict(fit_df: pd.DataFrame, es_df: pd.DataFrame, predict_df: pd.DataFrame,
                 task_col: str, seed: int, gpu_id: int = -1,
                 epochs: int = 100, patience: int = 20, batch_size: int = 32,
                 hidden_dim: int = 300, depth: int = 3, dropout: float = 0.0) -> np.ndarray:
    """Single-task D-MPNN: fit on fit_df, early-stop on es_df (val_loss),
    predict on predict_df. Returns predicted probabilities, same row order
    as predict_df. Mirrors chem_stage1_per_task_offset.py's other
    fit_predict_* candidates' contract exactly (fit_df/es_df/predict_df,
    task, seed, gpu_id) so the caller's dispatch is uniform."""
    import torch
    from chemprop import featurizers, models, nn as cpnn
    from chemprop.data import build_dataloader
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping

    seed_everything(seed, workers=True)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    fit_y = fit_df[task_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    es_y = es_df[task_col].to_numpy(dtype=np.float64).reshape(-1, 1)
    # predict_df may not carry the task column at all (e.g. cohort_union
    # rows for tasks where that compound has no label) -- labels are never
    # read back out of a prediction pass, only used to satisfy
    # MoleculeDatapoint's constructor, so a placeholder is fine here.
    predict_y = np.zeros((len(predict_df), 1), dtype=np.float64)

    fit_ds = _build_dataset(fit_df["smiles_canon"].tolist(), fit_y, featurizer)
    es_ds = _build_dataset(es_df["smiles_canon"].tolist(), es_y, featurizer)
    predict_ds = _build_dataset(predict_df["smiles_canon"].tolist(), predict_y, featurizer)

    fit_loader = build_dataloader(fit_ds, batch_size=batch_size, shuffle=True, seed=seed, num_workers=0)
    es_loader = build_dataloader(es_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    predict_loader = build_dataloader(predict_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    mp = cpnn.BondMessagePassing(d_h=hidden_dim, depth=depth, dropout=dropout)
    agg = cpnn.MeanAggregation()
    predictor = cpnn.BinaryClassificationFFN(n_tasks=1, input_dim=mp.output_dim)
    mpnn_model = models.MPNN(mp, agg, predictor, batch_norm=True,
                              metrics=[cpnn.BinaryAUROC(), cpnn.BinaryAUPRC()])

    use_gpu = torch.cuda.is_available() and gpu_id is not None and gpu_id >= 0
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=[gpu_id] if use_gpu else 1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience)],
        deterministic=False,
    )
    trainer.fit(mpnn_model, fit_loader, es_loader)

    raw_preds = trainer.predict(mpnn_model, predict_loader)
    preds = torch.cat(raw_preds).numpy().reshape(-1)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-csv", required=True)
    parser.add_argument("--es-csv", required=True)
    parser.add_argument("--predict-csv", required=True)
    parser.add_argument("--task-col", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    fit_df = pd.read_csv(args.fit_csv)
    es_df = pd.read_csv(args.es_csv)
    predict_df = pd.read_csv(args.predict_csv)

    preds = fit_predict(
        fit_df, es_df, predict_df, args.task_col, args.seed, gpu_id=args.gpu_id,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
    )
    pd.DataFrame({"pred": preds}).to_csv(args.out_csv, index=False)
    print(f"[chemprop-stage1-candidate] wrote {args.out_csv} ({len(preds)} predictions, "
          f"task={args.task_col}, seed={args.seed}, gpu_id={args.gpu_id})")


if __name__ == "__main__":
    main()
