# stage1/model.py
# %%
import os
import sys
import hashlib
import tempfile
import fcntl
from pathlib import Path
from abc import ABC
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

sys.path.append(str(Path(__file__).parents[1]))
from stage1.featurizer import (
    NUM_ATOMIC_NUM,
    NUM_CHIRALITY,
    NUM_BOND_TYPE,
    NUM_BOND_DIR,
)


# SNAP/Pretrain-GNN reserves extra embedding rows beyond atomic numbers 1..118
# for special tokens, while molecule featurization itself emits indices 0..117.
NUM_ATOM_TYPE = NUM_ATOMIC_NUM + 2
NUM_CHIRALITY_TAG = NUM_CHIRALITY
NUM_BOND_TYPE_EMBED = NUM_BOND_TYPE + 2
NUM_BOND_DIRECTION = NUM_BOND_DIR
SELF_LOOP_BOND_TYPE_INDEX = NUM_BOND_TYPE


def _task_family_indices(task_cols):
    task_cols = list(task_cols or [])
    families = []
    for prefix in ("NR-", "SR-"):
        idx = [i for i, task in enumerate(task_cols) if str(task).startswith(prefix)]
        if idx:
            families.append((prefix.rstrip("-"), idx))
    covered = {i for _, idx in families for i in idx}
    other = [i for i in range(len(task_cols)) if i not in covered]
    if other:
        families.append(("Other", other))
    return families


class TaskFamilyHead(nn.Module):
    """
    Shared-encoder, task-family-specific prediction heads.

    The input representation is shared. Endpoint logits are produced by
    independent heads for NR, SR, and any remaining task family, then scattered
    back to the original task column order.
    """

    def __init__(
        self,
        input_dim,
        task_cols,
        task_dim,
        head_type="linear",
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.task_dim = int(task_dim)
        self.families = _task_family_indices(task_cols)
        if not self.families:
            self.families = [("All", list(range(self.task_dim)))]
        if sum(len(idx) for _, idx in self.families) != self.task_dim:
            raise ValueError(
                "TaskFamilyHead requires task_cols to match task_dim. "
                f"Got {len(task_cols or [])} task columns and task_dim={self.task_dim}."
            )

        heads = {}
        for name, idx in self.families:
            out_dim = len(idx)
            if head_type == "mlp":
                heads[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, out_dim),
                )
            elif head_type == "linear":
                heads[name] = nn.Linear(input_dim, out_dim)
            else:
                raise ValueError(f"head_type must be 'linear' or 'mlp', got {head_type!r}")
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        out = x.new_empty((x.size(0), self.task_dim))
        for name, idx in self.families:
            out[:, idx] = self.heads[name](x)
        return out


class TaskFamilySequenceClassifier(nn.Module):
    """
    Shared ChemBERTa backbone with family-specific prediction heads.

    This mirrors the GNN family-head setup: a single backbone is shared across
    all tasks, while NR and SR endpoints are predicted by separate heads.
    """

    def __init__(self, backbone, classifier, pooling="cls"):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.backbone = backbone
        self.classifier = classifier
        self.pooling = pooling

    def _pool(self, last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "cls" or attention_mask is None:
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        enc_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if token_type_ids is not None:
            enc_kwargs["token_type_ids"] = token_type_ids
        enc_kwargs.update(kwargs)
        out = self.backbone(**enc_kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)
        if return_dict:
            return SequenceClassifierOutput(logits=logits)
        return (logits,)


def _self_loop_edge_attr(x, edge_attr):
    self_loop_attr = torch.zeros((x.size(0), 2), dtype=edge_attr.dtype, device=edge_attr.device)
    self_loop_attr[:, 0] = SELF_LOOP_BOND_TYPE_INDEX
    return self_loop_attr


class Fingerprint(nn.Module):

    def __init__(
        self,
        fp_bits=2048,
        radius=2,
        task_mode=None,
        num_labels=None,
        task_cols=None,
        task_family_heads=False,
        task_family_head_type="linear",
        task_family_hidden_dim=256,
        task_family_dropout=0.1,
        seed=42,
        save_path=None,
        device=None,
    ):
        super().__init__()

        self.fp_bits = fp_bits
        self.radius = radius
        self.task_mode = task_mode
        self.num_labels = num_labels
        self.task_cols = task_cols
        self.task_family_heads = bool(task_family_heads)
        self.task_family_head_type = task_family_head_type
        self.task_family_hidden_dim = int(task_family_hidden_dim)
        self.task_family_dropout = float(task_family_dropout)
        self.seed = seed
        self.save_path = save_path
        self.device = device

        self.register_buffer("_device_probe", torch.zeros(1))
        self.probe_head = None
        self._feature_model = None

    def encode(self, smiles_list):

        from rdkit import Chem
        from rdkit.Chem import AllChem

        fps = []

        for smi in smiles_list:

            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                fps.append(torch.zeros(self.fp_bits))

            else:
                fp = AllChem.GetMorganGenerator(
                    radius=self.radius,
                    fpSize=self.fp_bits,
                ).GetFingerprint(mol)

                fps.append(
                    torch.tensor(list(fp), dtype=torch.float32)
                )

        return torch.stack(fps).to(self._device_probe.device)

    def forward(self, smiles_list):
        return self.encode(smiles_list)

    @property
    def output_dim(self):
        return int(self.fp_bits)

    def _init_feature_model(self):
        if self.num_labels is None or self.task_mode is None:
            raise ValueError(
                "Fingerprint probe mode requires task_mode and num_labels to be set."
            )
        if self.probe_head is None:
            from stage1.probe import ProbeHead

            self.probe_head = ProbeHead(
                input_dim=self.output_dim,
                task_dim=self.num_labels,
                probe_type=self.task_family_head_type,
                hidden_dim=self.task_family_hidden_dim,
                dropout=self.task_family_dropout,
                task_cols=self.task_cols,
                task_family_heads=self.task_family_heads,
            )
        self._feature_model = FeatureProbeModel(
            probe_head=self.probe_head,
        )

    @property
    def model(self):
        if self._feature_model is None:
            self._init_feature_model()
        return self._feature_model

    @model.setter
    def model(self, value):
        self._feature_model = value

    def get_model(self):
        return self.model

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        if self.probe_head is None:
            raise RuntimeError("Fingerprint probe_head is not initialized.")
        torch.save(
            {
                "probe_state_dict": self.probe_head.state_dict(),
                "meta": {
                    "task_family_heads": self.task_family_heads,
                    "task_family_head_type": self.task_family_head_type,
                    "task_family_hidden_dim": self.task_family_hidden_dim,
                    "task_family_dropout": self.task_family_dropout,
                    "task_cols": self.task_cols,
                    "task_mode": self.task_mode,
                    "num_labels": self.num_labels,
                    "fp_bits": self.fp_bits,
                    "radius": self.radius,
                },
            },
            os.path.join(path, "fingerprint_probe.pt"),
        )

    def fit_probe(
        self,
        split,
        probe_type="linear",
        pooling="mean",
        pos_weight=None,
        **kwargs,
    ):
        from stage1.feature_probe import fit as feature_fit
        kwargs.setdefault("device", self.device)

        return feature_fit(
            split,
            self,
            probe_type=probe_type,
            pos_weight=pos_weight,
            **kwargs,
        )

    def predict_probe(self, test_df, **kwargs):
        from stage1.feature_probe import predict as feature_predict

        return feature_predict(self, test_df, **kwargs)


class FeatureProbeModel(nn.Module):
    def __init__(self, probe_head):
        super().__init__()
        self.probe_head = probe_head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.probe_head(features)

class PretrainGINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE_EMBED, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.cat((edge_attr, _self_loop_edge_attr(x, edge_attr)), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class PretrainGCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE_EMBED, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.cat((edge_attr, _self_loop_edge_attr(x, edge_attr)), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class PretrainGATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super().__init__(aggr=aggr, node_dim=0)
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE_EMBED, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.cat((edge_attr, _self_loop_edge_attr(x, edge_attr)), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out.mean(dim=1) + self.bias


class PretrainGraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE_EMBED, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = torch.cat((edge_attr, _self_loop_edge_attr(x, edge_attr)), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class UndirectedMPNN(nn.Module):

    def __init__(
        self,
        input_dim=None,
        hidden_dim=None,
        output_dim=None,
        n_layers=None,
        gnn_type=None,
        dropout=0.2,
        graph_pooling="mean",
        JK="last",
        task_cols=None,
        task_family_heads=False,
        task_family_head_type="linear",
        task_family_hidden_dim=256,
        task_family_dropout=0.1,
        task_mode="binary_classification",
        lr=1e-3,
        epochs=100,
        seed=42,
        save_path=None,
        patience=None,
        min_delta=0.0,
        **kwargs,
    ):
        super().__init__()

        self.gnns = None
        self.batch_norms = None
        self.project = None

        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.JK = JK
        self.task_cols = list(task_cols or [])
        self.task_family_heads = bool(task_family_heads)
        self.task_family_head_type = task_family_head_type
        self.task_family_hidden_dim = int(task_family_hidden_dim)
        self.task_family_dropout = float(task_family_dropout)

        self.task_mode = task_mode
        self.lr = lr
        self.epochs = epochs

        self.seed = seed

        self.save_path = save_path

        self.patience = patience
        self.min_delta = min_delta
        self.device = kwargs.get("device")

        if gnn_type is not None:
            if hidden_dim is None or output_dim is None or n_layers is None:
                raise ValueError(
                    "UndirectedMPNN requires hidden_dim, output_dim, and n_layers "
                    "when gnn_type is provided."
                )
            self._build_pretrain_gnn(n_layers, hidden_dim, gnn_type)
            self.project = self._make_prediction_head(
                self.graph_output_dim,
                output_dim,
            )

    def _make_prediction_head(self, input_dim, output_dim):
        if self.task_family_heads:
            return TaskFamilyHead(
                input_dim=input_dim,
                task_cols=self.task_cols,
                task_dim=output_dim,
                head_type=self.task_family_head_type,
                hidden_dim=self.task_family_hidden_dim,
                dropout=self.task_family_dropout,
            )
        return nn.Linear(input_dim, output_dim)

    def _build_pretrain_gnn(self, n_layers, hidden_dim, gnn_type):
        if n_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layer = int(n_layers)
        self.emb_dim = int(hidden_dim)
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, hidden_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, hidden_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        conv_map = {
            "gin": PretrainGINConv,
            "gcn": PretrainGCNConv,
            "gat": PretrainGATConv,
            "graphsage": PretrainGraphSAGEConv,
        }
        if gnn_type not in conv_map:
            raise ValueError(f"Unsupported gnn_type={gnn_type!r}")
        self.gnns = nn.ModuleList([conv_map[gnn_type](hidden_dim) for _ in range(n_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])

    @property
    def graph_output_dim(self):
        if self.JK == "concat":
            return (self.num_layer + 1) * self.emb_dim
        return self.emb_dim

    def _train_params(self):

        return dict(
            task_mode=self.task_mode,
            lr=self.lr,
            epochs=self.epochs,
            seed=self.seed,
            save_path=self.save_path,
            patience=self.patience,
            min_delta=self.min_delta,
            device=self.device,
        )

    def _predict_params(self):

        return dict(
            task_mode=self.task_mode,
            device=self.device,
        )

    def _pool(self, x, batch):
        if self.graph_pooling == "sum":
            return global_add_pool(x, batch)
        if self.graph_pooling == "mean":
            return global_mean_pool(x, batch)
        if self.graph_pooling == "max":
            return global_max_pool(x, batch)
        raise ValueError(f"Unsupported graph_pooling={self.graph_pooling!r}")

    def _encode_nodes(self, data):
        x = data.x.long()
        x = torch.stack([
            x[:, 0].clamp(min=0, max=NUM_ATOM_TYPE - 1),
            x[:, 1].clamp(min=0, max=NUM_CHIRALITY_TAG - 1),
        ], dim=1)
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], data.edge_index, data.edge_attr.long())
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            return torch.cat(h_list, dim=1)
        if self.JK == "last":
            return h_list[-1]
        if self.JK == "max":
            return torch.max(torch.stack(h_list, dim=0), dim=0)[0]
        if self.JK == "sum":
            return torch.sum(torch.stack(h_list, dim=0), dim=0)
        raise ValueError(f"Unsupported JK={self.JK!r}")

    def encode(self, data):
        x = self._encode_nodes(data)
        g = self._pool(x, data.batch)
        return g

    def forward(self, data):
        g = self.encode(data)
        out = self.project(g)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def fit(self, split, pos_weight=None, **kwargs):

        from stage1.trainer import UndirectedMPNN as Trainer

        trainer = Trainer(
            **self._train_params(),
            **kwargs,
        )

        return trainer.fit(
            split,
            self,
            pos_weight=pos_weight,
        )

    def predict(self, split, **kwargs):

        from stage1.trainer import UndirectedMPNN as Trainer

        trainer = Trainer(
            **self._predict_params(),
            **kwargs,
        )

        return trainer.predict(self, split)

    # ------------------------------------------------------------------
    # Probe methods: freeze backbone, train a lightweight head on the
    # matched bio+chem split (DataFrame), evaluate on matched test.
    # ------------------------------------------------------------------

    def _embed_df(self, df, smiles_col: str, device):
        """Convert a DataFrame of SMILES to frozen GNN graph embeddings [N, emb_dim]."""
        from stage1.loader import smiles_to_pyg
        from torch_geometric.data import Batch as PyGBatch

        smiles_list = df[smiles_col].astype(str).tolist()
        graphs = [smiles_to_pyg(s) for s in smiles_list]
        self.eval()
        with torch.no_grad():
            batch = PyGBatch.from_data_list(graphs).to(device)
            return self.encode(batch).cpu()   # [N, graph_output_dim]

    def fit_probe(
        self,
        df_split,
        smiles_col: str,
        task_cols: list,
        task_mode: str,
        probe_type: str = "linear",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 256,
        patience: int = None,
        min_delta: float = 0.0,
        pos_weight=None,
        save_path: str = None,
        seed: int = 42,
        chem_proj_dim: int = None,  # accepted for API compat; no longer used (projection folded into first MLP layer)
        device=None,
    ):
        """
        Freeze backbone; train a linear/MLP probe head on pre-computed GNN
        embeddings from the matched train split.  Evaluates early stopping
        on matched valid.  Saves head to save_path/gnn_probe_head.pt.
        """
        import copy
        from stage1.trainer import get_loss_fn, compute_metrics, init_best_score, is_better, primary_metric

        torch.manual_seed(seed)
        np.random.seed(seed)

        device = device or next(self.parameters()).device
        for p in self.parameters():
            p.requires_grad_(False)

        # Pre-compute embeddings (one pass, no grad)
        train_emb  = self._embed_df(df_split.train, smiles_col, device)  # [N_train, D]
        valid_emb  = self._embed_df(df_split.valid, smiles_col, device)

        def _labels(df):
            arr = df[task_cols].to_numpy(dtype=float)
            return torch.tensor(arr, dtype=torch.float32)

        train_lbl = _labels(df_split.train)
        valid_lbl = _labels(df_split.valid)

        in_dim = self.graph_output_dim
        n_tasks = len(task_cols)

        # No separate projection: proj + Linear compose to one linear map (LN between
        # them is not a non-linearity). The first head layer projects directly from
        # in_dim to hidden_dim, matching ProjectionMLP in bio+stage1/model.py.
        if probe_type == "linear" or hidden_dim <= 0:
            task_net = nn.Linear(in_dim, n_tasks)
        else:
            task_net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_tasks),
            )
        head = task_net
        head = head.to(device)

        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

        import torch.nn.functional as _F

        def _masked_bce_loss(logits, y, pw):
            # Per-task masking: avoids flattening [N,T]→[K] which breaks pos_weight[T].
            mask = ~y.isnan()
            y_filled = y.nan_to_num(0.0)
            task_losses = []
            for t in range(logits.shape[1]):
                tm = mask[:, t]
                if not tm.any():
                    continue
                pw_t = pw[t].unsqueeze(0) if pw is not None else None
                task_losses.append(
                    _F.binary_cross_entropy_with_logits(
                        logits[tm, t], y_filled[tm, t], pos_weight=pw_t
                    )
                )
            return torch.stack(task_losses).mean() if task_losses else logits.sum() * 0.0

        best_score = init_best_score(task_mode)
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            head.train()
            # mini-batch over pre-computed embeddings
            idx = torch.randperm(len(train_emb))
            train_loss_sum, n_batches = 0.0, 0
            for start in range(0, len(train_emb), batch_size):
                batch_idx = idx[start:start + batch_size]
                x = train_emb[batch_idx].to(device)
                y = train_lbl[batch_idx].to(device)
                if not (~y.isnan()).any():
                    continue
                optimizer.zero_grad()
                logits = head(x)
                loss = _masked_bce_loss(logits, y, pos_weight)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
                n_batches += 1

            # validation
            head.eval()
            with torch.no_grad():
                valid_logits = head(valid_emb.to(device))
            valid_probs = torch.sigmoid(valid_logits).cpu()
            val_metrics = compute_metrics(
                valid_probs.numpy(), valid_lbl.numpy(), task_mode,
            )
            score = primary_metric(val_metrics, task_mode)
            train_loss = train_loss_sum / max(n_batches, 1)

            auroc = val_metrics.get("auroc", float("nan"))
            auprc = val_metrics.get("auprc", float("nan"))
            print(
                f"[GNN-probe] epoch {epoch:3d} | loss={train_loss:.4f} | "
                f"val_auroc={auroc:.4f} | val_auprc={auprc:.4f}"
            )

            if is_better(score, best_score, task_mode, min_delta):
                best_score = score
                best_state = copy.deepcopy(head.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if patience and no_improve >= patience:
                print(f"[GNN-probe] early stopping at epoch {epoch}")
                break

        if best_state is not None:
            head.load_state_dict(best_state)

        self.probe_head = head
        self._probe_meta = {
            "smiles_col":    smiles_col,
            "task_cols":     task_cols,
            "task_mode":     task_mode,
            "chem_proj_dim": chem_proj_dim,
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(
                {"head_state_dict": best_state or head.state_dict(), "meta": self._probe_meta},
                os.path.join(save_path, "gnn_probe_head.pt"),
            )
            print(f"[GNN-probe] saved → {save_path}/gnn_probe_head.pt")

        return self

    def predict_probe(self, test_df, smiles_col: str = None, task_cols: list = None,
                      task_mode: str = None, device=None):
        """Embed test_df via frozen backbone + probe head → metrics dict."""
        from stage1.trainer import compute_metrics

        meta = getattr(self, "_probe_meta", {})
        smiles_col = smiles_col or meta.get("smiles_col")
        task_cols  = task_cols  or meta.get("task_cols")
        task_mode  = task_mode  or meta.get("task_mode")
        device     = device or next(self.parameters()).device

        test_emb = self._embed_df(test_df, smiles_col, device)
        with torch.no_grad():
            logits = self.probe_head(test_emb.to(device)).cpu()
        probs  = torch.sigmoid(logits).numpy()
        labels = test_df[task_cols].to_numpy(dtype=float)
        return compute_metrics(probs, labels, task_mode)


class GCN(UndirectedMPNN):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout=0.2,
        **kwargs,
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="gcn",
            dropout=dropout,
            **kwargs,
        )


class GAT(UndirectedMPNN):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        heads=4,
        dropout=0.2,
        **kwargs,
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="gat",
            dropout=dropout,
            **kwargs,
        )


class GIN(UndirectedMPNN):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout=0.0,
        mlp_dropout=0.2,
        **kwargs,
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="gin",
            dropout=dropout,
            **kwargs,
        )


class GraphSAGE(UndirectedMPNN):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout=0.2,
        **kwargs,
    ):

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            gnn_type="graphsage",
            dropout=dropout,
            **kwargs,
        )


class Transformer(nn.Module, ABC):

    _PROBE_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

    def __init__(
        self,
        model_name,
        task_mode="binary_classification",
        num_labels=1,
        frozen=False,
        checkpoint_path=None,
        lr=2e-5,
        epochs=10,
        batch_size=32,
        max_length=128,
        weight_decay=0.0,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        use_amp=False,
        patience=None,
        min_delta=0.0,
        smiles_col=None,
        task_cols=None,
        task_family_heads=False,
        task_family_head_type="linear",
        task_family_hidden_dim=256,
        task_family_dropout=0.1,
        seed=42,
        save_path=None,
        trust_remote_code=False,
        cache_dir=None,
        local_files_only=False,
        cache_size=100000,
        feature_cache_dir=None,
        device=None,
    ):

        super().__init__()

        self.model_name = model_name

        self.task_mode = task_mode
        self.num_labels = num_labels

        self.frozen = frozen
        self.checkpoint_path = checkpoint_path

        self.lr = lr
        self.epochs = epochs

        self.batch_size = batch_size
        self.max_length = max_length

        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm

        self.use_amp = use_amp

        self.patience = patience
        self.min_delta = min_delta

        self.smiles_col = smiles_col
        self.task_cols = task_cols
        self.task_family_heads = bool(task_family_heads)
        self.task_family_head_type = task_family_head_type
        self.task_family_hidden_dim = int(task_family_hidden_dim)
        self.task_family_dropout = float(task_family_dropout)

        self.seed = seed

        self.save_path = save_path
        self.device = device

        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.feature_cache_dir = feature_cache_dir

        self._cache_size = cache_size
        self._cache = {}
        self._disk_cache = None
        self._disk_cache_path = None
        self._disk_cache_lock_path = None

        self._tokenizer = None
        self._backbone = None
        self._seq_model = None
        self._family_head_path = None
        self._output_dim = None

        self.probe_head = None

        self._init_backbone()
        self._init_feature_cache()

    def _huggingface_kwargs(self):

        return dict(
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )

    def _init_backbone(self):

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **self._huggingface_kwargs(),
        )

        self._backbone = AutoModel.from_pretrained(
            self.model_name,
            **self._huggingface_kwargs(),
        )

        if self.frozen:

            for p in self._backbone.parameters():
                p.requires_grad_(False)

            self._backbone.eval()

        with torch.no_grad():

            tok = self._tokenizer(
                [self._PROBE_SMILES],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            out = self._backbone(**tok)

            self._output_dim = int(
                out.last_hidden_state[:, 0, :].shape[-1]
            )

        print(
            f"[{self.__class__.__name__}] "
            f"{self.model_name} "
            f"output_dim={self._output_dim}"
        )

    def _init_seq_model(self):

        config = AutoConfig.from_pretrained(
            self.model_name,
            **self._huggingface_kwargs(),
        )

        config.num_labels = self.num_labels

        if self.task_mode == "regression":
            config.problem_type = "regression"

        elif self.task_mode == "binary_classification":
            config.problem_type = "single_label_classification"

        elif self.task_mode == "multilabel_classification":
            config.problem_type = "multi_label_classification"

        else:
            raise ValueError(
                f"Unsupported task_mode: {self.task_mode!r}. "
                "Choose from: regression, binary_classification, multilabel_classification"
            )

        if self.task_family_heads:
            source = self.checkpoint_path or self.model_name
            backbone = AutoModel.from_pretrained(
                source,
                **self._huggingface_kwargs(),
            )
            classifier = TaskFamilyHead(
                input_dim=self._output_dim,
                task_cols=self.task_cols,
                task_dim=self.num_labels,
                head_type=self.task_family_head_type,
                hidden_dim=self.task_family_hidden_dim,
                dropout=self.task_family_dropout,
            )
            self._family_head_path = (
                os.path.join(self.checkpoint_path, "task_family_head.pt")
                if self.checkpoint_path
                else None
            )
            if self._family_head_path and os.path.exists(self._family_head_path):
                ckpt = torch.load(self._family_head_path, map_location="cpu")
                state_dict = ckpt.get("classifier_state_dict", ckpt.get("state_dict", ckpt))
                classifier.load_state_dict(state_dict)
            self._seq_model = TaskFamilySequenceClassifier(
                backbone=backbone,
                classifier=classifier,
                pooling="cls",
            )
        else:
            source = self.checkpoint_path or self.model_name
            self._seq_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    source,
                    config=config,
                    ignore_mismatched_sizes=(
                        self.checkpoint_path is None
                    ),
                    **self._huggingface_kwargs(),
                )
            )

    def _cache_key(self) -> str:
        source = self.checkpoint_path or self.model_name
        raw = f"{source}|maxlen={self.max_length}|frozen={int(self.frozen)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _init_feature_cache(self):
        if not self.frozen or not self.feature_cache_dir:
            return
        model_tag = str(self.model_name).replace(os.sep, "_").replace("/", "_")
        cache_dir = os.path.join(self.feature_cache_dir, model_tag, self._cache_key())
        os.makedirs(cache_dir, exist_ok=True)
        self._disk_cache_path = os.path.join(cache_dir, "smiles_embeddings.pt")
        self._disk_cache_lock_path = os.path.join(cache_dir, "smiles_embeddings.lock")
        if os.path.exists(self._disk_cache_path):
            try:
                cache_obj = torch.load(self._disk_cache_path, map_location="cpu")
                self._disk_cache = cache_obj if isinstance(cache_obj, dict) else {}
            except Exception:
                self._disk_cache = {}
        else:
            self._disk_cache = {}

    def _save_feature_cache(self):
        if not self.frozen or self._disk_cache is None or self._disk_cache_path is None:
            return
        os.makedirs(os.path.dirname(self._disk_cache_path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="smiles_embeddings_",
            suffix=".pt",
            dir=os.path.dirname(self._disk_cache_path),
        )
        os.close(tmp_fd)
        try:
            with open(self._disk_cache_lock_path, "a+b") as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                if os.path.exists(self._disk_cache_path):
                    try:
                        current = torch.load(self._disk_cache_path, map_location="cpu")
                        if isinstance(current, dict):
                            current.update(self._disk_cache)
                            self._disk_cache = current
                    except Exception:
                        pass
                torch.save(self._disk_cache, tmp_path)
                os.replace(tmp_path, self._disk_cache_path)
                fcntl.flock(lock_f, fcntl.LOCK_UN)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def backbone(self):
        return self._backbone

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def model(self):

        if self._seq_model is None:
            self._init_seq_model()

        return self._seq_model

    @model.setter
    def model(self, value):
        self._seq_model = value

    def get_model(self):
        return self.model

    @property
    def config(self):

        if self._seq_model is None:
            self._init_seq_model()

        return self._seq_model.config

    def get_tokenizer(self):
        return self.tokenizer

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        model = self.model
        for tensor in (*model.parameters(), *model.buffers()):
            tensor.data = tensor.data.contiguous()
        if self.task_family_heads:
            model.backbone.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            torch.save(
                {
                    "classifier_state_dict": model.classifier.state_dict(),
                    "meta": {
                        "task_family_heads": True,
                        "task_family_head_type": self.task_family_head_type,
                        "task_family_hidden_dim": self.task_family_hidden_dim,
                        "task_family_dropout": self.task_family_dropout,
                        "task_cols": self.task_cols,
                        "task_mode": self.task_mode,
                        "num_labels": self.num_labels,
                        "pooling": "cls",
                    },
                },
                os.path.join(path, "task_family_head.pt"),
            )
        else:
            model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def train(self, mode=True):

        super().train(mode)

        if self.frozen and self._backbone is not None:
            self._backbone.eval()

        return self

    @torch.no_grad()
    def _encode_cached(self, smiles, device):

        out = [None] * len(smiles)

        miss_idx = []
        miss_smiles = []

        for i, smi in enumerate(smiles):

            smi = str(smi)

            if smi in self._cache:

                out[i] = self._cache[smi]

            else:

                miss_idx.append(i)
                miss_smiles.append(smi)

        if miss_smiles and self._disk_cache is not None:
            still_missing_idx = []
            still_missing_smiles = []
            for rel_i, smi in enumerate(miss_smiles):
                vec = self._disk_cache.get(smi)
                if vec is None:
                    still_missing_idx.append(miss_idx[rel_i])
                    still_missing_smiles.append(smi)
                    continue
                out[miss_idx[rel_i]] = vec
            miss_idx = still_missing_idx
            miss_smiles = still_missing_smiles

        if miss_smiles:

            tok = self._tokenizer(
                miss_smiles,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)

            cls = self._backbone(
                **tok
            ).last_hidden_state[:, 0, :]

            for j, smi in enumerate(miss_smiles):

                vec = cls[j].detach().cpu()

                if len(self._cache) < self._cache_size:
                    self._cache[smi] = vec
                if self._disk_cache is not None:
                    self._disk_cache[smi] = vec

                out[miss_idx[j]] = vec

            self._save_feature_cache()

        return torch.stack([
            v.to(device)
            for v in out
        ])

    def encode(self, smiles, device=None):

        device = (
            device
            or next(self._backbone.parameters()).device
        )

        if self.frozen:
            return self._encode_cached(smiles, device)

        tok = self._tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        tok = {
            k: v.to(device)
            for k, v in tok.items()
        }

        return self._backbone(
            **tok
        ).last_hidden_state[:, 0, :]

    def _train_params(self):

        return dict(
            smiles_col=self.smiles_col,
            task_cols=self.task_cols,
            task_mode=self.task_mode,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            max_length=self.max_length,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            max_grad_norm=self.max_grad_norm,
            use_amp=self.use_amp,
            patience=self.patience,
            min_delta=self.min_delta,
            seed=self.seed,
            save_path=self.save_path,
            device=self.device,
        )

    def _predict_params(self):

        return dict(
            smiles_col=self.smiles_col,
            task_cols=self.task_cols,
            task_mode=self.task_mode,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_amp=self.use_amp,
            device=self.device,
        )

    def fit(self, split, pos_weight=None, **kwargs):

        from stage1.finetune import fit as finetune_fit

        return finetune_fit(
            split,
            self,
            pos_weight=pos_weight,
            **self._train_params(),
            **kwargs,
        )

    def predict(self, test_df, **kwargs):

        from stage1.finetune import predict as finetune_predict

        return finetune_predict(
            self,
            test_df,
            **self._predict_params(),
            **kwargs,
        )

    def fit_probe(
        self,
        split,
        probe_type="linear",
        pooling="mean",
        pos_weight=None,
        **kwargs,
    ):

        from stage1.probe import fit as probe_fit

        return probe_fit(
            split,
            self,
            probe_type=probe_type,
            pooling=pooling,
            pos_weight=pos_weight,
            **self._train_params(),
            **kwargs,
        )

    def predict_probe(self, test_df, **kwargs):

        from stage1.probe import predict as probe_predict

        return probe_predict(
            self,
            test_df,
            **self._predict_params(),
            **kwargs,
        )


class ChemBerta(Transformer):

    DEFAULT_MODEL = "DeepChem/ChemBERTa-100M-MLM"

    def __init__(
        self,
        model_name=None,
        **kwargs,
    ):

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            **kwargs,
        )


class MoLFormer(Transformer):

    DEFAULT_MODEL = "ibm-research/MoLFormer-XL-both-10pct"

    def __init__(
        self,
        model_name=None,
        **kwargs,
    ):

        super().__init__(
            model_name=model_name or self.DEFAULT_MODEL,
            **kwargs,
        )


GNN_ENCODERS = {
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "GraphSAGE": GraphSAGE,
}

TF_ENCODERS = {
    "Chemberta": ChemBerta,
    # "MoLFormer": MoLFormer,
}

CHEM_ENCODERS = {
    "Fingerprint": Fingerprint,
    **GNN_ENCODERS,
    **TF_ENCODERS,
}
