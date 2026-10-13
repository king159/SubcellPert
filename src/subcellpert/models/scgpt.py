"""scGPT network and benchmark adapter. Derived layers: MIT, see THIRD_PARTY_LICENSES.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel
from subcellpert.utils import (
    align_gene_ids,
    align_token_states,
    freeze,
    install_transformer_sdpa,
    resolve_model_directory,
)


def _load_encoder_weights(model, weights):
    """Require every encoder parameter; unused downstream heads may differ."""
    weights = {key.replace("Wqkv.", "in_proj_"): value for key, value in weights.items()}
    expected = model.state_dict()
    prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    missing = [
        key
        for key, value in expected.items()
        if key.startswith(prefixes) and (key not in weights or weights[key].shape != value.shape)
    ]
    if missing:
        raise ValueError(f"Pretrained encoder parameters are missing or incompatible: {missing[:5]}")
    model.load_state_dict(
        {key: value for key, value in weights.items() if key in expected and value.shape == expected[key].shape},
        strict=False,
    )


def deterministic_bins(values, n_bins=51, *, valid=None):
    """NumPy's linear quantiles and lower tie rule, evaluated on the input device."""
    values = torch.as_tensor(values, dtype=torch.float32)
    single = values.ndim == 1
    if single:
        values = values[None]
    if values.ndim != 2:
        raise ValueError("Quantile binning requires one or two count dimensions")
    valid = torch.ones_like(values, dtype=torch.bool) if valid is None else valid
    if not values.shape[1]:
        return values[0] if single else values
    ordered = values.double().masked_fill(~valid, float("inf")).sort(dim=1).values
    sizes = valid.sum(1).clamp_min(1)
    fractions = torch.as_tensor(np.linspace(0, 1, n_bins - 1), device=values.device)
    ranks = (sizes[:, None] - 1) * fractions
    lower = ordered.gather(1, ranks.floor().long())
    upper = ordered.gather(1, ranks.ceil().long())
    fraction = ranks - ranks.floor()
    delta = upper - lower
    boundaries = torch.where(fraction >= 0.5, upper - delta * (1 - fraction), lower + delta * fraction)
    boundaries = boundaries.masked_fill(~valid.any(1)[:, None], 0)
    bins = torch.searchsorted(boundaries.contiguous(), values.double().contiguous(), right=False).float() + 1
    return bins[0] if single else bins


def load_gene_vocabulary(GeneVocab, path):
    """Load the exact token order without quadratic insertion in the fallback.

    GeneVocab.from_dict inserts one token at a time; the pure-Python backend
    rebuilds its entire index on every insertion. Appending the already ordered
    contiguous vocabulary has the same token IDs and avoids that startup cost.
    """
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or not all(type(index) is int for index in mapping.values()):
        raise ValueError("Foundation vocabulary must map tokens to integer IDs")
    ordered = sorted(mapping, key=mapping.__getitem__)
    if [mapping[token] for token in ordered] != list(range(len(ordered))):
        raise ValueError("Foundation vocabulary IDs must be unique and contiguous")
    vocab = GeneVocab([])
    for token in ordered:
        vocab.append_token(token)
    if vocab.get_stoi() != mapping:
        raise ValueError("Foundation vocabulary token IDs changed during loading")
    return vocab


class GeneVocab:
    """Contiguous checkpoint vocabulary without a torchtext dependency."""

    def __init__(self, tokens):
        self.tokens = dict.fromkeys(tokens)
        self.tokens = {token: i for i, token in enumerate(self.tokens)}
        self.default_index = None

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.tokens

    def __getitem__(self, token):
        if token in self.tokens:
            return self.tokens[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError(token)

    def append_token(self, token):
        if token in self.tokens:
            raise ValueError(f"Duplicate vocabulary token: {token}")
        self.tokens[token] = len(self.tokens)

    def set_default_index(self, index):
        self.default_index = index

    def get_stoi(self):
        return dict(self.tokens)


class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class ScGPTNetwork(nn.Module):
    """Continuous-value Transformer used by the scGPT benchmark encoder.

    Derived from scGPT (MIT); decoder/pretraining heads are not part of the
    source-gene feature contract. Parameter names match the pretrained encoder.
    """

    def __init__(self, config, vocabulary_size, padding_id):
        super().__init__()
        width = config["embsize"]
        self.encoder = GeneEncoder(vocabulary_size, width, padding_id)
        self.value_encoder = ContinuousValueEncoder(width, config["dropout"])
        layer = nn.TransformerEncoderLayer(
            width, config["nheads"], config["d_hid"], config["dropout"], batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, config["nlayers"])

    def _encode(self, tokens, values, src_key_padding_mask):
        total = self.encoder(tokens) + self.value_encoder(values)
        return self.transformer_encoder(total, src_key_padding_mask=src_key_padding_mask)


class ScGPTGeneEncoder:
    network_class = ScGPTNetwork

    def __init__(self, root, metadata, device):
        directory = resolve_model_directory(root, metadata)
        config = json.loads((directory / "args.json").read_text())
        vocab = load_gene_vocabulary(GeneVocab, directory / "vocab.json")
        for special in ("<pad>", "<cls>", "<eoc>"):
            if special not in vocab:
                vocab.append_token(special)
        vocab.set_default_index(vocab[config["pad_token"]])
        self.model = self.network_class(config, len(vocab), vocab[config["pad_token"]])
        _load_encoder_weights(
            self.model, torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        )
        install_transformer_sdpa(self.model)
        freeze(self.model, device)
        self.vocabulary = vocab.get_stoi()
        self.dimension, self.device = (config["embsize"] + 1, device)
        self.pad_id, self.cls_id, self.pad_value = (vocab[config["pad_token"]], vocab["<cls>"], config["pad_value"])
        self.max_length = int(metadata.get("encoder_max_length", 1200))
        self.statistics = None

    def gene_ids(self, genes):
        return align_gene_ids(genes, self.vocabulary)

    def normalize_values(self, values, genes, selected):
        """scGPT bins raw eligible counts; spatial checkpoints override this step."""
        return values

    @torch.no_grad()
    def encode(self, matrix, genes, condition):
        del condition
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(self.gene_ids(genes), device=self.device)
        eligible = (ids >= 0)[None] & (matrix > 0)
        lengths = eligible.sum(1).clamp_max(self.max_length - 1)
        length = int(lengths.max()) if len(lengths) else 0
        positions = torch.arange(len(genes), device=self.device).expand_as(matrix)
        # Native token order is gene order, except when the expression cap applies.
        ranked = eligible.sum(1) >= self.max_length
        keys = torch.where(ranked[:, None], -matrix.double(), positions.double()).masked_fill(~eligible, float("inf"))
        selected = keys.argsort(dim=1, stable=True)[:, :length]
        valid_genes = torch.arange(length, device=self.device)[None] < lengths[:, None]
        values = matrix.gather(1, selected)
        values = self.normalize_values(values, genes, selected)
        binned = deterministic_bins(values, valid=valid_genes)
        tokens = torch.full((len(matrix), length + 1), self.pad_id, dtype=torch.long, device=self.device)
        expressions = torch.full(tokens.shape, self.pad_value, dtype=torch.float32, device=self.device)
        valid = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, 0] = self.cls_id
        tokens[:, 1:] = torch.where(valid_genes, ids[selected], self.pad_id)
        expressions[:, 1:] = torch.where(valid_genes, binned, self.pad_value)
        valid[:, 1:] = valid_genes
        states = self.model._encode(tokens, expressions, src_key_padding_mask=tokens.eq(self.pad_id))
        return align_token_states(states, tokens, ids, valid=valid)


class Model(Baseline):
    name = "scgpt"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = ScGPTGeneEncoder
