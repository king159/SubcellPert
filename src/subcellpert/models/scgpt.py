"""scgpt baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

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


class ScGPTGeneEncoder:
    def __init__(self, root, metadata, device):
        spatial = metadata["backbone"] == "scgpt_spatial"
        sys.path.insert(0, str(root / "external/scGPT"))
        if spatial:
            sys.path.insert(0, str(root / "external/scGPT-spatial"))
        package = "scgpt_spatial" if spatial else "scgpt"
        Model = importlib.import_module(f"{package}.model").TransformerModel
        GeneVocab = importlib.import_module("scgpt.tokenizer").GeneVocab
        directory = resolve_model_directory(root, metadata)
        config = json.loads((directory / "args.json").read_text())
        vocab = load_gene_vocabulary(GeneVocab, directory / "vocab.json")
        for special in ("<pad>", "<cls>", "<eoc>"):
            if special not in vocab:
                vocab.append_token(special)
        vocab.set_default_index(vocab[config["pad_token"]])
        options = {
            "ntoken": len(vocab),
            "d_model": config["embsize"],
            "nhead": config["nheads"],
            "d_hid": config["d_hid"],
            "nlayers": config["nlayers"],
            "nlayers_cls": config["n_layers_cls"],
            "n_cls": 1,
            "vocab": vocab,
            "dropout": config["dropout"],
            "pad_token": config["pad_token"],
            "pad_value": config["pad_value"],
            "do_mvc": True,
            "do_dab": False,
            "use_batch_labels": False,
            "explicit_zero_prob": False,
            "use_fast_transformer": False,
            "pre_norm": False,
        }
        if spatial:
            options.update(use_generative_training=False, use_MVC_impute=True, use_moe_dec=True)
        self.model = Model(**options)
        _load_encoder_weights(
            self.model, torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        )
        install_transformer_sdpa(self.model)
        freeze(self.model, device)
        self.vocabulary = vocab.get_stoi()
        self.dimension, self.device, self.spatial = (config["embsize"] + 1, device, spatial)
        self.pad_id, self.cls_id, self.pad_value = (vocab[config["pad_token"]], vocab["<cls>"], config["pad_value"])
        self.max_length = int(metadata.get("encoder_max_length", 1200))
        self.statistics = None
        self.gene_means = {}
        if spatial:
            import pandas as pd

            table = pd.read_csv(directory / "all_dict_mean_std.csv", index_col=0)
            self.gene_means = {int(key): float(value) for key, value in table["mean"].items()}

    def gene_ids(self, genes):
        return align_gene_ids(genes, self.vocabulary)

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
        if self.spatial:
            if self.statistics is None:
                raise ValueError("scGPT-spatial needs frozen training-source normalization statistics")
            scale = self.statistics["nonzero_mean"]
            fallback = np.asarray(self.statistics["gene_nonzero_mean"]) / scale
            axis_ids = self.gene_ids(genes)
            means = np.asarray(
                [self.gene_means.get(int(token), value) for token, value in zip(axis_ids, fallback, strict=True)]
            )
            means = torch.as_tensor(means, dtype=torch.float64, device=self.device)
            means = torch.where(torch.isfinite(means) & (means > 0), means, 1.0)
            values = (values / scale / means[selected]).float()
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
