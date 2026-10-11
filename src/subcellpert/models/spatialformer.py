"""spatialformer baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel
from subcellpert.utils import align_gene_ids, align_token_states, freeze


def install_spatial_sdpa(model):
    replaced = 0
    for module in model.modules():
        if type(module).__name__ == "AltAttention":
            module.forward = MethodType(spatial_sdpa_forward, module)
            replaced += 1
    if not replaced:
        raise ValueError("Expected SpatialFormer AltAttention implementation was not found")
    return replaced


def spatial_sdpa_forward(self, inputs, mask=None, alibi_bias=None):
    """Retain SpatialFormer's per-head interleaved QKV and dim**-0.5 scale."""
    batch, length, _ = inputs.shape
    qkv = self.qkv(inputs).view(batch, length, self.num_heads, self.dim * 3 // self.num_heads)
    q, k, v = qkv.permute(0, 2, 1, 3).split(self.dim // self.num_heads, dim=-1)
    bias = alibi_bias
    if mask is not None:
        if bias is None:
            bias = torch.zeros((batch, 1, 1, length), dtype=q.dtype, device=q.device)
        bias = bias.masked_fill(~mask[:, None, None, :], torch.finfo(q.dtype).min)
    values = F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, scale=self.scale, dropout_p=self.attn_drop.p if self.training else 0.0
    )
    self.attn_score = None
    return self.proj(values.transpose(1, 2).reshape(batch, length, self.dim))


def discard_attention_scores(module, inputs, output):
    """Feature extraction never consumes SpatialFormer's diagnostic matrices."""
    module.attn_score = None


class SpatialFormerGeneEncoder:
    def __init__(self, root, metadata, device):
        repository = root / "external/Spatialformer"
        sys.path.insert(0, str(repository))
        from spatialformer.model import Spaformer

        package = repository / "spatialformer"
        config = json.loads((package / "config/_config_train_large_pair.json").read_text())
        config["embedding_path"] = str(package / "spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl")
        self.model = Spaformer(
            **{
                key: config[key]
                for key in (
                    "dim_model",
                    "nheads",
                    "nlayers",
                    "dropout",
                    "masking_p",
                    "n_tokens",
                    "n_atokens",
                    "warmup",
                    "lr",
                    "max_epochs",
                    "mask_way",
                )
            },
            use_flash_attn=False,
            outer_config=config,
        )
        checkpoint = Path(metadata["model_checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True
        )
        install_spatial_sdpa(self.model.encoder)
        for module in self.model.encoder.modules():
            if hasattr(module, "attn_score"):
                module.register_forward_hook(discard_attention_scores)
        freeze(self.model, device)
        self.vocabulary = json.loads((package / "tokenizer/tokenv5.json").read_text())
        with (package / "data/gene_median.pkl").open("rb") as handle:
            self.medians = pickle.load(handle)
        self.tokens = metadata["spatialformer_tokens"]
        self.dimension, self.device = (int(config["dim_model"]) + 1, device)
        self.max_length = int(metadata.get("encoder_max_length", 1200))

    def gene_ids(self, genes):
        return align_gene_ids(genes, self.vocabulary)

    @torch.no_grad()
    def encode(self, matrix, genes, condition):
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(self.gene_ids(genes), device=self.device)
        conditions = [condition] * len(matrix) if isinstance(condition, str) else list(condition)
        if len(conditions) != len(matrix):
            raise ValueError("Source conditions must align with encoded cells")
        prefixes = []
        for source in conditions:
            source_token = self.tokens["condition_by_source"].get(source)
            if source_token is None:
                raise ValueError("SpatialFormer source condition has no registered token")
            prefixes.append(
                [
                    self.vocabulary[name]
                    for name in (
                        "<CLS>",
                        source_token,
                        self.tokens["tissue"],
                        self.tokens["species"],
                        self.tokens["assay"],
                    )
                ]
            )
        medians = torch.as_tensor(
            [self.medians.get(gene, 1.0) for gene in genes], dtype=torch.float64, device=self.device
        )
        medians = torch.where(torch.isfinite(medians) & (medians > 0), medians, 1.0)
        eligible = (matrix > 0) & (ids >= 0)[None]
        lengths = eligible.sum(1).clamp_max(self.max_length)
        length = int(lengths.max()) if len(lengths) else 0
        selected = (-(matrix / medians)).masked_fill(~eligible, float("inf")).argsort(dim=1, stable=True)[:, :length]
        valid_genes = torch.arange(length, device=self.device)[None] < lengths[:, None]
        tokens = torch.zeros((len(matrix), length + 6), dtype=torch.long, device=self.device)
        tokens[:, :5] = torch.as_tensor(prefixes, device=self.device)
        tokens[:, 5 : 5 + length] = torch.where(valid_genes, ids[selected], 0)
        tokens.scatter_(1, (lengths + 5)[:, None], self.vocabulary["<SEP>"])
        positions = torch.arange(length + 6, device=self.device)[None]
        attention = positions < (lengths + 6)[:, None]
        valid = (positions >= 5) & (positions < (lengths + 5)[:, None])
        embeddings = self.model.embeddings(tokens) + self.model.token_type_embeddings(attention.long())
        if self.model.outer_config["spatial_embedding"]:
            embeddings += self.model.spatialembeds(tokens)
        layers, attention_scores = self.model.encoder(embeddings, False, attention)
        result = align_token_states(layers[-1], tokens, ids, valid=valid)
        del layers, attention_scores
        for module in self.model.modules():
            for name in ("attentions", "attn_score"):
                if hasattr(module, name):
                    setattr(module, name, None)
        return result


class Model(Baseline):
    name = "spatialformer"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = SpatialFormerGeneEncoder
