"""STATE-SE network and benchmark adapter. Derived code: the repository LICENSE."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel


STATE_FEATURE_POLICY = "state_se_gene_tokens_deterministic_source_only_v1"


class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        Initializes the encoder layer.
        Args:
            d_model (int): model dimension.
            nhead (int): number of attention heads.
            dim_feedforward (int): dimension of the feed-forward network.
            dropout (float): dropout probability.
        """
        super().__init__()
        torch.backends.cuda.enable_flash_sdp(True)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, d_model)
            src_mask: (optional) attention mask.
            src_key_padding_mask: (optional) padding mask.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        residual = src
        qkv = self.qkv_proj(src)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        head_dim = self.d_model // self.nhead
        q = q.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        k = k.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        v = v.view(src.size(0), src.size(1), self.nhead, head_dim).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=False)
        attn_output = attn_output.transpose(1, 2).contiguous().view(src.size(0), src.size(1), self.d_model)
        attn_output = self.out_proj(attn_output)
        src = self.norm1(residual + self.dropout_layer(attn_output))
        residual2 = src
        ff_output = self.linear2(self.dropout_layer(F.gelu(self.linear1(src))))
        src = self.norm2(residual2 + self.dropout_layer(ff_output))
        return src


class FlashTransformerEncoder(nn.Module):
    def __init__(self, layers):
        """
        A simple encoder that applies a stack of FlashTransformerEncoderLayer instances.
        Args:
            layers (list[nn.Module]): list of FlashTransformerEncoderLayer instances.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Applies each encoder layer in sequence.
        Args:
            src: Tensor of shape (B, T, d_model)
            src_mask: (optional) attention mask.
            src_key_padding_mask: (optional) padding mask.
        Returns:
            Tensor of shape (B, T, d_model)
        """
        mask = src_key_padding_mask if src_key_padding_mask is not None else src_mask
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=mask)
        return output


class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))

        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


class StateEmbeddingModel(nn.Module):
    """STATE-SE inference network; derived under CC-BY-NC-SA-4.0.

    Keep the protein/count encoders, Transformer and gene-state decoder; omit
    downstream binary/dataset classifiers and the upstream training framework.
    """

    def __init__(self, token_dim, d_model, nhead, d_hid, nlayers, output_dim, dropout=0.0, compiled=False, cfg=None):
        super().__init__()
        if compiled:
            raise ValueError("Compile the complete frozen adapter after loading its checkpoint")
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))
        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.transformer_encoder = FlashTransformerEncoder(
            [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout) for _ in range(nlayers)]
        )
        self.decoder = nn.Sequential(SkipBlock(d_model), nn.Linear(d_model, output_dim))
        if cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(nn.Linear(1, 512), nn.LeakyReLU(), nn.Linear(512, 10))
        self.dataset_token = (
            nn.Parameter(torch.randn(1, token_dim)) if cfg.model.get("dataset_correction", False) else None
        )

    def forward(self, src: Tensor, mask: Tensor, counts=None, dataset_nums=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        if counts is not None:
            counts = counts.unsqueeze(-1)
            bin_weights = self.count_encoder(counts)
            bin_weights = F.softmax(bin_weights, dim=-1)
            bin_indices = torch.arange(10, device=src.device)
            bin_embeddings = self.bin_encoder(bin_indices)
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            if self.dataset_token is not None:
                dataset_count_emb = torch.zeros(count_emb.size(0), 1, count_emb.size(2), device=src.device)
                count_emb = torch.cat((count_emb, dataset_count_emb), dim=1)
            src = src + count_emb
        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)
        embedding = gene_output[:, 0, :]
        embedding = nn.functional.normalize(embedding, dim=1)
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]
        return (gene_output, embedding, dataset_emb)


class StateSEGeneEncoder:
    def __init__(self, root, metadata, device):
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        root = Path(root)

        if metadata.get("state_feature_policy") != STATE_FEATURE_POLICY:
            raise ValueError("Missing explicit STATE-SE inference/mapping policy")
        directory = root / metadata["model_directory"]
        cfg = OmegaConf.load(directory / "config.yaml")
        m = cfg.model
        self.model = StateEmbeddingModel(
            token_dim=cfg.tokenizer.token_dim,
            d_model=m.emsize,
            nhead=m.nhead,
            d_hid=m.d_hid,
            nlayers=m.nlayers,
            output_dim=m.output_dim,
            dropout=m.dropout,
            compiled=False,
            cfg=cfg,
        )
        weights = load_file(str(directory / "model.safetensors"))
        unused = ("pe_embedding.", "gene_embedding_layer.", "binary_decoder.", "dataset_embedder.", "dataset_encoder.")
        active = {key: value for key, value in weights.items() if not key.startswith(unused)}
        self.model.load_state_dict(active, strict=True)
        del weights
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)
        disable_state_attention_dropout(self.model)
        proteins = torch.load(directory / "protein_embeddings.pt", map_location="cpu", weights_only=True)
        self.vocabulary = {name: i for i, name in enumerate(proteins)}
        self.proteins = torch.stack(list(proteins.values())).float().to(self.device)
        if self.proteins.shape[1] != cfg.tokenizer.token_dim or not torch.isfinite(self.proteins).all():
            raise ValueError("STATE protein features are incompatible with the checkpoint")
        self.species = metadata["species"]
        self.homology = one_to_one_mouse_homology(directory / "HOM_MouseHumanSequence.rpt")
        self.max_length = int(metadata["encoder_max_length"])
        if self.max_length != cfg.dataset.pad_length:
            raise ValueError("STATE token budget must match the pinned checkpoint")
        self.dimension = int(m.output_dim) + 1
        self.statistics = None
        self.configure_compilation(metadata.get("compile_encoder", False))

    def configure_compilation(self, enabled):
        """Compile the frozen native forward while retaining its checkpoint keys."""
        if type(enabled) is not bool:
            raise ValueError("foundation.compile_encoder must be a boolean")
        self.compiled = enabled and self.device.type == "cuda"
        if self.compiled:
            self.model.compile(dynamic=True, fullgraph=True, options={"emulate_precision_casts": True})
            print("[state_se] Compiling the online encoder on first use | dynamic batches | bfloat16", flush=True)
        elif getattr(self.model, "_compiled_call_impl", None) is not None:
            self.model.compile(disable=True)

    def gene_ids(self, genes):
        return map_state_genes(genes, self.vocabulary, self.species, self.homology)

    @torch.no_grad()
    def encode(self, matrix, genes, condition):
        del condition
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        ids = self.gene_ids(genes)
        selected, normalized, nonempty = state_token_batch(matrix, genes, ids, self.max_length)
        result = torch.zeros((len(matrix), len(genes), self.dimension), dtype=torch.float32, device=self.device)
        present = nonempty.nonzero(as_tuple=True)[0]
        if not len(present):
            return result
        selected = selected[present]
        axis_ids = torch.as_tensor(ids, device=self.device)
        tokens = self.proteins[axis_ids[selected]].to(self.dtype)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        inputs = torch.cat((self.model.cls_token.expand(len(present), 1, -1), tokens), dim=1)
        counts = torch.cat((torch.zeros((len(present), 1), device=self.device), normalized[present]), dim=1).to(
            self.dtype
        )
        mask = torch.zeros(counts.shape, dtype=torch.bool, device=self.device)
        if self.model.dataset_token is not None:
            inputs = torch.cat((inputs, self.model.dataset_token.expand(len(present), 1, -1)), dim=1)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            states, _, _ = self.model(inputs, mask, counts=counts)
        result[present[:, None], selected, :-1] = states[:, 1 : tokens.shape[1] + 1].float()
        result[present[:, None], selected, -1] = 1
        return result


def disable_state_attention_dropout(model):
    """Upstream SDPA receives a scalar dropout even in eval; force inference mode."""
    model.eval().requires_grad_(False)
    for layer in model.transformer_encoder.layers:
        layer.dropout = 0.0


def map_state_genes(genes, vocabulary, species, homology=None):
    if species not in ("Human", "Mouse"):
        raise ValueError("STATE-SE requires an explicit Human or Mouse gene mapping")
    symbols = list(genes) if species == "Human" else [(homology or {}).get(g) for g in genes]
    ids = np.asarray([vocabulary.get(g, -1) for g in symbols], dtype=np.int64)
    matched = ids[ids >= 0]
    if len(set(matched)) != len(matched):
        raise ValueError("STATE-SE gene mapping must be one-to-one on the assay axis")
    return ids


def one_to_one_mouse_homology(path):
    groups = defaultdict(lambda: defaultdict(set))
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            taxon = row["NCBI Taxon ID"]
            if taxon in ("10090", "9606"):
                groups[row["DB Class Key"]][taxon].add(row["Symbol"])
    candidates = defaultdict(set)
    ambiguous = set()
    for species in groups.values():
        mouse, human = (species["10090"], species["9606"])
        if len(mouse) == len(human) == 1:
            candidates[next(iter(mouse))].update(human)
        elif mouse and human:
            ambiguous.update(mouse)
    mapping = {m: next(iter(h)) for m, h in candidates.items() if len(h) == 1 and m not in ambiguous}
    reverse = defaultdict(set)
    for mouse, human in mapping.items():
        reverse[human].add(mouse)
    return {mouse: human for mouse, human in mapping.items() if len(reverse[human]) == 1}


def state_token_inputs(matrix, genes, ids, max_length):
    """Expose ragged rows for callers that need them, with one batched host read."""
    selected, normalized, nonempty = state_token_batch(matrix, genes, ids, max_length)
    return [
        (selected[i], normalized[i]) if present else (selected[i, :0], normalized[i, :0])
        for i, present in enumerate(nonempty.tolist())
    ]


def state_token_batch(matrix, genes, ids, max_length):
    """Rank/normalize whole panels without a device synchronization per cell."""
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(genes) or not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("STATE-SE requires a finite nonnegative cell-by-gene count matrix")
    if max_length < 2:
        raise ValueError("STATE-SE needs room for CLS and at least one gene")
    matched = sorted((i for i, token in enumerate(ids) if token >= 0), key=lambda i: genes[i])
    matched = torch.as_tensor(matched, dtype=torch.long, device=matrix.device)
    values = matrix[:, matched]
    log_counts = values.log1p()
    totals = log_counts.sum(1)
    order = values.argsort(dim=1, descending=True, stable=True)[:, : max_length - 1]
    selected = matched[order]
    normalized = 100 * log_counts.gather(1, order) / totals.clamp_min(torch.finfo(torch.float32).tiny)[:, None]
    return selected, normalized, totals > 0


STATE_SOURCE_REVISION = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"


def state_binding_details(root, metadata):
    """Bind local inference choices separately from pretrained weight identity."""
    assets = json.loads((Path(root) / metadata["model_directory"] / "assets.json").read_text())
    return {
        "feature_policy": STATE_FEATURE_POLICY,
        "model_revision": assets["model_revision"],
        "source_revision": STATE_SOURCE_REVISION,
        "species": metadata["species"],
        "gene_mapping": "exact_human_symbols" if metadata["species"] == "Human" else "MGI_one_to_one_mouse_human",
        "tokens": "ranked_unique_measured_genes; no repeated padding; CLS_count_zero; optional_native_dataset_token",
        "normalization": "100*log1p(source_counts)/sum_log1p_mapped_measured_source_genes",
        "attention_dropout": 0.0,
        "precision": "bfloat16_cuda_float32_cpu; output_float32",
        "compile_encoder": metadata.get("compile_encoder", False),
        "native_state_transition_used": False,
        "missing_gene_policy": "unavailable_fm_features; retain_all_task_counts_and_RNA",
        "disk_feature_cache": False,
    }


class Model(Baseline):
    name = "state_se"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = StateSEGeneEncoder
