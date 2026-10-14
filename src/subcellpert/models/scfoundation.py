"""scFoundation MAE, Transformer, Performer and benchmark adapter.

Derived from BioMap scFoundation (Copyright 2023 BioMap); Apache-2.0,
see the repository LICENSE. Only the frozen benchmark computation is retained.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel
from subcellpert.utils import align_gene_ids, freeze, install_transformer_sdpa


def gather_padded_tokens(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)
    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device)
    data = torch.hstack([data, fake_data])
    fake_label = torch.full((labels.shape[0], max_num), 1, device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float("Inf"), device=labels.device)
    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data
    labels = torch.hstack([labels, fake_label])
    fake_label_gene_idx = labels.topk(max_num).indices
    new_data = torch.gather(data, 1, fake_label_gene_idx)
    padding_labels = new_data == pad_token_id
    return (new_data, padding_labels)


def convert_checkpoint(ckpt):
    newconfig = {}
    newconfig["config"] = {}
    model_type = ckpt["config"]["model"]
    for key, val in ckpt["config"]["model_config"][model_type].items():
        newconfig["config"][key] = val
    for key, val in ckpt["config"]["dataset_config"]["rnaseq"].items():
        newconfig["config"][key] = val
    if model_type == "performergau_resolution":
        model_type = "performer_gau"
    import collections

    d = collections.OrderedDict()
    for key, val in ckpt["state_dict"].items():
        d[str(key).split("model.")[1]] = val
    newconfig["config"]["model_type"] = model_type
    newconfig["model_state_dict"] = d
    newconfig["config"]["pos_embed"] = False
    newconfig["config"]["device"] = "cuda"
    return newconfig


class SoftBinEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id=None, pad_token_id=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()
        x = self.mlp(x)
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)
        bin_num_idx = self.bin_num_idx.to(x.device)
        token_emb = self.emb(bin_num_idx)
        x = torch.matmul(weight, token_emb)
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)
        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)
        if output_weight:
            return (x, weight)
        return x


class ScFoundationNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        embed_dim,
        decoder_embed_dim,
        tie_embed=False,
        bin_alpha=1.0,
        bin_num=10,
        pad_token_id=None,
        mask_token_id=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.token_emb = SoftBinEmbedding(
            embed_dim,
            max_seq_len,
            bin_num=bin_num,
            bin_alpha=bin_alpha,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
        )
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)
        self.encoder = None
        self.decoder = None
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.to_final = nn.Linear(decoder_embed_dim, 1)

    def forward(
        self,
        x,
        padding_label,
        encoder_position_gene_ids,
        encoder_labels,
        decoder_data,
        mask_gene_name,
        mask_labels,
        decoder_position_gene_ids,
        decoder_data_padding_labels,
        output_attentions=False,
        **kwargs,
    ):
        n = x.shape[1]
        assert n <= self.max_seq_len, (
            f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"
        )
        x = self.token_emb(torch.unsqueeze(x, 2), output_weight=0)
        if output_attentions:
            x.requires_grad_()
        position_emb = self.pos_emb(encoder_position_gene_ids)
        x += position_emb
        x = self.encoder(x, padding_mask=padding_label)
        decoder_data = self.token_emb(torch.unsqueeze(decoder_data, 2))
        position_emb = self.pos_emb(decoder_position_gene_ids)
        if mask_gene_name:
            raise ValueError("Gene-name masking is not part of benchmark inference")
        batch_idx, gen_idx = encoder_labels.nonzero(as_tuple=True)
        decoder_data[batch_idx, gen_idx] = x[~padding_label].to(decoder_data.dtype)
        decoder_data += position_emb
        decoder_data = self.decoder_embed(decoder_data)
        x = self.decoder(decoder_data, padding_mask=decoder_data_padding_labels)
        x = self.norm(x)
        if self.to_final is not None:
            x = self.to_final(x)
            return x.squeeze(2)
        else:
            return x
        return x


class ScFoundationTransformer(nn.Module):
    def __init__(self, max_seq_len, dim, depth, heads, ff_mult=4, norm_first=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        layers = []
        for i in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult, batch_first=True, norm_first=norm_first
                )
            )
        self.transformer_encoder = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, padding_mask):
        n = x.shape[1]
        assert n <= self.max_seq_len, (
            f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"
        )
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x


class FastAttention(nn.Module):
    """Checkpoint projection buffer; replaced by cell-independent attention after loading."""

    def __init__(self, dim_head):
        super().__init__()
        self.register_buffer("projection_matrix", torch.zeros(int(dim_head * math.log(dim_head)), dim_head))
        self.causal = self.generalized_attention = self.no_projection = False

    def forward(self, q, k, v):
        return CellIndependentAttention(self)(q, k, v)


class PerformerAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = heads
        self.fast_attention = FastAttention(dim_head)
        self.to_q = nn.Linear(dim, heads * dim_head)
        self.to_k = nn.Linear(dim, heads * dim_head)
        self.to_v = nn.Linear(dim, heads * dim_head)
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = (
            layer(x).reshape(batch, length, self.heads, -1).permute(0, 2, 1, 3)
            for layer in (self.to_q, self.to_k, self.to_v)
        )
        # The published scFoundation module ignores padding_mask in this stack;
        # preserve that behavior rather than silently changing checkpoint outputs.
        out = self.fast_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.dropout(self.to_out(out))


class PerformerFeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))


class FeedForwardChunk(nn.Module):
    """Preserve the checkpoint's fn nesting for the unchunked feed-forward path."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.fn = PerformerFeedForward(dim, dropout)

    def forward(self, x):
        return self.fn(x)


class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PerformerSequence(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["hidden_dim"]
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreLayerNorm(
                            dim,
                            PerformerAttention(
                                dim, config["heads"], config["dim_head"], config.get("attn_dropout", 0.0)
                            ),
                        ),
                        PreLayerNorm(dim, FeedForwardChunk(dim, config.get("ff_dropout", 0.0))),
                    ]
                )
                for _ in range(config["depth"])
            ]
        )

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = x + attention(x)
            x = x + feed_forward(x)
        return x


class FrozenPerformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = PerformerSequence(config)
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def forward(self, x):
        # Pretrained projections are frozen along with the feature encoder.
        return self.net(x)


class PerformerModule(nn.Module):
    def __init__(self, config, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.performer = FrozenPerformer(config)
        self.norm = nn.LayerNorm(config["hidden_dim"])

    def forward(self, x, padding_mask=None):
        if x.shape[1] > self.max_seq_len:
            raise ValueError("scFoundation sequence exceeds the checkpoint token budget")
        return self.norm(self.performer(x))


def build_scfoundation_network(config):
    if config["model"] != "mae_autobin":
        raise ValueError("Only the benchmark scFoundation MAE checkpoint is supported")
    modules = []
    for part in ("encoder", "decoder"):
        sub = config[part]
        if sub["module_type"] == "performer":
            module = PerformerModule(sub, config["seq_len"])
        elif sub["module_type"] == "transformer":
            module = ScFoundationTransformer(config["seq_len"], sub["hidden_dim"], sub["depth"], sub["heads"])
        else:
            raise ValueError(f"Unsupported scFoundation module: {sub['module_type']}")
        modules.append(module)
    model = ScFoundationNetwork(
        num_tokens=config["n_class"],
        max_seq_len=config["seq_len"],
        embed_dim=config["encoder"]["hidden_dim"],
        decoder_embed_dim=config["decoder"]["hidden_dim"],
        bin_alpha=config["bin_alpha"],
        bin_num=config["bin_num"],
        pad_token_id=config["pad_token_id"],
        mask_token_id=config["mask_token_id"],
    )
    model.encoder, model.decoder = modules
    return model


class CellIndependentAttention(torch.nn.Module):
    """Vectorize scFoundation's native singleton softmax Performer kernel.

    scFoundation's Performer key kernel subtracts a scalar maximum over the
    entire batch before adding epsilon. That epsilon makes a cell's output
    depend on its batch mates. Reduce over heads, tokens and random features
    separately for each cell. Keep the upstream epsilon, projection, scaling
    and linear attention equations, with no Python loop over cells.
    """

    def __init__(self, attention):
        super().__init__()
        if attention.causal or attention.generalized_attention or attention.no_projection:
            raise ValueError("Expected noncausal softmax Performer attention")
        self.attention = attention

    def _kernel(self, data, *, is_query):
        normalizer = data.shape[-1] ** (-0.25)
        projection = self.attention.projection_matrix.type_as(data)
        projection = projection.expand(data.shape[0], data.shape[1], -1, -1)
        projected = torch.einsum("...id,...jd->...ij", normalizer * data, projection)
        diagonal = ((data**2).sum(dim=-1) / 2.0 * normalizer**2).unsqueeze(-1)
        maximum = (
            projected.max(dim=-1, keepdim=True).values if is_query else projected.amax(dim=(1, 2, 3), keepdim=True)
        )
        return ((projected - diagonal - maximum).exp() + 0.0001) * projection.shape[-2] ** (-0.5)

    @staticmethod
    def _linear(q, k, v):
        inverse = 1.0 / torch.einsum("...nd,...d->...n", q, k.sum(dim=-2).type_as(q))
        context = torch.einsum("...nd,...ne->...de", k, v)
        return torch.einsum("...de,...nd,...n->...ne", context, q, inverse)

    def forward(self, q, k, v, output_attentions=False):
        q, k = (self._kernel(q, is_query=True), self._kernel(k, is_query=False))
        result = self._linear(q, k, v)
        if output_attentions:
            identity = torch.eye(v.shape[-2], device=v.device, dtype=torch.float16)
            identity = identity.expand(v.shape[0], v.shape[1], -1, -1)
            weights = self._linear(q.half(), k.half(), identity).detach().cpu() / q.shape[1]
            return (result, weights)
        return result


def isolate_attention_cells(module, attention_type):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, attention_type):
            setattr(module, name, CellIndependentAttention(child))
            replaced += 1
        else:
            replaced += isolate_attention_cells(child, attention_type)
    return replaced


class ScFoundationGeneEncoder:
    def __init__(self, root, metadata, device):
        checkpoint = Path(metadata["model_checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        repository = checkpoint.parent
        payload = convert_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False)["gene"])
        self.config = payload["config"]
        self.config.setdefault("ppi_edge", None)
        self.config["device"] = str(device)
        self.model = build_scfoundation_network(self.config)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to_final = None
        install_transformer_sdpa(self.model)
        if not isolate_attention_cells(self.model, FastAttention):
            raise ValueError("Expected scFoundation Performer attention implementation was not found")
        freeze(self.model, device)
        self.gather = gather_padded_tokens
        import pandas as pd

        names = pd.read_csv(repository / "OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"].tolist()
        if len(names) != 19264:
            raise ValueError("scFoundation requires its official 19,264-gene index")
        self.vocabulary = {name: index for index, name in enumerate(names)}
        self.dimension, self.device = (int(self.config["decoder"]["hidden_dim"]) + 1, device)
        self.gene_context = metadata.get("scfoundation_gene_context", "assay")
        if self.gene_context not in ("assay", "full"):
            raise ValueError("scFoundation gene context must be 'assay' or 'full'")

    def gene_ids(self, genes):
        return align_gene_ids(genes, self.vocabulary)

    @torch.no_grad()
    def encode(self, matrix, genes, condition):
        del condition
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(self.gene_ids(genes), device=self.device)
        columns = torch.nonzero(ids >= 0, as_tuple=True)[0]
        columns = columns[ids[columns].argsort(stable=True)]
        result = torch.zeros((len(matrix), len(genes), self.dimension), dtype=torch.float32, device=self.device)
        nonempty = torch.nonzero(matrix[:, columns].sum(1) > 0, as_tuple=True)[0]
        if not len(nonempty):
            return result
        raw = matrix[nonempty[:, None], columns]
        gene_positions = ids[columns]
        output_positions = torch.arange(len(columns), device=self.device)
        if self.gene_context == "full":
            full = torch.zeros((len(nonempty), 19264), device=self.device)
            full[:, gene_positions] = raw
            raw, gene_positions = full, torch.arange(19264, device=self.device)
            output_positions = ids[columns]
        total = raw.sum(1)
        normalized = torch.log1p(raw / total[:, None] * 10000)
        depth = torch.log10(total)[:, None]
        decoder = torch.cat((normalized, depth, depth), 1)
        labels = decoder > 0
        labels[:, -2:] = True
        positions = torch.cat((gene_positions, torch.tensor([19264, 19265], device=self.device)))
        positions = positions.expand(len(nonempty), -1)
        encoded, padding = self.gather(decoder, labels, self.config["pad_token_id"])
        encoder_positions, _ = self.gather(positions, labels, self.config["pad_token_id"])
        encoder_positions[padding] = self.config["seq_len"]
        states = self.model(
            x=encoded,
            padding_label=padding,
            encoder_position_gene_ids=encoder_positions,
            encoder_labels=labels,
            decoder_data=decoder,
            mask_gene_name=False,
            mask_labels=None,
            decoder_position_gene_ids=positions,
            decoder_data_padding_labels=torch.zeros_like(decoder, dtype=torch.bool),
        )
        result[nonempty[:, None], columns, :-1] = states[:, output_positions].float()
        result[nonempty[:, None], columns, -1] = 1
        return result


class Model(Baseline):
    name = "scfoundation"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = ScFoundationGeneEncoder
