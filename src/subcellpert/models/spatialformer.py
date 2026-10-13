"""SpatialFormer network and benchmark adapter. Derived layers: THIRD_PARTY_LICENSES.md."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

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


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, mask=None):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer("scale", torch.ones(d_model), persistent=True)
            self.register_buffer("bias", torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, dilation=1, bias=True, **kwargs):
        super().__init__(**kwargs)
        assert dilation == 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=groups, bias=bias
        )

    def calc_same_pad(self, i, k, s):
        if i % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - i % s, 0)
        return pad

    def forward(self, inputs):
        x = inputs
        i = x.size()[-1]
        pad = self.calc_same_pad(i=i, k=self.kernel_size, s=self.stride)
        x = F.pad(x, [pad // 2, pad - pad // 2])
        return self.conv(x)


class MaskedConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=17, stride=1, dilation=1, groups=1, bias=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = Conv1dSame(
            in_channels, out_channels, kernel_size, groups=groups, stride=stride, dilation=dilation, bias=bias
        )
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, :: self.stride]
        return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x.masked_fill(~mask[:, None, :], torch.tensor(0.0, dtype=x.dtype, device=x.device))
        x = self.conv(x)
        return x


def masked_batch_norm(
    input: Tensor,
    mask: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float = 1e-05,
) -> Tensor:
    """Applies Masked Batch Normalization for each channel in each data sample in a batch.
    See :class:`~MaskedBatchNorm1d`, :class:`~MaskedBatchNorm2d`, :class:`~MaskedBatchNorm3d` for details.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError("Expected running_mean and running_var to be not None when training=False")
    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims
    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements
        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = (running_mean, running_var)
    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)
    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]
    return out


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        "\n        Decide whether the mini-batch stats should be used for normalization rather than the buffers.\n        Mini-batch stats are used in training mode, and in eval mode when buffers are None.\n        "
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        "\n        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be\n        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are\n        used for normalization (i.e. in eval mode when buffers are not None).\n        "
        if mask is None:
            return F.batch_norm(
                input,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            return masked_batch_norm(
                input,
                mask,
                self.weight,
                self.bias,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                bn_training,
                exponential_average_factor,
                self.eps,
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    """Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.BatchNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        channels_last: bool = False,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.channels_last = channels_last

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :]
        out = super().forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0, 2, 1)
        return out


def get_act_fn(activation):
    if activation == "swish":
        return nn.SiLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "mish":
        return nn.Mish()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class GLU(nn.Module):
    def __init__(self, dim: int, activation: str = "sigmoid") -> None:
        super().__init__()
        self.dim = dim
        self.activation = get_act_fn(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * self.activation(gate)


class GLUMlp(nn.Module):
    def __init__(
        self, dim: int = 512, expand: int = 4, dropout: float = 0.1, bias: bool = True, activation: str = "gelu"
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(dim, dim * expand, bias=bias)
        self.glu = GLU(dim=-1, activation=activation)
        self.do1 = nn.Dropout(p=dropout)
        self.ffn2 = nn.Linear(dim * expand // 2, dim, bias=bias)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.glu(x)
        x = self.do1(x)
        x = self.ffn2(x)
        return x


class AltAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.attn_score = None
        self.scale = self.dim ** (-0.5)
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=True)

    forward = spatial_sdpa_forward


class AltBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        num_heads=4,
        expand=4,
        attn_dropout=0.2,
        mlp_dropout=0.2,
        drop_path=0.0,
        activation="gelu",
        prenorm=True,
        flash_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.flash_attn = flash_attn
        self.norm1 = nn.LayerNorm(dim)
        if not flash_attn:
            self.self_attn = AltAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)
        else:
            raise ValueError("SpatialFormer uses the integrated PyTorch SDPA implementation")
        self.drop1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GLUMlp(dim, expand, dropout=mlp_dropout, activation=activation)
        self.drop2 = DropPath(drop_path)
        self.prenorm = prenorm
        self.attn_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.self_attn(x, mask=mask, alibi_bias=alibi_bias)
        x = self.drop1(x)
        x = self.attn_scale(x)
        x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        attn_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        x = x + attn_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=17,
        groups=4,
        dilation=1,
        stride=1,
        conv_dropout=0.0,
        mlp_dropout=0.0,
        drop_path=0.0,
        expand=4,
        activation="swish",
        prenorm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prenorm = prenorm
        self.stride = stride
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.glu = GLU(dim=-1, activation=activation)
        self.expand_conv = nn.Linear(dim, 2 * dim)
        self.conv = MaskedConv1d(dim, dim, kernel_size=kernel_size, groups=groups)
        self.conv_norm = MaskedBatchNorm1d(dim, momentum=0.05)
        self.conv_act = get_act_fn(activation)
        self.conv_proj = nn.Linear(dim, dim)
        self.mlp = GLUMlp(dim, expand, mlp_dropout, activation=activation)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.drop1 = DropPath(drop_path)
        self.drop2 = DropPath(drop_path)
        self.conv_scale = ScaleBiasLayer(dim, adaptive_scale=True)
        self.mlp_scale = ScaleBiasLayer(dim, adaptive_scale=True)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.stride > 1:
                mask = mask[:, :: self.stride]
        return mask

    def forward(self, inputs, mask=None):
        x = inputs
        if self.prenorm:
            x = self.norm1(x)
        x = self.expand_conv(x)
        x = self.glu(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x, mask=mask)
        mask = self.compute_mask(inputs, mask)
        x = self.conv_norm(x, mask=mask)
        x = self.conv_act(x)
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)
        x = self.conv_proj(x)
        x = self.drop1(x)
        x = self.conv_scale(x)
        if self.stride == 1:
            x = x + inputs
        if not self.prenorm:
            x = self.norm1(x)
        conv_out = x
        if self.prenorm:
            x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.mlp_scale(x)
        if self.stride == 1:
            x = x + conv_out
        if not self.prenorm:
            x = self.norm2(x)
        return x


class SqueezeformerBlock(nn.Module):
    def __init__(
        self,
        dim=256,
        kernel_size=17,
        groups=4,
        num_heads=4,
        conv_expand=4,
        attn_expand=4,
        num_conv_block=1,
        num_attn_block=1,
        conv_dropout=0.1,
        attn_dropout=0.1,
        mlp_dropout=0.1,
        drop_path=0.1,
        activation="swish",
        prenorm=True,
        flash_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv_blocks = nn.ModuleList(
            [
                Conv1DBlock(
                    dim,
                    kernel_size,
                    groups,
                    1,
                    1,
                    conv_dropout,
                    mlp_dropout,
                    drop_path,
                    conv_expand,
                    activation,
                    prenorm,
                )
                for _ in range(num_conv_block)
            ]
        )
        self.attn_blocks = nn.ModuleList(
            [
                AltBlock(
                    dim, num_heads, attn_expand, attn_dropout, mlp_dropout, drop_path, activation, prenorm, flash_attn
                )
                for _ in range(num_attn_block)
            ]
        )

    def forward(self, inputs, mask=None, alibi_bias=None):
        x = inputs
        for block in self.conv_blocks:
            x = block(x, mask=mask)
        for block in self.attn_blocks:
            x = block(x, mask=mask, alibi_bias=alibi_bias)
        return x

    def get_attn(self):
        for block in self.attn_blocks:
            attns = block.self_attn.attn_score
        return attns


def get_alibi(max_positions, attention_heads):
    """SpatialFormer's symmetric 1-D ALiBi (derived from fairseq/data2vec)."""

    def slopes(n):
        def powers_of_two(count):
            start = 2 ** (-(2 ** -(math.log2(count) - 3)))
            return [start * start**i for i in range(count)]

        if math.log2(n).is_integer():
            return powers_of_two(n)
        power = 2 ** math.floor(math.log2(n))
        return powers_of_two(power) + slopes(2 * power)[0::2][: n - power]

    weights = torch.Tensor(slopes(attention_heads))
    positions = torch.arange(max_positions)
    distances = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1)) * -1
    return weights[:, None, None] * distances.unsqueeze(0).expand(attention_heads, -1, -1)


class SpaEncoder(nn.Module):
    """Convolution/attention stack with the benchmark's ALiBi and padding rules."""

    def __init__(self, dim, heads, layers):
        super().__init__()
        self.num_heads = heads
        self.emb_dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [SqueezeformerBlock(dim=dim, groups=dim, num_heads=heads, prenorm=False) for _ in range(layers)]
        )

    def forward(self, embeddings, bpp=None, mask=None):
        if bpp not in (None, False):
            raise ValueError("Benchmark SpatialFormer does not use adjacency inputs")
        x = self.norm(self.emb_dropout(embeddings))
        bias = get_alibi(x.size(1), self.num_heads).to(dtype=x.dtype, device=x.device).repeat(x.size(0), 1, 1, 1)
        outputs = []
        for layer in self.layers:
            x = layer(x, mask=mask, alibi_bias=bias)
            outputs.append(x)
        return outputs, [None] * len(outputs)


class GraphSAGESpatialEmbedding(nn.Module):
    def __init__(self, freeze, embedding_path):
        super().__init__()
        with open(embedding_path, "rb") as handle:
            self.emb = nn.Embedding.from_pretrained(pickle.load(handle), freeze=freeze)

    def forward(self, tokens):
        return self.emb(tokens)


class SpatialFormerNetwork(nn.Module):
    """Pretrained gene embeddings and SpatialFormer encoder, without training heads."""

    def __init__(self, config):
        super().__init__()
        width = config["dim_model"]
        self.outer_config = config
        self.encoder = SpaEncoder(width, config["nheads"], config["nlayers"])
        self.embeddings = nn.Embedding(config["n_tokens"] + config["n_atokens"], width, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(3, width)
        if config["spatial_embedding"]:
            self.spatialembeds = GraphSAGESpatialEmbedding(
                config["spatial_embedding_freeze"], config["embedding_path"]
            )


class SpatialFormerGeneEncoder:
    def __init__(self, root, metadata, device):
        checkpoint = root / metadata["model_checkpoint"]
        package = checkpoint.parent
        config = json.loads((package / "config/_config_train_large_pair.json").read_text())
        config["embedding_path"] = str(checkpoint.parent / "xenium_5k_pandavid_gene_embeddings.pkl")
        self.model = SpatialFormerNetwork(config)
        checkpoint = Path(metadata["model_checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        unused = (
            "classifier_head.",
            "pair_head.",
            "adjprojector.",
            "log_sigma_",
            "encoder.emb_proj.",
            "encoder.bpp_feature_proj.",
            "encoder.bpp_convnet.",
        )
        self.model.load_state_dict(
            {key: value for key, value in weights.items() if not key.startswith(unused)}, strict=True
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
