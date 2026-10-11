"""Shared tensor operations, transport math, configuration checks, and deterministic IO."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib import import_module
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F


if TYPE_CHECKING:
    from subcellpert.data.batching import CellRNABatch
    from subcellpert.models.shared import TensorCells


def get_model(name, setting):
    if name not in SUPPORTED_BASELINES:
        raise ValueError(f"Unknown baseline: {name}")
    return import_module(f"subcellpert.models.{name}").Model(setting)


CELL_CLASSICAL_SCHEMA = "complete_cell_distribution_classical_oracle"
QUANTILES = np.linspace(0, 1, 16)


def complete_cell_statistics(batch, *, source_fallback=False, include_radial=True, device="cpu"):
    """Exact moments and empirical radial quantiles, with every RNA retained."""
    if torch.device(device).type == "cuda":
        from subcellpert.classical_cuda import complete_cell_statistics_cuda

        return complete_cell_statistics_cuda(
            batch, device=device, source_fallback=source_fallback, include_radial=include_radial
        )
    batch.validated()
    b, g = batch.counts.shape
    mean = np.zeros((b, g, 2), dtype=np.float64)
    second = np.zeros((b, g, 2, 2), dtype=np.float64)
    radial = np.zeros((b, g, len(QUANTILES)), dtype=np.float64)
    valid = batch.counts > 0
    for cell in range(b):
        first, last = batch.cell_offsets[cell : cell + 2]
        xy, labels = (batch.coordinates[first:last].astype(np.float64), batch.molecule_genes[first:last])
        order = np.argsort(labels, kind="stable")
        ends = np.r_[0, np.cumsum(batch.counts[cell])]
        if source_fallback:
            mean[cell] = xy.mean(0)
            second[cell] = xy.T @ xy / len(xy)
            if include_radial:
                radial[cell] = np.quantile(np.linalg.norm(xy - mean[cell, 0], axis=1), QUANTILES)
        for gene in np.flatnonzero(batch.counts[cell]):
            points = xy[order[ends[gene] : ends[gene + 1]]]
            mean[cell, gene] = points.mean(0)
            second[cell, gene] = points.T @ points / len(points)
            if include_radial:
                radii = np.linalg.norm(points - mean[cell, gene], axis=1)
                radial[cell, gene] = np.quantile(radii, QUANTILES)
    return (mean, second, radial, np.ones_like(valid) if source_fallback else valid)


CELL_JOINT_CLASSICAL_SCHEMA = "factorized_complete_cell_joint_classical"
MAX_LOG_COUNT_SHIFT = float(np.log(4.0))


def integer_counts(log_counts):
    """Deterministic rounding; preserve zeros, including entirely empty outputs."""
    values = np.asarray(log_counts, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 40):
        raise ValueError("Invalid or numerically unsafe predicted log1p counts")
    return np.rint(np.expm1(values)).astype(np.int64)


RNA_COMPUTE_CHUNK = 65536


def cell_distribution_features(batch: TensorCells):
    """One vector per cell, preserving every gene's abundance and geometry."""
    b, g = batch.counts.shape
    result = batch.xy.new_zeros((b * g, 29))
    for first in range(0, len(batch.xy), RNA_COMPUTE_CHUNK):
        sl = slice(first, first + RNA_COMPUTE_CHUNK)
        labels = batch.cells[sl] * g + batch.genes[sl]
        result = result.index_add(0, labels, spatial_features(batch.xy[sl]))
    spatial = result.reshape(b, g, 29) / batch.counts.clamp_min(1)[..., None]
    return torch.cat((torch.log1p(batch.counts)[..., None], spatial), dim=2).flatten(1)


def predict_cell_batch(
    model,
    source_batch,
    oracle_counts,
    descriptor,
    *,
    rng,
    device,
    foundation=None,
    donor_rows=None,
    control_identity=False,
    joint_training=False,
):
    """Target geometry cannot be passed through this count-only interface."""
    from subcellpert.models.shared import TensorCells

    source = TensorCells.from_batch(source_batch, device)
    donors = (
        rng.integers(len(source_batch.cell_ids), size=len(oracle_counts))
        if donor_rows is None
        else np.asarray(donor_rows, dtype=np.int64)
    )
    if donors.shape != (len(oracle_counts),) or np.any(donors < 0) or np.any(donors >= len(source_batch.cell_ids)):
        raise ValueError("Invalid computational donor assignment")
    counts = torch.as_tensor(oracle_counts, dtype=torch.float32, device=device)
    condition = torch.as_tensor(
        np.broadcast_to(descriptor, (len(counts), model.descriptor_dimension)).copy(),
        dtype=torch.float32,
        device=device,
    )
    args = (source, counts, condition, torch.as_tensor(donors, device=device), foundation)
    if joint_training:
        params, predicted_log_counts = model(*args, joint_training=True)
    else:
        params = model(*args)
    xy, genes, cells, fallback = sample_donor_rna(
        source_batch, oracle_counts, donors, rng, control_identity=control_identity
    )
    xy = torch.as_tensor(xy, device=device)
    genes = torch.as_tensor(genes, device=device)
    cells = torch.as_tensor(cells, device=device)
    selected = params[cells, genes]
    predicted = torch.bmm(selected[:, :4].reshape(-1, 2, 2), xy[..., None]).squeeze(-1) + selected[:, 4:]
    result = (TensorCells(counts, predicted, genes, cells), donors, fallback)
    return (*result, predicted_log_counts) if joint_training else result


def sample_donor_rna(source: CellRNABatch, target_counts, donor_rows, rng, *, control_identity=False):
    """Generate all requested RNA from ONE coherent source cell per output cell.

    Absent genes use the same donor cell's all-RNA prior. This is an explicit
    generation fallback, not imputation of observed source expression.
    """
    target_counts = np.asarray(target_counts)
    donors = np.asarray(donor_rows)
    b, g = target_counts.shape
    if g != len(source.genes) or donors.shape != (b,) or donors.dtype.kind not in "iu":
        raise ValueError("Donors and complete target count axes must align")
    if donors.min() < 0 or donors.max() >= len(source.cell_ids):
        raise ValueError("Source donor row out of range")
    if target_counts.dtype.kind not in "iu" or np.any(target_counts < 0) or np.any(target_counts.sum(1) <= 0):
        raise ValueError("Oracle cells need nonnegative integer counts and at least one RNA")
    identity = np.broadcast_to(np.asarray(control_identity, dtype=bool), (b,))
    if not np.array_equal(target_counts[identity], source.counts[donors[identity]]):
        raise ValueError("Control identity examples must preserve the source cell count vector")
    total = int(target_counts.sum())
    # Stable cell/gene order matches the former per-donor sort, including
    # unsorted input labels. Missing genes still draw from raw donor RNA.
    labels = source.molecule_cells.astype(np.int64) * g + source.molecule_genes
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.cumsum(source.counts.ravel(), dtype=np.int64)]
    output_cells, output_genes = np.nonzero(target_counts)
    lengths = target_counts[output_cells, output_genes]
    source_cells = donors[output_cells]
    groups = source_cells * g + output_genes
    present = source.counts[source_cells, output_genes] > 0
    first = np.where(present, offsets[groups], source.cell_offsets[source_cells])
    last = np.where(present, offsets[groups + 1], source.cell_offsets[source_cells + 1])
    low, high = np.repeat(first, lengths), np.repeat(last, lengths)
    fixed = np.repeat(identity[output_cells], lengths)
    selected = np.empty(total, dtype=np.int64)
    # Broadcasting bounds consumes the same integer draws in cell/gene/RNA
    # order. Identity molecules consume no randomness, as before.
    sampled = ~fixed
    if np.any(sampled):
        selected[sampled] = rng.integers(low[sampled], high[sampled])
    if np.any(fixed):
        within_gene = np.arange(total) - np.repeat(np.r_[0, np.cumsum(lengths[:-1])], lengths)
        selected[fixed] = low[fixed] + within_gene[fixed]
    matched = np.repeat(present, lengths)
    selected[matched] = order[selected[matched]]
    return (
        source.coordinates[selected],
        np.repeat(output_genes, lengths),
        np.repeat(output_cells, lengths),
        int(lengths[~present].sum()),
    )


def spatial_features(xy):
    directions = xy.new_tensor([[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678], [0.70710678, -0.70710678]])
    phase = (xy @ directions.T)[..., None] * xy.new_tensor([1.0, 2.0, 4.0]) * torch.pi
    return torch.cat((xy, xy.square(), xy[:, :1] * xy[:, 1:], phase.sin().flatten(1), phase.cos().flatten(1)), dim=1)


def _additive_mask(mask, dtype):
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=dtype).masked_fill(mask, float("-inf"))
    return mask.to(dtype)


def install_transformer_sdpa(model):
    """Patch only ordinary PyTorch self-attention layers, preserving state keys."""
    replaced = 0
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoder):
            module.enable_nested_tensor = module.use_nested_tensor = False
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            attention = module.self_attn
            if not attention.batch_first or not attention._qkv_same_embed_dim or attention.add_zero_attn:
                raise ValueError("Unsupported foundation Transformer attention configuration")
            if attention.bias_k is not None or attention.bias_v is not None:
                raise ValueError("Foundation Transformer must not append key/value bias tokens")
            module.forward = MethodType(transformer_sdpa_forward, module)
            replaced += 1
    return replaced


def transformer_sdpa_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    """Dense batch-first TransformerEncoderLayer with fused SDPA self-attention."""
    if src.is_nested or src.ndim != 3 or (not self.self_attn.batch_first):
        raise ValueError("Foundation SDPA requires dense batch-first token states")

    def attend(x):
        attention = self.self_attn
        batch, length, width = x.shape
        heads = attention.num_heads
        qkv = F.linear(x, attention.in_proj_weight, attention.in_proj_bias)
        q, k, v = qkv.reshape(batch, length, 3, heads, width // heads).permute(2, 0, 3, 1, 4).unbind(0)
        mask = None
        if src_mask is not None:
            mask = _additive_mask(src_mask, q.dtype)
            if mask.ndim == 3:
                mask = mask.reshape(batch, heads, length, length)
        if src_key_padding_mask is not None:
            padding = _additive_mask(src_key_padding_mask, q.dtype)[:, None, None, :]
            mask = padding if mask is None else mask + padding
        if is_causal:
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            causal = _additive_mask(causal, q.dtype)
            mask = causal if mask is None else mask + causal
        values = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=attention.dropout if self.training else 0.0
        )
        values = values.transpose(1, 2).reshape(batch, length, width)
        return self.dropout1(attention.out_proj(values))

    def feedforward(x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    if self.norm_first:
        x = src + attend(self.norm1(src))
        return x + feedforward(self.norm2(x))
    x = self.norm1(src + attend(src))
    return self.norm2(x + feedforward(x))


GEARS_JOINT_SCHEMA = "gene_fm_gears_complete_cell_joint_v2"
GEARS_ORACLE_SCHEMA = "gene_fm_gears_complete_cell_oracle_v2"


def positive_neighbor_graph(similarity, neighbors=20):
    """Directed positive top-k edges; SGConv supplies each node's self-loop."""
    scores = np.asarray(similarity, dtype=np.float64).copy()
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or (not np.isfinite(scores).all()):
        raise ValueError("A GEARS similarity graph must be a finite square matrix")
    np.fill_diagonal(scores, 0)
    result = np.zeros_like(scores, dtype=np.float32)
    for target, row in enumerate(scores):
        sources = np.argsort(-row, kind="stable")[:neighbors]
        sources = sources[row[sources] > 0]
        result[sources, target] = row[sources]
    return result


def source_gene_graph(records, neighbors=20):
    """Coexpression from unique allowed SOURCE cells only, never responses."""
    genes = tuple(records[0].source[0].genes)
    g = len(genes)
    total, cross, n = (np.zeros(g), np.zeros((g, g)), 0)
    seen = set()
    for record in records:
        for population in record.source:
            if tuple(population.genes) != genes:
                raise ValueError("Source coexpression gene order differs between populations")
            for first in range(0, len(population.cell_ids), 256):
                rows = []
                for row in range(first, min(first + 256, len(population.cell_ids))):
                    key = (population.unit_id, population.condition, str(population.cell_ids[row]))
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                if not rows:
                    continue
                values = np.log1p(np.asarray(population.counts[rows], dtype=np.float64))
                if values.shape[1] != g or not np.isfinite(values).all():
                    raise ValueError("Source coexpression requires the complete finite gene axis")
                total += values.sum(0)
                cross += values.T @ values
                n += len(values)
    if not n:
        raise ValueError("GEARS coexpression needs nonempty source cells")
    covariance = cross - np.outer(total, total) / n
    scale = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
    return positive_neighbor_graph(np.clip(correlation, -1, 1), neighbors)


def cell_key(population, cell, counts):
    return (population.unit_id, population.condition, str(cell), hashlib.sha256(counts.tobytes()).digest())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def align_gene_ids(genes, vocabulary):
    lookup = {}
    for name, token in vocabulary.items():
        upper = str(name).upper()
        if upper in lookup and lookup[upper] != token:
            raise ValueError("Ambiguous case-insensitive foundation vocabulary")
        lookup[upper] = token
    result = np.asarray([vocabulary.get(gene, lookup.get(gene.upper(), -1)) for gene in genes], dtype=np.int64)
    matched = result[result >= 0]
    if len(np.unique(matched)) != len(matched):
        raise ValueError("Assay genes collide in the foundation vocabulary")
    return result


def align_token_states(states, token_ids, assay_token_ids, *, valid=None):
    """Align frozen token states on their device, including availability flags."""
    if not isinstance(states, torch.Tensor):
        raise TypeError("Foundation encoders must return device tensors")
    token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=states.device)
    assay_ids = torch.as_tensor(assay_token_ids, dtype=torch.long, device=states.device)
    if states.ndim != 3 or states.shape[:2] != token_ids.shape or not torch.isfinite(states).all():
        raise ValueError("Invalid contextual gene-token output")
    valid = (
        torch.ones_like(token_ids, dtype=torch.bool)
        if valid is None
        else torch.as_tensor(valid, dtype=torch.bool, device=states.device)
    )
    if valid.shape != token_ids.shape:
        raise ValueError("Token availability mask shape changed")
    result = torch.zeros(
        (len(states), len(assay_ids), states.shape[-1] + 1), dtype=torch.float32, device=states.device
    )
    columns = torch.nonzero(assay_ids >= 0, as_tuple=True)[0]
    if not len(columns):
        return result
    ordered, order = assay_ids[columns].sort()
    positions = torch.searchsorted(ordered, token_ids).clamp_max(len(ordered) - 1)
    rows, tokens = torch.nonzero(valid & (ordered[positions] == token_ids), as_tuple=True)
    genes = columns[order[positions[rows, tokens]]]
    flat = rows * len(assay_ids) + genes
    if len(flat.unique()) != len(flat):
        raise ValueError("Duplicate gene token in a source-cell representation")
    result[rows, genes, :-1] = states[rows, tokens].float()
    result[rows, genes, -1] = 1
    return result


def freeze(model, device):
    model.to(device).eval()
    model.requires_grad_(False)
    return model


def resolve_model_directory(root, metadata):
    """Resolve the configured pretrained directory without legacy path fallbacks."""
    root = Path(root).resolve()
    directory = Path(metadata["model_directory"])
    return directory if directory.is_absolute() else root / directory


def _bounded_matrix(matrix: np.ndarray, minimum: float = 0.1, maximum: float = 10.0) -> np.ndarray:
    u, values, vh = np.linalg.svd(matrix)
    return u * np.clip(values, minimum, maximum)[..., None, :] @ vh


def _gaussian_map(source_covariance: np.ndarray, target_covariance: np.ndarray) -> np.ndarray:
    root = _matrix_power(source_covariance, 0.5)
    inverse = _matrix_power(source_covariance, -0.5)
    return inverse @ _matrix_power(root @ target_covariance @ root, 0.5) @ inverse


def _masked_spatial_ridge(features, outputs, valid, weights, regularization, device):
    """Share each solve among all genes with the same observed-target mask."""
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights)).clip(0.1)
    x = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    coefficients = np.zeros((x.shape[1], outputs.shape[1], outputs.shape[2]), dtype=np.float32)
    patterns, inverse = np.unique(valid.T, axis=0, return_inverse=True)
    for pattern_id, mask in enumerate(patterns):
        genes = np.flatnonzero(inverse == pattern_id)
        if not mask.any():
            continue
        root = np.sqrt(weights[mask] / weights[mask].mean())[:, None]
        xt = torch.as_tensor(x[mask] * root, dtype=torch.float64, device=device)
        yt = torch.as_tensor(
            outputs[mask][:, genes].reshape(mask.sum(), -1) * root, dtype=torch.float64, device=device
        )
        if len(xt) < xt.shape[1]:
            fitted = xt.T @ torch.linalg.solve(
                xt @ xt.T + regularization * torch.eye(len(xt), device=device, dtype=xt.dtype), yt
            )
        else:
            fitted = torch.linalg.solve(
                xt.T @ xt + regularization * torch.eye(xt.shape[1], device=device, dtype=xt.dtype), xt.T @ yt
            )
        coefficients[:, genes] = fitted.cpu().numpy().reshape(x.shape[1], len(genes), outputs.shape[2])
    return (mean.astype(np.float32), scale.astype(np.float32), coefficients)


def _matrix_power(matrix: np.ndarray, exponent: float, floor: float = 0.0001) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(matrix)
    return vectors * np.maximum(eigenvalues, floor)[..., None, :] ** exponent @ np.swapaxes(vectors, -1, -2)


BASELINES = (
    "count_matched_resampling",
    "gaussian_affine_transport",
    "radial_quantile_transport",
    "scgpt_spatial",
    "spatialformer",
    "scgpt",
    "scfoundation",
    "state_se",
)
CELL_EVALUATION_SCHEMA = "sealed_unpaired_complete_cell_oracle_evaluation"
CELL_PREDICTION_SCHEMA = "complete_cell_oracle_hdf5"
CELL_PROTOCOL_ID = "scperturbench_scfoundation_cell_oracle"
CLASSICAL_BASELINES = BASELINES[:3]
FOUNDATION_BASELINES = BASELINES[3:]
SUPPORTED_BASELINES = BASELINES
SUPPORTED_FOUNDATION_BASELINES = FOUNDATION_BASELINES


def validate_cell_configuration(configuration):
    """Reject obsolete options instead of silently ignoring them."""
    from subcellpert.data.splits import FIT_FAMILY_POLICY, SCHEDULE_MANIFEST, VALIDATION_MANIFEST
    from subcellpert.evaluate_one_step import GENERATION_POLICY

    if configuration.get("training_protocol") != CELL_PROTOCOL_ID:
        raise ValueError("Use the current complete-cell Oracle configuration")
    if configuration.get("evaluation_mode") != "oracle_count":
        raise ValueError("Oracle configuration requires evaluation_mode: oracle_count")
    if configuration.get("initialization") != "independent_from_joint_checkpoints":
        raise ValueError("The cell Oracle requires independently fitted adapters")
    model = configuration.get("model", {})
    model_contract = {
        "architecture": "gene_fm_gears_shared_spatial_v2",
        "hidden_dimension": 512,
        "gene_graph": "training_source_coexpression",
        "condition_graph": "registered_descriptor_cosine",
        "spatial_head": "shared_count_conditioned_affine",
        "foundation_features": "frozen_contextual_gene_states_with_availability",
    }
    if any((model.get(key) != value for key, value in model_contract.items())):
        raise ValueError("Use the shared GEARS spatial architecture with contextual gene FM states")
    schedule = configuration.get("schedule", {})
    if (
        schedule.get("split_manifest") != SCHEDULE_MANIFEST
        or schedule.get("split_seeds") != [1, 2, 3]
        or schedule.get("model_repeats_per_split") != 1
        or (schedule.get("fits_per_dataset_baseline") != 3)
        or (schedule.get("seed_semantics") != "one_train_validation_test_partition_per_seed_and_fit_family")
        or (schedule.get("fit_family_policy") != FIT_FAMILY_POLICY)
    ):
        raise ValueError("Use three data-split seeds with one independent model per split, fit family and mode")
    lung = configuration.get("dataset_overrides", {}).get("lung-drug", {})
    if lung.get("fit_families") != ["drug"] or lung.get("fits_per_dataset_baseline") != 3:
        raise ValueError("Lung jointly fits AZD3965 and VB124 responses from Bleomycin for each split seed and mode")
    train = configuration["training"]
    foundation = configuration.get("foundation", {})
    from subcellpert.precision import validate_precision

    validate_precision(train.get("precision", "fp32"))
    if "contextual_feature_mode" in foundation:
        raise ValueError("This release supports the configured baseline encoders only")
    if any(key in train for key in ("batch_selection", "batch_candidates")):
        raise ValueError("This release requires a fixed training batch size")
    if train.get("matmul_precision", "highest") not in ("highest", "high"):
        raise ValueError("training.matmul_precision must be highest or high (TF32 on CUDA)")
    if type(foundation.get("compile_encoder", False)) is not bool:
        raise ValueError("foundation.compile_encoder must be a boolean")
    parallel = configuration.get("distributed_execution")
    expected_encoder_batch = train.get("batch_size")
    if parallel is not None:
        from subcellpert.distributed import local_batch_size

        size = parallel.get("world_size")
        if type(size) is not int or size < 2:
            raise ValueError("Distributed execution requires an integer world_size >= 2")
        expected_encoder_batch = local_batch_size(train.get("batch_size"), size)
        if (
            parallel.get("global_batch_size") != train.get("batch_size")
            or parallel.get("local_batch_size") != expected_encoder_batch
            or parallel.get("normalization") != "global_batchnorm"
            or parallel.get("loss_weighting") != "global_condition_groups"
        ):
            raise ValueError("Distributed execution must preserve the global batch and condition weighting")
    encoder_batch = foundation.get("encoder_batch_size", expected_encoder_batch)
    if (
        type(expected_encoder_batch) is not int
        or type(encoder_batch) is not int
        or not 1 <= encoder_batch <= expected_encoder_batch
    ):
        raise ValueError("foundation.encoder_batch_size must be positive and no larger than per-rank training.batch_size")
    if "encoder_batch_candidates" in foundation:
        raise ValueError("Use a fixed foundation.encoder_batch_size")
    prohibited = {"points_per_gene", "steps_per_epoch", "cells_per_group", "batch_groups_per_gpu"}
    if prohibited.intersection(train):
        raise ValueError("Population/RNA sampling options are invalid for cell training")
    fixed = {
        "epochs": 5,
        "accumulation_steps": 1,
        "optimizer": "Adam",
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "scheduler_gamma": 0.5,
        "gradient_clip_value": 1.0,
        "assignment_seed": 42,
        "objective": "gears_gene_supervision_with_local_RNA_spatial_extension",
        "direction_lambda": 0.1,
        "expression_scale": "raw_counts_log1p",
        "spatial_loss": "cell_gene_complete_RNA_moment_Fourier_L4",
        "include_initialization_in_validation": True,
    }
    if any((train.get(key) != value for key, value in fixed.items())):
        raise ValueError("Training settings differ from the frozen scFoundation sampling contract")
    if "spatial_projections" in train:
        raise ValueError("benchmark gene-wise training must not use fixed random projections")
    if train.get("development_access") != "train_only" or train.get("checkpoint_selection") != "best_validation":
        raise ValueError("Training must exclude validation and select checkpoints on validation only")
    validation = configuration.get("validation", {})
    if (
        validation.get("split_manifest") != VALIDATION_MANIFEST
        or validation.get("metric") != "cell_balanced_spatial_swd"
        or validation.get("gene_panel") != "all_gene"
        or (validation.get("seed") != 42)
        or (validation.get("gradient_updates") is not False)
        or (validation.get("test_used_for_selection") is not False)
        or (validation.get("tie_break") != "earliest_epoch")
        or (validation.get("cell_cap_per_response", 0) < 1)
        or (validation.get("spatial_projections", 0) < 1)
    ):
        raise ValueError("Independent Oracle validation configuration is missing or invalid")
    if type(train.get("batch_size")) is not int or train["batch_size"] < 1:
        raise ValueError("Cell batch size must be a positive integer")
    if configuration["evaluation"].get("rna_subsampling") is not False:
        raise ValueError("Current cell evaluation must consume every RNA in selected cells")
    if configuration.get("inference", {}).get("generation_policy") != GENERATION_POLICY:
        raise ValueError("Oracle and Joint require a common target-sized budget fixed before generation")
    if (
        configuration["evaluation"].get("cell_cap_per_group") != 2000
        or configuration["evaluation"].get("base_seed") != 42
    ):
        raise ValueError("Evaluation cell selection must retain the frozen 2000-cell/seed42 policy")
    return configuration


JOINT_EVALUATION_SCHEMA = "sealed_unpaired_complete_cell_joint_evaluation"
JOINT_PREDICTION_SCHEMA = "complete_cell_joint_hdf5"
JOINT_PROTOCOL_ID = "scperturbench_scfoundation_cell_joint"


def validate_joint_configuration(configuration):
    if configuration.get("training_protocol") != JOINT_PROTOCOL_ID or configuration.get("evaluation_mode") != "joint":
        raise ValueError("Use the current complete-cell Joint configuration")
    if configuration.get("initialization") != "independent_from_oracle_checkpoints":
        raise ValueError("Joint count and spatial heads must be fitted independently of Oracle checkpoints")
    oracle = dict(
        configuration,
        training_protocol="scperturbench_scfoundation_cell_oracle",
        evaluation_mode="oracle_count",
        initialization="independent_from_joint_checkpoints",
        validation={
            **configuration.get("validation", {}),
            "metric": "cell_balanced_spatial_swd",
            "gene_panel": "all_gene",
        },
    )
    validate_cell_configuration(oracle)
    if (
        configuration.get("validation", {}).get("metric") != "mse"
        or configuration["validation"].get("gene_panel") != "top_20_deg"
    ):
        raise ValueError("Joint checkpoints must be selected by independent validation top-20-DEG MSE")
    expected = {
        "factorization": "development_counts_condition_spatial_decoder; predicted_counts_at_inference",
        "count_loss": "gears_gene_L4_plus_direction",
        "count_loss_weight": 1.0,
        "spatial_loss_weight": 1.0,
        "max_log_count_shift": None,
        "count_discretization": "round_expm1_preserve_zeros",
        "output_cell_policy": "preselected_source_donors_for_target_sized_budget",
        "target_counts_at_inference": False,
        "target_cell_count_at_inference": True,
    }
    if configuration.get("joint") != expected:
        raise ValueError("Joint factorization, count objective, or source-only information contract changed")
    evaluation = configuration["evaluation"]
    if evaluation.get("schema") != JOINT_EVALUATION_SCHEMA or evaluation.get("counts") != "predicted_zero_inclusive":
        raise ValueError("Joint evaluation must score predicted counts including zero-expression cells")
    return configuration


def _check_binding(saved, binding, label):
    if saved.get("binding") != binding:
        raise ValueError(f"{label} belongs to different code, data, configuration, model or frozen features")


def cell_lineage(root, spec, mode="oracle_count"):
    if mode not in ("oracle_count", "joint"):
        raise ValueError("Unknown complete-cell evaluation mode")
    joint = mode == "joint"
    ("intervention_ood_" if spec.axis == "perturbation_ood" else "context_ood_") + ("joint" if joint else "oracle")
    return {
        **spec.lineage(),
        "training_protocol": JOINT_PROTOCOL_ID if joint else CELL_PROTOCOL_ID,
        "prediction_storage": JOINT_PREDICTION_SCHEMA if joint else CELL_PREDICTION_SCHEMA,
        "mode": mode,
        "joint_checkpoint_reused": False,
        **({"oracle_checkpoint_reused": False} if joint else {}),
    }


def gears_implementation_binding():
    """Bind checkpoints to every public runtime source file, including adapters."""
    directory = Path(__file__).resolve().parent
    return {path.relative_to(directory).as_posix(): sha256_file(path) for path in sorted(directory.rglob("*.py"))}


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write formatted JSON through a unique same-directory temporary file."""
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_rng(seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(seed_from_parts(seed, *parts))


def seed_from_parts(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    words = np.frombuffer(payload + b"\x00" * (-len(payload) % 4), dtype="<u4")
    return int(np.random.SeedSequence(words.tolist()).generate_state(1, dtype=np.uint64)[0])


def reset_run_output(output):
    """Discard generated run artifacts before an unconditional CLI rerun."""
    import shutil

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = (
        "run_config.json",
        "resolved_config.json",
        "batch_profile.json",
        "encoder_batch_profile.json",
        "cell_model.pt",
        "cell_resume.pt",
        "cell_best_validation.pt",
        "result.json",
        "initial_validation.json",
        "validation_selection.json",
        "validation_split.json",
        "training_history.json",
        "training_objective.json",
        "status.json",
        "events.jsonl",
        "failure.json",
        "feature_access_execution.json",
        "foundation_cache.json",
        "foundation_execution.json",
        "completion_audit.json",
    )
    paths = [output / name for name in files]
    paths.append(output / "responses")
    paths.extend(output.glob("cell_validation_*"))
    for name in files:
        paths.extend(output.glob(f".{name}.*.tmp"))
        if name.endswith(".pt"):
            paths.extend(output.glob(f"{name}.*.tmp"))
    for path in paths:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
