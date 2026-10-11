"""Shared baseline contracts, analytic state, and trainable GEARS response models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SGConv

from subcellpert.precision import neural_autocast, validate_precision
from subcellpert.utils import QUANTILES, RNA_COMPUTE_CHUNK, integer_counts, positive_neighbor_graph


if TYPE_CHECKING:
    from subcellpert.data.batching import CellRNABatch
    from subcellpert.data.populations import FullGenePopulation
    from subcellpert.data.sampling import OracleCounts


GEARS_HIDDEN_SIZE = 512


@dataclass(frozen=True)
class ModelInput:
    """Observed sources and a requested condition; never target coordinates.

    Oracle receives exact coordinate-free counts. Joint receives preselected
    source donor IDs and the frozen target population-size budget.
    """

    sources: tuple[FullGenePopulation, ...]
    descriptor: np.ndarray
    oracle_counts: OracleCounts | None = None
    source_cell_ids: tuple[str, ...] | None = None
    generation_budget: int | None = None


@dataclass(frozen=True)
class ModelOutput:
    """A sealed HDF5 prediction with cell IDs, full-gene counts and RNA geometry."""

    path: Path
    prediction: object


class Baseline:
    """Shared API implemented by every named baseline module."""

    name: str
    foundation: bool = False

    def __init__(self, setting):
        if setting not in {"joint", "oracle", "oracle_count"}:
            raise ValueError(f"Unknown setting: {setting}")
        self.setting = "oracle_count" if setting == "oracle" else setting

    @property
    def model_class(self):
        return self.joint_class if self.setting == "joint" else self.oracle_class

    def build(self, genes, descriptor_dimension, *, foundation_dimension=None, state=None, **options):
        """Construct an adapter or restore an analytically fitted transport."""
        if self.foundation:
            if foundation_dimension is None:
                raise ValueError(f"{self.name} requires source gene foundation features")
            return self.model_class(len(genes), descriptor_dimension, foundation_dimension, **options)
        if state is None:
            raise ValueError("Classical models must be fitted or restored from their analytic state")
        return self.model_class(self.name, tuple(genes), descriptor_dimension, state)

    def fit_analytic(self, records, **options):
        if self.foundation:
            raise ValueError("Foundation adapters are optimized with train_one_step")
        return self.model_class.fit(self.name, records, **options)

    def predict(
        self,
        model,
        inputs: ModelInput,
        output,
        *,
        seed,
        device="cpu",
        foundation_store=None,
        provenance=None,
        batch_size=256,
    ) -> ModelOutput:
        """Generate using the same input and output objects for every baseline."""
        from subcellpert.data.sampling import OracleCounts
        from subcellpert.evaluate_one_step import write_cell_prediction, write_joint_prediction

        if not isinstance(inputs, ModelInput) or not inputs.sources:
            raise ValueError("Prediction requires ModelInput with nonempty observed sources")
        if self.foundation != (foundation_store is not None):
            raise ValueError("Foundation feature access does not match the selected baseline")
        options = {
            "seed": seed,
            "device": device,
            "foundation_store": foundation_store,
            "provenance": provenance,
            "batch_size": batch_size,
        }
        output = Path(output)
        if self.setting == "joint":
            if inputs.oracle_counts is not None or len(inputs.sources) != 1:
                raise ValueError("Joint prediction consumes one source unit and no target counts")
            if inputs.source_cell_ids is None or inputs.generation_budget is None:
                raise ValueError("Joint donors and population-size budget must be frozen before generation")
            prediction = write_joint_prediction(
                model,
                inputs.sources[0],
                inputs.descriptor,
                output,
                source_cell_ids=inputs.source_cell_ids,
                generation_budget=inputs.generation_budget,
                **options,
            )
        else:
            if not isinstance(inputs.oracle_counts, OracleCounts):
                raise TypeError("Oracle prediction requires coordinate-free OracleCounts")
            if inputs.source_cell_ids is not None or inputs.generation_budget is not None:
                raise ValueError("Oracle donor assignment and output budget follow OracleCounts")
            prediction = write_cell_prediction(
                model, inputs.oracle_counts, inputs.sources, inputs.descriptor, output, **options
            )
        return ModelOutput(output, prediction)


class ClassicalModel:
    """Marker for analytic transports with count and geometry prediction methods."""


class _Moments:
    def __init__(self, genes):
        self.weight = np.zeros(genes)
        self.mean = np.zeros((genes, 2))
        self.second = np.zeros((genes, 2, 2))
        self.radial = np.zeros((genes, len(QUANTILES)))

    def add(self, statistics):
        mean, second, radial, valid = statistics
        self.weight += valid.sum(0)
        self.mean += (mean * valid[..., None]).sum(0)
        self.second += (second * valid[..., None, None]).sum(0)
        self.radial += (radial * valid[..., None]).sum(0)

    def result(self):
        weight = self.weight.clip(1)
        mean = self.mean / weight[:, None]
        covariance = self.second / weight[:, None, None] - mean[..., None] * mean[:, None, :]
        return (mean, covariance, self.radial / weight[:, None], self.weight > 0)


@dataclass(frozen=True)
class TensorCells:
    counts: torch.Tensor
    xy: torch.Tensor
    genes: torch.Tensor
    cells: torch.Tensor

    @classmethod
    def from_batch(cls, batch: CellRNABatch, device):
        batch.validated()
        return cls(
            torch.as_tensor(batch.counts, dtype=torch.float32, device=device),
            torch.as_tensor(batch.coordinates, dtype=torch.float32, device=device),
            torch.as_tensor(batch.molecule_genes, dtype=torch.long, device=device),
            torch.as_tensor(batch.molecule_cells, dtype=torch.long, device=device),
        )

    def select_cells(self, rows):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.counts.device)
        remap = torch.full((len(self.counts),), -1, dtype=torch.long, device=self.counts.device)
        remap[rows] = torch.arange(len(rows), device=rows.device)
        labels = remap[self.cells]
        keep = labels >= 0
        return TensorCells(self.counts[rows], self.xy[keep], self.genes[keep], labels[keep])


class _BatchNorm(nn.BatchNorm1d):
    """GEARS BatchNorm with defined training behavior for singleton cells."""

    def forward(self, values):
        if self.training and getattr(self, "synchronize_distributed", False):
            from subcellpert.distributed import synchronized_batch_norm, world_size

            if world_size() > 1:
                return synchronized_batch_norm(self, values)
        if self.training and len(values) == 1:
            return F.batch_norm(
                values, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps
            )
        return super().forward(values)


def gears_mlp(widths):
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.extend((nn.Linear(left, right), _BatchNorm(right)))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def descriptor_graph(vectors, neighbors=20):
    values = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(values, norm, out=np.zeros_like(values), where=norm > 0)
    return positive_neighbor_graph(normalized @ normalized.T, neighbors)


class GearsResponse(nn.Module):
    """Gene/condition graph propagation, recovery MLP, and GEARS readouts."""

    def __init__(self, n_genes, descriptor_dimension, foundation_dimension, hidden_size, condition_count):
        super().__init__()
        h = hidden_size
        self.hidden_size = h
        self.gene_identity = nn.Embedding(n_genes, h)
        self.gene_position = nn.Embedding(n_genes, h)
        self.foundation_projection = nn.Sequential(
            nn.LayerNorm(foundation_dimension - 1), nn.Linear(foundation_dimension - 1, h)
        )
        self.availability = nn.Linear(1, h, bias=False)
        self.source_projection = nn.Linear(17, h)
        self.gene_graph_layers = nn.ModuleList([SGConv(h, h, K=1)])
        self.condition_identity = nn.Embedding(condition_count, h)
        self.condition_projection = nn.Linear(descriptor_dimension, h, bias=False)
        self.condition_graph_layers = nn.ModuleList([SGConv(h, h, K=1) for _ in range(1)])
        self.base_norm = _BatchNorm(h)
        self.base_transform = gears_mlp([h, h, h])
        self.condition_fusion = gears_mlp([h, h, h])
        self.response_norm = _BatchNorm(h)
        self.recovery = gears_mlp([h, 2 * h, h])
        self.gene_readout_weight = nn.Parameter(torch.empty(n_genes, h))
        self.gene_readout_bias = nn.Parameter(torch.zeros(n_genes))
        self.cross_gene = gears_mlp([n_genes, h, h])
        nn.init.xavier_normal_(self.gene_readout_weight)

    @staticmethod
    def propagate(values, adjacency, layers):
        edges = adjacency.nonzero().T
        weights = adjacency[edges[0], edges[1]]
        for index, layer in enumerate(layers):
            values = layer(values, edges, weights)
            if index < len(layers) - 1:
                values = values.relu()
        return values

    def forward(
        self,
        source_features,
        source_log_counts,
        foundation,
        descriptor,
        condition_vectors,
        gene_graph,
        condition_graph,
    ):
        b, g = source_log_counts.shape
        available = foundation[..., -1:]
        fm = self.foundation_projection(foundation[..., :-1]) * available
        local = self.gene_identity.weight[None] + fm + self.availability(available)
        local = local + self.source_projection(torch.cat((source_features, source_log_counts[..., None]), -1))
        position = self.propagate(self.gene_position.weight, gene_graph, self.gene_graph_layers)
        base = self.base_norm(local.flatten(0, 1)).relu() + 0.2 * position.repeat(b, 1)
        base = self.base_transform(base).reshape(b, g, -1)
        distances = (descriptor[:, None] - condition_vectors[None]).abs().amax(-1)
        nearest = distances.argmin(1)
        if torch.any(distances.gather(1, nearest[:, None]) > 1e-05):
            raise ValueError("Unregistered perturbation descriptor passed to GEARS")
        condition = self.condition_identity.weight + self.condition_projection(condition_vectors)
        condition = self.propagate(condition, condition_graph, self.condition_graph_layers)
        base = base + self.condition_fusion(condition[nearest])[:, None]
        recovered = self.recovery(self.response_norm(base.flatten(0, 1)).relu()).reshape(b, g, -1)
        gene_response = (recovered * self.gene_readout_weight).sum(-1) + self.gene_readout_bias
        cross = self.cross_gene(gene_response)
        return (recovered, gene_response, cross)


class GearsCellOracleModel(nn.Module):
    """Gene-level frozen FM -> GEARS -> count-conditioned spatial parameters."""

    def __init__(
        self,
        n_genes,
        descriptor_dimension,
        foundation_dimension,
        *,
        condition_vectors,
        gene_graph=None,
        hidden_dimension=GEARS_HIDDEN_SIZE,
        maximum_displacement=0.5,
        compute_precision="fp32",
    ):
        super().__init__()
        if foundation_dimension is None or foundation_dimension < 2:
            raise ValueError("GEARS requires per-gene FM features and an availability channel")
        vectors = torch.as_tensor(np.asarray(condition_vectors), dtype=torch.float32).clone()
        if (
            vectors.ndim != 2
            or vectors.shape[1] != descriptor_dimension
            or (not len(vectors))
            or (not torch.isfinite(vectors).all())
        ):
            raise ValueError("GEARS needs the registered finite condition descriptor vocabulary")
        self.n_genes, self.descriptor_dimension, self.foundation_dimension = (
            n_genes,
            descriptor_dimension,
            foundation_dimension,
        )
        self.maximum_displacement = maximum_displacement
        self.compute_precision = validate_precision(compute_precision)
        self.hidden_dimension = hidden_dimension
        self.register_buffer("condition_vectors", vectors)
        graph = np.zeros((n_genes, n_genes), dtype=np.float32) if gene_graph is None else np.asarray(gene_graph)
        if graph.shape != (n_genes, n_genes) or not np.isfinite(graph).all() or np.any(graph < 0):
            raise ValueError("Invalid source-only gene coexpression graph")
        self.register_buffer("gene_graph", torch.as_tensor(graph, dtype=torch.float32).clone())
        self.register_buffer("condition_graph", torch.from_numpy(descriptor_graph(vectors.numpy())))
        self.point_encoder = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 16))
        self.response = GearsResponse(
            n_genes, descriptor_dimension, foundation_dimension, hidden_dimension, len(vectors)
        )
        h = hidden_dimension
        self.spatial_head = nn.Sequential(nn.Linear(2 * h + 2, h), nn.ReLU(), nn.Linear(h, 6))
        nn.init.zeros_(self.spatial_head[-1].weight)
        nn.init.zeros_(self.spatial_head[-1].bias)

    def encode_source(self, source: TensorCells):
        b, g = source.counts.shape
        if g != self.n_genes or torch.any(source.counts < 0) or torch.any(source.counts.sum(1) <= 0):
            raise ValueError("GEARS source cells must retain the full gene axis and nonempty RNA")
        sums = source.xy.new_zeros((b * g, 16))
        for first in range(0, len(source.xy), RNA_COMPUTE_CHUNK):
            sl = slice(first, first + RNA_COMPUTE_CHUNK)
            with neural_autocast(self.compute_precision, source.xy.device):
                features = self.point_encoder(source.xy[sl])
            sums = sums.index_add(0, source.cells[sl] * g + source.genes[sl], features.float())
        return sums.reshape(b, g, 16) / source.counts.clamp_min(1)[..., None]

    def response_state(self, source, encoded, descriptor, donors, foundation):
        b = len(donors)
        if descriptor.shape != (b, self.descriptor_dimension) or donors.shape != (b,) or donors.dtype != torch.long:
            raise ValueError("Descriptors and donors must follow the output cell axis")
        if torch.any(donors < 0) or torch.any(donors >= len(source.counts)):
            raise ValueError("Source donor outside the observed population")
        if foundation is None or foundation.shape != (len(source.counts), self.n_genes, self.foundation_dimension):
            raise ValueError("GEARS needs aligned per-cell PER-GENE FM states, not pooled cell embeddings")
        if not torch.isfinite(foundation).all() or torch.any((foundation[..., -1] != 0) & (foundation[..., -1] != 1)):
            raise ValueError("Invalid gene FM features or availability mask")
        with neural_autocast(self.compute_precision, source.counts.device):
            return self.response(
                encoded[donors],
                torch.log1p(source.counts[donors]),
                foundation[donors],
                descriptor,
                self.condition_vectors,
                self.gene_graph,
                self.condition_graph,
            )

    def spatial_parameters(self, state, counts):
        recovered, gene_response, cross = state
        if counts.shape != gene_response.shape or not torch.isfinite(counts).all() or torch.any(counts < 0):
            raise ValueError("Spatial conditioning must preserve nonnegative complete-cell counts")
        with neural_autocast(self.compute_precision, counts.device):
            raw = self.spatial_head(
                torch.cat(
                    (
                        recovered,
                        cross[:, None].expand(-1, self.n_genes, -1),
                        gene_response[..., None],
                        torch.log1p(counts)[..., None],
                    ),
                    -1,
                )
            )
        # Keep near-identity transforms and the coordinate loss in FP32.
        raw = raw.float()
        identity = torch.eye(2, device=raw.device, dtype=raw.dtype).reshape(1, 1, 4)
        return torch.cat((identity + 0.1 * raw[..., :4].tanh(), self.maximum_displacement * raw[..., 4:].tanh()), -1)

    def forward(self, source, conditional_counts, descriptor, donor_rows, foundation=None):
        state = self.response_state(source, self.encode_source(source), descriptor, donor_rows, foundation)
        return self.spatial_parameters(state, conditional_counts)


class GearsCellJointModel(GearsCellOracleModel):
    """GEARS gene-specific count readout; target counts only enter spatial head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_weight = nn.Parameter(torch.zeros(self.n_genes, self.hidden_dimension + 1))
        self.count_bias = nn.Parameter(torch.zeros(self.n_genes))

    def count_from_state(self, source, donors, state):
        _, gene_response, cross = state
        features = torch.cat((gene_response[..., None], cross[:, None].expand(-1, self.n_genes, -1)), -1)
        delta = (features * self.count_weight).sum(-1) + self.count_bias
        return (torch.log1p(source.counts[donors]) + delta).clamp_min(0)

    def count_log1p(self, source, descriptor, donors, foundation=None):
        state = self.response_state(source, self.encode_source(source), descriptor, donors, foundation)
        return self.count_from_state(source, donors, state)

    def joint_inference_parameters(self, source, descriptor, donors, foundation=None):
        """One shared response state for predicted counts and spatial decoding."""
        state = self.response_state(source, self.encode_source(source), descriptor, donors, foundation)
        counts = integer_counts(self.count_from_state(source, donors, state).cpu().numpy())
        params = self.spatial_parameters(state, torch.as_tensor(counts, device=source.counts.device))
        return (counts, params)

    def forward(self, source, conditional_counts, descriptor, donor_rows, foundation=None, *, joint_training=False):
        state = self.response_state(source, self.encode_source(source), descriptor, donor_rows, foundation)
        spatial = self.spatial_parameters(state, conditional_counts)
        if joint_training:
            return (spatial, self.count_from_state(source, donor_rows, state))
        return spatial
