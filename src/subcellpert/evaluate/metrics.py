"""Complete-cell count and spatial evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import ttest_ind_from_stats, wasserstein_distance

from subcellpert.data.arrays import H5Array, PackedCoordinates
from subcellpert.data.batching import PopulationCellReader
from subcellpert.data.populations import (
    CellBalancedFullGenePointCloud,
    FullGenePopulation,
    subset_full_gene_population,
)
from subcellpert.data.sampling import deterministic_cell_sample
from subcellpert.models.shared import TensorCells
from subcellpert.utils import cell_distribution_features


def _full_gene_measure(populations, gene):
    """Equal supported units, equal expressing cells, every RNA within each cell."""
    clouds = [getattr(p, "streamed_point_cloud", None) or p.point_cloud for p in populations]
    clouds = [c for c in clouds if c.gene_entry_lengths[gene] > 0]
    points, weights = ([], [])
    for cloud in clouds:
        first = int(cloud.gene_entry_starts[gene])
        last = first + int(cloud.gene_entry_lengths[gene])
        lengths = cloud.entry_lengths[first:last]
        for entry, length in zip(range(first, last), lengths, strict=True):
            start = int(cloud.entry_starts[entry])
            points.append(np.asarray(cloud.coordinates[start : start + int(length)], dtype=np.float64))
            weights.append(np.full(int(length), 1 / (len(clouds) * len(lengths) * int(length))))
    return (np.concatenate(points), np.concatenate(weights)) if points else (np.empty((0, 2)), np.empty(0))


def _materialize_evaluation_cells(population):
    """Read selected RNA once, avoiding repeated HDF5 decompression per gene."""
    prediction = hasattr(population, "streamed_point_cloud")
    cloud = population.streamed_point_cloud if prediction else population.point_cloud
    from subcellpert.data.populations import LazyGeneCloud

    if isinstance(cloud, LazyGeneCloud):
        cloud = cloud.cloud
    packed = cloud.coordinates
    if isinstance(packed, np.ndarray):
        return population
    if isinstance(packed, PackedCoordinates) and isinstance(packed.source, H5Array):
        packed = PackedCoordinates(packed.source[:], packed.physical_starts, packed.lengths)
    coordinates = np.empty((len(packed), 2), dtype=np.float32)
    for first in range(0, len(coordinates), 262144):
        coordinates[first : first + 262144] = packed[first : first + 262144]
    materialized = replace(cloud, coordinates=coordinates)
    if prediction:
        return replace(population, coordinates=coordinates, streamed_point_cloud=materialized)
    return replace(population, point_cloud=materialized)


def _project_cells(populations, panels, projections):
    vocabulary = populations[0].genes
    directions = {}
    for panel in panels:
        generator = torch.Generator().manual_seed(42)
        directions[panel.name] = torch.nn.functional.normalize(
            torch.randn(len(panel.genes) * 30, projections, generator=generator), dim=0
        )
    outputs = {p.name: [] for p in panels}
    with torch.inference_mode():
        for population in populations:
            if hasattr(population, "streamed_point_cloud"):
                view = SimpleNamespace(
                    point_cloud=population.streamed_point_cloud,
                    counts=population.counts,
                    genes=population.genes,
                    cell_ids=population.cell_ids,
                )
            else:
                view = population
            reader = PopulationCellReader(view)
            for first in range(0, len(view.cell_ids), 32):
                batch = reader.read(np.arange(first, min(first + 32, len(view.cell_ids))))
                features = cell_distribution_features(TensorCells.from_batch(batch, "cpu")).reshape(
                    len(batch.cell_ids), len(vocabulary), 30
                )
                for panel in panels:
                    projected = features[:, panel.indices].flatten(1) @ directions[panel.name]
                    outputs[panel.name].append(projected.numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def _weighted_sliced_w2(predicted, target, pw, tw):
    values = []
    for k in range(predicted.shape[1]):
        pi, ti = (np.argsort(predicted[:, k]), np.argsort(target[:, k]))
        pc, tc = (np.cumsum(pw[pi]), np.cumsum(tw[ti]))
        pc[-1] = tc[-1] = 1.0
        cuts = np.unique(np.r_[0.0, pc, tc])
        mid = (cuts[:-1] + cuts[1:]) / 2
        px = predicted[pi[np.searchsorted(pc, mid).clip(max=len(pi) - 1)], k]
        tx = target[ti[np.searchsorted(tc, mid).clip(max=len(ti) - 1)], k]
        values.append(float(np.sum((px - tx) ** 2 * np.diff(cuts))))
    return float(np.mean(values))


def exact_spatial_metrics(predictions, targets, *, projections=64):
    """The existing spatial W1 estimand, using its exact weighted RNA measure."""
    if projections < 1 or not predictions or len(predictions) != len(targets):
        raise ValueError("Spatial evaluation needs aligned, nonempty biological units")
    genes = predictions[0].genes
    if any(p.genes != genes for p in [*predictions, *targets]):
        raise ValueError("Spatial evaluation must retain the complete shared vocabulary")
    angles = np.linspace(0, np.pi, projections, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    swd, centroid, radial = ([], [], [])
    for gene in range(len(genes)):
        px, pw = _full_gene_measure(predictions, gene)
        tx, tw = _full_gene_measure(targets, gene)
        if bool(len(px)) != bool(len(tx)):
            raise ValueError("Oracle output must cover every target-present gene")
        if not len(tx):
            swd.append(None)
            centroid.append(None)
            radial.append(None)
            continue
        pp, tp = (px @ directions.T, tx @ directions.T)
        swd.append(float(np.mean([wasserstein_distance(pp[:, k], tp[:, k], pw, tw) for k in range(projections)])))
        centroid.append(float(np.mean((np.average(px, axis=0, weights=pw) - np.average(tx, axis=0, weights=tw)) ** 2)))
        radial.append(float(wasserstein_distance(np.linalg.norm(px, axis=1), np.linalg.norm(tx, axis=1), pw, tw)))
    return {
        "per_gene_swd": swd,
        "per_gene_centroid_mse": centroid,
        "per_gene_radial_wasserstein": radial,
        "rna_subsampling": False,
        "weighting": "equal_supported_unit_then_equal_expressing_cell_then_equal_RNA",
    }


def score_cell_predictions(predictions, targets, panels, *, projections=64):
    exact = exact_spatial_metrics(predictions, targets, projections=projections)
    pp, tp = (_project_cells(populations, panels, projections) for populations in (predictions, targets))
    pw, tw = (unit_cell_weights([len(p.cell_ids) for p in populations]) for populations in (predictions, targets))
    rows = []
    for panel in panels:

        def average(key):
            values = [exact[key][int(i)] for i in panel.indices if exact[key][int(i)] is not None]
            return float(np.mean(values)) if values else None

        centroid = average("per_gene_centroid_mse")
        rows.append(
            {
                "gene_panel": panel.name,
                "gene_count": len(panel.genes),
                "cell_balanced_sliced_wasserstein": average("per_gene_swd"),
                "spatial_centroid_rmse": None if centroid is None else float(np.sqrt(centroid)),
                "radial_wasserstein": average("per_gene_radial_wasserstein"),
                "unpaired_cell_spatial_feature_sliced_wasserstein": _weighted_sliced_w2(
                    pp[panel.name], tp[panel.name], pw, tw
                ),
                "cell_feature_distance_power": 2,
                "rna_subsampling": False,
            }
        )
    return (rows, exact)


def _normalized_log_moments(populations: list[FullGenePopulation]) -> tuple[int, np.ndarray, np.ndarray]:
    genes = len(populations[0].genes)
    total, sums, squares = (0, np.zeros(genes), np.zeros(genes))
    for population in populations:
        for start in range(0, len(population.cell_ids), 4096):
            counts = population.counts[start : start + 4096].astype(np.float32)
            library = counts.sum(axis=1)
            counts /= np.where(library > 0, library / 10000, 1)[:, None]
            values = np.log1p(counts).astype(np.float64)
            total += len(values)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
    mean = sums / total
    variance = np.maximum((squares - total * mean**2) / max(total - 1, 1), 0)
    return (total, mean, variance)


def streaming_deg_panels(
    sources: list[FullGenePopulation], targets: list[FullGenePopulation], top_k: int = 100
) -> ScPertBenchDEGResult:
    """Scanpy-equivalent Welch t-test using every observed cell without concatenation."""
    if not sources or not targets:
        raise ValueError("DEG populations must contain source and target cells")
    genes = sources[0].genes
    if any(p.genes != genes for p in [*sources, *targets]):
        raise ValueError("DEG populations must share the complete gene vocabulary")
    ns, ms, vs = _normalized_log_moments(sources)
    nt, mt, vt = _normalized_log_moments(targets)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores, pvalues = ttest_ind_from_stats(mt, np.sqrt(vt), nt, ms, np.sqrt(vs), ns, equal_var=False)
    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max).astype(
        np.float32
    )
    pvalues = np.nan_to_num(pvalues, nan=1.0)
    p_order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[p_order] = np.minimum.accumulate((pvalues[p_order] * len(genes) / np.arange(1, len(genes) + 1))[::-1])[
        ::-1
    ].clip(0, 1)
    logfold = np.log2((np.expm1(mt) + 1e-09) / (np.expm1(ms) + 1e-09))
    order = np.lexsort((np.asarray(genes), -np.abs(scores)))
    selected = order[: min(top_k, len(genes))]
    selected_genes = tuple(genes[i] for i in selected)
    ranking = tuple(
        {
            "gene": genes[i],
            "score": float(scores[i]),
            "abs_score": float(abs(scores[i])),
            "pvals_adj": float(adjusted[i]),
            "logfoldchanges": float(logfold[i]),
        }
        for i in order
    )
    return ScPertBenchDEGResult(
        GenePanel("all_gene", genes, np.arange(len(genes))),
        GenePanel("top_100_deg", selected_genes, selected),
        ranking,
    )


def unit_cell_weights(lengths: list[int]) -> np.ndarray:
    if not lengths or min(lengths) <= 0:
        raise ValueError("each represented biological unit must contain cells")
    return np.concatenate([np.full(n, 1 / (len(lengths) * n)) for n in lengths])


@dataclass(frozen=True)
class GenePanel:
    name: str
    genes: tuple[str, ...]
    indices: np.ndarray


SC_PERT_BENCH_DEG_METHOD = "scanpy_rank_genes_groups_t_test_abs_score"


@dataclass(frozen=True)
class ScPertBenchDEGResult:
    all_gene: GenePanel
    top_100_deg: GenePanel
    ranking: tuple[dict[str, float | str], ...]
    method: str = SC_PERT_BENCH_DEG_METHOD

    @property
    def panels(self) -> tuple[GenePanel, GenePanel]:
        return (self.all_gene, self.top_100_deg)


JOINT_SPATIAL_METRIC_POLICY = "shared_nonempty_gene_w1_v1"


@dataclass(frozen=True)
class JointEvaluationPopulation:
    dataset: str
    condition: str
    context: str
    unit_id: str
    cell_ids: np.ndarray
    genes: tuple
    counts: object
    point_cloud: CellBalancedFullGenePointCloud


def _count_population(populations):
    values = np.concatenate([np.asarray(p.counts[:], dtype=np.float64) for p in populations])
    return (np.log1p(values), unit_cell_weights([len(p.cell_ids) for p in populations]))


def _materialize_selected(population):
    """Read just the selected cells' RNA once, without loading a whole output file."""
    cloud = population.point_cloud
    from subcellpert.data.populations import LazyGeneCloud

    if isinstance(cloud, LazyGeneCloud):
        cloud = cloud.cloud
    packed = cloud.coordinates
    coordinates = np.empty((len(packed), 2), dtype=np.float32)
    for first in range(0, len(packed), 262144):
        coordinates[first : first + 262144] = packed[first : first + 262144]
    return JointEvaluationPopulation(
        population.dataset,
        population.condition,
        population.context,
        population.unit_id,
        population.cell_ids,
        population.genes,
        population.counts,
        replace(cloud, coordinates=coordinates).validated(),
    )


def _pearson(x, y):
    x, y = (x - x.mean(), y - y.mean())
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denominator, -1, 1)) if denominator > 1e-12 else None


def _prediction_population(prediction, dataset, condition):
    meta = prediction.storage_manifest
    return JointEvaluationPopulation(
        dataset,
        condition,
        meta["context"],
        meta["source_unit_id"],
        prediction.cell_ids,
        prediction.genes,
        prediction.counts,
        prediction.streamed_point_cloud,
    )


def _subset_prediction(population, ids):
    lookup = {str(cell): i for i, cell in enumerate(population.cell_ids)}
    rows = np.asarray([lookup[str(cell)] for cell in ids])
    remap = np.full(len(population.cell_ids), -1, dtype=np.int64)
    remap[rows] = np.arange(len(rows))
    cloud = population.point_cloud
    keep = remap[cloud.entry_cell_indices] >= 0
    coordinates = cloud.coordinates.subset_entries(keep)
    gene_lengths = np.bincount(cloud.entry_gene_indices[keep], minlength=len(population.genes))
    selected = CellBalancedFullGenePointCloud(
        coordinates,
        cloud.entry_gene_indices[keep],
        remap[cloud.entry_cell_indices[keep]],
        coordinates.starts,
        cloud.entry_lengths[keep],
        np.cumsum(gene_lengths) - gene_lengths,
        gene_lengths,
    ).validated()
    return JointEvaluationPopulation(
        population.dataset,
        population.condition,
        population.context,
        population.unit_id,
        np.asarray(ids),
        population.genes,
        population.counts[rows],
        selected,
    )


def joint_count_metrics(predicted, targets, sources, panels, *, projections=64):
    if not predicted or not targets or (not sources) or (projections < 1):
        raise ValueError("Joint count evaluation requires source, prediction and target populations")
    genes = predicted[0].genes
    if any(p.genes != genes for p in [*predicted, *targets, *sources]):
        raise ValueError("Joint count evaluation must preserve the complete shared gene axis")
    px, pw = _count_population(predicted)
    tx, tw = _count_population(targets)
    sx, sw = _count_population(sources)
    rows = []
    for panel in panels:
        x, y, s = (px[:, panel.indices], tx[:, panel.indices], sx[:, panel.indices])
        pm, tm, sm = (np.average(v, axis=0, weights=w) for v, w in ((x, pw), (y, tw), (s, sw)))
        rows.append(
            {
                "gene_panel": panel.name,
                "gene_count": len(panel.genes),
                "mse": float(np.mean((pm - tm) ** 2)),
                "pcc_delta": _pearson(pm - sm, tm - sm),
            }
        )
    return rows


def joint_spatial_metrics(predictions, targets, panels, *, projections=64):
    """Mean gene-conditioned RNA W1, skipping genes empty on either side."""
    if projections < 1 or not predictions or (not targets):
        raise ValueError("Joint spatial evaluation requires nonempty population lists")
    genes = predictions[0].genes
    if any(p.genes != genes for p in [*predictions, *targets]):
        raise ValueError("Joint spatial evaluation must retain the shared gene vocabulary")
    angles = np.linspace(0, np.pi, projections, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    swd, radial, centroid = ([], [], [])
    for gene in range(len(genes)):
        px, pw = _full_gene_measure(predictions, gene)
        tx, tw = _full_gene_measure(targets, gene)
        if not len(tx) or not len(px):
            swd.append(None)
            radial.append(None)
            centroid.append(None)
            continue
        pp, tp = (px @ directions.T, tx @ directions.T)
        swd.append(float(np.mean([wasserstein_distance(pp[:, k], tp[:, k], pw, tw) for k in range(projections)])))
        radial.append(float(wasserstein_distance(np.linalg.norm(px, axis=1), np.linalg.norm(tx, axis=1), pw, tw)))
        centroid.append(float(np.mean((np.average(px, axis=0, weights=pw) - np.average(tx, axis=0, weights=tw)) ** 2)))
    rows = []
    for panel in panels:

        def average(values):
            selected = [values[i] for i in panel.indices if values[i] is not None]
            return float(np.mean(selected)) if selected else None

        centroid_mean = average(centroid)
        rows.append(
            {
                "cell_balanced_spatial_swd": average(swd),
                "radial_wasserstein": average(radial),
                "spatial_centroid_rmse": float(np.sqrt(centroid_mean)) if centroid_mean is not None else None,
            }
        )
    return (
        rows,
        {
            "policy": JOINT_SPATIAL_METRIC_POLICY,
            "per_gene_swd": swd,
            "per_gene_radial_wasserstein": radial,
            "per_gene_centroid_mse": centroid,
            "rna_subsampling": False,
            "weighting": "equal_supported_unit_then_equal_expressing_cell_then_equal_RNA",
        },
    )


def sample_joint_evaluation(populations, *, response_id, namespace, cap=2000):
    """Independent seed-42, unit-stratified source/prediction/target selections."""
    populations = sorted(populations, key=lambda p: p.unit_id)
    if not populations or cap < len(populations):
        raise ValueError("Joint evaluation sampling must retain every represented biological unit")
    sizes = np.asarray([len(p.cell_ids) for p in populations])
    allocations = np.zeros(len(sizes), dtype=np.int64)
    remaining = min(cap, int(sizes.sum()))
    while remaining:
        selected = np.flatnonzero(allocations < sizes)[:remaining]
        allocations[selected] += 1
        remaining -= len(selected)
    sampled, audit = ([], [])
    for population, count in zip(populations, allocations, strict=True):
        ids = deterministic_cell_sample(
            population.cell_ids,
            dataset=population.dataset,
            fold=response_id,
            transition=population.condition,
            group=f"{namespace}/{population.unit_id}",
            cap=int(count),
            base_seed=42,
        )
        subset = (
            _subset_prediction(population, ids)
            if isinstance(population, JointEvaluationPopulation)
            else subset_full_gene_population(population, ids)
        )
        sampled.append(subset)
        audit.append(
            {
                "unit_id": population.unit_id,
                "available_cells": len(population.cell_ids),
                "selected_cells": len(ids),
                "cell_ids": ids.tolist(),
            }
        )
    return (sampled, audit)


def score_joint_predictions(predictions, targets, sources, panels, *, projections=64):
    counts = joint_count_metrics(predictions, targets, sources, panels, projections=projections)
    spatial, exact = joint_spatial_metrics(predictions, targets, panels, projections=projections)
    return ([dict(c, **s) for c, s in zip(counts, spatial, strict=True)], exact)
