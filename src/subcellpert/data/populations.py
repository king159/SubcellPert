"""Full-gene populations, point-cloud views, and population selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from subcellpert.data.arrays import GeneBackedCounts, PackedCoordinates, SparseCounts
from subcellpert.data.storage import (
    CELL_MAJOR_FORMAT,
    CellMajorCounts,
    CellMajorStore,
    csr_entries,
    load_inference_cell_index_manifest,
    load_population_index_manifest,
)
from subcellpert.utils import (
    deterministic_rng,
)


class LazyGeneCloud:
    """Construct a gene-oriented evaluator view only if a consumer needs it."""

    def __init__(self, store, rows):
        self.store, self.rows = (store, np.asarray(rows, dtype=np.int64))

    @cached_property
    def cloud(self):
        entries, lengths, local = csr_entries(self.store.indptr, self.rows)
        genes = np.asarray(self.store.gene_ids[entries])
        counts = np.asarray(self.store.values[entries])
        cells = np.repeat(np.arange(len(self.rows), dtype=np.int64), lengths)
        local_rna_starts = np.r_[0, np.cumsum(counts, dtype=np.int64)[:-1]]
        first_rna = local_rna_starts[local[:-1]]
        starts = self.store.rna_indptr[self.rows][cells].astype(np.int64) + local_rna_starts - first_rna[cells]
        order = np.argsort(genes, kind="stable")
        gene_lengths = np.bincount(genes, minlength=self.store.n_genes)
        coordinates = PackedCoordinates(self.store.coordinates, starts[order], counts[order])
        return CellBalancedFullGenePointCloud(
            coordinates,
            genes[order],
            cells[order],
            coordinates.starts,
            counts[order],
            np.cumsum(gene_lengths) - gene_lengths,
            gene_lengths,
        ).validated()

    def __getattr__(self, key):
        return getattr(self.cloud, key)

    def validated(self):
        return self.cloud.validated()


def iter_cell_major_populations(
    index_dir, manifest, *, conditions=None, contexts=None, units=None, cell_cap=None, cell_seed=42
):
    from subcellpert.utils import deterministic_rng

    genes = tuple(sorted(manifest["inventory"]["genes"]))
    for shard in sorted(manifest["shards"], key=lambda s: int(s["unit_index"])):
        if units is not None and str(shard["unit_id"]) not in set(units):
            continue
        cells = pl.scan_parquet(Path(index_dir) / shard["cells"])
        if conditions is not None:
            cells = cells.filter(pl.col("condition").is_in(list(conditions)))
        if contexts is not None:
            cells = cells.filter(pl.col("context").is_in(list(contexts)))
        frame = cells.collect()
        if frame.is_empty():
            continue
        store = CellMajorStore(index_dir, shard, len(genes))
        groups = frame.partition_by(["context", "condition", "unit_id"], as_dict=True)
        for key in sorted(groups):
            context, condition, unit = (str(v) for v in key)
            selected = groups[key].sort("cell_id")
            if cell_cap is not None and selected.height > cell_cap:
                if cell_cap < 1:
                    raise ValueError("Cell cap must be positive")
                rows = np.sort(
                    deterministic_rng(cell_seed, manifest["dataset"], context, condition, unit).choice(
                        selected.height, cell_cap, replace=False
                    )
                )
                selected = selected[rows.tolist()]
            rows = selected["cell_key"].to_numpy().astype(np.int64)
            yield FullGenePopulation(
                manifest["dataset"],
                context,
                condition,
                unit,
                genes,
                selected["cell_id"].to_numpy(),
                store.counts(rows),
                LazyGeneCloud(store, rows),
                selected,
                store,
                rows,
            ).validated()


@dataclass(frozen=True)
class CellBalancedFullGenePointCloud:
    """Packed coordinates with cell-balanced sampling for every gene at once."""

    coordinates: np.ndarray
    entry_gene_indices: np.ndarray
    entry_cell_indices: np.ndarray
    entry_starts: np.ndarray
    entry_lengths: np.ndarray
    gene_entry_starts: np.ndarray
    gene_entry_lengths: np.ndarray

    def validated(self) -> CellBalancedFullGenePointCloud:
        coordinates = (
            self.coordinates
            if isinstance(self.coordinates, PackedCoordinates)
            else np.asarray(self.coordinates, dtype=np.float32)
        )
        entry_gene_indices = np.asarray(self.entry_gene_indices)
        entry_cell_indices = np.asarray(self.entry_cell_indices)
        entry_starts = np.asarray(self.entry_starts)
        entry_lengths = np.asarray(self.entry_lengths)
        gene_entry_starts = np.asarray(self.gene_entry_starts, dtype=np.int64)
        gene_entry_lengths = np.asarray(self.gene_entry_lengths, dtype=np.int64)
        if any(
            v.dtype.kind not in "iu" for v in (entry_gene_indices, entry_cell_indices, entry_starts, entry_lengths)
        ):
            raise ValueError("point entry indices and lengths must be integers")
        entry_count = len(entry_starts)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1:] != (2,)
            or (not isinstance(coordinates, PackedCoordinates) and (not np.isfinite(coordinates).all()))
        ):
            raise ValueError("full-gene coordinates must have finite shape [molecules, 2]")
        if any(value.shape != (entry_count,) for value in (entry_gene_indices, entry_cell_indices, entry_lengths)):
            raise ValueError("full-gene point entries are not aligned")
        if gene_entry_starts.ndim != 1 or gene_entry_lengths.shape != gene_entry_starts.shape:
            raise ValueError("gene entry runs are not aligned")
        if entry_count:
            if np.any(entry_lengths <= 0) or np.any(entry_starts < 0):
                raise ValueError("point entry offsets and lengths must be positive")
            expected_starts = (
                coordinates.starts
                if isinstance(coordinates, PackedCoordinates)
                else np.concatenate(([0], np.cumsum(entry_lengths[:-1])))
            )
            if not np.array_equal(entry_starts, expected_starts):
                raise ValueError("point entries must be contiguous")
            if int(entry_starts[-1] + entry_lengths[-1]) != len(coordinates):
                raise ValueError("point entries do not cover the packed coordinates")
            if np.any(entry_gene_indices < 0) or np.any(entry_gene_indices >= len(gene_entry_starts)):
                raise ValueError("point entries reference an unknown gene")
            if np.any(entry_cell_indices < 0):
                raise ValueError("point entries reference a negative cell")
        elif len(coordinates):
            raise ValueError("coordinates exist without nonempty cell-gene entries")
        if np.any(gene_entry_lengths < 0) or np.any(gene_entry_starts < 0):
            raise ValueError("gene entry runs must be non-negative")
        if len(gene_entry_starts):
            expected_starts = np.concatenate(([0], np.cumsum(gene_entry_lengths[:-1])))
            if not np.array_equal(gene_entry_starts, expected_starts):
                raise ValueError("gene entry runs must be contiguous")
            if int(gene_entry_starts[-1] + gene_entry_lengths[-1]) != entry_count:
                raise ValueError("gene entry runs do not cover the point entries")
            for gene, (start, length) in enumerate(zip(gene_entry_starts, gene_entry_lengths, strict=True)):
                if length and np.any(entry_gene_indices[start : start + length] != gene):
                    raise ValueError("point entries are not sorted by the frozen gene axis")
        return CellBalancedFullGenePointCloud(
            coordinates=coordinates,
            entry_gene_indices=entry_gene_indices,
            entry_cell_indices=entry_cell_indices,
            entry_starts=entry_starts,
            entry_lengths=entry_lengths,
            gene_entry_starts=gene_entry_starts,
            gene_entry_lengths=gene_entry_lengths,
        )

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_entry_starts))

    def sample_all_genes(self, points_per_gene: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample cells uniformly, then one molecule, for all genes vectorially."""
        if points_per_gene <= 0:
            raise ValueError("points_per_gene must be positive")
        output = np.zeros((self.n_genes, points_per_gene, 2), dtype=np.float32)
        defined = self.gene_entry_lengths > 0
        if not np.any(defined):
            return (output, np.zeros((self.n_genes, points_per_gene), dtype=bool))
        safe_entry_lengths = np.maximum(self.gene_entry_lengths, 1)
        entry_offsets = np.floor(rng.random((self.n_genes, points_per_gene)) * safe_entry_lengths[:, None]).astype(
            np.int64
        )
        selected_entries = self.gene_entry_starts[:, None] + entry_offsets
        selected_entries[~defined] = 0
        selected_lengths = self.entry_lengths[selected_entries]
        point_offsets = np.floor(rng.random((self.n_genes, points_per_gene)) * selected_lengths).astype(np.int64)
        selected_points = self.entry_starts[selected_entries] + point_offsets
        output[defined] = self.coordinates[selected_points[defined]]
        mask = np.broadcast_to(defined[:, None], (self.n_genes, points_per_gene)).copy()
        return (output, mask)

    @cached_property
    def cell_sampling_index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Index the all-RNA source prior by expressing cell, then molecule."""
        order = np.argsort(self.entry_cell_indices, kind="stable")
        cells, first = np.unique(self.entry_cell_indices[order], return_index=True)
        lengths = self.entry_lengths[order]
        cumulative = np.cumsum(lengths, dtype=np.int64)
        starts = np.concatenate(([0], cumulative[:-1]))[first] if len(first) else np.empty(0, dtype=np.int64)
        totals = np.add.reduceat(lengths, first) if len(first) else np.empty(0, dtype=np.int64)
        return (order, cumulative, starts, totals)

    def sample_source_prior(self, count: int, rng: np.random.Generator) -> np.ndarray:
        """Sample source cells equally, then their RNA, without target information."""
        if not len(self.coordinates):
            raise ValueError("an all-zero source population has no RNA geometry prior")
        order, cumulative, starts, totals = self.cell_sampling_index
        cell = rng.integers(0, len(totals), size=count)
        offsets = starts[cell] + np.floor(rng.random(count) * totals[cell]).astype(np.int64)
        positions = np.searchsorted(cumulative, offsets, side="right")
        entries = order[positions]
        previous = np.where(positions > 0, cumulative[np.maximum(positions - 1, 0)], 0)
        return self.coordinates[self.entry_starts[entries] + offsets - previous]

    def sample_source_all_genes(self, points_per_gene: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Complete source geometry with an explicit all-RNA prior for absent genes.

        The prior is still cell-balanced and uses observed source RNA only. The
        summary's availability flag remains zero for genes needing this prior.
        """
        points, mask = self.sample_all_genes(points_per_gene, rng)
        absent = self.gene_entry_lengths == 0
        if absent.any() and len(self.coordinates):
            points[absent] = self.sample_source_prior(int(absent.sum()) * points_per_gene, rng).reshape(
                -1, points_per_gene, 2
            )
            mask[absent] = True
        return (points, mask)

    @cached_property
    def cell_balanced_moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact equal-expressing-cell moments, shared by training and inference.

        Each nonempty cell/gene pattern has unit mass regardless of RNA count.
        Missing genes use the observed source all-RNA cell-balanced prior; their
        availability flag stays false. Coordinate blocks bound temporary memory.
        """
        entries = len(self.entry_starts)
        gene_mean = np.zeros((self.n_genes, 2), dtype=np.float64)
        gene_second = np.zeros((self.n_genes, 3), dtype=np.float64)
        defined = self.gene_entry_lengths > 0
        use_prior = entries and (not defined.all())
        n_cells = int(self.entry_cell_indices.max()) + 1 if use_prior else 0
        cell_sums = np.zeros((n_cells, 5), dtype=np.float64)
        cell_counts = np.zeros(n_cells, dtype=np.int64)
        entry_ends = self.entry_starts + self.entry_lengths
        for start in range(0, len(self.coordinates), 2000000):
            stop = min(start + 2000000, len(self.coordinates))
            first, last = np.searchsorted(entry_ends, [start, stop - 1], side="right")
            last += 1
            values = self.coordinates[start:stop].astype(np.float64)
            offsets = np.r_[0, self.entry_starts[first + 1 : last] - start]
            sums = np.add.reduceat(values, offsets, axis=0)
            quadratic = np.column_stack((values[:, 0] ** 2, values[:, 0] * values[:, 1], values[:, 1] ** 2))
            products = np.add.reduceat(quadratic, offsets, axis=0)
            lengths = self.entry_lengths[first:last]
            genes = self.entry_gene_indices[first:last]
            for axis in range(2):
                gene_mean[:, axis] += np.bincount(genes, weights=sums[:, axis] / lengths, minlength=self.n_genes)
            for axis in range(3):
                gene_second[:, axis] += np.bincount(genes, weights=products[:, axis] / lengths, minlength=self.n_genes)
            if use_prior:
                cells = self.entry_cell_indices[first:last]
                for axis in range(5):
                    weights = sums[:, axis] if axis < 2 else products[:, axis - 2]
                    cell_sums[:, axis] += np.bincount(cells, weights=weights, minlength=n_cells)
                np.add.at(cell_counts, cells, np.diff(np.r_[offsets, stop - start]))
        denominator = np.maximum(self.gene_entry_lengths, 1)[:, None]
        gene_mean /= denominator
        gene_second /= denominator
        if use_prior:
            occupied = cell_counts > 0
            prior = (cell_sums[occupied] / cell_counts[occupied, None]).mean(axis=0)
            gene_mean[~defined] = prior[:2]
            gene_second[~defined] = prior[2:]
        covariance = np.zeros((self.n_genes, 2, 2), dtype=np.float64)
        covariance[:, 0, 0] = np.maximum(gene_second[:, 0] - gene_mean[:, 0] ** 2, 0)
        covariance[:, 1, 1] = np.maximum(gene_second[:, 2] - gene_mean[:, 1] ** 2, 0)
        covariance[:, 0, 1] = covariance[:, 1, 0] = gene_second[:, 1] - gene_mean[:, 0] * gene_mean[:, 1]
        return (gene_mean.astype(np.float32), covariance.astype(np.float32), defined)

    @cached_property
    def spatial_summary(self) -> np.ndarray:
        mean, covariance, defined = self.cell_balanced_moments
        scale = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-08))
        return np.column_stack((mean, scale, defined)).astype(np.float32)


FULL_GENE_POPULATION_LOADER = "cell_major_full_gene_v4"


@dataclass(frozen=True)
class FullGenePopulation:
    dataset: str
    context: str
    condition: str
    unit_id: str
    genes: tuple[str, ...]
    cell_ids: np.ndarray
    counts: np.ndarray
    point_cloud: CellBalancedFullGenePointCloud
    cell_metadata: pl.DataFrame | None = None
    cell_store: Any | None = None
    cell_rows: np.ndarray | None = None

    def validated(self) -> FullGenePopulation:
        cells = np.asarray(self.cell_ids, dtype=str)
        if self.cell_store is not None:
            if not isinstance(self.counts, CellMajorCounts) or self.cell_rows is None:
                raise ValueError("Cell-major populations require their CSR count and cell-row views")
            if (
                self.counts.shape != (len(cells), len(self.genes))
                or not len(cells)
                or len(np.unique(cells)) != len(cells)
            ):
                raise ValueError("Cell-major population cell/gene axes disagree")
            if not np.array_equal(self.counts.rows, self.cell_rows):
                raise ValueError("CSR counts do not follow the population cell rows")
            if np.any(self.cell_rows < 0) or np.any(self.cell_rows >= len(self.cell_store.indptr) - 1):
                raise ValueError("Cell-major population row is out of range")
            if np.any(self.cell_store.rna_indptr[self.cell_rows + 1] <= self.cell_store.rna_indptr[self.cell_rows]):
                raise ValueError("Zero-RNA cells must be removed during preprocessing")
            if self.cell_metadata is not None and (
                not np.array_equal(self.cell_metadata["cell_id"].to_numpy(), cells)
            ):
                raise ValueError("Cell metadata does not follow the cell-major population")
            return self
        raw_counts = (
            np.empty(0, dtype=np.int64)
            if isinstance(self.counts, GeneBackedCounts)
            else self.counts.matrix.data
            if isinstance(self.counts, SparseCounts)
            else np.asarray(self.counts)
        )
        if raw_counts.dtype.kind not in "iu" and (
            not np.isfinite(raw_counts).all() or np.any(raw_counts != np.floor(raw_counts))
        ):
            raise ValueError("full-gene counts must be finite integers")
        counts = self.counts if isinstance(self.counts, SparseCounts) else np.asarray(self.counts, dtype=np.int64)
        if not self.genes or len(self.genes) != len(set(self.genes)):
            raise ValueError("full-gene population vocabulary must be unique and nonempty")
        if cells.ndim != 1 or not len(cells) or len(cells) != len(set(cells.tolist())):
            raise ValueError("full-gene population cell IDs must be nonempty and unique")
        if counts.shape != (len(cells), len(self.genes)) or (raw_counts.size and raw_counts.min() < 0):
            raise ValueError("full-gene counts must have non-negative shape [cells, G]")
        point_cloud = self.point_cloud.validated()
        if point_cloud.n_genes != len(self.genes):
            raise ValueError("point-cloud vocabulary does not match the count matrix")
        if len(point_cloud.entry_cell_indices) and int(point_cloud.entry_cell_indices.max()) >= len(cells):
            raise ValueError("point cloud references a cell outside the population")
        if isinstance(counts, GeneBackedCounts):
            counts.validate_cloud(point_cloud)
        if int(counts.sum()) != len(point_cloud.coordinates):
            raise ValueError("full-gene counts and cell-gene point lengths disagree")
        for first in range(0, 0 if isinstance(counts, GeneBackedCounts) else len(point_cloud.entry_lengths), 2000000):
            stop = first + 2000000
            values = counts[point_cloud.entry_cell_indices[first:stop], point_cloud.entry_gene_indices[first:stop]]
            if not np.array_equal(values, point_cloud.entry_lengths[first:stop]):
                raise ValueError("full-gene counts and cell-gene point lengths disagree")
        if self.cell_metadata is not None and (not np.array_equal(self.cell_metadata["cell_id"].to_numpy(), cells)):
            raise ValueError("cell metadata does not follow the population cell axis")
        return FullGenePopulation(
            dataset=self.dataset,
            context=self.context,
            condition=self.condition,
            unit_id=self.unit_id,
            genes=self.genes,
            cell_ids=cells,
            counts=counts,
            point_cloud=point_cloud,
            cell_metadata=self.cell_metadata,
        )


def _assemble_population(
    *,
    dataset: str,
    genes: tuple[str, ...],
    cell_frame: pl.DataFrame,
    metadata_rows: list[dict[str, Any]],
    cell_keys: np.ndarray,
    cell_counts: np.ndarray,
    points: np.ndarray,
    cell_cap: int | None = None,
    cell_seed: int = 42,
) -> FullGenePopulation:
    context, condition, unit_id = _group_key(metadata_rows[0])
    cells = cell_frame.sort("cell_id")
    all_cell_ids = cells["cell_id"].to_numpy().astype(str)
    selected_indices = np.arange(len(all_cell_ids))
    if cell_cap is not None and len(selected_indices) > cell_cap:
        if cell_cap < 1:
            raise ValueError("cell_cap must be positive when supplied")
        rng = deterministic_rng(cell_seed, dataset, context, condition, unit_id)
        selected_indices = np.sort(rng.choice(len(all_cell_ids), size=cell_cap, replace=False))
    cell_ids = all_cell_ids[selected_indices]
    selected_lookup = np.full(len(all_cell_ids), -1, dtype=np.int32)
    selected_lookup[selected_indices] = np.arange(len(selected_indices))
    declared_keys = cells["cell_key"].to_numpy().astype(np.uint32)
    if len(declared_keys) != len(np.unique(declared_keys)):
        raise ValueError("declared-cell sidecar contains duplicate cell keys")
    key_order = np.argsort(declared_keys, kind="stable")
    sorted_keys = declared_keys[key_order]
    key_to_cell = key_order
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    rows_by_gene = {str(row["gene"]): row for row in metadata_rows}
    if set(rows_by_gene) != set(genes) or len(rows_by_gene) != len(metadata_rows):
        raise ValueError(f"population metadata does not cover each frozen gene once: {context}/{condition}/{unit_id}")
    physical_starts: list[np.ndarray] = []
    entry_gene_indices: list[np.ndarray] = []
    entry_cell_indices: list[np.ndarray] = []
    entry_lengths: list[np.ndarray] = []
    gene_entry_lengths = np.zeros(len(genes), dtype=np.int64)
    for gene in genes:
        row = rows_by_gene[gene]
        if int(row["declared_cell_count"]) != len(all_cell_ids):
            raise ValueError("gene population disagrees with the declared cell universe")
        gene_index = gene_to_index[gene]
        cell_start = int(row["cell_start"])
        cell_stop = cell_start + int(row["nonempty_cell_count"])
        nonempty_keys = np.asarray(cell_keys[cell_start:cell_stop], dtype=np.uint32)
        nonempty_counts = np.asarray(cell_counts[cell_start:cell_stop])
        if len(nonempty_keys) > 1 and np.any(nonempty_keys[1:] <= nonempty_keys[:-1]):
            raise ValueError("backed count lookup requires strictly sorted cell keys per gene")
        positions = np.searchsorted(sorted_keys, nonempty_keys)
        if len(positions) and (
            np.any(positions >= len(sorted_keys)) or not np.array_equal(sorted_keys[positions], nonempty_keys)
        ):
            raise ValueError("nonempty count key is absent from the declared-cell sidecar")
        cell_indices = selected_lookup[key_to_cell[positions]]
        point_start = int(row["point_start"])
        point_stop = point_start + int(row["point_length"])
        if int(nonempty_counts.sum()) != point_stop - point_start:
            raise ValueError("gene counts do not sum to its point slice")
        kept = cell_indices >= 0
        original_starts = (
            point_start + np.concatenate(([0], np.cumsum(nonempty_counts[:-1])))
            if len(nonempty_counts)
            else np.empty(0, dtype=np.int64)
        )
        selected_starts = original_starts[kept]
        nonempty_counts = nonempty_counts[kept]
        cell_indices = cell_indices[kept]
        if len(nonempty_counts):
            physical_starts.append(
                selected_starts.astype(np.uint32 if len(points) <= np.iinfo(np.uint32).max else np.uint64)
            )
            entry_gene_indices.append(
                np.full(len(nonempty_counts), gene_index, dtype=np.uint16 if len(genes) <= 65536 else np.uint32)
            )
            entry_cell_indices.append(cell_indices.astype(np.uint32, copy=False))
            entry_lengths.append(nonempty_counts)
            gene_entry_lengths[gene_index] = len(nonempty_counts)
    entry_lengths_array = np.concatenate(entry_lengths) if entry_lengths else np.empty(0, dtype=np.int64)
    gene_entry_starts = np.concatenate(([0], np.cumsum(gene_entry_lengths[:-1], dtype=np.int64)))
    coordinates = PackedCoordinates(
        points,
        np.concatenate(physical_starts) if physical_starts else np.empty(0, dtype=np.int64),
        entry_lengths_array,
    )
    entry_genes = np.concatenate(entry_gene_indices) if entry_gene_indices else np.empty(0, dtype=np.int64)
    entry_cells = np.concatenate(entry_cell_indices) if entry_cell_indices else np.empty(0, dtype=np.int64)
    del entry_lengths, entry_gene_indices, entry_cell_indices, physical_starts
    counts = GeneBackedCounts(
        cell_keys,
        cell_counts,
        [rows_by_gene[gene]["cell_start"] for gene in genes],
        [rows_by_gene[gene]["nonempty_cell_count"] for gene in genes],
        declared_keys[selected_indices],
        full_cell_universe=len(selected_indices) == len(all_cell_ids),
    )
    point_cloud = CellBalancedFullGenePointCloud(
        coordinates=coordinates,
        entry_gene_indices=entry_genes,
        entry_cell_indices=entry_cells,
        entry_starts=coordinates.starts,
        entry_lengths=entry_lengths_array,
        gene_entry_starts=gene_entry_starts,
        gene_entry_lengths=gene_entry_lengths,
    ).validated()
    return FullGenePopulation(
        dataset=dataset,
        context=context,
        condition=condition,
        unit_id=unit_id,
        genes=genes,
        cell_ids=cell_ids,
        counts=counts,
        point_cloud=point_cloud,
        cell_metadata=cells[selected_indices.tolist()],
    ).validated()


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["context"]), str(row["condition"]), str(row["unit_id"]))


def _selected_vocabulary(manifest: dict[str, Any], genes: Sequence[str] | None) -> tuple[str, ...]:
    inventory = tuple(sorted(str(value) for value in manifest.get("inventory", {}).get("genes", [])))
    if not inventory:
        raise ValueError("population index has no frozen gene inventory")
    if genes is None:
        return inventory
    requested = tuple(sorted({str(value) for value in genes}))
    if requested != inventory:
        raise ValueError(
            f"The complete-cell protocol requires the complete frozen gene vocabulary; requested {len(requested)} of {len(inventory)} genes"
        )
    return inventory


def iter_full_gene_populations(
    index_dir: Path,
    *,
    conditions: Sequence[str],
    contexts: Sequence[str],
    genes: Sequence[str] | None = None,
    expected_dataset: str | None = None,
    expected_context_column: str | None = None,
    cell_cap: int | None = None,
    cell_seed: int = 42,
    units: Sequence[str] | None = None,
) -> Iterator[FullGenePopulation]:
    """Yield exact zero-inclusive populations with the complete gene vocabulary."""
    selected_conditions = tuple(sorted({str(value) for value in conditions}))
    selected_contexts = tuple(sorted({str(value) for value in contexts}))
    if not selected_conditions or not selected_contexts:
        raise ValueError("conditions and contexts must be nonempty")
    manifest = load_population_index_manifest(
        index_dir, expected_dataset=expected_dataset, expected_context_column=expected_context_column
    )
    vocabulary = _selected_vocabulary(manifest, genes)
    inventory = manifest.get("inventory", {})
    checks = {
        "conditions": (selected_conditions, set(inventory.get("conditions", []))),
        "contexts": (selected_contexts, set(inventory.get("contexts", []))),
    }
    for label, (requested, available) in checks.items():
        if unknown := set(requested).difference(available):
            raise ValueError(f"population index lacks requested {label}: {sorted(unknown)}")
    if manifest.get("format") == CELL_MAJOR_FORMAT:
        yield from iter_cell_major_populations(
            index_dir,
            manifest,
            conditions=selected_conditions,
            contexts=selected_contexts,
            units=units,
            cell_cap=cell_cap,
            cell_seed=cell_seed,
        )
        return
    inference_manifest = load_inference_cell_index_manifest(
        index_dir, expected_dataset=str(manifest["dataset"]), expected_context_column=str(manifest["context_column"])
    )
    cell_shards = {int(row["unit_index"]): row for row in inference_manifest["shards"]}
    for shard in sorted(manifest["shards"], key=lambda row: int(row["unit_index"])):
        shard_index = int(shard["unit_index"])
        predicate = pl.col("condition").is_in(selected_conditions) & pl.col("context").is_in(selected_contexts)
        if units is not None:
            predicate = predicate & pl.col("unit_id").is_in(units)
        metadata = pl.scan_parquet(index_dir / str(shard["populations"])).filter(predicate).collect(engine="streaming")
        if metadata.is_empty():
            continue
        cell_row = cell_shards.get(shard_index)
        if cell_row is None:
            raise ValueError(f"inference cell index lacks shard {shard_index}")
        declared = pl.scan_parquet(index_dir / str(cell_row["cells"])).filter(predicate).collect(engine="streaming")
        cell_keys = np.load(index_dir / str(shard["cell_keys"]), mmap_mode="r")
        cell_counts = np.load(index_dir / str(shard["cell_counts"]), mmap_mode="r")
        points = np.load(index_dir / str(shard["points"]), mmap_mode="r")
        metadata_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in metadata.iter_rows(named=True):
            metadata_groups[_group_key(row)].append(row)
        declared_groups = {
            (str(raw_key[0]), str(raw_key[1]), str(raw_key[2])): frame
            for raw_key, frame in declared.partition_by(
                ["context", "condition", "unit_id"], as_dict=True, maintain_order=True
            ).items()
        }
        if set(metadata_groups) != set(declared_groups):
            raise ValueError("population metadata and declared-cell groups do not match")
        for key in sorted(metadata_groups):
            yield _assemble_population(
                dataset=str(manifest["dataset"]),
                genes=vocabulary,
                cell_frame=declared_groups[key],
                metadata_rows=metadata_groups[key],
                cell_keys=cell_keys,
                cell_counts=cell_counts,
                points=points,
                cell_cap=cell_cap,
                cell_seed=cell_seed,
            )


def subset_full_gene_population(population: FullGenePopulation, cell_ids: Sequence[str]) -> FullGenePopulation:
    """Select cells while preserving the complete gene axis and packed geometry."""
    population = population.validated()
    selected = np.asarray([str(value) for value in cell_ids], dtype=str)
    if selected.ndim != 1 or not len(selected) or len(selected) != len(set(selected.tolist())):
        raise ValueError("selected full-gene cell IDs must be nonempty and unique")
    cell_to_index = {cell: index for index, cell in enumerate(population.cell_ids.tolist())}
    missing = [cell for cell in selected if cell not in cell_to_index]
    if missing:
        raise ValueError(f"selected full-gene cells are absent from the population: {missing[:3]}")
    old_indices = np.asarray([cell_to_index[cell] for cell in selected], dtype=np.int64)
    if population.cell_store is not None:
        from dataclasses import replace

        rows = population.cell_rows[old_indices]
        return replace(
            population,
            cell_ids=selected,
            counts=population.counts.subset(old_indices),
            point_cloud=LazyGeneCloud(population.cell_store, rows),
            cell_rows=rows,
            cell_metadata=population.cell_metadata[old_indices.tolist()]
            if population.cell_metadata is not None
            else None,
        ).validated()
    old_to_new = np.full(len(population.cell_ids), -1, dtype=np.int64)
    old_to_new[old_indices] = np.arange(len(selected), dtype=np.int64)
    cloud = population.point_cloud
    keep_entries = old_to_new[cloud.entry_cell_indices] >= 0
    kept_lengths = cloud.entry_lengths[keep_entries]
    selected_starts = cloud.entry_starts[keep_entries]
    coordinates = (
        cloud.coordinates.subset_entries(keep_entries)
        if isinstance(cloud.coordinates, PackedCoordinates)
        else PackedCoordinates(cloud.coordinates, selected_starts, kept_lengths)
    )
    entry_starts = coordinates.starts
    entry_genes = cloud.entry_gene_indices[keep_entries]
    gene_entry_lengths = np.bincount(entry_genes, minlength=len(population.genes)).astype(np.int64, copy=False)
    gene_entry_starts = np.concatenate(([0], np.cumsum(gene_entry_lengths[:-1], dtype=np.int64)))
    point_cloud = CellBalancedFullGenePointCloud(
        coordinates=coordinates,
        entry_gene_indices=entry_genes,
        entry_cell_indices=old_to_new[cloud.entry_cell_indices[keep_entries]],
        entry_starts=entry_starts,
        entry_lengths=kept_lengths,
        gene_entry_starts=gene_entry_starts,
        gene_entry_lengths=gene_entry_lengths,
    ).validated()
    return FullGenePopulation(
        dataset=population.dataset,
        context=population.context,
        condition=population.condition,
        unit_id=population.unit_id,
        genes=population.genes,
        cell_ids=selected,
        counts=population.counts.subset(old_indices)
        if isinstance(population.counts, SparseCounts)
        else SparseCounts(population.counts[old_indices]),
        point_cloud=point_cloud,
        cell_metadata=population.cell_metadata[old_indices.tolist()] if population.cell_metadata is not None else None,
    ).validated()
