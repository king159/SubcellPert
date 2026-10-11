"""Complete-cell RNA reads, donor deduplication, collation, and bounded prefetch."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch


if TYPE_CHECKING:
    from subcellpert.data.populations import FullGenePopulation


class CompleteCellCollator:
    """One batched read per population, including duplicate donors and controls.

    Returned batches contain arrays and small audit records, with no population
    object or full backing store carried into the training step.
    """

    def __init__(self, examples, *, foundation_store=None, read=None):
        self.examples, self.foundation_store, self.custom_read = (examples, foundation_store, read)
        self.readers = {}

    def _read(self, population, rows):
        if self.custom_read is not None:
            return self.custom_read(population, rows)
        key = id(population)
        if key not in self.readers:
            self.readers[key] = PopulationCellReader(population)
        return self.readers[key].read(rows)

    def __call__(self, indices):
        return self.encode(self.prepare(indices))

    def encode(self, batch):
        """Run the frozen encoder on the consuming thread and its CUDA stream."""
        if self.foundation_store is None:
            return batch
        return replace(
            batch,
            foundation=self.foundation_store.encode_counts(
                batch.source.counts, batch.source.genes, batch.source_conditions
            ),
        )

    def prepare(self, indices):
        """Read and collate CPU arrays without invoking any device operations."""
        indices = np.asarray(indices, dtype=np.int64)
        examples = [self.examples.example(int(i)) for i in indices]
        populations, requested = ({}, {})
        for example in examples:
            for population, row in ((example.source, example.source_row), (example.target, example.target_row)):
                key = id(population)
                populations[key] = population
                requested.setdefault(key, {})[row] = None
        batches, lookup, identities = ([], {}, [])
        for key, rows in requested.items():
            rows = np.asarray(list(rows), dtype=np.int64)
            batch = self._read(populations[key], rows)
            first = len(identities)
            for i, row in enumerate(rows):
                lookup[key, int(row)] = first + i
                identities.append(f"cell_{first + i}")
            batches.append(batch)
        all_cells = concatenate_cell_batches(batches, identities)
        source_rows, donor_lookup, donors, source_conditions = ([], {}, [], [])
        if self.foundation_store is not None:
            sources = {id(example.source): example.source for example in examples}
            for population in sources.values():
                self.foundation_store.validate_population(population)
        for example in examples:
            row = lookup[id(example.source), example.source_row]
            if row not in donor_lookup:
                donor_lookup[row] = len(source_rows)
                source_rows.append(row)
                source_conditions.append(example.source.condition)
            donors.append(donor_lookup[row])
        target_rows = [lookup[id(e.target), e.target_row] for e in examples]
        source = select_batch_cells(all_cells, source_rows, [f"source_{i}" for i in range(len(source_rows))])
        target = select_batch_cells(all_cells, target_rows, [f"example_{int(i)}" for i in indices])
        audit = []
        for i, example in enumerate(examples):
            source_rna = int(source.counts[donors[i]].sum())
            target_rna = int(target.counts[i].sum())
            fallback = int(target.counts[i, source.counts[donors[i]] == 0].sum())
            audit.append(
                {
                    "example_index": int(indices[i]),
                    "source_cell_id": str(example.source.cell_ids[example.source_row]),
                    "target_cell_id": str(example.target.cell_ids[example.target_row]),
                    "control_identity": example.control_identity,
                    "source_unit": example.source.unit_id,
                    "target_unit": example.target.unit_id,
                    "input_rna_molecules": source_rna + target_rna,
                    "fallback_rna_molecules": fallback,
                }
            )
        return CompleteCellExamplesBatch(
            indices,
            source,
            target,
            np.asarray(donors, dtype=np.int64),
            np.stack([e.descriptor for e in examples]),
            np.asarray([e.control_identity for e in examples]),
            tuple(e.condition_key for e in examples),
            None,
            tuple(audit),
            sum(row["input_rna_molecules"] for row in audit),
            len(all_cells.coordinates),
            tuple(source_conditions),
        )


@dataclass(frozen=True)
class CompleteCellExamplesBatch:
    indices: np.ndarray
    source: CellRNABatch
    target: CellRNABatch
    donors: np.ndarray
    descriptors: np.ndarray
    identity: np.ndarray
    condition_keys: tuple
    foundation: torch.Tensor | None
    audit: tuple
    logical_rna_molecules: int
    physical_rna_molecules: int
    source_conditions: tuple[str, ...] = ()


def concatenate_cell_batches(batches, identities):
    genes = batches[0].genes
    if any(b.genes != genes for b in batches):
        raise ValueError("Complete gene axes differ across cell populations")
    counts = np.concatenate([b.counts for b in batches])
    lengths = counts.sum(1, dtype=np.int64)
    return CellRNABatch(
        genes,
        np.asarray(identities),
        counts,
        np.concatenate([b.coordinates for b in batches]),
        np.concatenate([b.molecule_genes for b in batches]),
        np.repeat(np.arange(len(counts), dtype=np.int32), lengths),
        np.r_[0, np.cumsum(lengths)],
    ).validated()


def prefetch_cell_batches(indices, collator, *, depth=2):
    """Bound CPU prefetch without forking CUDA/Polars or copying full stores.

    Collation is deterministic and never touches training RNG. One worker owns
    the readers and produces batches in the original DataLoader shuffle order.
    """
    if depth < 1:
        for rows in indices:
            yield collator(rows)
        return
    source = iter(indices)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cell-read") as pool:
        pending = deque()
        for _ in range(depth):
            rows = next(source, None)
            if rows is None:
                break
            pending.append(pool.submit(collator, rows))
        while pending:
            result = pending.popleft().result()
            rows = next(source, None)
            if rows is not None:
                pending.append(pool.submit(collator, rows))
            yield result


def select_batch_cells(batch, rows, identities):
    rows = np.asarray(rows, dtype=np.int64)
    lengths = np.diff(batch.cell_offsets)[rows]
    offsets = np.r_[0, np.cumsum(lengths)]
    positions = np.repeat(batch.cell_offsets[rows] - offsets[:-1], lengths) + np.arange(int(offsets[-1]))
    return CellRNABatch(
        batch.genes,
        np.asarray(identities),
        batch.counts[rows],
        batch.coordinates[positions],
        batch.molecule_genes[positions],
        np.repeat(np.arange(len(rows), dtype=np.int32), lengths),
        offsets,
    ).validated()


@dataclass(frozen=True)
class CellRNABatch:
    """Ragged cell-major RNA arrays; no molecule or gene subsampling."""

    genes: tuple[str, ...]
    cell_ids: np.ndarray
    counts: np.ndarray
    coordinates: np.ndarray
    molecule_genes: np.ndarray
    molecule_cells: np.ndarray
    cell_offsets: np.ndarray

    def validated(self):
        cells, genes = self.counts.shape
        if cells != len(self.cell_ids) or genes != len(self.genes):
            raise ValueError("Cell batch count axes must match complete cell/gene identities")
        if len(set(self.cell_ids.tolist())) != cells or len(set(self.genes)) != genes:
            raise ValueError("Cell batch identities must be unique")
        if self.counts.dtype.kind not in "iu" or np.any(self.counts < 0):
            raise ValueError("Cell batch counts must be nonnegative integers")
        if np.any(self.counts.sum(axis=1, dtype=np.int64) == 0):
            raise ValueError("Zero-RNA cells must be removed during preprocessing")
        molecules = int(self.counts.sum(dtype=np.int64))
        if self.coordinates.shape != (molecules, 2) or not np.isfinite(self.coordinates).all():
            raise ValueError("Cell batch must contain all finite RNA coordinates")
        if self.molecule_genes.shape != (molecules,) or self.molecule_cells.shape != (molecules,):
            raise ValueError("Every RNA must carry a gene label and cell membership")
        if any(a.dtype.kind not in "iu" for a in (self.molecule_genes, self.molecule_cells, self.cell_offsets)):
            raise ValueError("RNA labels and cell offsets must be integers")
        expected = np.r_[0, np.cumsum(self.counts.sum(axis=1, dtype=np.int64))]
        if not np.array_equal(self.cell_offsets, expected):
            raise ValueError("RNA cell offsets must agree with complete cell counts")
        if molecules:
            if (
                self.molecule_cells.min() < 0
                or self.molecule_cells.max() >= cells
                or self.molecule_genes.min() < 0
                or (self.molecule_genes.max() >= genes)
            ):
                raise ValueError("RNA labels fall outside the batch axes")
            if not np.array_equal(self.molecule_cells, np.repeat(np.arange(cells), np.diff(expected))):
                raise ValueError("RNA arrays must preserve cell-major membership")
        observed = np.bincount(
            self.molecule_cells.astype(np.int64) * genes + self.molecule_genes, minlength=cells * genes
        ).reshape(cells, genes)
        if not np.array_equal(observed, self.counts):
            raise ValueError("RNA coordinates and cell-by-gene counts disagree")
        return self


class PopulationCellReader:
    """Index cell-gene entries once, then read only selected complete cells."""

    def __init__(self, population: FullGenePopulation):
        self.population = population
        if getattr(population, "cell_store", None) is not None:
            return
        cloud = population.point_cloud
        self.entry_order = np.argsort(cloud.entry_cell_indices, kind="stable")
        sizes = np.bincount(cloud.entry_cell_indices, minlength=len(population.cell_ids))
        self.cell_entry_offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]

    def read(self, rows):
        rows = np.asarray(rows)
        population = self.population
        if rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu":
            raise ValueError("Select a nonempty integer cell-index vector")
        if rows.min() < 0 or rows.max() >= len(population.cell_ids) or len(np.unique(rows)) != len(rows):
            raise ValueError("Cell indices must be distinct and in range")
        if getattr(population, "cell_store", None) is not None:
            return population.cell_store.read(
                population.cell_rows[rows], genes=population.genes, cell_ids=population.cell_ids[rows]
            )
        cloud = population.point_cloud
        counts = np.zeros((len(rows), len(population.genes)), dtype=np.int64)
        for i, row in enumerate(rows):
            entries = self.entry_order[self.cell_entry_offsets[row] : self.cell_entry_offsets[row + 1]]
            counts[i, cloud.entry_gene_indices[entries]] = cloud.entry_lengths[entries]
        offsets = np.r_[0, np.cumsum(counts.sum(axis=1), dtype=np.int64)]
        points = np.empty((int(offsets[-1]), 2), dtype=np.float32)
        molecule_genes = np.empty(len(points), dtype=np.int32)
        for batch_row, original_row in enumerate(rows):
            entries = self.entry_order[
                self.cell_entry_offsets[original_row] : self.cell_entry_offsets[original_row + 1]
            ]
            lengths = cloud.entry_lengths[entries].astype(np.int64)
            destination = int(offsets[batch_row])
            local_starts = np.r_[0, np.cumsum(lengths[:-1])] if len(entries) else np.empty(0, dtype=np.int64)
            total = int(lengths.sum())
            if total != offsets[batch_row + 1] - offsets[batch_row]:
                raise ValueError("Indexed source cell RNA does not match its counts")
            if not total:
                continue
            molecule_genes[destination : destination + total] = np.repeat(cloud.entry_gene_indices[entries], lengths)
            for first in range(0, total, 262144):
                local = np.arange(first, min(total, first + 262144), dtype=np.int64)
                selected = np.searchsorted(local_starts + lengths, local, side="right")
                logical = cloud.entry_starts[entries[selected]] + local - local_starts[selected]
                points[destination + first : destination + first + len(local)] = cloud.coordinates[logical]
        return CellRNABatch(
            population.genes,
            population.cell_ids[rows].copy(),
            counts,
            points,
            molecule_genes,
            np.repeat(np.arange(len(rows), dtype=np.int32), np.diff(offsets)),
            offsets,
        ).validated()
