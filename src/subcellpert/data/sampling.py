"""Deterministic evaluation sampling, oracle counts, and joint cell-axis budgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from subcellpert.data.arrays import GeneBackedCounts
from subcellpert.data.storage import (
    CELL_MAJOR_FORMAT,
    CellMajorCounts,
    CellMajorStore,
    load_inference_cell_index_manifest,
    load_population_index_manifest,
)
from subcellpert.utils import (
    seed_from_parts,
)


@dataclass(frozen=True)
class OracleCounts:
    dataset: str
    context: str
    condition: str
    unit_id: str
    genes: tuple[str, ...]
    cell_ids: np.ndarray
    counts: object

    def validated(self):
        if not len(self.cell_ids) or len(set(self.cell_ids.tolist())) != len(self.cell_ids):
            raise ValueError("Oracle cell axis must be nonempty and unique")
        if self.counts.shape != (len(self.cell_ids), len(self.genes)) or len(set(self.genes)) != len(self.genes):
            raise ValueError("Oracle counts must cover the complete cell and gene axes")
        if isinstance(self.counts, CellMajorCounts):
            self.counts.validate_nonempty()
            return self
        for first in range(0, len(self.cell_ids), 128):
            values = self.counts[first : first + 128]
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
                raise ValueError("Oracle counts must be finite nonnegative integers")
            if np.any(values.sum(axis=1) <= 0):
                raise ValueError("Zero-RNA cells must be removed during preprocessing")
        return self

    def subset(self, ids):
        lookup = {cell: i for i, cell in enumerate(self.cell_ids)}
        rows = np.array([lookup[cell] for cell in ids], dtype=np.int64)
        counts = self.counts.subset(rows) if hasattr(self.counts, "subset") else self.counts[rows]
        return OracleCounts(
            self.dataset, self.context, self.condition, self.unit_id, self.genes, self.cell_ids[rows], counts
        )


def iter_oracle_counts(index_dir: Path, *, condition: str, context: str):
    """Only read cell IDs/keys and count arrays; never open RNA or transform data."""
    manifest = load_population_index_manifest(index_dir)
    if manifest.get("format") == CELL_MAJOR_FORMAT:
        genes = tuple(sorted(manifest["inventory"]["genes"]))
        for shard in sorted(manifest["shards"], key=lambda s: int(s["unit_index"])):
            frame = (
                pl.scan_parquet(index_dir / shard["cells"])
                .filter((pl.col("condition") == condition) & (pl.col("context") == context))
                .select("unit_id", "cell_id", "cell_key")
                .sort("cell_id")
                .collect()
            )
            if frame.is_empty():
                continue
            store = CellMajorStore(index_dir, shard, len(genes))
            yield OracleCounts(
                manifest["dataset"],
                context,
                condition,
                str(shard["unit_id"]),
                genes,
                frame["cell_id"].to_numpy(),
                store.counts(frame["cell_key"].to_numpy()),
            ).validated()
        return
    inference = load_inference_cell_index_manifest(
        index_dir, expected_dataset=manifest["dataset"], expected_context_column=manifest["context_column"]
    )
    cell_shards = {int(s["unit_index"]): s for s in inference["shards"]}
    genes = tuple(manifest["inventory"]["genes"])
    predicate = (pl.col("condition") == condition) & (pl.col("context") == context)
    for shard in sorted(manifest["shards"], key=lambda s: int(s["unit_index"])):
        metadata = (
            pl.scan_parquet(index_dir / shard["populations"])
            .filter(predicate)
            .select("condition", "context", "unit_id", "gene", "cell_start", "nonempty_cell_count")
            .collect()
        )
        if metadata.is_empty():
            continue
        cells = (
            pl.scan_parquet(index_dir / cell_shards[int(shard["unit_index"])]["cells"])
            .filter(predicate)
            .select("condition", "context", "unit_id", "cell_id", "cell_key")
            .collect()
        )
        keys = np.load(index_dir / shard["cell_keys"], mmap_mode="r", allow_pickle=False)
        values = np.load(index_dir / shard["cell_counts"], mmap_mode="r", allow_pickle=False)
        for unit in sorted(metadata["unit_id"].unique().to_list()):
            rows = {r["gene"]: r for r in metadata.filter(pl.col("unit_id") == unit).iter_rows(named=True)}
            if set(rows) != set(genes) or len(rows) != len(genes):
                raise ValueError("Oracle gene vocabulary does not match the count index")
            declared = cells.filter(pl.col("unit_id") == unit).sort("cell_id")
            counts = GeneBackedCounts(
                keys,
                values,
                [rows[g]["cell_start"] for g in genes],
                [rows[g]["nonempty_cell_count"] for g in genes],
                declared["cell_key"].to_numpy(),
                full_cell_universe=True,
            )
            yield OracleCounts(
                manifest["dataset"], context, condition, str(unit), genes, declared["cell_id"].to_numpy(), counts
            )


def sample_oracle_counts(populations, *, response_id, cap=2000):
    populations = sorted(populations, key=lambda p: p.unit_id)
    rows = sample_cell_axes(populations, response_id=response_id, namespace="target", cap=cap)
    sampled = [p.subset(r["cell_ids"]).validated() for p, r in zip(populations, rows, strict=True)]
    return (sampled, rows)


SC_PERT_BENCH_BASE_SEED = 42
SC_PERT_BENCH_CELL_CAP = 2000


def deterministic_cell_sample(
    cell_ids: np.ndarray | list[str] | tuple[str, ...],
    *,
    dataset: str,
    fold: str,
    transition: str,
    group: str,
    cap: int = SC_PERT_BENCH_CELL_CAP,
    base_seed: int = SC_PERT_BENCH_BASE_SEED,
) -> np.ndarray:
    """Sample one evaluation group without replacement using the sampling namespace."""
    cells = np.asarray(cell_ids, dtype=str)
    if cells.ndim != 1 or not len(cells) or len(cells) != len(set(cells.tolist())):
        raise ValueError("cell IDs must be a nonempty unique vector")
    if cap <= 0:
        raise ValueError("cell cap must be positive")
    ordered = np.sort(cells)
    if len(ordered) <= cap:
        return ordered
    namespace = f"cell-sample-v1|{base_seed}|{dataset}|{fold}|{transition}|{group}"
    seed = seed_from_parts(namespace)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(ordered), size=cap, replace=False))
    return ordered[indices]


@dataclass(frozen=True)
class CellAxis:
    dataset: str
    condition: str
    unit_id: str
    cell_ids: np.ndarray


def iter_cell_axes(index_dir, *, condition, context):
    """Read IDs and population membership, never expression arrays or coordinates."""
    index_dir = Path(index_dir)
    manifest = load_population_index_manifest(index_dir)
    membership = (
        manifest
        if manifest.get("format") == CELL_MAJOR_FORMAT
        else load_inference_cell_index_manifest(
            index_dir, expected_dataset=manifest["dataset"], expected_context_column=manifest["context_column"]
        )
    )
    for shard in sorted(membership["shards"], key=lambda s: int(s["unit_index"])):
        frame = (
            pl.scan_parquet(index_dir / shard["cells"])
            .filter((pl.col("condition") == condition) & (pl.col("context") == context))
            .select("unit_id", "cell_id")
            .sort("cell_id")
            .collect()
        )
        for unit in sorted(frame["unit_id"].unique().to_list()):
            yield CellAxis(
                manifest["dataset"],
                condition,
                str(unit),
                frame.filter(pl.col("unit_id") == unit)["cell_id"].to_numpy(),
            )


def joint_output_ids(donors):
    return np.asarray([f"joint:{cell}:draw:{i}" for i, cell in enumerate(donors)])


def sample_cell_axes(populations, *, response_id, namespace, cap):
    """Shared unit-stratified sampling without replacement, before generation."""
    populations = sorted(populations, key=lambda p: p.unit_id)
    if not populations or cap < len(populations) or len({p.unit_id for p in populations}) != len(populations):
        raise ValueError("Cell selection needs unique nonempty units and enough budget to retain them")
    sizes = np.array([len(p.cell_ids) for p in populations])
    if np.any(sizes <= 0):
        raise ValueError("Cell selection cannot include an empty population")
    allocations = np.zeros(len(sizes), dtype=int)
    remaining = min(cap, int(sizes.sum()))
    while remaining:
        selected = np.flatnonzero(allocations < sizes)[:remaining]
        allocations[selected] += 1
        remaining -= len(selected)
    rows = []
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
        rows.append(
            {
                "unit_id": population.unit_id,
                "available_cells": len(population.cell_ids),
                "selected_cells": len(ids),
                "cell_ids": ids.tolist(),
            }
        )
    return rows
