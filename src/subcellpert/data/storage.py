"""Cell-major storage readers and population/inference manifest validation."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from subcellpert.data.arrays import SparseCounts
from subcellpert.data.batching import CellRNABatch


CELL_MAJOR_ARRAY_KEYS = ("points", "cell_counts", "cell_genes", "cell_indptr", "cell_rna_indptr")
CELL_MAJOR_ARTIFACT_KEYS = (*CELL_MAJOR_ARRAY_KEYS, "cells", "populations")
CELL_MAJOR_FORMAT = "cell_major_rna_csr_numpy_parquet_v3"
CELL_MAJOR_SCHEMA = 3


class CellMajorCounts(SparseCounts):
    """Counts-only CSR view: no coordinate path, metadata or geometry handle."""

    def __init__(self, indptr, gene_ids, values, rows, n_genes):
        self.indptr, self.gene_ids, self.values = (indptr, gene_ids, values)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.shape = (len(self.rows), int(n_genes))
        self.dtype, self.ndim = (np.dtype("int64"), 2)
        self.size = int(np.prod(self.shape))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, columns = index
            if isinstance(rows, np.ndarray) and isinstance(columns, np.ndarray):
                rows, columns = np.broadcast_arrays(rows, columns)
                unique, inverse = np.unique(rows, return_inverse=True)
                return self[unique][inverse.reshape(rows.shape), columns]
            return self[rows][..., columns]
        scalar = isinstance(index, (int, np.integer))
        local_rows = np.arange(*index.indices(len(self))) if isinstance(index, slice) else np.atleast_1d(index)
        if local_rows.dtype == bool:
            local_rows = np.flatnonzero(local_rows)
        selected = self.rows[local_rows]
        output = np.zeros((len(selected), self.shape[1]), dtype=np.int64)
        entries, lengths, _ = csr_entries(self.indptr, selected)
        output[np.repeat(np.arange(len(selected)), lengths), self.gene_ids[entries]] = self.values[entries]
        return output[0] if scalar else output

    def subset(self, rows):
        return CellMajorCounts(self.indptr, self.gene_ids, self.values, self.rows[rows], self.shape[1])

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def sum(self, axis=None):
        if axis not in (None, 0, 1):
            raise ValueError("Unsupported CSR count reduction")
        result = np.zeros(self.shape[1], dtype=np.int64) if axis == 0 else []
        for first in range(0, len(self), 4096):
            entries, lengths, ptr = csr_entries(self.indptr, self.rows[first : first + 4096])
            values = self.values[entries].astype(np.int64)
            if axis == 0:
                np.add.at(result, self.gene_ids[entries], values)
            else:
                sums = np.zeros(len(lengths), dtype=np.int64)
                np.add.at(sums, np.repeat(np.arange(len(lengths)), lengths), values)
                result.append(sums)
        if axis == 0:
            return result
        values = np.concatenate(result) if result else np.empty(0, dtype=np.int64)
        return values if axis == 1 else int(values.sum())

    def mean(self, axis=None):
        if axis == 0:
            return self.sum(0) / len(self)
        if axis == 1:
            return self.sum(1) / self.shape[1]
        return self.sum() / (len(self) * self.shape[1])

    def prevalence(self):
        observed = np.zeros(self.shape[1], dtype=np.int64)
        for first in range(0, len(self), 4096):
            entries, _, _ = csr_entries(self.indptr, self.rows[first : first + 4096])
            observed += np.bincount(self.gene_ids[entries], minlength=self.shape[1])
        return observed / len(self)

    def validate_nonempty(self):
        """Check selected sparse counts without constructing their dense gene axis."""
        if (
            self.values.dtype.kind != "u"
            or self.gene_ids.shape != self.values.shape
            or np.any(self.rows < 0)
            or np.any(self.rows + 1 >= len(self.indptr))
        ):
            raise ValueError("Invalid cell-major count arrays or selected rows")
        for first in range(0, len(self), 4096):
            rows = self.rows[first : first + 4096]
            starts, stops = (self.indptr[rows], self.indptr[rows + 1])
            if np.any(stops <= starts) or np.any(stops > len(self.values)):
                raise ValueError("Zero-RNA cells or invalid CSR count pointers")
            entries, _, _ = csr_entries(self.indptr, rows)
            if np.any(self.values[entries] == 0) or np.any(self.gene_ids[entries] >= self.shape[1]):
                raise ValueError("Invalid nonzero gene counts")
        return self


class CellMajorStore:
    """One physical RNA sequence per cell and one CSR run per cell's genes."""

    def __init__(self, index_dir, shard, n_genes):
        self.index_dir, self.shard, self.n_genes = (Path(index_dir), shard, n_genes)
        self.indptr = self._array("cell_indptr")
        self.rna_indptr = self._array("cell_rna_indptr")
        self.gene_ids = self._array("cell_genes")
        self.values = self._array("cell_counts")
        n, e, m = (int(shard["declared_cells"]), int(shard["nonempty_cell_gene_rows"]), int(shard["molecules"]))
        if (
            self.indptr.shape != (n + 1,)
            or self.rna_indptr.shape != (n + 1,)
            or self.gene_ids.shape != (e,)
            or (self.values.shape != (e,))
        ):
            raise ValueError("Cell-major CSR shape mismatch")
        if self.indptr.dtype != np.dtype("uint64") or self.rna_indptr.dtype != np.dtype("uint64"):
            raise ValueError("Cell-major pointer arrays must use uint64")
        if self.gene_ids.dtype.kind != "u" or self.values.dtype.name not in ("uint16", "uint32"):
            raise ValueError("Cell-major gene/count arrays must use compact unsigned integers")
        if self.indptr[0] != 0 or self.rna_indptr[0] != 0 or self.indptr[-1] != e or (self.rna_indptr[-1] != m):
            raise ValueError("Cell-major pointers do not cover complete cells and RNA")

    def _array(self, key):
        return np.load(self.index_dir / self.shard[key], mmap_mode="r", allow_pickle=False)

    @cached_property
    def coordinates(self):
        values = self._array("points")
        if values.dtype != np.dtype("float32") or values.shape != (int(self.shard["molecules"]), 2):
            raise ValueError("Cell-major RNA coordinate shape/dtype mismatch")
        return values

    def counts(self, rows):
        return CellMajorCounts(self.indptr, self.gene_ids, self.values, rows, self.n_genes)

    def read(self, rows, *, genes, cell_ids):
        rows = np.asarray(rows, dtype=np.int64)
        counts = self.counts(rows)[:]
        lengths = np.asarray(self.rna_indptr[rows + 1] - self.rna_indptr[rows], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
        coordinates = np.empty((int(offsets[-1]), 2), dtype=np.float32)
        labels = np.empty(len(coordinates), dtype=np.int32)
        for i, row in enumerate(rows):
            start, stop = (int(v) for v in self.rna_indptr[row : row + 2])
            coordinates[offsets[i] : offsets[i + 1]] = self.coordinates[start:stop]
            first, last = (int(v) for v in self.indptr[row : row + 2])
            labels[offsets[i] : offsets[i + 1]] = np.repeat(self.gene_ids[first:last], self.values[first:last])
        return CellRNABatch(
            genes,
            np.asarray(cell_ids).copy(),
            counts,
            coordinates,
            labels,
            np.repeat(np.arange(len(rows), dtype=np.int32), lengths),
            offsets,
        ).validated()


def csr_entries(indptr, rows):
    rows = np.asarray(rows, dtype=np.int64)
    starts = np.asarray(indptr[rows], dtype=np.int64)
    lengths = np.asarray(indptr[rows + 1], dtype=np.int64) - starts
    local = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    entries = np.repeat(starts - local[:-1], lengths) + np.arange(int(local[-1]), dtype=np.int64)
    return (entries, lengths, local)


def _shared_array_path(index_dir, payload, key, relative):
    """Permit a frozen view to reference its own dataset inside a release."""
    if key not in CELL_MAJOR_ARRAY_KEYS or payload.get("cell_identity_view", {}).get("schema") != "cell_identity_view":
        return False
    dataset = payload.get("dataset")
    if not isinstance(dataset, str) or Path(dataset).name != dataset or dataset in {".", ".."}:
        return False
    target = (Path(index_dir) / relative).resolve()
    for root in Path(index_dir).resolve().parents:
        if (root / "manifest.json").is_file() and (root / "datasets").is_dir():
            population = root / "datasets" / dataset / "population"
            # Stop at the nearest release, even when it is nested inside a checkout.
            return target.is_relative_to(population) and population.resolve().is_relative_to(root)
    return False


def validate_cell_major_manifest(index_dir, payload, *, expected_dataset=None, expected_context_column=None):
    if payload.get("schema_version") != CELL_MAJOR_SCHEMA or payload.get("status") != "complete":
        raise ValueError("Incomplete cell-major population manifest")
    if payload.get("format") != CELL_MAJOR_FORMAT or not payload.get("shards"):
        raise ValueError("Invalid cell-major population format or shards")
    for key, expected in [("dataset", expected_dataset), ("context_column", expected_context_column)]:
        if expected is not None and payload.get(key) != expected:
            raise ValueError(f"Cell-major population {key} mismatch")
    units = set()
    for shard in payload["shards"]:
        unit = int(shard["unit_index"])
        if (
            unit in units
            or shard.get("format") != CELL_MAJOR_FORMAT
            or shard.get("schema_version") != CELL_MAJOR_SCHEMA
        ):
            raise ValueError("Invalid or duplicate cell-major shard")
        units.add(unit)
        for key in CELL_MAJOR_ARTIFACT_KEYS:
            relative = Path(shard[key])
            if (
                relative.is_absolute()
                or (".." in relative.parts and not _shared_array_path(index_dir, payload, key, relative))
                or not (Path(index_dir) / relative).is_file()
            ):
                raise ValueError(f"Cell-major artifact is missing or unsafe: {relative}")
    return payload


INFERENCE_CELL_INDEX_DIRECTORY = "inference_cells"
INFERENCE_CELL_INDEX_FORMAT = "population_index_declared_cells_parquet"
INFERENCE_CELL_INDEX_SCHEMA_VERSION = 2
POPULATION_INDEX_CELL_KEY_SEMANTICS = "shard_local_lexical_cell_rank"
POPULATION_INDEX_FORMAT = "compact_population_index_numpy_parquet"
POPULATION_INDEX_LOADED_CELL_KEY_SEMANTICS = "unit_index_uint32_shifted_plus_lexical_cell_rank"
POPULATION_INDEX_SCHEMA_VERSION = 2
_CELL_COUNT_DTYPES = {"uint16": np.dtype(np.uint16), "uint32": np.dtype(np.uint32)}
_CELL_KEY_DTYPE = np.dtype(np.uint32)


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"


def inference_cell_index_path(index_dir: Path) -> Path:
    """Return the cell-membership sidecar used by indexed inference.

    The compact training index deliberately stores only nonempty cell/gene
    rows. Oracle-count inference additionally needs the exact identities of
    retained cells, including zeros for individual genes. They live in this sidecar so inference can
    preserve the public per-cell contract without reopening canonical molecule
    Parquets.  Keeping the sidecar manifest separate also leaves the hash of an
    already-trained population index unchanged.
    """
    return index_dir / INFERENCE_CELL_INDEX_DIRECTORY


def load_inference_cell_index_manifest(
    index_dir: Path, *, expected_dataset: str | None = None, expected_context_column: str | None = None
) -> dict[str, Any]:
    path = inference_cell_index_path(index_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != INFERENCE_CELL_INDEX_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("format") != INFERENCE_CELL_INDEX_FORMAT
    ):
        raise ValueError(f"inference cell index is absent, stale, or incomplete: {path}")
    if (
        payload.get("cell_key_dtype") != _CELL_KEY_DTYPE.name
        or payload.get("cell_key_semantics") != POPULATION_INDEX_CELL_KEY_SEMANTICS
        or payload.get("loaded_cell_key_semantics") != POPULATION_INDEX_LOADED_CELL_KEY_SEMANTICS
        or (payload.get("loaded_cell_key_dtype") != np.dtype(np.uint64).name)
    ):
        raise ValueError(f"inference cell index has incompatible cell keys: {path}")
    checks = {"dataset": expected_dataset, "context_column": expected_context_column}
    for key, expected in checks.items():
        if expected is not None and payload.get(key) != expected:
            raise ValueError(f"inference cell index {key} mismatch: {payload.get(key)!r} != {expected!r}")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("complete inference cell index has no shards")
    for shard in shards:
        if (
            shard.get("schema_version") != INFERENCE_CELL_INDEX_SCHEMA_VERSION
            or shard.get("format") != INFERENCE_CELL_INDEX_FORMAT
            or shard.get("cell_key_dtype") != _CELL_KEY_DTYPE.name
        ):
            raise ValueError("inference cell-index shard has incompatible metadata")
        relative = shard.get("cells")
        if not isinstance(relative, str):
            raise ValueError("inference cell-index shard has no cell table")
        cell_path = index_dir / relative
        if not cell_path.is_file():
            raise ValueError(f"inference cell-index shard artifact is missing: {relative!r}")
    return payload


def load_population_index_manifest(
    index_dir: Path, *, expected_dataset: str | None = None, expected_context_column: str | None = None
) -> dict[str, Any]:
    path = _manifest_path(index_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("format") == CELL_MAJOR_FORMAT:
        return validate_cell_major_manifest(
            index_dir, payload, expected_dataset=expected_dataset, expected_context_column=expected_context_column
        )
    if (
        payload.get("schema_version") != POPULATION_INDEX_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("format") != POPULATION_INDEX_FORMAT
    ):
        raise ValueError(f"population index is absent, stale, or incomplete: {path}")
    count_dtypes = payload.get("cell_count_dtypes")
    if (
        payload.get("cell_key_dtype") != _CELL_KEY_DTYPE.name
        or payload.get("cell_key_semantics") != POPULATION_INDEX_CELL_KEY_SEMANTICS
        or payload.get("loaded_cell_key_semantics") != POPULATION_INDEX_LOADED_CELL_KEY_SEMANTICS
        or (payload.get("loaded_cell_key_dtype") != np.dtype(np.uint64).name)
        or (not isinstance(count_dtypes, list))
        or (not count_dtypes)
        or any(value not in _CELL_COUNT_DTYPES for value in count_dtypes)
        or (payload.get("cell_count_dtype") not in {*_CELL_COUNT_DTYPES, "mixed"})
    ):
        raise ValueError(f"population index has incompatible compact dtypes: {path}")
    checks = {"dataset": expected_dataset, "context_column": expected_context_column}
    for key, expected in checks.items():
        if expected is not None and payload.get(key) != expected:
            raise ValueError(f"population index {key} mismatch: {payload.get(key)!r} != {expected!r}")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("complete population index has no shards")
    unit_indices: set[int] = set()
    shard_count_dtypes: set[str] = set()
    shard_observed_maxima: list[int] = []
    for shard in shards:
        try:
            unit_index = int(shard["unit_index"])
            count_dtype = str(shard["cell_count_dtype"])
            observed_max = int(shard["cell_count_observed_max"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("population-index shard lacks compact dtype metadata") from error
        if (
            shard.get("schema_version") != POPULATION_INDEX_SCHEMA_VERSION
            or shard.get("format") != POPULATION_INDEX_FORMAT
            or shard.get("cell_key_dtype") != _CELL_KEY_DTYPE.name
            or (shard.get("cell_key_semantics") != POPULATION_INDEX_CELL_KEY_SEMANTICS)
            or (count_dtype not in _CELL_COUNT_DTYPES)
            or (not 0 <= unit_index < 2**32)
            or (unit_index in unit_indices)
            or (observed_max < 0)
            or (observed_max > np.iinfo(_CELL_COUNT_DTYPES[count_dtype]).max)
        ):
            raise ValueError("population-index shard has incompatible compact metadata")
        unit_indices.add(unit_index)
        shard_count_dtypes.add(count_dtype)
        shard_observed_maxima.append(observed_max)
        for key in ("points", "cell_keys", "cell_counts", "populations"):
            relative = shard.get(key)
            if not isinstance(relative, str) or not (index_dir / relative).is_file():
                raise ValueError(f"population-index shard artifact is missing: {relative!r}")
    expected_count_dtype = next(iter(shard_count_dtypes)) if len(shard_count_dtypes) == 1 else "mixed"
    if (
        sorted(str(value) for value in count_dtypes) != sorted(shard_count_dtypes)
        or payload.get("cell_count_dtype") != expected_count_dtype
        or payload.get("cell_count_observed_max") != max(shard_observed_maxima)
    ):
        raise ValueError(f"population index has inconsistent compact dtype summary: {path}")
    return payload
