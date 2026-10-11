"""Sparse counts and lazy RNA arrays backed by NumPy, CSR, or HDF5 storage."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


class PackedCoordinates:
    """Logical concatenation of entry slices, with no RNA array concatenation."""

    def __init__(self, source, physical_starts, lengths):
        self.source = source
        self.lengths = np.asarray(lengths)
        if self.lengths.dtype.kind not in "iu" or (self.lengths.size and self.lengths.min() <= 0):
            raise ValueError("packed coordinate lengths must be integers")
        index_dtype = (
            np.uint32
            if max(len(source), int(self.lengths.sum(dtype=np.uint64))) <= np.iinfo(np.uint32).max
            else np.uint64
        )
        raw_starts = np.asarray(physical_starts)
        if raw_starts.dtype.kind not in "iu" or (
            raw_starts.size and (raw_starts.min() < 0 or raw_starts.max() > len(source))
        ):
            raise ValueError("packed coordinate offsets must be nonnegative integers")
        self.physical_starts = raw_starts.astype(index_dtype, copy=False)
        if self.physical_starts.ndim != 1 or self.lengths.shape != self.physical_starts.shape:
            raise ValueError("packed coordinate offsets and lengths must align")
        if source.shape[1:] != (2,) or source.dtype != np.dtype("float32"):
            raise ValueError("invalid packed coordinate source or entry range")
        for first in range(0, len(self.lengths), 2000000):
            ends = self.physical_starts[first : first + 2000000].astype(np.uint64) + self.lengths[
                first : first + 2000000
            ].astype(np.uint64)
            if ends.max() > len(source):
                raise ValueError("packed coordinate entry exceeds source bounds")
        self.ends = np.cumsum(self.lengths, dtype=index_dtype)
        self.starts = self.ends - self.lengths
        self.shape = (int(self.ends[-1]) if len(self.ends) else 0, 2)
        self.dtype = np.dtype("float32")
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        columns = None
        if isinstance(index, tuple):
            index, columns = index
        if isinstance(index, slice):
            first, stop, step = index.indices(len(self))
            if step == 1 and stop > first:
                begin, end = np.searchsorted(self.ends, [first, stop - 1], side="right")
                shift = self.physical_starts[begin : end + 1].astype(np.int64) - self.starts[begin : end + 1].astype(
                    np.int64
                )
                if np.all(shift == shift[0]):
                    value = self.source[first + int(shift[0]) : stop + int(shift[0])]
                    return value if columns is None else value[..., columns]
        indices = (
            np.arange(*index.indices(len(self)), dtype=np.int64) if isinstance(index, slice) else np.asarray(index)
        )
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
        indices = np.where(indices < 0, indices + len(self), indices)
        if np.any((indices < 0) | (indices >= len(self))):
            raise IndexError("coordinate index out of range")
        entries = np.searchsorted(self.ends, indices, side="right")
        physical = self.physical_starts[entries] + indices - self.starts[entries]
        value = self.source[physical]
        return value if columns is None else value[..., columns]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def subset_entries(self, selected):
        return PackedCoordinates(self.source, self.physical_starts[selected], self.lengths[selected])


class SparseCounts:
    """Keep zero-inclusive counts sparse; only requested cell rows become dense."""

    def __init__(self, matrix):
        self.matrix = sparse.csr_matrix(matrix)
        self.shape = self.matrix.shape
        self.ndim = 2
        self.dtype = self.matrix.dtype
        self.size = int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        value = self.matrix[index]
        if sparse.issparse(value):
            result = value.toarray().astype(np.int64, copy=False)
            if isinstance(index, (int, np.integer)):
                return result[0]
            return result
        return np.asarray(value).reshape(-1) if np.ndim(value) else value

    def subset(self, rows):
        return SparseCounts(self.matrix[rows])

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.matrix.toarray(), dtype=np.int64 if dtype is None else dtype)

    def sum(self, axis=None):
        value = self.matrix.sum(axis=axis)
        return np.asarray(value).ravel() if axis is not None else value

    def mean(self, axis=None):
        value = self.matrix.mean(axis=axis)
        return np.asarray(value).ravel() if axis is not None else value

    def prevalence(self):
        return np.asarray((self.matrix > 0).mean(axis=0)).ravel()


class H5Array:
    """Reopen on demand, avoiding persistent file descriptors across populations."""

    def __init__(self, path, key):
        self.path, self.key = (Path(path), key)
        with h5py.File(path, "r") as f:
            self.shape, self.dtype = (f[key].shape, f[key].dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as f:
            if isinstance(index, tuple) and all(isinstance(i, np.ndarray) for i in index):
                rows, columns = np.broadcast_arrays(*index)
                result = np.empty(rows.shape, dtype=self.dtype)
                blocks = rows // 128
                for block in np.unique(blocks):
                    selected = blocks == block
                    first = int(block) * 128
                    result[selected] = f[self.key][first : first + 128][rows[selected] - first, columns[selected]]
                return result
            if isinstance(index, np.ndarray):
                shape = index.shape
                unique, inverse = np.unique(index, return_inverse=True)
                result = np.empty((len(unique), *self.shape[1:]), dtype=self.dtype)
                chunk_rows = f[self.key].chunks[0] if f[self.key].chunks else 128
                blocks = unique // chunk_rows
                for block in np.unique(blocks):
                    select = blocks == block
                    start = int(block) * chunk_rows
                    result[select] = f[self.key][start : start + chunk_rows][unique[select] - start]
                return result[inverse].reshape((*shape, *self.shape[1:]))
            return f[self.key][index]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def sum(self, axis=None):
        if axis not in (None, 0, 1):
            raise ValueError("unsupported reduction axis")
        if axis == 1:
            return np.concatenate([self[first : first + 128].sum(axis=1) for first in range(0, len(self), 128)])
        result = 0
        for first in range(0, len(self), 128):
            result = result + self[first : first + 128].sum(axis=axis)
        return result

    def all(self):
        return all(self[first : first + 128].all() for first in range(0, len(self), 128))


MOLECULE_BATCH_SIZE = 262144


class GeneBackedCounts(SparseCounts):
    """Read full-gene cell batches directly from sorted, memory-mapped gene runs.

    Only gene offsets and the selected cell keys are resident. No full CSR or
    dense matrix is assembled. Batched binary searches support arbitrary rows.
    """

    def __init__(self, keys, values, starts, lengths, row_keys, *, full_cell_universe=False):
        self.keys, self.values = (keys, values)
        self.starts, self.lengths = (np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64))
        self.row_keys = np.asarray(row_keys, dtype=np.uint32)
        self.shape, self.ndim, self.dtype = ((len(row_keys), len(starts)), 2, np.dtype("int64"))
        self.size = int(np.prod(self.shape))
        self.full_cell_universe = full_cell_universe
        self._sorted_rows = len(self.row_keys) < 2 or np.all(self.row_keys[1:] > self.row_keys[:-1])
        self._sums = self._nonempty = None

    def _lookup(self, rows, genes):
        rows, genes = np.broadcast_arrays(rows, genes)
        wanted = self.row_keys[rows]
        lower, end = self._lower_bounds(wanted, genes)
        if not len(self.keys):
            return np.zeros(rows.shape, dtype=np.int64)
        positions = np.minimum(lower, len(self.keys) - 1)
        found = (lower < end) & (self.keys[positions] == wanted)
        return np.where(found, self.values[positions], 0).astype(np.int64)

    def _lower_bounds(self, wanted, genes):
        wanted, genes = np.broadcast_arrays(wanted, genes)
        lower = self.starts[genes].copy()
        end = self.starts[genes] + self.lengths[genes]
        upper = end.copy()
        if not len(self.keys):
            return (lower, end)
        while np.any(lower < upper):
            middle = (lower + upper) // 2
            before = self.keys[np.minimum(middle, len(self.keys) - 1)] < wanted
            active = lower < upper
            lower = np.where(active & before, middle + 1, lower)
            upper = np.where(active & ~before, middle, upper)
        return (lower, end)

    def _contiguous_rows(self, first, stop):
        bounds, _ = self._lower_bounds(
            np.asarray([int(self.row_keys[first]), int(self.row_keys[stop - 1]) + 1])[:, None],
            np.arange(self.shape[1])[None, :],
        )
        lengths = bounds[1] - bounds[0]
        starts = np.cumsum(lengths) - lengths
        genes = np.repeat(np.arange(self.shape[1]), lengths)
        positions = np.repeat(bounds[0] - starts, lengths) + np.arange(int(lengths.sum()))
        rows = np.searchsorted(self.row_keys, self.keys[positions]) - first
        result = np.zeros((stop - first, self.shape[1]), dtype=np.int64)
        result[rows, genes] = self.values[positions]
        return result

    def __getitem__(self, index):
        if isinstance(index, tuple):
            rows, columns = index
            if isinstance(rows, np.ndarray) and isinstance(columns, np.ndarray):
                shape = np.broadcast_shapes(rows.shape, columns.shape)
                rows, columns = (a.ravel() for a in np.broadcast_arrays(rows, columns))
                result = np.empty(len(rows), dtype=np.int64)
                for first in range(0, len(rows), 262144):
                    result[first : first + 262144] = self._lookup(
                        rows[first : first + 262144], columns[first : first + 262144]
                    )
                return result.reshape(shape)
            return self[rows][..., columns]
        scalar = isinstance(index, (int, np.integer))
        rows = np.arange(*index.indices(len(self))) if isinstance(index, slice) else np.atleast_1d(index)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        result = np.empty((len(rows), self.shape[1]), dtype=np.int64)
        genes = np.arange(self.shape[1])
        for first in range(0, len(rows), 128):
            selected = rows[first : first + 128]
            if (
                self.full_cell_universe
                and self._sorted_rows
                and (len(selected) > 1)
                and np.all(np.diff(selected) == 1)
            ):
                values = self._contiguous_rows(int(selected[0]), int(selected[-1]) + 1)
            else:
                values = self._lookup(selected[:, None], genes[None, :])
            result[first : first + len(selected)] = values
        return result[0] if scalar else result

    def subset(self, rows):
        return GeneBackedCounts(
            self.keys,
            self.values,
            self.starts,
            self.lengths,
            self.row_keys[rows],
            full_cell_universe=self.full_cell_universe and len(rows) == len(self),
        )

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def validate_cloud(self, cloud):
        sums = np.zeros(self.shape[1], dtype=np.int64)
        nonempty = np.zeros(self.shape[1], dtype=np.int64)
        for gene in range(self.shape[1]):
            start, length = (self.starts[gene], self.lengths[gene])
            keys, values = (self.keys[start : start + length], self.values[start : start + length])
            first, n = (cloud.gene_entry_starts[gene], cloud.gene_entry_lengths[gene])
            observed_keys = self.row_keys[cloud.entry_cell_indices[first : first + n]]
            observed_lengths = cloud.entry_lengths[first : first + n]
            if not self.full_cell_universe:
                positions = np.searchsorted(keys, self.row_keys)
                valid = positions < len(keys)
                if len(keys):
                    valid &= keys[np.minimum(positions, len(keys) - 1)] == self.row_keys
                positions = np.sort(positions[valid])
                keys, values = (keys[positions], values[positions])
            if not np.array_equal(observed_keys, keys) or not np.array_equal(observed_lengths, values):
                raise ValueError("full-gene counts and cell-gene point lengths disagree")
            sums[gene], nonempty[gene] = (values.sum(dtype=np.int64), len(values))
        self._sums, self._nonempty = (sums, nonempty)

    def _statistics(self):
        if self._sums is None:
            sums = np.zeros(self.shape[1], dtype=np.int64)
            nonempty = sums.copy()
            for first in range(0, len(self), 128):
                values = self[first : first + 128]
                sums += values.sum(axis=0)
                nonempty += (values > 0).sum(axis=0)
            self._sums, self._nonempty = (sums, nonempty)
        return (self._sums, self._nonempty)

    def sum(self, axis=None):
        if axis == 1:
            return np.concatenate([self[first : first + 128].sum(axis=1) for first in range(0, len(self), 128)])
        sums = self._statistics()[0]
        return sums if axis == 0 else sums.sum()

    def mean(self, axis=None):
        if axis == 0:
            return self.sum(axis=0) / len(self)
        if axis == 1:
            return self.sum(axis=1) / self.shape[1]
        return self.sum() / self.size

    def prevalence(self):
        return self._statistics()[1] / len(self)
