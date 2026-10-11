"""Complete-cell PyTorch datasets, training records, and deterministic data loaders."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Keep the dataset/loader in this file and import helpers from the sibling data/ directory.
__path__ = [str(Path(__file__).with_suffix(""))]

from subcellpert.data.batching import CompleteCellCollator, prefetch_cell_batches
from subcellpert.data.populations import iter_full_gene_populations
from subcellpert.data.records import FullGeneTrainingRecord, build_full_gene_training_records
from subcellpert.data.splits import FullGeneFitSpec, partition_population, split_axes


@dataclass(frozen=True)
class CellExample:
    source: object
    source_row: int
    target: object
    target_row: int
    descriptor: np.ndarray
    condition_key: tuple
    control_identity: bool


SCFOUNDATION_ACCUMULATION_STEPS = 1
SCFOUNDATION_BATCH_SIZE = 2
SCFOUNDATION_EPOCHS = 5
SCFOUNDATION_SAMPLING_SCHEMA = "scperturbench_scfoundation_fixed_control_per_cell"


class CellTrainingExamples:
    """Compact cached random assignments; RNA is read only when a batch is used.

    Source pools obey the benchmark's existing access boundary. Within each
    permitted pool, controls are sampled uniformly over CELLS, with replacement,
    once at construction. Each eligible target yields exactly one example.
    """

    def __init__(self, records, seed, *, include_controls=True, control_descriptors=None):
        if not records:
            raise ValueError("Cell sampling requires development records")
        self.records = records
        self.controls = []
        self.assignments = []
        rng = np.random.RandomState(seed)
        sizes = []
        seen = set()
        explicit_controls = {
            (r.target.dataset, r.target.context, r.target.condition, r.target.unit_id)
            for r in records
            if r.identity_only
        }
        for record in records:
            lengths = np.array([len(p.cell_ids) for p in record.source], dtype=np.int64)
            if not len(lengths) or np.any(lengths < 1):
                raise ValueError("Every allowed source population must be nonempty")
            ends = np.cumsum(lengths)
            rows = rng.randint(0, int(ends[-1]), size=len(record.target.cell_ids))
            source_ids = np.searchsorted(ends, rows, side="right")
            source_rows = rows - np.r_[0, ends[:-1]][source_ids]
            if record.identity_only:
                source_ids = np.zeros(len(record.target.cell_ids), dtype=np.int64)
                source_rows = np.arange(len(record.target.cell_ids))
            self.assignments.append((source_ids, source_rows))
            sizes.append(len(record.target.cell_ids))
            if include_controls:
                for population in record.source:
                    key = (population.dataset, population.context, population.condition, population.unit_id)
                    if key in seen or key in explicit_controls:
                        continue
                    seen.add(key)
                    descriptor = (
                        control_descriptors[population.condition]
                        if control_descriptors is not None
                        else np.zeros_like(record.descriptor)
                    )
                    self.controls.append((population, np.asarray(descriptor, dtype=np.float32)))
        self.record_ends = np.cumsum(sizes, dtype=np.int64)
        self.target_examples = int(self.record_ends[-1])
        self.control_ends = np.cumsum([len(p.cell_ids) for p, _ in self.controls], dtype=np.int64)

    def __len__(self):
        return self.target_examples + (int(self.control_ends[-1]) if len(self.control_ends) else 0)

    def example(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.target_examples:
            i = int(np.searchsorted(self.record_ends, index, side="right"))
            row = index - (int(self.record_ends[i - 1]) if i else 0)
            record = self.records[i]
            sources, source_rows = self.assignments[i]
            return CellExample(
                record.source[int(sources[row])],
                int(source_rows[row]),
                record.target,
                row,
                record.descriptor,
                (record.context, record.condition),
                record.identity_only,
            )
        index -= self.target_examples
        i = int(np.searchsorted(self.control_ends, index, side="right"))
        row = index - (int(self.control_ends[i - 1]) if i else 0)
        population, descriptor = self.controls[i]
        return CellExample(
            population, row, population, row, descriptor, (population.context, population.condition), True
        )

    def epoch_batches(self, epoch, seed, batch_size=SCFOUNDATION_BATCH_SIZE, *, rank=0, world_size=1):
        if epoch < 0 or batch_size < 1 or len(self) < batch_size:
            raise ValueError("A shuffled drop-last epoch requires at least one full cell batch")
        if world_size < 1 or not 0 <= rank < world_size or batch_size % world_size:
            raise ValueError("Global cell batch must divide equally across valid distributed ranks")
        local_size = batch_size // world_size
        generator = torch.Generator().manual_seed(int(seed) + int(epoch))
        loader = DataLoader(
            range(len(self)), batch_size=batch_size, shuffle=True, drop_last=True, generator=generator, num_workers=0
        )
        for indices in loader:
            yield indices.numpy()[rank * local_size : (rank + 1) * local_size]


class CompleteCellDataset(Dataset):
    """An item is a complete one-cell batch; batched reads deduplicate donors."""

    def __init__(self, records, *, assignment_seed=42, control_descriptors=None, foundation_store=None):
        self.examples = CellTrainingExamples(records, assignment_seed, control_descriptors=control_descriptors)
        self.collator = CompleteCellCollator(self.examples, foundation_store=foundation_store)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.collator([index])

    def __getitems__(self, indices):
        return self.collator(indices)


def _already_collated(batch):
    return batch


def iter_training_batches(dataset, *, batch_size, seed, epoch=0, prefetch=2, rank=0, world_size=1):
    """Overlap CPU RNA preparation with owner-thread online encoding/training."""
    if world_size < 1 or not 0 <= rank < world_size or batch_size % world_size:
        raise ValueError("Global cell batch must divide equally across valid distributed ranks")
    if dataset.collator.foundation_store is not None:
        dataset.collator.foundation_store.set_training_batch_size(batch_size // world_size)
    if prefetch < 1 and world_size == 1:
        yield from make_dataloader(dataset, batch_size=batch_size, seed=seed, epoch=epoch)
    else:
        prepared = prefetch_cell_batches(
            dataset.examples.epoch_batches(epoch, seed, batch_size, rank=rank, world_size=world_size),
            dataset.collator.prepare,
            depth=prefetch,
        )
        with closing(prepared):
            for batch in prepared:
                yield dataset.collator.encode(batch)


def make_dataloader(dataset, *, batch_size, seed, epoch=0):
    """Use fixed donor assignments, seeded epoch shuffling, and drop_last=True.

    A single process owns the RNA readers and optional online GPU encoder.
    The batch sampler preserves the existing benchmark shuffle order.
    """
    if epoch < 0 or batch_size < 1 or len(dataset) < batch_size:
        raise ValueError("A shuffled drop-last epoch requires at least one full cell batch")
    if dataset.collator.foundation_store is not None:
        dataset.collator.foundation_store.set_training_batch_size(batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_already_collated,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed) + int(epoch)),
    )


def load_fit_records(
    spec: FullGeneFitSpec, *, smoke: bool = False, split: str = "train"
) -> list[FullGeneTrainingRecord]:
    contexts, transitions = split_axes(spec, split)
    contexts = contexts[:2] if smoke else contexts
    transitions = transitions[:2] if smoke else transitions
    conditions = tuple(sorted({p for t in transitions for p in (t.source_condition, t.target_condition)}))
    index_dir = spec.index_dir
    if spec.validation_split.get("axis") == "cell_view":
        index_dir = Path(spec.metadata["project_root"]) / spec.validation_split["index_views"][split]
    populations = list(
        iter_full_gene_populations(
            index_dir,
            conditions=conditions,
            contexts=contexts,
            expected_dataset=spec.dataset,
            expected_context_column=spec.context_column,
            cell_cap=None,
        )
    )
    populations = [
        selected
        for population in populations
        if (selected := partition_population(spec, population, split, cell_cap=32 if smoke else None)) is not None
    ]
    if split == "train":
        for population in populations:
            spec.validate_fit_population(population)
    records = build_full_gene_training_records(
        populations, list(transitions), spec.descriptors.condition_vectors, source_pool=spec.source_pool
    )
    if split == "train" and spec.validation_split.get("axis") == "cell_view":
        for population in populations:
            if population.condition in spec.validation_split["source_conditions"]:
                records.append(
                    FullGeneTrainingRecord(
                        transition_id=f"identity/{population.condition}",
                        condition=population.condition,
                        context=population.context,
                        target_unit=population.unit_id,
                        descriptor=spec.descriptors.condition_vectors[population.condition],
                        source=(population,),
                        target=population,
                        identity_only=True,
                    ).validated()
                )
    return records
