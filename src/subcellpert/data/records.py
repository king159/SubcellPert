"""Source/target training records and complete-gene transition contracts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from subcellpert.data.populations import FullGenePopulation


@dataclass(frozen=True)
class FullGeneTrainingRecord:
    transition_id: str
    condition: str
    context: str
    target_unit: str
    descriptor: np.ndarray
    source: tuple[FullGenePopulation, ...]
    target: FullGenePopulation
    foundation_context: np.ndarray | None = None
    source_foundation_contexts: tuple[np.ndarray, ...] | None = None
    identity_only: bool = False

    def validated(self) -> FullGeneTrainingRecord:
        descriptor = np.asarray(self.descriptor, dtype=np.float32)
        if descriptor.ndim != 1 or not len(descriptor) or (not np.isfinite(descriptor).all()):
            raise ValueError("training descriptor must be a finite vector")
        if not self.source:
            raise ValueError("full-gene training record requires a source population")
        genes = self.target.genes
        for population in self.source:
            if population.genes != genes:
                raise ValueError("source and target populations have different gene vocabularies")
        foundation = (
            np.asarray(self.foundation_context, dtype=np.float32) if self.foundation_context is not None else None
        )
        if foundation is not None and (foundation.ndim != 1 or not np.isfinite(foundation).all()):
            raise ValueError("foundation context must be a finite vector")
        source_foundations = self.source_foundation_contexts
        if source_foundations is not None:
            source_foundations = tuple(np.asarray(v, dtype=np.float32) for v in source_foundations)
            if len(source_foundations) != len(self.source) or any(
                v.ndim != 1 or not len(v) or (not np.isfinite(v).all()) for v in source_foundations
            ):
                raise ValueError("source foundation vectors must align with source populations")
        return FullGeneTrainingRecord(
            transition_id=self.transition_id,
            condition=self.condition,
            context=self.context,
            target_unit=self.target_unit,
            descriptor=descriptor,
            source=self.source,
            target=self.target,
            foundation_context=foundation,
            source_foundation_contexts=source_foundations,
            identity_only=self.identity_only,
        )


@dataclass(frozen=True)
class FullGeneTransition:
    transition_id: str
    source_condition: str
    target_condition: str


def build_full_gene_training_records(
    populations: list[FullGenePopulation],
    transitions: list[FullGeneTransition],
    descriptors: dict[str, np.ndarray],
    *,
    source_pool: str,
) -> list[FullGeneTrainingRecord]:
    """Bind complete source/target populations without opening held-out contexts."""
    if source_pool not in {"same_unit_then_pooled", "pooled_reference"}:
        raise ValueError(f"unknown source_pool={source_pool!r}")
    by_context_condition: dict[tuple[str, str], list[FullGenePopulation]] = defaultdict(list)
    for population in populations:
        by_context_condition[population.context, population.condition].append(population.validated())
    records = []
    contexts = sorted({population.context for population in populations})
    for transition in transitions:
        if transition.target_condition not in descriptors:
            raise ValueError(f"missing descriptor for {transition.target_condition!r}")
        for context in contexts:
            sources = sorted(
                by_context_condition.get((context, transition.source_condition), []), key=lambda value: value.unit_id
            )
            targets = sorted(
                by_context_condition.get((context, transition.target_condition), []), key=lambda value: value.unit_id
            )
            for target in targets:
                selected = sources
                if source_pool == "same_unit_then_pooled":
                    same_unit = [value for value in sources if value.unit_id == target.unit_id]
                    selected = same_unit or sources
                if not selected:
                    raise ValueError(f"no source population for {transition.transition_id}/{context}/{target.unit_id}")
                records.append(
                    FullGeneTrainingRecord(
                        transition_id=transition.transition_id,
                        condition=transition.target_condition,
                        context=context,
                        target_unit=target.unit_id,
                        descriptor=descriptors[transition.target_condition],
                        source=tuple(selected),
                        target=target,
                    ).validated()
                )
    if not records:
        raise ValueError("no full-gene development records were constructed")
    return records
