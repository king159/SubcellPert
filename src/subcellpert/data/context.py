"""Cellular-context task/split readers and known-condition descriptor access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from subcellpert.data.intervention import (
    ConditionDescriptorStore,
    load_intervention_descriptors,
    load_intervention_split,
    load_intervention_task,
)


CONTEXT_OOD_PROTOCOLS = frozenset({"context_ood_oracle", "context_ood_joint"})
CONTEXT_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ContextFitGroup:
    """One independently fitted model shared by one or more response folds."""

    fit_group_id: str
    fold_ids: tuple[str, ...]
    transition_ids: tuple[str, ...]
    set2contexts: dict[str, tuple[str, ...]]
    evaluation_context_group: str

    @property
    def training_contexts(self) -> tuple[str, ...]:
        return self.set2contexts["train"]

    @property
    def validation_contexts(self) -> tuple[str, ...]:
        return self.set2contexts["val"]

    @property
    def held_out_contexts(self) -> tuple[str, ...]:
        return self.set2contexts["test"]


@dataclass(frozen=True)
class ContextFitGroupDataAccess:
    training_contexts: tuple[str, ...]
    validation_contexts: tuple[str, ...]
    held_out_contexts: tuple[str, ...]
    transition_ids: tuple[str, ...]
    source_condition: str
    target_conditions: tuple[str, ...]
    training_loaded_conditions: tuple[str, ...]


@dataclass(frozen=True)
class ContextFold:
    fold_id: str
    seed: int
    transition_id: str
    source_condition: str
    target_condition: str
    set2contexts: dict[str, tuple[str, ...]]
    test_contexts: tuple[str, ...]
    test_subgroup: dict[str, tuple[str, ...]]

    @property
    def training_contexts(self) -> tuple[str, ...]:
        return self.set2contexts["train"]

    @property
    def validation_contexts(self) -> tuple[str, ...]:
        return self.set2contexts["val"]

    @property
    def held_out_context(self) -> str:
        if len(self.test_contexts) != 1:
            raise ValueError(f"fold {self.fold_id} must hold out exactly one context")
        return self.test_contexts[0]

    @property
    def test_conditions(self) -> tuple[str, ...]:
        return (self.target_condition,)


@dataclass(frozen=True)
class ContextOODTask:
    task_id: str
    dataset: str
    track: str
    context_key: str
    cache_context_column: str
    cache_dir: Path
    split_path: Path
    source_pool: str
    descriptor_task_path: Path | None
    descriptor_dataset: str | None
    descriptor_source: dict[str, Any]
    manifest_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContextSplit:
    split_id: str
    dataset: str
    context_key: str
    contexts: tuple[str, ...]
    condition_components: dict[str, tuple[str, ...]]
    transitions: dict[str, ContextTransition]
    transition_contexts: dict[str, tuple[str, ...]]
    folds: dict[str, ContextFold]
    fit_groups: dict[str, ContextFitGroup]
    metadata: dict[str, Any]

    def fold(self, fold_id: str) -> ContextFold:
        try:
            return self.folds[fold_id]
        except KeyError as error:
            raise KeyError(f"unknown fold {fold_id!r} for split {self.split_id}") from error

    def fit_group(self, fit_group_id: str) -> ContextFitGroup:
        try:
            return self.fit_groups[fit_group_id]
        except KeyError as error:
            raise KeyError(f"unknown fit group {fit_group_id!r} for split {self.split_id}") from error

    def fit_group_for_fold(self, fold_id: str) -> ContextFitGroup:
        matches = [group for group in self.fit_groups.values() if fold_id in group.fold_ids]
        if len(matches) != 1:
            raise ValueError(f"fold {fold_id!r} belongs to {len(matches)} fit groups in {self.split_id}")
        return matches[0]


@dataclass(frozen=True)
class ContextTransition:
    transition_id: str
    source_condition: str
    target_condition: str


@dataclass(frozen=True)
class KnownConditionDescriptorStore(ConditionDescriptorStore):
    """Backward-compatible name for a known-condition descriptor store."""


def _unique_strings(values: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{location} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{location} contains duplicates")
    return tuple(values)


def context_fit_group_data_access(split: ContextSplit, fit_group: ContextFitGroup) -> ContextFitGroupDataAccess:
    """Derive the joint training boundary for one independently fitted model."""
    transitions = [split.transitions[value] for value in fit_group.transition_ids]
    source_conditions = {transition.source_condition for transition in transitions}
    if len(source_conditions) != 1:
        raise ValueError(f"fit group {fit_group.fit_group_id} must have one shared source condition")
    source_condition = next(iter(source_conditions))
    target_conditions = tuple(sorted(transition.target_condition for transition in transitions))
    return ContextFitGroupDataAccess(
        training_contexts=tuple(sorted(set(fit_group.training_contexts).union(fit_group.validation_contexts))),
        validation_contexts=fit_group.validation_contexts,
        held_out_contexts=fit_group.held_out_contexts,
        transition_ids=fit_group.transition_ids,
        source_condition=source_condition,
        target_conditions=target_conditions,
        training_loaded_conditions=tuple(sorted({source_condition, *target_conditions})),
    )


def load_context_descriptor_source(task: ContextOODTask) -> tuple[Any, Any, Any]:
    """Load and cross-check the immutable known-condition descriptor.

    Cellular or patient context is the OOD variable in this track; intervention
    identity is observed in development contexts.  The formal tasks therefore use
    deterministic condition one-hot vectors, while the legacy artifact-binding
    branch remains readable for archived task manifests.
    """
    context_split = load_context_split(task.split_path)
    if task.descriptor_source.get("provider") == "known_condition_one_hot":
        conditions = sorted(context_split.condition_components)
        dimension = len(conditions)
        vectors = {condition: np.eye(dimension, dtype=np.float32)[index] for index, condition in enumerate(conditions)}
        {
            "schema_version": 1,
            "provider": "known_condition_one_hot",
            "conditions": conditions,
            "condition_components": {
                condition: list(context_split.condition_components[condition]) for condition in conditions
            },
            "target_response_data_used": False,
            "semantics": "condition identity is observed in development contexts; held-out variable is context",
        }
        {"conditions": conditions, "vectors": [vectors[condition].tolist() for condition in conditions]}
        store = KnownConditionDescriptorStore(
            condition_vectors=vectors,
            component_vectors=vectors,
            component_texts={condition: condition for condition in conditions},
            dimension=dimension,
            provider="known_condition_one_hot",
            artifact_path=task.split_path,
            metadata_path=task.manifest_path,
        )
        return (task, context_split, store)
    assert task.descriptor_task_path is not None and task.descriptor_dataset is not None
    project_root = task.descriptor_task_path.parents[2]
    descriptor_task = load_intervention_task(
        task.descriptor_task_path, task.descriptor_dataset, project_root=project_root
    )
    descriptor_split = load_intervention_split(descriptor_task.split_path)
    if task.dataset != descriptor_task.dataset:
        raise ValueError("context task and descriptor source must refer to the same dataset")
    if context_split.condition_components != descriptor_split.condition_components:
        raise ValueError("context split conditions disagree with the frozen descriptor source")
    store = load_intervention_descriptors(
        descriptor_task, descriptor_split, task_manifest_path=task.descriptor_task_path
    )
    return (descriptor_task, descriptor_split, store)


def load_context_split(path: Path) -> ContextSplit:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("split_axis") != "cellular_context":
        raise ValueError(f"{path} must be a schema-version-1 cellular-context split")
    contexts = _unique_strings(raw.get("contexts"), location="contexts")
    component_rows = raw.get("condition_components")
    if not isinstance(component_rows, dict) or not component_rows:
        raise ValueError(f"{path} must define condition_components")
    components = {
        str(condition): _unique_strings(values, location=f"condition_components.{condition}")
        for condition, values in component_rows.items()
    }
    transition_rows = raw.get("transitions")
    if not isinstance(transition_rows, list):
        raise ValueError(f"{path} transitions must be a list")
    transitions: dict[str, ContextTransition] = {}
    for index, row in enumerate(transition_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} transition {index} is not a mapping")
        transition_id = row.get("id")
        if not isinstance(transition_id, str) or not transition_id or transition_id in transitions:
            raise ValueError(f"{path} transition {index} has an invalid id")
        transitions[transition_id] = ContextTransition(
            transition_id=transition_id,
            source_condition=str(row["source_condition"]),
            target_condition=str(row["target_condition"]),
        )
    raw_transition_contexts = raw.get("transition_contexts")
    if raw_transition_contexts is None:
        transition_contexts = dict.fromkeys(transitions, contexts)
    else:
        if not isinstance(raw_transition_contexts, dict) or set(raw_transition_contexts) != set(transitions):
            raise ValueError(f"{path} transition_contexts must define every transition exactly once")
        transition_contexts = {
            transition_id: _unique_strings(
                raw_transition_contexts[transition_id], location=f"transition_contexts.{transition_id}"
            )
            for transition_id in transitions
        }
        for transition_id, values in transition_contexts.items():
            if len(values) < 2:
                raise ValueError(f"transition {transition_id} requires at least two contexts")
            if unknown := set(values).difference(contexts):
                raise ValueError(f"transition {transition_id} has unknown contexts: {sorted(unknown)}")
    fold_rows = raw.get("folds")
    if not isinstance(fold_rows, list):
        raise ValueError(f"{path} folds must be a list")
    folds: dict[str, ContextFold] = {}
    for index, row in enumerate(fold_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} fold {index} is not a mapping")
        fold_id = row.get("id")
        if not isinstance(fold_id, str) or not fold_id or fold_id in folds:
            raise ValueError(f"{path} fold {index} has an invalid id")
        blocks = row.get("set2contexts")
        if not isinstance(blocks, dict) or set(blocks) != set(CONTEXT_SPLIT_NAMES):
            raise ValueError(f"fold {fold_id} set2contexts must define train, val, and test")
        subgroup = row.get("test_subgroup")
        if not isinstance(subgroup, dict):
            raise ValueError(f"fold {fold_id} must define test_subgroup")
        folds[fold_id] = ContextFold(
            fold_id=fold_id,
            seed=int(row.get("seed", 0)),
            transition_id=str(row["transition"]),
            source_condition=str(row["source_condition"]),
            target_condition=str(row["target_condition"]),
            set2contexts={
                name: _unique_strings(blocks[name], location=f"folds.{fold_id}.{name}") for name in CONTEXT_SPLIT_NAMES
            },
            test_contexts=_unique_strings(row.get("test_contexts"), location=f"folds.{fold_id}.test_contexts"),
            test_subgroup={
                str(name): _unique_strings(values, location=f"folds.{fold_id}.test_subgroup.{name}")
                for name, values in subgroup.items()
            },
        )
    fit_group_rows = raw.get("fit_groups")
    fit_groups: dict[str, ContextFitGroup] = {}
    if fit_group_rows is None:
        fit_group_rows = [
            {
                "id": fold.fold_id,
                "folds": [fold.fold_id],
                "transitions": [fold.transition_id],
                "set2contexts": {name: list(values) for name, values in fold.set2contexts.items()},
                "evaluation_context_group": fold.held_out_context,
            }
            for fold in folds.values()
        ]
    if not isinstance(fit_group_rows, list):
        raise ValueError(f"{path} fit_groups must be a list")
    for index, row in enumerate(fit_group_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} fit group {index} is not a mapping")
        fit_group_id = row.get("id")
        if not isinstance(fit_group_id, str) or not fit_group_id or fit_group_id in fit_groups:
            raise ValueError(f"{path} fit group {index} has an invalid id")
        blocks = row.get("set2contexts")
        if not isinstance(blocks, dict) or set(blocks) != set(CONTEXT_SPLIT_NAMES):
            raise ValueError(f"fit group {fit_group_id} set2contexts must define train, val, and test")
        fit_groups[fit_group_id] = ContextFitGroup(
            fit_group_id=fit_group_id,
            fold_ids=_unique_strings(row.get("folds"), location=f"fit_groups.{fit_group_id}.folds"),
            transition_ids=_unique_strings(row.get("transitions"), location=f"fit_groups.{fit_group_id}.transitions"),
            set2contexts={
                name: _unique_strings(blocks[name], location=f"fit_groups.{fit_group_id}.{name}")
                for name in CONTEXT_SPLIT_NAMES
            },
            evaluation_context_group=str(row.get("evaluation_context_group", "")),
        )
    split = ContextSplit(
        split_id=str(raw["id"]),
        dataset=str(raw["dataset"]),
        context_key=str(raw["context_key"]),
        contexts=contexts,
        condition_components=components,
        transitions=transitions,
        transition_contexts=transition_contexts,
        folds=folds,
        fit_groups=fit_groups,
        metadata={
            key: value
            for key, value in raw.items()
            if key not in {"condition_components", "transitions", "transition_contexts", "folds", "fit_groups"}
        },
    )
    validate_context_split(split)
    return split


def load_context_task(path: Path, dataset: str, *, project_root: Path) -> ContextOODTask:
    raw = yaml.safe_load(path.read_text())
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 2
        or raw.get("generalization_axis") != "cellular_context"
    ):
        raise ValueError(f"{path} must be a schema-version-2 cellular-context task manifest")
    declared_protocols = set(raw.get("protocols", []))
    if declared_protocols != CONTEXT_OOD_PROTOCOLS:
        raise ValueError(f"context task must declare exactly {sorted(CONTEXT_OOD_PROTOCOLS)}")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise KeyError(f"dataset {dataset!r} is not defined in {path}")
    row = datasets[dataset]
    if not isinstance(row, dict):
        raise ValueError(f"dataset {dataset!r} context task must be a mapping")
    split_path = project_root / str(row["split_manifest"])
    split = load_context_split(split_path)
    if split.dataset != dataset:
        raise ValueError(f"task dataset {dataset!r} does not match split dataset {split.dataset!r}")
    cache_context_column = str(row.get("cache_context_column", "cell_type"))
    if split.context_key != "cell_type" and "cache_context_column" not in row:
        raise ValueError("non-cell-type context tasks must declare cache_context_column")
    if not cache_context_column:
        raise ValueError("cache_context_column must be nonempty")
    source_pool = str(row.get("source_pool", "same_unit_then_pooled"))
    if source_pool not in {"same_unit_then_pooled", "pooled_reference"}:
        raise ValueError(f"unsupported source_pool={source_pool!r}")
    descriptor = row.get("descriptor_source")
    if not isinstance(descriptor, dict):
        raise ValueError("context task must bind a frozen descriptor_source")
    provider = descriptor.get("provider")
    if provider == "known_condition_one_hot":
        if set(descriptor) != {"provider"}:
            raise ValueError("known_condition_one_hot takes no response-derived configuration")
        descriptor_task_path = None
        descriptor_dataset = None
    else:
        if not isinstance(descriptor.get("task_manifest"), str) or not isinstance(descriptor.get("dataset"), str):
            raise ValueError(
                "descriptor_source must be known_condition_one_hot or bind an intervention task_manifest and dataset"
            )
        descriptor_task_path = project_root / str(descriptor["task_manifest"])
        descriptor_dataset = str(descriptor["dataset"])
    return ContextOODTask(
        task_id=str(raw["id"]),
        dataset=dataset,
        track=str(row["track"]),
        context_key=split.context_key,
        cache_context_column=cache_context_column,
        cache_dir=project_root / str(row["cache_dir"]),
        split_path=split_path,
        source_pool=source_pool,
        descriptor_task_path=descriptor_task_path,
        descriptor_dataset=descriptor_dataset,
        descriptor_source=descriptor,
        manifest_path=path,
        metadata=row,
    )


def validate_context_split(split: ContextSplit) -> None:
    """Reject context leakage, missing transitions, and incomplete supported grids."""
    context_universe = set(split.contexts)
    if len(context_universe) < 2:
        raise ValueError("context OOD requires at least two cellular contexts")
    if not split.context_key:
        raise ValueError("context OOD requires a named context_key")
    if not split.transitions or not split.folds:
        raise ValueError("context split requires transitions and folds")
    conditions = set(split.condition_components)
    for transition in split.transitions.values():
        if transition.source_condition == transition.target_condition:
            raise ValueError(f"transition {transition.transition_id} has identical source and target")
        if {transition.source_condition, transition.target_condition}.difference(conditions):
            raise ValueError(f"transition {transition.transition_id} uses an unknown condition")
    observed_pairs: set[tuple[str, str]] = set()
    for fold in split.folds.values():
        if fold.transition_id not in split.transitions:
            raise ValueError(f"fold {fold.fold_id} references unknown transition {fold.transition_id!r}")
        transition = split.transitions[fold.transition_id]
        if (
            fold.source_condition != transition.source_condition
            or fold.target_condition != transition.target_condition
        ):
            raise ValueError(f"fold {fold.fold_id} disagrees with its transition")
        transition_universe = set(split.transition_contexts[fold.transition_id])
        blocks = {name: set(fold.set2contexts[name]) for name in CONTEXT_SPLIT_NAMES}
        if any(
            (blocks[left] & blocks[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test")))
        ):
            raise ValueError(f"fold {fold.fold_id} assigns a context twice")
        if set().union(*blocks.values()) != transition_universe:
            raise ValueError(f"fold {fold.fold_id} does not partition its transition contexts")
        if set(fold.test_contexts) != blocks["test"] or len(fold.test_contexts) != 1:
            raise ValueError(f"fold {fold.fold_id} must expose one matching test context")
        if set(fold.test_subgroup) != {"unseen_context"} or set(fold.test_subgroup["unseen_context"]) != {
            fold.target_condition
        }:
            raise ValueError(f"fold {fold.fold_id} must label its target as unseen_context")
        pair = (fold.transition_id, fold.held_out_context)
        if pair in observed_pairs:
            raise ValueError(f"duplicate transition/context fold: {pair}")
        observed_pairs.add(pair)
    complete_pairs = {
        (transition_id, context)
        for transition_id, contexts in split.transition_contexts.items()
        for context in contexts
    }
    unsupported_rows = split.metadata.get("unsupported_response_pairs", [])
    if not isinstance(unsupported_rows, list):
        raise ValueError("unsupported_response_pairs must be a list")
    unsupported_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(unsupported_rows):
        if not isinstance(row, dict):
            raise ValueError(f"unsupported_response_pairs[{index}] must be a mapping")
        pair = (str(row.get("transition", "")), str(row.get("context", "")))
        if pair not in complete_pairs:
            raise ValueError(f"unsupported response pair is outside the declared grid: {pair}")
        if pair in unsupported_pairs:
            raise ValueError(f"duplicate unsupported response pair: {pair}")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"unsupported response pair {pair} requires a reason")
        unsupported_pairs.add(pair)
    expected_pairs = complete_pairs.difference(unsupported_pairs)
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(observed_pairs))
        extra = sorted(observed_pairs.difference(expected_pairs))
        raise ValueError(f"context fold grid is incomplete; missing={missing[:3]}, extra={extra[:3]}")
    grouped_folds: set[str] = set()
    for group in split.fit_groups.values():
        if not group.fold_ids:
            raise ValueError(f"fit group {group.fit_group_id} has no response folds")
        if unknown := set(group.fold_ids).difference(split.folds):
            raise ValueError(f"fit group {group.fit_group_id} references unknown folds: {sorted(unknown)}")
        if overlap := grouped_folds.intersection(group.fold_ids):
            raise ValueError(f"response folds occur in multiple fit groups: {sorted(overlap)}")
        grouped_folds.update(group.fold_ids)
        folds = [split.folds[fold_id] for fold_id in group.fold_ids]
        if set(group.transition_ids) != {fold.transition_id for fold in folds}:
            raise ValueError(f"fit group {group.fit_group_id} transition list disagrees with its folds")
        if set(group.held_out_contexts) != {fold.held_out_context for fold in folds}:
            raise ValueError(f"fit group {group.fit_group_id} held-out contexts disagree with its folds")
        blocks = {name: set(group.set2contexts[name]) for name in CONTEXT_SPLIT_NAMES}
        if any(
            (blocks[left] & blocks[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test")))
        ):
            raise ValueError(f"fit group {group.fit_group_id} assigns a context twice")
        for transition_id in group.transition_ids:
            transition_universe = set(split.transition_contexts[transition_id])
            if blocks["train"] | blocks["val"] | blocks["test"] != transition_universe:
                raise ValueError(
                    f"fit group {group.fit_group_id} does not partition transition {transition_id} contexts"
                )
        expected_group_pairs = {
            (transition_id, context)
            for transition_id in group.transition_ids
            for context in group.held_out_contexts
            if context in split.transition_contexts[transition_id]
        }
        observed_group_pairs = {(fold.transition_id, fold.held_out_context) for fold in folds}
        if observed_group_pairs != expected_group_pairs:
            raise ValueError(f"fit group {group.fit_group_id} does not contain its complete response grid")
        if not group.evaluation_context_group:
            raise ValueError(f"fit group {group.fit_group_id} has no evaluation context group")
        context_fit_group_data_access(split, group)
    if grouped_folds != set(split.folds):
        missing = sorted(set(split.folds).difference(grouped_folds))
        raise ValueError(f"response folds are absent from fit groups: {missing[:3]}")
