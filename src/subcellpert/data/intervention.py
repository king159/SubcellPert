"""Intervention task/split readers and condition descriptor contracts."""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class ConditionDescriptorStore:
    """Validated vectors with frozen provenance fields.

    The dataclass is shallowly frozen; contained arrays and mappings retain their
    normal mutability for compatibility with existing loaders.
    """

    condition_vectors: dict[str, np.ndarray]
    component_vectors: dict[str, np.ndarray]
    component_texts: dict[str, str]
    dimension: int
    provider: str
    artifact_path: Path
    metadata_path: Path

    @property
    def checkpoint_binding(self) -> dict[str, Any]:
        return {"intervention_descriptor_provider": self.provider, "intervention_descriptor_dimension": self.dimension}


DESCRIPTOR_SCHEMA = "intervention_component_descriptors"


@dataclass(frozen=True)
class InterventionDescriptorStore(ConditionDescriptorStore):
    """Backward-compatible name for an intervention descriptor store."""


def component_texts(components: set[str], contract: dict[str, Any]) -> dict[str, str]:
    """Render the pre-registered experimental-design text for every component."""
    template = contract.get("component_prompt_template")
    if not isinstance(template, str) or "{component}" not in template:
        raise ValueError("intervention descriptor requires component_prompt_template containing {component}")
    raw_overrides = contract.get("component_descriptions", {})
    if not isinstance(raw_overrides, dict) or not all(
        (isinstance(key, str) and isinstance(value, str) and value.strip() for key, value in raw_overrides.items())
    ):
        raise ValueError("component_descriptions must map strings to nonempty strings")
    if unknown := set(raw_overrides).difference(components):
        raise ValueError(f"component_descriptions contains unknown components: {sorted(unknown)}")
    rendered = {
        component: str(raw_overrides.get(component, template.format(component=component))).strip()
        for component in sorted(components)
    }
    if any(not text for text in rendered.values()):
        raise ValueError("every intervention component must have nonempty registered text")
    return rendered


def compose_condition_descriptors(
    condition_components: dict[str, tuple[str, ...]], component_vectors: dict[str, np.ndarray], *, dimension: int
) -> dict[str, np.ndarray]:
    """Add unit-normalized component vectors; references receive the zero vector."""
    if dimension <= 0:
        raise ValueError("descriptor dimension must be positive")
    output: dict[str, np.ndarray] = {}
    for condition, components in condition_components.items():
        if unknown := set(components).difference(component_vectors):
            raise ValueError(
                f"condition {condition!r} has components absent from the descriptor artifact: {sorted(unknown)}"
            )
        if not components:
            output[condition] = np.zeros(dimension, dtype=np.float32)
            continue
        value = np.sum(
            [np.asarray(component_vectors[component], dtype=np.float32) for component in components],
            axis=0,
            dtype=np.float32,
        )
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError(f"condition {condition!r} produced an invalid descriptor")
        output[condition] = value
    return output


def load_intervention_descriptors(
    task: InterventionTask, split: InterventionSplit, *, task_manifest_path: Path
) -> InterventionDescriptorStore:
    """Load an artifact only after validating all anti-leakage and provenance binds."""
    artifact_path = task.descriptor_artifact
    metadata_path = task.descriptor_metadata
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Frozen intervention descriptors are incomplete; download the complete pinned SubcellPert dataset bundle for {task.dataset}: {artifact_path}, {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "schema_version": 1,
        "schema": DESCRIPTOR_SCHEMA,
        "status": "complete",
        "task_id": task.task_id,
        "dataset": task.dataset,
        "split_id": split.split_id,
        "provider": task.descriptor_provider,
        "dimension": task.descriptor_dimension,
        "component_normalization": "l2",
        "condition_composition": "additive_sum",
        "target_response_data_used": False,
    }
    changed = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if changed:
        raise ValueError(f"intervention descriptor metadata mismatch: {changed}")
    required_components = {component for values in split.condition_components.values() for component in values}
    registered_texts = component_texts(required_components, task.descriptor_contract)
    if metadata.get("component_texts") != registered_texts:
        raise ValueError("descriptor component text does not match the registered task text")
    with np.load(artifact_path, allow_pickle=False) as payload:
        if set(payload.files) != {"components", "vectors"}:
            raise ValueError("descriptor artifact must contain only components and vectors")
        components = [str(value) for value in payload["components"].tolist()]
        vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if len(components) != len(set(components)) or set(components) != required_components:
        raise ValueError("descriptor artifact component universe does not match the split")
    if vectors.shape != (len(components), task.descriptor_dimension):
        raise ValueError("descriptor artifact matrix shape is invalid")
    if not np.isfinite(vectors).all():
        raise ValueError("descriptor artifact contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if len(norms) and (not np.allclose(norms, 1.0, atol=2e-05, rtol=2e-05)):
        raise ValueError("descriptor artifact component vectors are not L2-normalized")
    component_vectors = {component: vectors[index].copy() for index, component in enumerate(components)}
    conditions = compose_condition_descriptors(
        split.condition_components, component_vectors, dimension=task.descriptor_dimension
    )
    return InterventionDescriptorStore(
        condition_vectors=conditions,
        component_vectors=component_vectors,
        component_texts=registered_texts,
        dimension=task.descriptor_dimension,
        provider=task.descriptor_provider,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
    )


CUSTOM_TEST_SUBGROUPS = frozenset({"unseen_condition"})
FORMAL_DESCRIPTOR_PROVIDERS = frozenset(
    {"biomedbert_non_special_mean", "gears_go_source_coexpression", "component_multihot"}
)
GEARS_TEST_SUBGROUPS = frozenset({"unseen_single", "combo_seen0", "combo_seen1", "combo_seen2"})


@dataclass(frozen=True)
class InterventionFold:
    """One frozen intervention-identity split."""

    fold_id: str
    seed: int
    set2conditions: dict[str, tuple[str, ...]]
    test_perts: tuple[str, ...]
    test_subgroup: dict[str, tuple[str, ...]]

    @property
    def train_conditions(self) -> tuple[str, ...]:
        return self.set2conditions["train"]

    @property
    def validation_conditions(self) -> tuple[str, ...]:
        return self.set2conditions["val"]

    @property
    def test_conditions(self) -> tuple[str, ...]:
        return self.set2conditions["test"]


@dataclass(frozen=True)
class InterventionSplit:
    """Validated split manifest shared by dataset, baseline, and evaluator code."""

    split_id: str
    dataset: str
    reference_conditions: tuple[str, ...]
    condition_components: dict[str, tuple[str, ...]]
    folds: dict[str, InterventionFold]
    metadata: dict[str, Any]

    def fold(self, fold_id: str) -> InterventionFold:
        try:
            return self.folds[fold_id]
        except KeyError as error:
            raise KeyError(f"unknown fold {fold_id!r} for split {self.split_id}") from error


@dataclass(frozen=True)
class InterventionTask:
    """Dataset-specific execution contract for intervention-OOD prediction."""

    task_id: str
    dataset: str
    track: str
    cache_dir: Path
    split_path: Path
    default_source_condition: str | None
    source_by_target: dict[str, str]
    source_pool: str
    cell_type_mode: str
    constant_cell_type: str | None
    descriptor_artifact: Path
    descriptor_metadata: Path
    descriptor_provider: str
    descriptor_dimension: int
    descriptor_contract: dict[str, Any]
    metadata: dict[str, Any]

    def source_for(self, target: str) -> str:
        if target in self.source_by_target:
            return self.source_by_target[target]
        if self.default_source_condition is None:
            raise KeyError(f"task {self.dataset} has no source condition for {target!r}")
        return self.default_source_condition


SPLIT_NAMES = ("train", "val", "test")


def _as_unique_strings(values: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{location} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{location} contains duplicate conditions")
    return tuple(values)


def _expected_gears_subgroup(condition: str, components: tuple[str, ...], train_components: set[str]) -> str:
    seen = sum(component in train_components for component in components)
    if len(components) == 1:
        if seen:
            raise ValueError(f"test single {condition!r} has a component visible in training")
        return "unseen_single"
    if seen == 0:
        return "combo_seen0"
    if seen == 1:
        return "combo_seen1"
    return "combo_seen2"


def artifact_path_token(value: str) -> str:
    """Stable readable path component shared by planners and block workers."""
    stem = re.sub("[^A-Za-z0-9_.-]+", "-", value).strip("-.")[:48] or "value"
    digest = f"{zlib.crc32(value.encode()):08x}"
    return f"{stem}-{digest}"


def load_intervention_split(path: Path) -> InterventionSplit:
    """Load and fully validate a frozen intervention split YAML."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"{path} must be an intervention split with schema_version=1")
    components_raw = raw.get("condition_components")
    if not isinstance(components_raw, dict) or not components_raw:
        raise ValueError(f"{path} must define condition_components")
    components = {
        str(condition): _as_unique_strings(values, location=f"condition_components.{condition}")
        for condition, values in components_raw.items()
    }
    references = _as_unique_strings(raw.get("reference_conditions"), location="reference_conditions")
    fold_rows = raw.get("folds")
    if not isinstance(fold_rows, list):
        raise ValueError(f"{path} folds must be a list")
    folds: dict[str, InterventionFold] = {}
    for index, row in enumerate(fold_rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} fold {index} is not a mapping")
        fold_id = row.get("id")
        if not isinstance(fold_id, str) or not fold_id or fold_id in folds:
            raise ValueError(f"{path} fold {index} has a missing or duplicate id")
        set2conditions_raw = row.get("set2conditions")
        if not isinstance(set2conditions_raw, dict) or set(set2conditions_raw) != set(SPLIT_NAMES):
            raise ValueError(f"fold {fold_id} set2conditions must define train, val, and test")
        set2conditions = {
            name: _as_unique_strings(set2conditions_raw[name], location=f"folds.{fold_id}.{name}")
            for name in SPLIT_NAMES
        }
        subgroup_raw = row.get("test_subgroup")
        if not isinstance(subgroup_raw, dict):
            raise ValueError(f"fold {fold_id} must define test_subgroup")
        subgroup = {
            str(name): _as_unique_strings(values, location=f"folds.{fold_id}.test_subgroup.{name}")
            for name, values in subgroup_raw.items()
        }
        folds[fold_id] = InterventionFold(
            fold_id=fold_id,
            seed=int(row.get("seed", 0)),
            set2conditions=set2conditions,
            test_perts=_as_unique_strings(row.get("test_perts"), location=f"folds.{fold_id}.test_perts"),
            test_subgroup=subgroup,
        )
    split = InterventionSplit(
        split_id=str(raw["id"]),
        dataset=str(raw["dataset"]),
        reference_conditions=references,
        condition_components=components,
        folds=folds,
        metadata={key: value for key, value in raw.items() if key not in {"condition_components", "folds"}},
    )
    validate_intervention_split(split)
    return split


def load_intervention_task(path: Path, dataset: str, *, project_root: Path) -> InterventionTask:
    """Load one dataset from the task manifest."""
    raw = yaml.safe_load(path.read_text())
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != 2
        or raw.get("generalization_axis") != "perturbation_identity"
    ):
        raise ValueError(f"{path} must be a schema-version-2 perturbation-identity task")
    if set(raw.get("protocols", [])) != {"intervention_ood_oracle", "intervention_ood_joint"}:
        raise ValueError("intervention task must declare exactly oracle and joint modes")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or dataset not in datasets:
        raise KeyError(f"dataset {dataset!r} is not defined in {path}")
    row = datasets[dataset]
    if not isinstance(row, dict):
        raise ValueError(f"dataset {dataset!r} task specification must be a mapping")
    source_mapping_raw = row.get("source_by_target", {})
    if not isinstance(source_mapping_raw, dict):
        raise ValueError(f"dataset {dataset!r} source_by_target must be a mapping")
    split_path = project_root / str(row["split_manifest"])
    split = load_intervention_split(split_path)
    if split.dataset != dataset:
        raise ValueError(f"task dataset {dataset!r} does not match split dataset {split.dataset!r}")
    default_source = row.get("default_source_condition")
    if default_source is not None and default_source not in split.reference_conditions:
        raise ValueError(f"default source {default_source!r} is not a reference condition")
    source_mapping = {str(target): str(source) for target, source in source_mapping_raw.items()}
    if unknown_sources := set(source_mapping.values()).difference(split.condition_components):
        raise ValueError(f"source_by_target contains unknown source conditions: {sorted(unknown_sources)}")
    source_pool = str(row.get("source_pool", "pooled_reference"))
    if source_pool not in {"same_unit_then_pooled", "pooled_reference"}:
        raise ValueError(f"unsupported source_pool={source_pool!r}")
    cell_type_mode = str(row.get("cell_type_mode", "observed"))
    constant_cell_type = row.get("constant_cell_type")
    if cell_type_mode not in {"observed", "constant"}:
        raise ValueError(f"unsupported cell_type_mode={cell_type_mode!r}")
    if cell_type_mode == "constant" and (not isinstance(constant_cell_type, str)):
        raise ValueError("constant cell_type_mode requires constant_cell_type")
    descriptor_raw = row.get("intervention_descriptor")
    if not isinstance(descriptor_raw, dict):
        raise ValueError(f"dataset {dataset!r} must define a frozen intervention_descriptor")
    required_descriptor_fields = {
        "artifact",
        "metadata",
        "provider",
        "dimension",
        "component_normalization",
        "condition_composition",
        "target_response_data_used",
        "component_prompt_template",
    }
    if missing := required_descriptor_fields.difference(descriptor_raw):
        raise ValueError(f"dataset {dataset!r} intervention_descriptor is missing {sorted(missing)}")
    descriptor_dimension = int(descriptor_raw["dimension"])
    if descriptor_dimension <= 0:
        raise ValueError("intervention descriptor dimension must be positive")
    descriptor_provider = str(descriptor_raw["provider"])
    if descriptor_provider not in FORMAL_DESCRIPTOR_PROVIDERS:
        raise ValueError(f"unsupported formal intervention descriptor provider: {descriptor_provider!r}")
    if descriptor_raw.get("component_normalization") != "l2":
        raise ValueError("formal component descriptors must be L2-normalized")
    if descriptor_raw.get("condition_composition") != "additive_sum":
        raise ValueError("formal condition descriptors must use additive_sum")
    if descriptor_raw.get("target_response_data_used") is not False:
        raise ValueError("formal descriptors cannot use target response data")
    provider_fields = {
        "biomedbert_non_special_mean": {"model_id", "model_revision", "pooling", "maximum_tokens"},
        "gears_go_source_coexpression": {
            "go_annotations",
            "go_neighbors",
            "coexpression_neighbors",
            "coexpression_threshold",
            "source_normalization",
        },
        "component_multihot": set(),
    }[descriptor_provider]
    if missing := provider_fields.difference(descriptor_raw):
        raise ValueError(
            f"dataset {dataset!r} descriptor provider {descriptor_provider!r} is missing {sorted(missing)}"
        )
    task = InterventionTask(
        task_id=str(raw["id"]),
        dataset=dataset,
        track=str(row["track"]),
        cache_dir=project_root / str(row["cache_dir"]),
        split_path=split_path,
        default_source_condition=str(default_source) if default_source is not None else None,
        source_by_target=source_mapping,
        source_pool=source_pool,
        cell_type_mode=cell_type_mode,
        constant_cell_type=str(constant_cell_type) if constant_cell_type is not None else None,
        descriptor_artifact=project_root / str(descriptor_raw["artifact"]),
        descriptor_metadata=project_root / str(descriptor_raw["metadata"]),
        descriptor_provider=descriptor_provider,
        descriptor_dimension=descriptor_dimension,
        descriptor_contract=descriptor_raw,
        metadata=row,
    )
    for condition in split.condition_components:
        if condition in split.reference_conditions:
            continue
        task.source_for(condition)
    return task


def validate_intervention_split(split: InterventionSplit) -> None:
    """Reject target leakage and malformed GEARS/Systema interfaces."""
    if not split.folds:
        raise ValueError("intervention split must define at least one fold")
    references = set(split.reference_conditions)
    if not references:
        raise ValueError("intervention split requires at least one reference condition")
    if references.difference(split.condition_components):
        raise ValueError("every reference condition must be present in condition_components")
    for condition, components in split.condition_components.items():
        if condition in references:
            if components:
                raise ValueError(f"reference condition {condition!r} must have no functional components")
        elif not components:
            raise ValueError(f"non-reference condition {condition!r} has no functional components")
        if len(components) != len(set(components)):
            raise ValueError(f"condition {condition!r} repeats a functional component")
    universe = set(split.condition_components)
    for fold in split.folds.values():
        blocks = {name: set(fold.set2conditions[name]) for name in SPLIT_NAMES}
        if blocks["train"] & blocks["val"] or blocks["train"] & blocks["test"] or blocks["val"] & blocks["test"]:
            raise ValueError(f"fold {fold.fold_id} assigns a condition to multiple splits")
        assigned = set().union(*blocks.values())
        if assigned != universe:
            missing = sorted(universe.difference(assigned))
            extra = sorted(assigned.difference(universe))
            raise ValueError(f"fold {fold.fold_id} condition universe mismatch; missing={missing}, extra={extra}")
        if not references.issubset(blocks["train"]):
            raise ValueError(f"fold {fold.fold_id} must keep all references in train")
        if references & (blocks["val"] | blocks["test"]):
            raise ValueError(f"fold {fold.fold_id} leaks a reference into validation/test")
        if set(fold.test_perts) != blocks["test"]:
            raise ValueError(f"fold {fold.fold_id} test_perts must exactly equal set2conditions.test")
        subgroup_conditions = [condition for values in fold.test_subgroup.values() for condition in values]
        if len(subgroup_conditions) != len(set(subgroup_conditions)):
            raise ValueError(f"fold {fold.fold_id} repeats a test condition across subgroups")
        if set(subgroup_conditions) != blocks["test"]:
            raise ValueError(f"fold {fold.fold_id} test_subgroup must partition the test conditions")
        unknown_subgroups = set(fold.test_subgroup).difference(GEARS_TEST_SUBGROUPS | CUSTOM_TEST_SUBGROUPS)
        if unknown_subgroups:
            raise ValueError(f"fold {fold.fold_id} has unknown test subgroups: {sorted(unknown_subgroups)}")
        development_components = {
            component
            for condition in (blocks["train"] | blocks["val"]).difference(references)
            for component in split.condition_components[condition]
        }
        for subgroup, conditions in fold.test_subgroup.items():
            if subgroup in CUSTOM_TEST_SUBGROUPS:
                continue
            for condition in conditions:
                expected = _expected_gears_subgroup(
                    condition, split.condition_components[condition], development_components
                )
                if subgroup != expected:
                    raise ValueError(f"fold {fold.fold_id} labels {condition!r} as {subgroup}, expected {expected}")
