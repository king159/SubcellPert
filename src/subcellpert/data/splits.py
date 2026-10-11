"""Frozen fit families, split seeds, validation boundaries, and population partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from subcellpert.data.context import load_context_descriptor_source, load_context_split, load_context_task
from subcellpert.data.intervention import (
    ConditionDescriptorStore,
    load_intervention_descriptors,
    load_intervention_split,
    load_intervention_task,
)
from subcellpert.data.populations import FULL_GENE_POPULATION_LOADER, FullGenePopulation, subset_full_gene_population
from subcellpert.data.records import FullGeneTransition
from subcellpert.data.registry import CONTEXT_OOD, DATASETS
from subcellpert.data.storage import load_population_index_manifest
from subcellpert.utils import (
    deterministic_rng,
)


@dataclass(frozen=True)
class FullGeneFitSpec:
    code: str
    dataset: str
    task_id: str
    task_path: Path
    split_path: Path
    index_dir: Path
    axis: str
    fit_id: str
    context_column: str
    training_contexts: tuple[str, ...]
    training_conditions: tuple[str, ...]
    held_out_contexts: tuple[str, ...]
    held_out_conditions: tuple[str, ...]
    transitions: tuple[FullGeneTransition, ...]
    source_pool: str
    responses: tuple[ResponseSpec, ...]
    descriptors: ConditionDescriptorStore
    metadata: dict[str, Any]
    validation_split: dict[str, Any] = field(default_factory=dict)

    def validate_fit_population(self, population: FullGenePopulation) -> None:
        if population.dataset != self.dataset:
            raise ValueError("fit population belongs to another dataset")
        if population.context not in self.training_contexts or population.condition not in self.training_conditions:
            raise ValueError("population is outside the frozen fitting access boundary")
        allowed_control = (
            self.validation_split.get("axis") == "cell_view"
            and population.condition in self.validation_split["source_conditions"]
        )
        if (
            self.axis == "cellular_context_ood"
            and population.context in self.held_out_contexts
            and (not allowed_control)
        ):
            raise ValueError("held-out context entered supervised fitting")
        if self.axis == "perturbation_ood" and population.condition in self.held_out_conditions:
            raise ValueError("held-out perturbation entered fitting")

    def lineage(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_code": self.code,
            "task_id": self.task_id,
            "generalization_axis": self.axis,
            "fit_group": self.fit_id,
            "fit_family": self.metadata.get("fit_family"),
            "fit_family_policy": self.metadata.get("fit_family_policy"),
            "split_seed": self.metadata.get("split_seed"),
            "schedule_manifest": self.metadata.get("schedule_manifest"),
            "split_policy": self.metadata.get("split_policy"),
            "test_index_view": self.metadata.get("test_index_view"),
            "population_loader": FULL_GENE_POPULATION_LOADER,
            "prediction_storage": "complete_cell_oracle_hdf5",
            "training_contexts": list(self.training_contexts),
            "training_conditions": list(self.training_conditions),
            "held_out_contexts": list(self.held_out_contexts),
            "held_out_conditions": list(self.held_out_conditions),
            "test_response_count": len(self.responses),
            "training_protocol": "scperturbench_scfoundation_cell_oracle",
            "gene_axis_partitioned": False,
            "target_response_used_in_fit": False,
            "split_reference": "https://github.com/bm2-lab/scPerturBench",
            "validation_split": self.validation_split,
            "validation_used_in_fit": False,
        }


@dataclass(frozen=True)
class ResponseSpec:
    response_id: str
    context: str
    source_condition: str
    target_condition: str


def _support(index_dir: Path) -> set[tuple[str, str]]:
    manifest = load_population_index_manifest(index_dir)
    support = set()
    for shard in manifest["shards"]:
        frame = (
            pl.scan_parquet(index_dir / shard["populations"])
            .select("context", "condition", "declared_cell_count")
            .unique()
            .collect()
        )
        support.update(((str(c), str(p)) for c, p, n in frame.iter_rows() if n > 0))
    return support


def resolve_fit_specs(root: Path, code: str, *, apply_validation: bool = True) -> list[FullGeneFitSpec]:
    if apply_validation:
        return load_seed_specs(root, code)
    binding = DATASETS[code]
    task_path = root / binding.task_manifests[binding.default_task]
    index_dir = root / binding.cache_dir / "population"
    manifest = load_population_index_manifest(index_dir, expected_dataset=binding.dataset_id)
    specs = []
    if binding.default_task == CONTEXT_OOD:
        task = load_context_task(task_path, binding.dataset_id, project_root=root)
        split = load_context_split(task.split_path)
        _, _, descriptors = load_context_descriptor_source(task)
        for group in split.fit_groups.values():
            contexts = tuple(sorted(set(group.training_contexts) | set(group.validation_contexts)))
            transitions = tuple(
                FullGeneTransition(t, split.transitions[t].source_condition, split.transitions[t].target_condition)
                for t in group.transition_ids
            )
            conditions = tuple(sorted({p for t in transitions for p in (t.source_condition, t.target_condition)}))
            responses = tuple(
                ResponseSpec(f.fold_id, f.held_out_context, f.source_condition, f.target_condition)
                for fid in group.fold_ids
                for f in [split.fold(fid)]
            )
            if set(contexts) & set(group.held_out_contexts):
                raise ValueError("context split leaks held-out contexts into final fitting")
            if any(r.target_condition not in conditions for r in responses):
                raise ValueError("context OOD must evaluate a known transition")
            specs.append(
                FullGeneFitSpec(
                    code,
                    binding.dataset_id,
                    task.task_id,
                    task_path,
                    task.split_path,
                    index_dir,
                    "cellular_context_ood",
                    group.fit_group_id,
                    task.cache_context_column,
                    contexts,
                    conditions,
                    group.held_out_contexts,
                    (),
                    transitions,
                    task.source_pool,
                    responses,
                    descriptors,
                    task.metadata,
                )
            )
    else:
        task = load_intervention_task(task_path, binding.dataset_id, project_root=root)
        split = load_intervention_split(task.split_path)
        descriptors = load_intervention_descriptors(task, split, task_manifest_path=task_path)
        contexts = tuple(sorted(manifest["inventory"]["contexts"]))
        support = _support(index_dir)
        for fold in split.folds.values():
            development = sorted(
                (set(fold.train_conditions) | set(fold.validation_conditions)) - set(split.reference_conditions)
            )
            transitions = tuple(
                FullGeneTransition(f"{task.source_for(t)}_to_{t}", task.source_for(t), t) for t in development
            )
            sources = {task.source_for(t) for t in [*development, *fold.test_conditions]}
            conditions = tuple(sorted(sources | set(development)))
            if set(conditions) & set(fold.test_conditions):
                raise ValueError(
                    "perturbation split leaks held-out target condition into fitting or source covariates"
                )
            responses = tuple(
                ResponseSpec(f"{fold.fold_id}/{target}/{context}", context, task.source_for(target), target)
                for target in fold.test_conditions
                for context in contexts
                if (context, target) in support and (context, task.source_for(target)) in support
            )
            if not responses:
                raise ValueError("perturbation OOD has no supported held-out response")
            specs.append(
                FullGeneFitSpec(
                    code,
                    binding.dataset_id,
                    task.task_id,
                    task_path,
                    task.split_path,
                    index_dir,
                    "perturbation_ood",
                    fold.fold_id,
                    str(manifest["context_column"]),
                    contexts,
                    conditions,
                    (),
                    fold.test_conditions,
                    transitions,
                    task.source_pool,
                    responses,
                    descriptors,
                    task.metadata,
                )
            )
    return specs


CRC_SUPPORT_POLICY = "same_context_regimen_support_v1"
DUAL_SUPPORT_DATASETS = frozenset(
    {
        "atm-rt",
        "ddx54-pd1",
        "lung-drug",
        "sickness",
        "dss-colitis",
        "adipose",
        "uc-vdz",
        "kidney-iri",
        "gut-microbiota",
        "cardiac-mi",
        "breast-fus",
        "gbm-ohsv",
        "spinal-ko",
    }
)
DUAL_SUPPORT_POLICY = "same_context_dual_min50_v1"
PF_SUPPORT_POLICY = "dual_min50_single_guide_perturbations_v1"
REAUDITED_SUPPORT_DATASETS = frozenset({"gut-microbiota", "cardiac-mi", "breast-fus", "gbm-ohsv", "spinal-ko"})
SICKNESS_SUPPORT_POLICY = "same_context_pbs_support_v1"


def response_support_policy(code):
    if code in DUAL_SUPPORT_DATASETS:
        return DUAL_SUPPORT_POLICY
    if code in REAUDITED_SUPPORT_DATASETS:
        return "same_context_registered_source_support_v1"
    return {"sickness": SICKNESS_SUPPORT_POLICY, "crc-tx": CRC_SUPPORT_POLICY, "pf-astro": PF_SUPPORT_POLICY}.get(code)


FIT_FAMILY_POLICY = "independent_source_target_families_v1"
LUNG_FAMILY_TRANSITIONS = {"drug": ("azd3965_addon", "vb124_addon")}
NO_VALIDATION_DATASETS = frozenset({"atm-rt", "ddx54-pd1"})
SCHEDULE_MANIFEST = "benchmark/splits/cell_seed_schedule_dual_min50.json"
SPLIT_SEEDS = (1, 2, 3)


def fit_families_for_dataset(code):
    return tuple(LUNG_FAMILY_TRANSITIONS) if code == "lung-drug" else ("default",)


def fit_id_for(seed):
    return f"seed{seed}"


def load_seed_specs(root, code):
    root = Path(root)
    manifest = json.loads((root / SCHEDULE_MANIFEST).read_text())
    rows = validate_seed_schedule(manifest, code)
    templates = {s.fit_id: s for s in resolve_fit_specs(root, code, apply_validation=False)}
    specs = []
    for row in rows:
        base = templates[row["template_fit_id"]]
        spec = replace(
            base,
            fit_id=row["fit_id"],
            axis=row["axis"],
            context_column=row["context_column"],
            index_dir=root / row["index_dir"],
            source_pool="pooled_reference",
            **{
                key: tuple(row[key])
                for key in ("training_contexts", "training_conditions", "held_out_contexts", "held_out_conditions")
            },
            transitions=tuple(FullGeneTransition(**r) for r in row["transitions"]),
            responses=tuple(ResponseSpec(**r) for r in row["responses"]),
            metadata={
                **base.metadata,
                "split_seed": row["split_seed"],
                "fit_family": row["fit_family"],
                "fit_family_policy": FIT_FAMILY_POLICY,
                "schedule_manifest": SCHEDULE_MANIFEST,
                "split_policy": "task_splits",
                "test_index_view": row["index_dir"],
                "project_root": str(root),
            },
        )
        specs.append(apply_validation_plan(root, spec))
    return specs


def seeds_for_dataset(code):
    return (1,) if code in NO_VALIDATION_DATASETS else SPLIT_SEEDS


def validate_seed_schedule(manifest, code):
    if (
        manifest.get("schema") != "three_seed_cell_schedule"
        or manifest.get("split_policy") != "task_splits"
        or manifest.get("fit_family_policy") != FIT_FAMILY_POLICY
    ):
        raise ValueError("Missing current independent-fit-family seed schedule; regenerate the split views")
    rows = manifest["datasets"][code]
    expected = [
        (seed, family, fit_id_for(seed))
        for seed, family in product(seeds_for_dataset(code), fit_families_for_dataset(code))
    ]
    if [(r["split_seed"], r.get("fit_family"), r["fit_id"]) for r in rows] != expected:
        raise ValueError("Dataset split seeds or fit families do not match the current schedule")
    if any(r.get("response_support_policy") != response_support_policy(code) for r in rows):
        raise ValueError("Obsolete response admission policy; regenerate the dataset split views")
    return rows


VALIDATION_MANIFEST = "benchmark/splits/cell_validation_dual_min50.json"


def apply_validation_plan(root, spec):
    manifest = json.loads((Path(root) / VALIDATION_MANIFEST).read_text())
    if manifest.get("schema") != "independent_cell_validation":
        raise ValueError("Missing independent validation split schema")
    plan = manifest["datasets"][spec.code][spec.fit_id]
    if plan.get("axis") == "cell_view":
        if (
            plan.get("fit_family_policy") != FIT_FAMILY_POLICY
            or plan.get("fit_family") != spec.metadata.get("fit_family")
            or set(plan["source_conditions"]) & set(plan["target_conditions"])
        ):
            raise ValueError("Validation plan does not preserve independent source/target fit families")
        if plan.get("response_admission", {}).get("policy") != response_support_policy(spec.code):
            raise ValueError("Validation plan has an obsolete response admission policy")
        declared = set(plan["source_conditions"]) | set(plan["target_conditions"])
        if set(plan["family_conditions"]) != declared or any(
            t.source_condition not in plan["source_conditions"] or t.target_condition not in plan["target_conditions"]
            for t in (*spec.transitions, *spec.responses)
        ):
            raise ValueError("A response leaves its frozen fit-family access boundary")
        if plan.get("donor_patient_split") is not False or plan.get("response_identity_disjoint") is not True:
            raise ValueError("The benchmark split must exclude response leakage without donor grouping")
        splits = ("train", "validation", "test") if plan.get("enabled", True) else ("train", "test")
        for split in splits:
            view = load_population_index_manifest(
                Path(root) / plan["index_views"][split],
                expected_dataset=spec.dataset,
                expected_context_column=spec.context_column,
            )
            if view.get("cell_identity_view", {}).get("schema") != "cell_identity_view":
                raise ValueError("Missing frozen benchmark cell identities")
        if any(not set(plan[key]) <= declared for key in ("training_conditions", "validation_conditions")):
            raise ValueError("Training/validation conditions leave the fit family")
        return replace(
            spec,
            training_contexts=tuple(plan["training_contexts"]),
            training_conditions=tuple(plan["training_conditions"]),
            transitions=tuple(FullGeneTransition(**r) for r in plan["training_transitions"]),
            validation_split=plan,
        )
    if set(plan["outer_development_contexts"]) != set(spec.training_contexts) or set(
        plan["outer_development_conditions"]
    ) != set(spec.training_conditions):
        raise ValueError("Validation plan no longer matches the outer development split")
    train_contexts = tuple(plan["training_contexts"])
    train_conditions = tuple(plan["training_conditions"])
    val_contexts = tuple(plan["validation_contexts"])
    val_conditions = tuple(plan["validation_conditions"])
    if not train_contexts or not train_conditions or (not val_contexts) or (not val_conditions):
        raise ValueError("Both training and validation must be nonempty")
    if (set(train_contexts) | set(val_contexts)) & set(spec.held_out_contexts) or (
        set(train_conditions) | set(val_conditions)
    ) & set(spec.held_out_conditions):
        raise ValueError("Validation plan includes an outer test response")
    if not set(train_contexts) | set(val_contexts) <= set(spec.training_contexts) or not set(train_conditions) | set(
        val_conditions
    ) <= set(spec.training_conditions):
        raise ValueError("Validation plan leaves the permitted development data")
    axis = plan["axis"]
    if axis == "context" and set(train_contexts) & set(val_contexts):
        raise ValueError("Training and validation contexts overlap")
    if axis == "condition":
        targets = {row["target_condition"] for row in plan["validation_transitions"]}
        if targets & set(train_conditions):
            raise ValueError("Validation intervention responses enter training")
    elif axis not in {"context", "unit", "cell"}:
        raise ValueError("Unknown independent validation axis")
    if axis == "unit" and (not plan.get("validation_units_by_condition")):
        raise ValueError("Unit validation needs explicit sample identities")
    if axis == "cell" and (not 0 < plan.get("validation_fraction", 0) < 1):
        raise ValueError("Cell validation needs a proper holdout fraction")
    transitions = tuple(t for t in spec.transitions if t.target_condition in train_conditions)
    if not transitions or not plan["validation_transitions"]:
        raise ValueError("Training or validation has no response transition")
    return replace(
        spec,
        training_contexts=train_contexts,
        training_conditions=train_conditions,
        transitions=transitions,
        validation_split=plan,
    )


def partition_population(spec, population, split, *, cell_cap=None):
    rows = population_rows(spec, population, split)
    if cell_cap is not None:
        rows = rows[:cell_cap]
    if not len(rows):
        return None
    if len(rows) == len(population.cell_ids) and np.array_equal(rows, np.arange(len(rows))):
        return population
    return subset_full_gene_population(population, population.cell_ids[rows])


def population_rows(spec, population, split):
    """Return disjoint cell identities; never use RNA values to choose a split."""
    contexts, transitions = split_axes(spec, split)
    conditions = {c for t in transitions for c in (t.source_condition, t.target_condition)}
    if population.context not in contexts or population.condition not in conditions:
        return np.empty(0, dtype=np.int64)
    plan = spec.validation_split
    rows = np.arange(len(population.cell_ids))
    if plan["axis"] == "unit":
        units = plan["validation_units_by_condition"].get(population.condition)
        if units is None:
            raise ValueError("Population condition has no frozen validation unit assignment")
        in_validation = population.unit_id in units
        if in_validation != (split == "validation"):
            return rows[:0]
    elif plan["axis"] == "cell":
        order = np.argsort(population.cell_ids.astype(str), kind="stable")
        rng = deterministic_rng(
            plan["seed"],
            "validation_cells",
            spec.dataset,
            spec.fit_id,
            population.context,
            population.condition,
            population.unit_id,
        )
        order = order[rng.permutation(len(order))]
        n_validation = min(len(order) - 1, max(1, round(len(order) * plan["validation_fraction"])))
        if n_validation < 1:
            raise ValueError("Within-sample validation requires at least two cells per population")
        rows = order[:n_validation] if split == "validation" else order[n_validation:]
    return rows[np.argsort(population.cell_ids[rows].astype(str), kind="stable")]


def split_axes(spec, split):
    if split not in {"train", "validation"}:
        raise ValueError("Only training or independent validation records may be loaded")
    if not spec.validation_split:
        raise ValueError("An independent validation plan is required")
    if split == "train":
        return (spec.training_contexts, spec.transitions)
    plan = spec.validation_split
    return (
        tuple(plan["validation_contexts"]),
        tuple(FullGeneTransition(**row) for row in plan["validation_transitions"]),
    )
