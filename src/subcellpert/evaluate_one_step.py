"""Sealed prediction generation and one-response joint/oracle evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from subcellpert.data.arrays import MOLECULE_BATCH_SIZE, H5Array, PackedCoordinates
from subcellpert.data.batching import PopulationCellReader, concatenate_cell_batches, select_batch_cells
from subcellpert.data.populations import (
    CellBalancedFullGenePointCloud,
    iter_full_gene_populations,
    subset_full_gene_population,
)
from subcellpert.data.sampling import OracleCounts, iter_cell_axes, joint_output_ids, sample_cell_axes
from subcellpert.evaluate.metrics import (
    JOINT_SPATIAL_METRIC_POLICY,
    _materialize_evaluation_cells,
    _materialize_selected,
    _prediction_population,
    exact_spatial_metrics,
    sample_joint_evaluation,
    score_cell_predictions,
    score_joint_predictions,
    streaming_deg_panels,
    unit_cell_weights,
)
from subcellpert.models.shared import ClassicalModel as CellClassical
from subcellpert.models.shared import ClassicalModel as CellJointClassical
from subcellpert.models.shared import TensorCells
from subcellpert.utils import (
    CELL_EVALUATION_SCHEMA,
    CELL_PREDICTION_SCHEMA,
    CELL_PROTOCOL_ID,
    JOINT_EVALUATION_SCHEMA,
    JOINT_PREDICTION_SCHEMA,
    JOINT_PROTOCOL_ID,
    _save_torch,
    atomic_write_json,
    integer_counts,
    sample_donor_rna,
    seed_from_parts,
)
from subcellpert.utils import CELL_PREDICTION_SCHEMA as ORACLE_PREDICTION_SCHEMA


def evaluate_cell_response(
    spec, response, oracles, paths, directory, *, projections=64, scientific_result=True, generation_plan=None
):
    """Validate every seal and exact count matrix BEFORE opening target geometry."""
    predictions = [load_oracle_prediction(path) for path in paths]
    if not predictions or len(predictions) != len(oracles):
        raise ValueError("All Oracle units must be sealed before evaluating")
    if any(
        p.storage_manifest.get("schema") != CELL_PREDICTION_SCHEMA
        or p.storage_manifest.get("training_protocol") != CELL_PROTOCOL_ID
        for p in predictions
    ):
        raise ValueError("A population-level prediction cannot enter current cell evaluation")
    identities = [validate_oracle_count_identity(p, o) for p, o in zip(predictions, oracles, strict=True)]
    if scientific_result and generation_plan is None:
        raise ValueError("Scientific Oracle evaluation requires a pre-generation selection")
    if generation_plan is not None:
        if sum(len(p.cell_ids) for p in predictions) != generation_plan["output_cells"]:
            raise ValueError("Oracle output count differs from the common generation budget")
        if any(
            p.storage_manifest["provenance"].get("generation_plan_sha256") != plan_digest(generation_plan)
            for p in predictions
        ):
            raise ValueError("Oracle prediction is not bound to the pre-generation selection")
        if [{"unit_id": o.unit_id, "cell_ids": o.cell_ids.tolist()} for o in oracles] != [
            {"unit_id": r["unit_id"], "cell_ids": r["cell_ids"]} for r in generation_plan["target"]
        ]:
            raise ValueError("Oracle inputs differ from the preselected target cell IDs")
    full_targets = list(
        iter_full_gene_populations(
            spec.index_dir,
            conditions=[response.target_condition],
            contexts=[response.context],
            expected_dataset=spec.dataset,
            expected_context_column=spec.context_column,
        )
    )
    by_unit = {p.unit_id: p for p in full_targets}
    targets = [subset_full_gene_population(by_unit[o.unit_id], o.cell_ids) for o in oracles]
    for target, oracle in zip(targets, oracles, strict=True):
        for first in range(0, len(oracle.cell_ids), 128):
            if not np.array_equal(target.counts[first : first + 128], oracle.counts[first : first + 128]):
                raise ValueError("Evaluator geometry no longer matches the frozen Oracle input")
    sources = list(
        iter_full_gene_populations(
            spec.index_dir,
            conditions=[response.source_condition],
            contexts=[response.context],
            expected_dataset=spec.dataset,
            expected_context_column=spec.context_column,
        )
    )
    panels = streaming_deg_panels(sources, full_targets)
    if generation_plan is not None:
        validate_generation_plan(generation_plan, spec, response, sources, full_targets, cap=generation_plan["cap"])
    atomic_write_json(
        directory / "gene_panels.json",
        {
            "ranking": panels.ranking,
            "source": "all_observed_source_and_target_counts_after_prediction_seal",
            "prediction_used": False,
            "panels": [{"name": p.name, "genes": list(p.genes)} for p in panels.panels],
        },
    )
    predictions = [_materialize_evaluation_cells(p) for p in predictions]
    targets = [_materialize_evaluation_cells(p) for p in targets]
    units = []
    with tqdm(
        total=1 + len(targets),
        desc=f"Evaluate oracle | {response.response_id}",
        unit="comparison",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
        metrics, exact = score_cell_predictions(predictions, targets, panels.panels, projections=projections)
        progress.update(1)
        for prediction, target in zip(predictions, targets, strict=True):
            rows, _ = score_cell_predictions([prediction], [target], panels.panels, projections=projections)
            units.append({"unit_id": target.unit_id, "cells": len(target.cell_ids), "metrics": rows})
            progress.update(1)
    result = {
        "status": "complete",
        "schema": CELL_EVALUATION_SCHEMA,
        "training_protocol": CELL_PROTOCOL_ID,
        "mode": "oracle_count",
        "response_id": response.response_id,
        "context": response.context,
        "source_condition": response.source_condition,
        "target_condition": response.target_condition,
        "scientific_result": scientific_result,
        "gene_panels": metrics,
        "metrics": metrics[0],
        "spatial_metrics": exact,
        "target_unit_rows": units,
        "count_integrity": identities,
        **(
            {
                "generation_policy": GENERATION_POLICY,
                "generation_budget": generation_plan["output_cells"],
                "generation_plan_sha256": plan_digest(generation_plan),
                "sampling": {
                    "target": generation_plan["target"],
                    "selection_stage": "before_generation",
                    "prediction_subsampling": False,
                    "cap_per_population": generation_plan["cap"],
                    "base_seed": 42,
                },
            }
            if generation_plan is not None
            else {}
        ),
        "target_access_stage": "count_only_prediction_then_sealed_coordinate_evaluation",
        "target_geometry_used_for_prediction": False,
        "target_counts_used_for_prediction": True,
        "prediction_unit_coverage": 1.0,
        "count_performance_metrics": "not_applicable_exact_oracle_constraint",
        "biological_cell_pairing": False,
        "rna_subsampling": False,
    }
    atomic_write_json(directory / "metrics.json", result)
    return result


class BestValidationCheckpoint:
    """Persist selected weights separately from the latest optimizer-resume state."""

    def __init__(self, path, binding):
        self.path, self.binding = (Path(path), binding)
        self.selection = None
        if self.path.exists():
            saved = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
            if saved.get("binding") != binding:
                raise ValueError("Validation checkpoint belongs to another training/validation split")
            self.selection = saved["selection"]

    def consider(self, model, validation, *, epoch, complete_epoch):
        score = validation["score"]
        if not np.isfinite(score):
            raise ValueError("Cannot select a checkpoint with a nonfinite validation score")
        improved = self.selection is None or score < self.selection["validation"]["score"]
        if improved:
            self.selection = {
                "epoch": epoch,
                "complete_epoch": complete_epoch,
                "validation": validation,
                "tie_break": "earliest_epoch",
            }
            _save_torch(self.path, {"binding": self.binding, "model": model.state_dict(), "selection": self.selection})
        return improved

    def restore(self, model):
        if self.selection is None or not self.path.exists():
            raise ValueError("No independently validated checkpoint is available")
        saved = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
        if saved.get("binding") != self.binding or saved.get("selection") != self.selection:
            raise ValueError("Selected validation checkpoint metadata changed")
        model.load_state_dict(saved["model"], strict=True)
        return self.selection


def _mean_log_counts(populations):
    weights = unit_cell_weights([len(p.cell_ids) for p in populations])
    values = np.concatenate([np.log1p(np.asarray(p.counts[:], dtype=np.float64)) for p in populations])
    return np.average(values, axis=0, weights=weights)


def _predict_validation_counts(
    model, source, descriptor, *, device, foundation_store, batch_size=256, source_cell_ids=None
):
    """Mirror Joint inference rounding without opening any target coordinates."""
    reader = PopulationCellReader(source)
    if source_cell_ids is None:
        donor_rows, output_ids = (np.arange(len(source.cell_ids)), source.cell_ids)
    else:
        lookup = {str(cell): i for i, cell in enumerate(source.cell_ids)}
        donor_rows = np.asarray([lookup[cell] for cell in source_cell_ids])
        output_ids = joint_output_ids(source_cell_ids)
    values = []

    def compute(first, last):
        selected, assignment = np.unique(donor_rows[first:last], return_inverse=True)
        batch = reader.read(selected)
        if isinstance(model, CellJointClassical):
            counts = model.predict_counts(batch, descriptor)[assignment]
        else:
            foundation = (
                None
                if foundation_store is None
                else torch.as_tensor(foundation_store.get(source, batch.cell_ids), device=device)
            )
            counts = integer_counts(
                model.count_log1p(
                    TensorCells.from_batch(batch, device),
                    torch.as_tensor(np.tile(descriptor, (last - first, 1)), dtype=torch.float32, device=device),
                    torch.as_tensor(assignment, device=device),
                    foundation,
                )
                .cpu()
                .numpy()
            )
        return counts

    for _, counts in prediction_batches(len(donor_rows), batch_size, compute, show_progress=False):
        values.append(counts)
    return SimpleNamespace(cell_ids=output_ids, counts=np.concatenate(values))


def assert_independent_records(training, validation):
    """No validation response cell may occur in any gradient-training input."""
    if not training or not validation:
        raise ValueError("Training and validation records must both be nonempty")
    populations = {}
    for record in training:
        for population in (*record.source, record.target):
            key = (population.dataset, population.context, population.condition, population.unit_id)
            populations.setdefault(key, []).append(population)
    for record in validation:
        target = record.target
        key = (target.dataset, target.context, target.condition, target.unit_id)
        for train in populations.get(key, []):
            if np.intersect1d(train.cell_ids, target.cell_ids, assume_unique=True).size:
                raise ValueError("A validation response cell occurs in gradient-training data")


def evaluate_validation(
    model,
    records,
    *,
    mode,
    device="cpu",
    foundation_store=None,
    cell_cap=128,
    projections=64,
    scratch_directory=None,
    gene_panel="all_gene",
):
    """Macro-average response scores; fixed samples and geometry across epochs.

    Joint selects by all-gene MSE under source-only predicted counts. Oracle
    selects by exact cell-balanced spatial W1 under supplied validation counts.
    This function does not update parameters or use any outer test response.
    """
    if not records or mode not in {"joint", "oracle_count"}:
        raise ValueError("Independent validation records and a supported mode are required")
    if cell_cap < 1 or projections < 1:
        raise ValueError("Validation must retain cells and spatial projections")
    if gene_panel not in {"all_gene", "top_20_deg"} or (mode != "joint" and gene_panel != "all_gene"):
        raise ValueError("Unsupported independent validation gene panel")
    grouped = defaultdict(list)
    for record in records:
        grouped[record.transition_id, record.context].append(record)
    was_training = model.training if isinstance(model, torch.nn.Module) else None
    if was_training is not None:
        model.eval()
    rows = []
    try:
        with (
            torch.inference_mode(),
            tqdm(
                total=len(grouped),
                desc=f"Validation ({mode})",
                unit="response",
                dynamic_ncols=True,
                mininterval=1.0,
            ) as progress,
        ):
            for (transition, context), group in sorted(grouped.items()):
                response_id = f"validation/{transition}/{context}"
                targets, target_sampling = sample_joint_evaluation(
                    [r.target for r in group], response_id=response_id, namespace="validation_target", cap=cell_cap
                )
                budget = sum(len(p.cell_ids) for p in targets)
                if mode == "joint":
                    unique = {(p.unit_id, p.condition, p.context): p for record in group for p in record.source}
                    sources = {p.unit_id: p for p in unique.values()}
                    source_sampling = select_joint_donors(
                        list(sources.values()), response_id=response_id, budget=budget
                    )
                    predictions = [
                        _predict_validation_counts(
                            model,
                            sources[draw["unit_id"]],
                            group[0].descriptor,
                            device=device,
                            foundation_store=foundation_store,
                            source_cell_ids=draw["source_cell_ids"],
                        )
                        for draw in source_sampling
                    ]
                    errors = (_mean_log_counts(predictions) - _mean_log_counts(targets)) ** 2
                    selected_genes = np.arange(len(errors))
                    if gene_panel == "top_20_deg":
                        panels = streaming_deg_panels(list(unique.values()), [r.target for r in group], top_k=20)
                        selected_genes = panels.panels[1].indices
                    score = float(np.mean(errors[selected_genes]))
                else:
                    by_unit = {record.target.unit_id: record for record in group}
                    source_sampling = "one_fixed_uniform_allowed_source_donor_per_validation_target_cell"
                    with TemporaryDirectory(prefix="cell_validation_", dir=scratch_directory) as temporary:
                        predictions = []
                        for i, target in enumerate(targets):
                            record = by_unit[target.unit_id]
                            oracle = OracleCounts(
                                target.dataset,
                                target.context,
                                target.condition,
                                target.unit_id,
                                target.genes,
                                target.cell_ids,
                                target.counts,
                            )
                            path = Path(temporary) / f"prediction_{i}.h5"
                            write_cell_prediction(
                                model,
                                oracle,
                                list(record.source),
                                record.descriptor,
                                path,
                                seed=seed_from_parts(42, "validation_geometry", response_id, target.unit_id),
                                device=device,
                                foundation_store=foundation_store,
                                provenance={"purpose": "independent_validation"},
                            )
                            predictions.append(_materialize_evaluation_cells(load_oracle_prediction(path)))
                        measured = [_materialize_evaluation_cells(p) for p in targets]
                        metrics = exact_spatial_metrics(predictions, measured, projections=projections)
                        supported = [v for v in metrics["per_gene_swd"] if v is not None]
                        if not supported:
                            raise ValueError("Validation response has no spatially supported genes")
                        score = float(np.mean(supported))
                if sum(len(p.cell_ids) for p in predictions) != budget:
                    raise ValueError("Validation prediction count differs from the common target-sized budget")
                if not np.isfinite(score):
                    raise ValueError("Validation score must be finite before checkpoint selection")
                rows.append(
                    {
                        "transition": transition,
                        "context": context,
                        "score": score,
                        "target_sampling": target_sampling,
                        "source_sampling": source_sampling,
                        "generation_policy": GENERATION_POLICY,
                        "generation_budget": budget,
                        "prediction_cells": sum(len(p.cell_ids) for p in predictions),
                        **(
                            {
                                "all_gene_mse": float(np.mean(errors)),
                                "selection_genes": [group[0].target.genes[i] for i in selected_genes],
                            }
                            if mode == "joint"
                            else {}
                        ),
                    }
                )
                progress.set_postfix(score=f"{score:.6g}", refresh=False)
                progress.update(1)
    finally:
        if was_training is not None:
            model.train(was_training)
    return {
        "metric": "mse" if mode == "joint" else "cell_balanced_spatial_swd",
        "score": float(np.mean([r["score"] for r in rows])),
        "direction": "minimize",
        "gene_panel": gene_panel,
        "cell_cap_per_response": cell_cap,
        "sampling_seed": 42,
        "model_seed_used_for_validation_sampling": False,
        "aggregation": "equal_responses; equal_units_then_cells_within_response",
        "validation_used_in_gradient_training": False,
        "outer_test_used": False,
        "responses": rows,
    }


def evaluate_joint_response(
    spec,
    response,
    paths,
    directory,
    *,
    projections=64,
    cap=2000,
    scientific_result=True,
    expected_provenance=None,
    generation_plan=None,
):
    """Validate EVERY prediction before loading ANY held-out target data."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("Joint evaluation requires sealed predictions")
    predictions = [load_joint_prediction(path) for path in paths]
    identities = [validate_joint_prediction(p) for p in predictions]
    if scientific_result and generation_plan is None:
        raise ValueError("Scientific Joint evaluation requires a pre-generation selection")
    if generation_plan is not None:
        if sum(r["cells"] for r in identities) != generation_plan["output_cells"]:
            raise ValueError("Joint output count differs from the common generation budget")
        for prediction in predictions:
            meta = prediction.storage_manifest
            if (
                meta.get("generation_policy") != GENERATION_POLICY
                or meta.get("generation_budget") != generation_plan["output_cells"]
                or meta.get("target_cell_count_used_for_prediction") is not True
                or (meta["provenance"].get("generation_plan_sha256") != plan_digest(generation_plan))
            ):
                raise ValueError("Joint prediction is not bound to the pre-generation selection")
    if len({p.storage_manifest["source_unit_id"] for p in predictions}) != len(predictions):
        raise ValueError("Joint prediction units must be unique")
    for prediction in predictions:
        meta = prediction.storage_manifest
        if meta["context"] != response.context or meta["source_condition"] != response.source_condition:
            raise ValueError("Joint prediction belongs to another source response")
        if expected_provenance is not None and meta["provenance"] != expected_provenance:
            raise ValueError("Joint prediction checkpoint or response bindings changed")
    if expected_provenance is not None:
        input_path = Path(directory) / "joint_source_input.json"
        inputs = json.loads(input_path.read_text())
        if any(inputs.get(key) is not False for key in ("target_counts_loaded", "target_geometry_loaded")):
            raise ValueError("Joint input contains prohibited target information")
        if (
            generation_plan is None
            or inputs.get("target_cell_count_loaded") is not True
            or inputs.get("generation_plan_sha256") != plan_digest(generation_plan)
            or (inputs.get("generation_budget") != generation_plan["output_cells"])
        ):
            raise ValueError("Joint input does not declare the common population-size budget")
        expected_units = {row["unit_id"]: row for row in inputs["donors"]}
        if set(expected_units) != {p.storage_manifest["source_unit_id"] for p in predictions}:
            raise ValueError("All preselected source units must be predicted before target expression access")
        for prediction in predictions:
            meta = prediction.storage_manifest
            expected = expected_units[meta["source_unit_id"]]
            if (
                meta["source_cell_count"] != expected["selected_cells"]
                or meta["source_donor_cell_ids"] != expected["source_cell_ids"]
            ):
                raise ValueError("Joint prediction source population differs from the frozen input")
    full_targets = list(
        iter_full_gene_populations(
            spec.index_dir,
            conditions=[response.target_condition],
            contexts=[response.context],
            expected_dataset=spec.dataset,
            expected_context_column=spec.context_column,
        )
    )
    full_sources = list(
        iter_full_gene_populations(
            spec.index_dir,
            conditions=[response.source_condition],
            contexts=[response.context],
            expected_dataset=spec.dataset,
            expected_context_column=spec.context_column,
        )
    )
    if generation_plan is not None:
        validate_generation_plan(generation_plan, spec, response, full_sources, full_targets, cap=cap)
    elif {p.unit_id for p in full_sources} != {p.storage_manifest["source_unit_id"] for p in predictions}:
        raise ValueError("Joint predictions do not cover the source biological units")
    panels = streaming_deg_panels(full_sources, full_targets)
    directory = Path(directory)
    atomic_write_json(
        directory / "gene_panels.json",
        {
            "ranking": panels.ranking,
            "source": "all_observed_source_and_target_counts_after_prediction_seal",
            "prediction_used": False,
            "panels": [{"name": p.name, "genes": list(p.genes)} for p in panels.panels],
        },
    )
    predicted_populations = [_prediction_population(p, spec.dataset, response.target_condition) for p in predictions]
    if generation_plan is None:
        predicted, pred_sampling = sample_joint_evaluation(
            predicted_populations, response_id=response.response_id, namespace="prediction", cap=cap
        )
        targets, target_sampling = sample_joint_evaluation(
            full_targets, response_id=response.response_id, namespace="target", cap=cap
        )
        sources, source_sampling = sample_joint_evaluation(
            full_sources, response_id=response.response_id, namespace="source", cap=cap
        )
    else:
        predicted = predicted_populations
        pred_sampling = [
            {
                "unit_id": p.unit_id,
                "available_cells": len(p.cell_ids),
                "selected_cells": len(p.cell_ids),
                "cell_ids": p.cell_ids.tolist(),
            }
            for p in predicted
        ]
        target_sampling, source_sampling = (generation_plan["target"], generation_plan["source_reference"])
        target_by_unit, source_by_unit = ({p.unit_id: p for p in full_targets}, {p.unit_id: p for p in full_sources})
        targets = [subset_full_gene_population(target_by_unit[r["unit_id"]], r["cell_ids"]) for r in target_sampling]
        sources = [subset_full_gene_population(source_by_unit[r["unit_id"]], r["cell_ids"]) for r in source_sampling]
    predicted = [_materialize_selected(p) for p in predicted]
    targets = [_materialize_selected(p) for p in targets]
    target_units, source_units = ([], [])
    with tqdm(
        total=1 + len(targets) + len(predicted),
        desc=f"Evaluate joint | {response.response_id}",
        unit="comparison",
        dynamic_ncols=True,
        mininterval=1.0,
    ) as progress:
        rows, exact = score_joint_predictions(predicted, targets, sources, panels.panels, projections=projections)
        progress.update(1)
        for target in targets:
            metrics, _ = score_joint_predictions(predicted, [target], sources, panels.panels, projections=projections)
            target_units.append(
                {
                    "unit_id": target.unit_id,
                    "cells": len(target.cell_ids),
                    "comparison": "this_target_unit_vs_all_prediction_units",
                    "metrics": metrics,
                }
            )
            progress.update(1)
        for prediction in predicted:
            metrics, _ = score_joint_predictions(
                [prediction], targets, sources, panels.panels, projections=projections
            )
            source_units.append(
                {
                    "unit_id": prediction.unit_id,
                    "cells": len(prediction.cell_ids),
                    "comparison": "this_prediction_source_unit_vs_all_target_units",
                    "metrics": metrics,
                }
            )
            progress.update(1)
    result = {
        "status": "complete",
        "schema": JOINT_EVALUATION_SCHEMA,
        "training_protocol": JOINT_PROTOCOL_ID,
        "mode": "joint",
        "spatial_metric_policy": JOINT_SPATIAL_METRIC_POLICY,
        "response_id": response.response_id,
        "context": response.context,
        "source_condition": response.source_condition,
        "target_condition": response.target_condition,
        "scientific_result": scientific_result,
        "expression_metric_definition": {
            "metrics": ["mse", "pcc_delta"],
            "scale": "raw_counts_log1p",
            "aggregation": "equal_biological_units_then_cells",
            "mse": "mean_squared_difference_of_population_mean_log1p_vectors",
            "pcc_delta": "pearson_of_prediction_minus_source_and_target_minus_source_means",
            "constant_delta_policy": "null",
        },
        "gene_panels": rows,
        "metrics": rows[0],
        "spatial_metrics": exact,
        "target_unit_rows": target_units,
        "prediction_source_unit_rows": source_units,
        "sampling": {
            "prediction": pred_sampling,
            "target": target_sampling,
            "source": source_sampling,
            "cap_per_population": cap,
            "base_seed": 42,
            "selection_stage": "before_generation" if generation_plan is not None else "historical_after_generation",
            "prediction_subsampling": generation_plan is None,
        },
        **(
            {
                "generation_policy": GENERATION_POLICY,
                "generation_budget": generation_plan["output_cells"],
                "generation_plan_sha256": plan_digest(generation_plan),
            }
            if generation_plan is not None
            else {}
        ),
        "count_integrity": identities,
        "target_access_stage": "metadata_budget_then_source_prediction_then_sealed_expression_and_coordinate_evaluation"
        if generation_plan is not None
        else "source_only_prediction_then_sealed_count_and_coordinate_evaluation",
        "target_geometry_used_for_prediction": False,
        "target_counts_used_for_prediction": False,
        "target_cell_count_used_for_prediction": generation_plan is not None,
        "prediction_unit_coverage": 1.0,
        "predicted_empty_cell_rate": sum(r["predicted_empty_cells"] for r in identities)
        / sum(r["cells"] for r in identities),
        "count_performance_metrics": "predicted_zero_inclusive",
        "biological_cell_pairing": False,
        "rna_subsampling": False,
        "uncertainty": "biological_unit_clustered_analysis_pending; cells_genes_and_model_seeds_are_not_replicates",
    }
    atomic_write_json(directory / "metrics.json", result)
    return result


@torch.inference_mode()
def evaluate_one_step(
    spec,
    response,
    predictions,
    output,
    *,
    setting,
    oracle_counts=None,
    projections=64,
    cap=2000,
    scientific_result=True,
    expected_provenance=None,
    generation_plan=None,
):
    """Return one response's metric record; targets are opened only by evaluators."""
    if setting == "joint":
        if oracle_counts is not None:
            raise ValueError("Joint evaluation does not accept OracleCounts as prediction inputs")
        return evaluate_joint_response(
            spec,
            response,
            predictions,
            output,
            projections=projections,
            cap=cap,
            scientific_result=scientific_result,
            expected_provenance=expected_provenance,
            generation_plan=generation_plan,
        )
    if setting not in {"oracle", "oracle_count"}:
        raise ValueError(f"Unknown setting: {setting}")
    if oracle_counts is None:
        raise ValueError("Oracle evaluation requires the coordinate-free counts used for generation")
    return evaluate_cell_response(
        spec,
        response,
        oracle_counts,
        predictions,
        output,
        projections=projections,
        scientific_result=scientific_result,
        generation_plan=generation_plan,
    )


def write_cell_prediction(
    model,
    oracle,
    sources,
    descriptor,
    output,
    *,
    seed,
    device="cpu",
    foundation_store=None,
    provenance=None,
    batch_size=256,
):
    if not isinstance(oracle, OracleCounts):
        raise TypeError("Cell prediction accepts coordinate-free OracleCounts only")
    oracle.validated()
    if not sources or any(p.genes != oracle.genes or not len(p.cell_ids) for p in sources):
        raise ValueError("Source cells and Oracle counts must share the frozen population and full gene axis")
    sources = sorted(sources, key=lambda p: p.unit_id)
    ends = np.cumsum([len(p.cell_ids) for p in sources])
    assignment_rng = np.random.default_rng(np.random.SeedSequence([seed, 41417]))
    rows = assignment_rng.integers(int(ends[-1]), size=len(oracle.cell_ids))
    units = np.searchsorted(ends, rows, side="right")
    donors = rows - np.r_[0, ends[:-1]][units]
    identities = [
        {
            "target_cell_id": str(cell),
            "source_unit": sources[int(unit)].unit_id,
            "source_cell_id": str(sources[int(unit)].cell_ids[int(donor)]),
        }
        for cell, unit, donor in zip(oracle.cell_ids, units, donors, strict=True)
    ]
    metadata = {
        "training_protocol": CELL_PROTOCOL_ID,
        "unit_id": oracle.unit_id,
        "context": oracle.context,
        "target_condition": oracle.condition,
        "geometry_prior": "one_uniform_source_cell_per_output_cell_all_genes; missing_gene_uses_same_donor_all_RNA",
        "geometry_rng": "cell_id_seeded_complete_RNA",
        "source_cell_assignment": identities,
        "biological_cell_pairing": False,
        "source_rna_subsampling": False,
    }
    readers = {}
    execution = {"requested_batch_size": batch_size}
    metadata["execution"] = execution

    def compute(first, last):
        batches, feature_requests, identities = ([], [], [])
        donor_lookup = {}
        offset = 0
        for unit in np.unique(units[first:last]):
            population = sources[int(unit)]
            if int(unit) not in readers:
                readers[int(unit)] = PopulationCellReader(population)
            selected = np.unique(donors[first:last][units[first:last] == unit])
            batch = readers[int(unit)].read(selected)
            batches.append(batch)
            identities.extend(f"{unit}:{cell}" for cell in batch.cell_ids)
            donor_lookup.update((((int(unit), int(row)), offset + i) for i, row in enumerate(selected)))
            if foundation_store is not None:
                feature_requests.append((population, batch.cell_ids, list(range(offset, offset + len(selected)))))
            offset += len(selected)
        source = concatenate_cell_batches(batches, identities)
        selected = np.asarray(
            [donor_lookup[int(u), int(d)] for u, d in zip(units[first:last], donors[first:last], strict=True)]
        )
        counts = np.asarray(oracle.counts[first:last], dtype=np.int64)
        if isinstance(model, CellClassical):
            if foundation_store is not None:
                raise ValueError("Classical controls do not consume foundation features")
            return (source, selected, counts, None)
        foundation = None if foundation_store is None else foundation_store.get_batch(feature_requests, source.genes)
        params = model(
            TensorCells.from_batch(source, device),
            torch.as_tensor(counts, dtype=torch.float32, device=device),
            torch.as_tensor(np.tile(descriptor, (last - first, 1)), dtype=torch.float32, device=device),
            torch.as_tensor(selected, device=device),
            foundation,
        )
        return (source, selected, counts, params.cpu().numpy())

    def geometry():
        if not isinstance(model, CellClassical):
            model.eval()
        with torch.inference_mode():
            for first, (source, selected, all_counts, parameters) in prediction_batches(
                len(oracle.cell_ids), batch_size, compute, execution, desc="Predict oracle"
            ):
                for local, donor in enumerate(selected):
                    cell = first + local
                    counts = all_counts[local : local + 1]
                    rng = np.random.default_rng(seed_from_parts(*[seed, str(oracle.cell_ids[cell])]))
                    if isinstance(model, CellClassical):
                        single = select_batch_cells(source, [donor], source.cell_ids[donor : donor + 1])
                        xy, genes, cells, fallback = sample_donor_rna(single, counts, np.array([0]), rng)
                        xy = apply_transform(
                            model.transform_cell(single, descriptor, device=device), xy, genes, device=device
                        ).astype(np.float32)
                    else:
                        xy, genes, cells, fallback = sample_donor_rna(source, counts, np.array([donor]), rng)
                        xy = apply_transform({"kind": "affine", "parameters": parameters[local]}, xy, genes).astype(
                            np.float32
                        )
                    yield (cells + cell, genes, xy, fallback)

    return write_oracle_prediction(
        oracle,
        sources,
        output,
        seed=seed,
        provenance={**(provenance or {}), "training_protocol": CELL_PROTOCOL_ID},
        geometry_iterator=geometry(),
        cell_metadata=metadata,
    )


GENERATION_POLICY = "matched_target_budget_before_generation_v1"


def freeze_generation_plan(directory, plan):
    path = Path(directory) / "generation_plan.json"
    sealed = {"sha256": plan_digest(plan), "plan": plan}
    if path.exists() and json.loads(path.read_text()) != sealed:
        raise ValueError("Pre-generation selection changed; predictions cannot be reused")
    if not path.exists():
        atomic_write_json(path, sealed)
    return sealed["sha256"]


def load_generation_plan(directory):
    sealed = json.loads((Path(directory) / "generation_plan.json").read_text())
    plan = sealed["plan"]
    if sealed["sha256"] != plan_digest(plan) or plan.get("policy") != GENERATION_POLICY:
        raise ValueError("Pre-generation selection seal changed")
    return plan


def make_generation_plan(spec, response, *, cap=2000):
    targets = list(iter_cell_axes(spec.index_dir, condition=response.target_condition, context=response.context))
    sources = list(iter_cell_axes(spec.index_dir, condition=response.source_condition, context=response.context))
    if any(p.dataset != spec.dataset for p in [*sources, *targets]):
        raise ValueError("Generation budget belongs to another dataset")
    target_rows = sample_cell_axes(targets, response_id=response.response_id, namespace="target", cap=cap)
    source_rows = sample_cell_axes(sources, response_id=response.response_id, namespace="source", cap=cap)
    return {
        "policy": GENERATION_POLICY,
        "dataset": spec.dataset,
        "response_id": response.response_id,
        "context": response.context,
        "source_condition": response.source_condition,
        "target_condition": response.target_condition,
        "cap": cap,
        "base_seed": 42,
        "output_cells": sum(row["selected_cells"] for row in target_rows),
        "target": target_rows,
        "source_reference": source_rows,
        "selection_stage": "before_generation",
        "target_expression_read": False,
        "target_geometry_read": False,
    }


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_joint_donors(sources, *, response_id, budget):
    """Only the scalar output budget crosses into source-donor selection.

    Distribute draws equally across source units; draw uniformly with replacement
    within each unit. If the budget cannot cover all units, choose units by seed.
    """
    sources = sorted(sources, key=lambda p: p.unit_id)
    if budget < 1 or not sources or any(not len(p.cell_ids) for p in sources):
        raise ValueError("Joint donor selection needs sources and a positive output budget")
    rng = np.random.default_rng(seed_from_parts(42, sources[0].dataset, response_id, "joint_donor_units"))
    allocations = np.full(len(sources), budget // len(sources), dtype=int)
    allocations[rng.permutation(len(sources))[: budget % len(sources)]] += 1
    rows = []
    for source, count in zip(sources, allocations, strict=True):
        if not count:
            continue
        ids = np.sort(source.cell_ids.astype(str))
        rng = np.random.default_rng(seed_from_parts(42, source.dataset, response_id, "joint_donors", source.unit_id))
        donors = ids[rng.integers(len(ids), size=int(count))].tolist()
        rows.append(
            {
                "unit_id": source.unit_id,
                "available_cells": len(ids),
                "selected_cells": len(donors),
                "source_cell_ids": donors,
            }
        )
    return rows


def validate_generation_plan(plan, spec, response, sources, targets, *, cap):
    """Reconcile a frozen selection with full observations after predictions seal."""
    expected = {
        "dataset": spec.dataset,
        "response_id": response.response_id,
        "context": response.context,
        "source_condition": response.source_condition,
        "target_condition": response.target_condition,
        "cap": cap,
        "base_seed": 42,
        "policy": GENERATION_POLICY,
        "selection_stage": "before_generation",
        "target_expression_read": False,
        "target_geometry_read": False,
    }
    if any((plan.get(k) != v for k, v in expected.items())):
        raise ValueError("Pre-generation selection response or information access changed")
    for name, populations, namespace in (("target", targets, "target"), ("source_reference", sources, "source")):
        if plan[name] != sample_cell_axes(populations, response_id=response.response_id, namespace=namespace, cap=cap):
            raise ValueError("Pre-generation cell selection differs from the frozen population")
    if plan["output_cells"] != sum(row["selected_cells"] for row in plan["target"]):
        raise ValueError("Generation budget differs from selected target cell count")


def load_joint_prediction(path):
    prediction = load_sealed_cell_prediction(path, schema=JOINT_PREDICTION_SCHEMA, mode="joint")
    metadata = prediction.storage_manifest
    if metadata.get("training_protocol") != JOINT_PROTOCOL_ID or any(
        metadata.get(key) is not False
        for key in ("target_counts_used_for_prediction", "target_coordinates_used_for_prediction")
    ):
        raise ValueError("Joint prediction violated source-only information access")
    budget_used = metadata.get("target_cell_count_used_for_prediction")
    if budget_used is not False and (
        budget_used is not True
        or metadata.get("generation_policy") != GENERATION_POLICY
        or (not isinstance(metadata.get("generation_budget"), int))
        or (metadata["generation_budget"] < metadata["cell_count"])
    ):
        raise ValueError("Joint target population size is permitted only as a frozen generation budget")
    validate_joint_prediction(prediction)
    return prediction


def validate_joint_prediction(prediction):
    """Multiplicity agrees with PREDICTED counts; no target data is an argument."""
    n, g = prediction.counts.shape
    metadata = prediction.storage_manifest
    if (
        not n
        or not g
        or n != len(prediction.cell_ids)
        or (g != len(prediction.genes))
        or (len(set(prediction.cell_ids.tolist())) != n)
    ):
        raise ValueError("Joint prediction axes are invalid")
    if any(not str(cell).startswith("joint:") for cell in prediction.cell_ids):
        raise ValueError("Joint output IDs no longer follow the sealed source cell axis")
    if n != metadata["source_cell_count"] or n != metadata["cell_count"]:
        raise ValueError("Joint output cell count must match its selected source assignments")
    if metadata.get("generation_policy") == GENERATION_POLICY:
        donors = metadata.get("source_donor_cell_ids", [])
        if len(donors) != n or not np.array_equal(prediction.cell_ids, joint_output_ids(donors)):
            raise ValueError("Joint generated IDs differ from the preselected source assignments")
    cloud = prediction.streamed_point_cloud
    if len(cloud.entry_cell_indices) and cloud.entry_cell_indices.max() >= n:
        raise ValueError("Joint RNA belongs to an unknown generated cell")
    order = np.argsort(cloud.entry_cell_indices, kind="stable")
    cells = cloud.entry_cell_indices[order]
    total = empty = 0
    for first in range(0, n, 128):
        counts = prediction.counts[first : first + 128]
        if counts.dtype.kind not in "iu" or np.any(counts < 0):
            raise ValueError("Joint count predictions must be nonnegative integers")
        observed = np.zeros_like(counts)
        begin, end = np.searchsorted(cells, [first, first + len(counts)])
        indices = order[begin:end]
        np.add.at(
            observed,
            (cloud.entry_cell_indices[indices] - first, cloud.entry_gene_indices[indices]),
            cloud.entry_lengths[indices],
        )
        if not np.array_equal(counts, observed):
            raise ValueError("Joint geometry multiplicities differ from predicted counts")
        total += int(counts.sum())
        empty += int(np.sum(counts.sum(1) == 0))
    if total != metadata["molecule_count"] or total != len(prediction.coordinates):
        raise ValueError("Joint RNA total disagrees with the sealed counts")
    return {
        "exact_predicted_cell_gene_counts": True,
        "cells": n,
        "genes": g,
        "molecules": total,
        "predicted_empty_cells": empty,
    }


def write_joint_prediction(
    model,
    source,
    descriptor,
    output,
    *,
    seed,
    device="cpu",
    foundation_store=None,
    provenance=None,
    batch_size=256,
    source_cell_ids=None,
    generation_budget=None,
):
    """One cell per preselected donor draw; no target expression or geometry input."""
    if not len(source.cell_ids):
        raise ValueError("Joint generation requires a nonempty source population")
    output = Path(output)
    sidecar = output.with_suffix(".h5.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    reader = PopulationCellReader(source)
    planned = source_cell_ids is not None
    if planned:
        lookup = {str(cell): i for i, cell in enumerate(source.cell_ids)}
        if not source_cell_ids or any(str(cell) not in lookup for cell in source_cell_ids):
            raise ValueError("Joint donors must be selected from the permitted source population")
        if not isinstance(generation_budget, int) or generation_budget < len(source_cell_ids):
            raise ValueError("Joint source assignments exceed the response generation budget")
        donor_rows = np.asarray([lookup[str(cell)] for cell in source_cell_ids])
        cell_ids = joint_output_ids(source_cell_ids)
    else:
        if generation_budget is not None:
            raise ValueError("A generation budget requires preselected source donors")
        donor_rows = np.arange(len(source.cell_ids))
        cell_ids = np.asarray([f"joint:{c}" for c in source.cell_ids])
    n, g = (len(cell_ids), len(source.genes))
    offset = entries = fallback = 0
    if not isinstance(model, CellJointClassical):
        model.eval()
    execution = {"requested_batch_size": batch_size}

    def compute(first, last):
        selected, assignment = np.unique(donor_rows[first:last], return_inverse=True)
        batch = reader.read(selected)
        if isinstance(model, CellJointClassical):
            if foundation_store is not None:
                raise ValueError("Classical Joint controls cannot consume frozen embeddings")
            return (batch, assignment, model.predict_counts(batch, descriptor)[assignment], None)
        foundation = (
            None
            if foundation_store is None
            else torch.as_tensor(foundation_store.get(source, batch.cell_ids), device=device)
        )
        tensor = TensorCells.from_batch(batch, device)
        condition = torch.as_tensor(np.tile(descriptor, (last - first, 1)), dtype=torch.float32, device=device)
        donors = torch.as_tensor(assignment, device=device)
        if hasattr(model, "joint_inference_parameters"):
            counts, params = model.joint_inference_parameters(tensor, condition, donors, foundation)
        else:
            counts = integer_counts(model.count_log1p(tensor, condition, donors, foundation).cpu().numpy())
            params = model(
                tensor, torch.as_tensor(counts, dtype=torch.float32, device=device), condition, donors, foundation
            )
        return (batch, assignment, counts, params.cpu().numpy())

    try:
        with h5py.File(temporary, "w") as f, torch.inference_mode():
            string = h5py.string_dtype("utf-8")
            f.create_dataset("cell_ids", data=cell_ids.astype(object), dtype=string)
            f.create_dataset("genes", data=np.asarray(source.genes, dtype=object), dtype=string)
            counts_ds = f.create_dataset("counts", (n, g), dtype="int64", chunks=(min(n, 128), g), compression="lzf")
            defined_ds = f.create_dataset(
                "spatial_defined", (n, g), dtype="bool", chunks=(min(n, 128), g), compression="lzf"
            )
            coordinates = f.create_dataset(
                "coordinates",
                (0, 2),
                maxshape=(None, 2),
                dtype="float32",
                chunks=(MOLECULE_BATCH_SIZE, 2),
                compression="lzf",
            )
            entry_ds = {
                key: f.create_dataset(key, (0,), maxshape=(None,), dtype="int64", chunks=(65536,))
                for key in ("entry_cells", "entry_genes", "entry_starts", "entry_lengths")
            }
            for first, (batch, assignment, batch_counts, parameters) in prediction_batches(
                n, batch_size, compute, execution, desc="Predict joint"
            ):
                for local in range(len(batch_counts)):
                    cell = first + local
                    donor = int(assignment[local])
                    rng_id = str(cell_ids[cell]) if planned else str(source.cell_ids[cell])
                    rng = np.random.default_rng(seed_from_parts(*[seed, source.unit_id, rng_id]))
                    counts = batch_counts[local : local + 1]
                    counts_ds[cell] = counts[0]
                    defined_ds[cell] = True
                    if not counts.sum():
                        continue
                    if isinstance(model, CellJointClassical):
                        from subcellpert.data.batching import select_batch_cells

                        single = select_batch_cells(batch, [donor], batch.cell_ids[donor : donor + 1])
                        xy, genes, _, missing = sample_donor_rna(single, counts, np.asarray([0]), rng)
                        xy = apply_transform(
                            model.transform_cell(single, descriptor, device=device), xy, genes, device=device
                        ).astype(np.float32)
                    else:
                        xy, genes, _, missing = sample_donor_rna(batch, counts, np.asarray([donor]), rng)
                        xy = apply_transform({"kind": "affine", "parameters": parameters[local]}, xy, genes).astype(
                            np.float32
                        )
                    if xy.shape != (int(counts.sum()), 2) or not np.isfinite(xy).all():
                        raise ValueError("Joint generator did not emit all finite predicted RNA")
                    coordinates.resize((offset + len(xy), 2))
                    coordinates[offset : offset + len(xy)] = xy
                    present = np.flatnonzero(counts[0])
                    lengths = counts[0, present]
                    if not np.array_equal(genes, np.repeat(present, lengths)):
                        raise ValueError("Joint generator changed the complete gene axis")
                    values = {
                        "entry_cells": np.full(len(present), cell),
                        "entry_genes": present,
                        "entry_starts": offset + np.r_[0, np.cumsum(lengths[:-1])],
                        "entry_lengths": lengths,
                    }
                    for key, array in values.items():
                        entry_ds[key].resize((entries + len(present),))
                        entry_ds[key][entries : entries + len(present)] = array
                    entries += len(present)
                    offset += len(xy)
                    fallback += int(missing)
        temporary.replace(output)
        atomic_write_json(
            sidecar,
            {
                "schema": JOINT_PREDICTION_SCHEMA,
                "training_protocol": JOINT_PROTOCOL_ID,
                "evaluation_mode": "joint",
                "status": "complete",
                "prediction": str(output.resolve()),
                "size_bytes": output.stat().st_size,
                "provenance": provenance or {},
                "source_unit_id": source.unit_id,
                "source_condition": source.condition,
                "context": source.context,
                "source_cell_count": n,
                "available_source_cell_count": len(source.cell_ids),
                "cell_count": n,
                "gene_count": g,
                "molecule_count": offset,
                "target_counts_used_for_prediction": False,
                "target_coordinates_used_for_prediction": False,
                "target_cell_count_used_for_prediction": planned,
                **(
                    {
                        "generation_policy": GENERATION_POLICY,
                        "generation_budget": generation_budget,
                        "source_donor_cell_ids": list(source_cell_ids),
                    }
                    if planned
                    else {}
                ),
                "count_policy": "predicted_zero_inclusive",
                "source_assignment": "preselected_donor_draws_all_genes"
                if planned
                else "one_output_per_source_cell_all_genes",
                "biological_cell_pairing": False,
                "rna_subsampling": False,
                "gene_axis_partitioned": False,
                "prior_fallback_molecules": fallback,
                "seed": seed,
                "random_seed_scheme": "seed_sequence_utf8_labels",
                "execution": execution,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)
    return load_joint_prediction(output)


@dataclass(frozen=True)
class OraclePrediction:
    cell_ids: np.ndarray
    genes: tuple[str, ...]
    counts: H5Array
    coordinates: PackedCoordinates
    spatial_defined: H5Array
    streamed_point_cloud: CellBalancedFullGenePointCloud
    storage_manifest: dict


def apply_transform(transform, xy, genes, *, device="cpu"):
    if torch.device(device).type == "cuda":
        from subcellpert.classical_cuda import apply_transform_cuda

        return apply_transform_cuda(transform, xy, genes, device=device)
    if transform["kind"] == "affine":
        params = transform["parameters"][genes]
        return np.einsum("mij,mj->mi", params[:, :4].reshape(-1, 2, 2), xy) + params[:, 4:]
    if transform["kind"] != "radial":
        raise ValueError("Unknown Oracle spatial transform")
    centered = xy - transform["centers"][genes]
    radius = np.linalg.norm(centered, axis=1)
    source_q = transform["source_quantiles"][genes]
    target_q = transform["target_quantiles"][genes]
    low = np.clip((radius[:, None] >= source_q).sum(1) - 1, 0, source_q.shape[1] - 2)
    rows = np.arange(len(genes))
    fraction = np.clip(
        (radius - source_q[rows, low]) / np.maximum(source_q[rows, low + 1] - source_q[rows, low], 1e-06), 0, 1
    )
    mapped = target_q[rows, low] + fraction * (target_q[rows, low + 1] - target_q[rows, low])
    return (
        transform["centers"][genes]
        + transform["shift"][genes]
        + centered * (mapped / np.maximum(radius, 1e-06))[:, None]
    )


def load_oracle_prediction(path):
    return load_sealed_cell_prediction(path, schema=ORACLE_PREDICTION_SCHEMA, mode="oracle_count")


def load_sealed_cell_prediction(path, *, schema, mode):
    """Read shared cell/RNA storage only after validating the selected mode's seal."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".h5.json").read_text())
    if manifest.get("schema") != schema or manifest.get("evaluation_mode") != mode:
        raise ValueError(f"Prediction is not a dedicated {mode} output")
    if manifest.get("size_bytes") != path.stat().st_size:
        raise ValueError("Prediction payload changed size or is incomplete")
    with h5py.File(path, "r") as f:
        cells = f["cell_ids"].asstr()[:]
        genes = tuple(f["genes"].asstr()[:])
        ec, eg, es, el = (f[k][:] for k in ["entry_cells", "entry_genes", "entry_starts", "entry_lengths"])
    order = np.lexsort((ec, eg))
    coordinates = PackedCoordinates(H5Array(path, "coordinates"), es[order], el[order])
    gene_lengths = np.bincount(eg, minlength=len(genes))
    cloud = CellBalancedFullGenePointCloud(
        coordinates,
        eg[order],
        ec[order],
        coordinates.starts,
        el[order],
        np.cumsum(gene_lengths) - gene_lengths,
        gene_lengths,
    ).validated()
    return OraclePrediction(
        cells, genes, H5Array(path, "counts"), coordinates, H5Array(path, "spatial_defined"), cloud, manifest
    )


def validate_oracle_count_identity(prediction, oracle):
    if prediction.genes != oracle.genes or not np.array_equal(prediction.cell_ids, oracle.cell_ids):
        raise ValueError("Oracle prediction axes changed")
    cloud = prediction.streamed_point_cloud
    order = np.argsort(cloud.entry_cell_indices, kind="stable")
    sorted_cells = cloud.entry_cell_indices[order]
    for first in range(0, len(oracle.cell_ids), 128):
        truth = oracle.counts[first : first + 128]
        if not np.array_equal(prediction.counts[first : first + 128], truth):
            raise ValueError("Oracle predicted counts differ from the supplied target counts")
        observed = np.zeros(truth.shape, dtype=np.int64)
        begin, end = np.searchsorted(sorted_cells, [first, first + len(truth)])
        indices = order[begin:end]
        np.add.at(
            observed,
            (cloud.entry_cell_indices[indices] - first, cloud.entry_gene_indices[indices]),
            cloud.entry_lengths[indices],
        )
        if not np.array_equal(observed, truth):
            raise ValueError("Oracle geometry multiplicities differ from exact target counts")
    return {
        "exact_cell_gene_counts": True,
        "molecules": int(prediction.counts.sum()),
        "cells": len(oracle.cell_ids),
        "genes": len(oracle.genes),
    }


def write_oracle_prediction(oracle, sources, output, *, seed, provenance, geometry_iterator, cell_metadata):
    if not isinstance(oracle, OracleCounts):
        raise TypeError("The predictor cannot receive a target geometry population")
    oracle.validated()
    output = Path(output)
    sidecar = output.with_suffix(".h5.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.tmp")
    n, g = oracle.counts.shape
    try:
        with h5py.File(temporary, "w") as f:
            string = h5py.string_dtype("utf-8")
            f.create_dataset("cell_ids", data=np.asarray(oracle.cell_ids, dtype=object), dtype=string)
            f.create_dataset("genes", data=np.asarray(oracle.genes, dtype=object), dtype=string)
            counts = f.create_dataset("counts", (n, g), dtype="int64", chunks=(min(128, n), g), compression="lzf")
            defined = f.create_dataset(
                "spatial_defined", (n, g), dtype="bool", chunks=(min(128, n), g), compression="lzf"
            )
            total = 0
            for first in range(0, n, 128):
                values = oracle.counts[first : first + 128]
                counts[first : first + 128] = values
                defined[first : first + 128] = True
                total += int(values.sum())
            coordinates = f.create_dataset(
                "coordinates",
                (total, 2),
                dtype="float32",
                maxshape=(None, 2),
                chunks=(MOLECULE_BATCH_SIZE, 2),
                compression="lzf",
            )
            entries = {
                key: f.create_dataset(key, (0,), maxshape=(None,), chunks=(65536,), dtype="int64")
                for key in ["entry_cells", "entry_genes", "entry_starts", "entry_lengths"]
            }
            offset, entry_count, fallback = (0, 0, 0)
            previous = None
            for cells, genes, xy, missing in geometry_iterator:
                if (
                    not len(xy)
                    or xy.shape != (len(cells), 2)
                    or len(genes) != len(cells)
                    or (not np.isfinite(xy).all())
                ):
                    raise ValueError("Invalid complete-cell prediction geometry")
                coordinates[offset : offset + len(xy)] = xy
                starts = np.flatnonzero(np.r_[True, (cells[1:] != cells[:-1]) | (genes[1:] != genes[:-1])])
                lengths = np.diff(np.r_[starts, len(xy)])
                ec, eg = (cells[starts], genes[starts])
                if previous == (int(ec[0]), int(eg[0])):
                    entries["entry_lengths"][entry_count - 1] += lengths[0]
                    starts, lengths, ec, eg = (starts[1:], lengths[1:], ec[1:], eg[1:])
                stop = entry_count + len(starts)
                for key, values in [
                    ("entry_cells", ec),
                    ("entry_genes", eg),
                    ("entry_starts", offset + starts),
                    ("entry_lengths", lengths),
                ]:
                    entries[key].resize((stop,))
                    entries[key][entry_count:stop] = values
                offset += len(xy)
                entry_count = stop
                fallback += missing
                previous = (int(cells[-1]), int(genes[-1]))
            if offset != total:
                raise ValueError("Oracle RNA total does not match exact target counts")
        temporary.replace(output)
        manifest = {
            "schema": ORACLE_PREDICTION_SCHEMA,
            "evaluation_mode": "oracle_count",
            "status": "complete",
            "prediction": str(output.resolve()),
            "size_bytes": output.stat().st_size,
            "provenance": provenance,
            "gene_count": g,
            "cell_count": n,
            "molecule_count": total,
            "target_coordinates_used_for_prediction": False,
            "target_counts_used_for_prediction": True,
            "count_policy": "exact_target_per_cell_gene",
            "count_head": "absent",
            "gene_axis_partitioned": False,
            "source_unit_ids": [s.unit_id for s in sources],
            "prior_fallback_molecules": fallback,
            "seed": seed,
            "random_seed_scheme": "seed_sequence_utf8_labels",
        }
        manifest.update(cell_metadata)
        manifest["schema"] = ORACLE_PREDICTION_SCHEMA
        atomic_write_json(sidecar, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    prediction = load_oracle_prediction(output)
    validate_oracle_count_identity(prediction, oracle)
    return prediction


def prediction_batches(length, batch_size, compute, report=None, *, desc="Predict", show_progress=True):
    """Report completed cells, including geometry writes, across adaptive batches."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Prediction batch size must be a positive integer")
    first = 0
    with tqdm(
        total=length,
        desc=desc,
        unit="cell",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not show_progress,
    ) as progress:
        while first < length:
            size = min(batch_size, length - first)
            try:
                result = compute(first, first + size)
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                batch_size = max(1, size // 2)
                gc.collect()
                torch.cuda.empty_cache()
                if report is not None:
                    report["oom_backoffs"] = report.get("oom_backoffs", 0) + 1
                progress.set_postfix(batch_size=batch_size, refresh=False)
                continue
            if report is not None:
                report["model_batches"] = report.get("model_batches", 0) + 1
                report["largest_batch"] = max(report.get("largest_batch", 0), size)
            yield (first, result)
            del result
            first += size
            progress.update(size)
