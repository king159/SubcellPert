"""Manifest status, result formatting, and completed-result integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from subcellpert.data.intervention import artifact_path_token
from subcellpert.evaluate_one_step import (
    GENERATION_POLICY,
    load_generation_plan,
    load_joint_prediction,
    load_oracle_prediction,
    make_generation_plan,
    plan_digest,
)
from subcellpert.utils import (
    CELL_CLASSICAL_SCHEMA,
    CELL_EVALUATION_SCHEMA,
    CELL_JOINT_CLASSICAL_SCHEMA,
    CELL_PROTOCOL_ID,
    CLASSICAL_BASELINES,
    GEARS_JOINT_SCHEMA,
    GEARS_ORACLE_SCHEMA,
    JOINT_EVALUATION_SCHEMA,
    JOINT_PROTOCOL_ID,
    cell_lineage,
    gears_implementation_binding,
)


def audit_cell_result(root, directory, spec, baseline, seed, configuration, *, smoke):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    joint = configuration.get("evaluation_mode") == "joint"
    mode = "joint" if joint else "oracle_count"
    protocol = JOINT_PROTOCOL_ID if joint else CELL_PROTOCOL_ID
    evaluation_schema = JOINT_EVALUATION_SCHEMA if joint else CELL_EVALUATION_SCHEMA
    prediction_loader = load_joint_prediction if joint else load_oracle_prediction
    expected = {
        **cell_lineage(root, spec, mode),
        "baseline": baseline,
        "model_seed": seed,
        "smoke": smoke,
        "configuration_values": configuration,
    }
    if baseline not in CLASSICAL_BASELINES:
        from subcellpert.data.features import load_cell_foundation

        expected.update(
            gene_foundation=load_cell_foundation(root, spec, baseline, configuration=configuration).binding,
            gears_implementation=gears_implementation_binding(),
        )
    if any((result.get(k) != v or result.get("binding", {}).get(k) != v for k, v in expected.items())):
        raise ValueError("Result bindings do not match the current frozen cell job")
    if result.get("status") != "complete" or result.get("scientific_result") is not (not smoke):
        raise ValueError("Result completion or smoke/scientific status is invalid")
    if joint and (
        result.get("target_cell_count_in_prediction") is not True
        or result.get("target_counts_in_prediction") is not False
    ):
        raise ValueError("Joint result summary must declare population-size access without target expression access")
    checkpoint = directory / "cell_model.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = CELL_CLASSICAL_SCHEMA if baseline in CLASSICAL_BASELINES else GEARS_ORACLE_SCHEMA
    if joint:
        schema = CELL_JOINT_CLASSICAL_SCHEMA if baseline in CLASSICAL_BASELINES else GEARS_JOINT_SCHEMA
    if saved.get("schema") != schema or saved.get("binding") != result.get("binding"):
        raise ValueError("Checkpoint schema or bindings disagree with the result")
    selection = saved.get("validation_selection", {})
    validation = selection.get("validation", {})
    validation_enabled = spec.validation_split.get("enabled", True)
    if result.get("validation_selection") != selection:
        raise ValueError("Checkpoint selection evidence disagrees with the result")
    if validation_enabled:
        if (
            validation.get("metric") != ("mse" if joint else "cell_balanced_spatial_swd")
            or validation.get("validation_used_in_gradient_training") is not False
            or validation.get("outer_test_used") is not False
            or (not validation.get("responses"))
        ):
            raise ValueError("Independent validation evidence is missing or inconsistent")
    elif (
        spec.code not in {"atm-rt", "ddx54-pd1"}
        or seed != 1
        or validation != {"status": "disabled", "reason": "protocol_no_validation"}
    ):
        raise ValueError("Invalid single-seed/no-validation experiment")
    expected_selection = (
        "fixed_analytic_train_only"
        if baseline in CLASSICAL_BASELINES
        else "best_validation"
        if validation_enabled
        else "final_epoch"
    )
    if (
        saved.get("checkpoint_selection") != expected_selection
        or result.get("checkpoint_selection") != expected_selection
    ):
        raise ValueError("Checkpoint selection does not follow the independent validation protocol")
    if baseline not in CLASSICAL_BASELINES:
        history = saved.get("history", [])
        if not history or history[-1]["optimizer_steps"] != result.get("optimizer_steps"):
            raise ValueError("Actual optimizer history is missing or inconsistent")
        if validation_enabled:
            if any("validation" not in h for h in history):
                raise ValueError("Every trained epoch needs independent validation")
            initial = saved.get("initial_validation")
            initial_path = directory / "initial_validation.json"
            if not initial or not initial_path.exists() or json.loads(initial_path.read_text()) != initial:
                raise ValueError("Initialization validation evidence is missing or changed")
            candidates = [{"epoch": 0, "validation": initial}, *history]
            best = min(candidates, key=lambda h: (h["validation"]["score"], h["epoch"]))
            if selection.get("epoch") != best["epoch"] or validation != best["validation"]:
                raise ValueError("Prediction checkpoint is not the best validation epoch")
        elif (
            any("validation" in h for h in history)
            or selection.get("epoch") != history[-1]["epoch"]
            or selection.get("selection") != "final_epoch"
            or (selection.get("test_used_for_selection") is not False)
        ):
            raise ValueError("No-validation fits must use the fixed final epoch without test selection")
        if smoke:
            if result["optimizer_steps"] != 2:
                raise ValueError("Cell neural smoke requires two actual optimizer updates")
        else:
            epochs = configuration["training"]["epochs"]
            if len(history) != epochs or any(
                (not h["complete_epoch"] or h["epoch"] != i + 1 for i, h in enumerate(history))
            ):
                raise ValueError(f"Formal fitting requires all {epochs} complete cell epochs")
            batch = saved["batch_profile"]["batch_size"]
            expected_steps = history[0]["training_examples"] // batch * epochs
            if result["optimizer_steps"] != expected_steps:
                raise ValueError("Optimizer steps do not match the real shuffled cell epochs")
    expected_responses = spec.responses[:1] if smoke else spec.responses
    if result.get("completed_responses") != len(expected_responses) or len(result.get("responses", [])) != len(
        expected_responses
    ):
        raise ValueError("Response matrix is incomplete")
    for expected_response, row in zip(expected_responses, result["responses"], strict=True):
        if row.get("schema") != evaluation_schema or row.get("training_protocol") != protocol:
            raise ValueError("Evaluation does not implement the current cell protocol")
        if row.get("response_id") != expected_response.response_id or row.get("scientific_result") is not (not smoke):
            raise ValueError("Evaluation response bindings changed")
        if row.get("target_geometry_used_for_prediction") is not False or row.get("rna_subsampling") is not False:
            raise ValueError("Evaluation violates the full-RNA count-only prediction boundary")
        if [p.get("gene_panel") for p in row.get("gene_panels", [])] != ["all_gene", "top_100_deg"]:
            raise ValueError("Both frozen evaluation panels are required")
        integrity_key = "exact_predicted_cell_gene_counts" if joint else "exact_cell_gene_counts"
        if not row.get("count_integrity") or not all(r.get(integrity_key) is True for r in row["count_integrity"]):
            raise ValueError("Exact per-cell/gene RNA integrity was not verified")
        if joint and (
            row.get("target_counts_used_for_prediction") is not False
            or row.get("target_cell_count_used_for_prediction") is not True
        ):
            raise ValueError("Joint may use the population-size budget, but never target expression counts")
        response_dir = directory / "responses" / artifact_path_token(expected_response.response_id)
        if json.loads((response_dir / "metrics.json").read_text()) != row:
            raise ValueError("Evaluation metrics differ from the completed result")
        plan = load_generation_plan(response_dir)
        cap = 64 if smoke else configuration["evaluation"]["cell_cap_per_group"]
        if plan != make_generation_plan(spec, expected_response, cap=cap):
            raise ValueError("Generation selection differs from the frozen metadata-only budget")
        if (
            row.get("generation_policy") != GENERATION_POLICY
            or row.get("generation_budget") != plan["output_cells"]
            or row.get("generation_plan_sha256") != plan_digest(plan)
            or (row.get("sampling", {}).get("selection_stage") != "before_generation")
            or (row.get("sampling", {}).get("prediction_subsampling") is not False)
            or (sum(r["cells"] for r in row["count_integrity"]) != plan["output_cells"])
        ):
            raise ValueError("Generated and evaluated cell counts must match the pre-generation target budget")
        paths = sorted(response_dir.glob("prediction_*.h5"))
        if len(paths) != len(row["count_integrity"]):
            raise ValueError("Sealed prediction units are missing")
        actual_cells = 0
        if joint:
            inputs = json.loads((response_dir / "joint_source_input.json").read_text())
            if (
                inputs.get("generation_plan_sha256") != plan_digest(plan)
                or inputs.get("generation_budget") != plan["output_cells"]
                or inputs.get("target_cell_count_loaded") is not True
                or (inputs.get("target_counts_loaded") is not False)
                or (inputs.get("target_geometry_loaded") is not False)
            ):
                raise ValueError("Joint source input violates the population-size-only metadata boundary")
            expected_units = {r["unit_id"]: r for r in inputs["donors"]}
        else:
            expected_units = {r["unit_id"]: r for r in plan["target"]}
        seen_units = set()
        for path in paths:
            prediction = prediction_loader(path)
            actual_cells += len(prediction.cell_ids)
            meta = prediction.storage_manifest
            unit = meta["source_unit_id"] if joint else meta["unit_id"]
            if unit not in expected_units or unit in seen_units:
                raise ValueError("Generated units differ from the pre-generation selection")
            seen_units.add(unit)
            selected = expected_units[unit]
            if len(prediction.cell_ids) != selected["selected_cells"]:
                raise ValueError("Generated unit cell count differs from its pre-generation allocation")
            if joint:
                if meta.get("source_donor_cell_ids") != selected["source_cell_ids"]:
                    raise ValueError("Joint donors differ from the pre-generation source assignments")
            elif prediction.cell_ids.tolist() != selected["cell_ids"]:
                raise ValueError("Oracle cell IDs differ from the pre-generation selection")
            provenance = prediction.storage_manifest["provenance"]
            if provenance.get("binding") != result["binding"]:
                raise ValueError("Prediction does not belong to the audited checkpoint")
            if provenance.get("generation_plan_sha256") != plan_digest(plan):
                raise ValueError("Prediction does not belong to the audited generation selection")
            if joint and provenance.get("response_id") != expected_response.response_id:
                raise ValueError("Joint prediction source input or response changed")
        if actual_cells != plan["output_cells"] or seen_units != set(expected_units):
            raise ValueError("Sealed outputs do not exactly exhaust the common generation budget")
    return {
        "status": "passed",
        "dataset_code": spec.code,
        "baseline": baseline,
        "fit_group": spec.fit_id,
        "seed": seed,
        "responses": len(expected_responses),
        "smoke": smoke,
    }


def format_run_summary(result):
    """Keep CLI completion messages consistent across models and settings."""
    keys = ("status", "dataset", "baseline", "optimizer_steps", "completed_responses")
    return json.dumps({key: result[key] for key in keys}, sort_keys=True)


def read_benchmark_status(root):
    """Derive observation and plan counts without inferring completion or upload.

    This reads JSON metadata only. It does not claim to verify RNA payload bytes,
    completed model results, remote Hugging Face commits, or publication status.
    """
    root = Path(root).resolve()
    evidence = {}

    def read(relative):
        relative = Path(relative)
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"Manifest path must stay within the release: {relative}")
        payload = path.read_bytes()
        evidence[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {relative}")
        return value

    release = read("manifest.json")
    if release.get("schema_version") != 1 or release.get("layout") != "population_only_public_release":
        raise ValueError("Unsupported release manifest schema or layout")
    rows, pending, seen = ([], [], set())
    for entry in release["datasets"]:
        dataset = entry["id"]
        if dataset in seen:
            raise ValueError(f"Duplicate admitted dataset: {dataset}")
        seen.add(dataset)
        metadata = read(entry["metadata_manifest"])
        if metadata.get("dataset") != dataset:
            raise ValueError(f"Metadata identity differs from the release: {dataset}")
        if entry.get("status", "complete") != "complete" or metadata.get("status") != "complete":
            pending.append(dataset)
            continue
        population = read(Path(entry["population"]) / "manifest.json")
        inference = read(Path(entry["population"]) / "inference_cells/manifest.json")
        if population.get("status") != "complete" or inference.get("status") != "complete":
            raise ValueError(f"An admitted materialized dataset has incomplete population metadata: {dataset}")
        if population.get("dataset") != dataset or inference.get("dataset") != dataset:
            raise ValueError(f"Population identity differs from the release: {dataset}")
        totals = population["totals"]
        cells, molecules = (totals["declared_cells"], totals["molecules"])
        if any(type(value) is not int or value < 0 for value in (cells, molecules)):
            raise ValueError(f"Invalid population totals: {dataset}")
        if inference["totals"]["declared_cells"] != cells:
            raise ValueError(f"Declared-cell and inference-cell totals disagree: {dataset}")
        genes = population["inventory"]["genes"]
        rows.append({"dataset": dataset, "cells": cells, "rna_molecules": molecules, "genes": len(genes)})
    schedule_path = "benchmark/schedules/cell_current.json"
    schedule = read(schedule_path) if (root / schedule_path).is_file() else None
    plan = None
    if schedule is not None:
        jobs = schedule["jobs"]
        identities = [(j["dataset"], j["baseline"], j["mode"], j["seed"], j["fit_group"]) for j in jobs]
        if len(identities) != len(set(identities)) or schedule["formal_fits"] != len(jobs):
            raise ValueError("Scheduled fit identities are duplicated or disagree with formal_fits")
        plan = {
            "status": schedule["status"],
            "fits": len(jobs),
            "datasets": len({j["dataset"] for j in jobs}),
            "baselines": len({j["baseline"] for j in jobs}),
            "split_family_instances": len({(j["dataset"], j["seed"], j["fit_group"]) for j in jobs}),
        }
    return {
        "release_id": release["release_id"],
        "release_status": release["status"],
        "observation_inventory": {
            "admitted_datasets": len(seen),
            "materialized_datasets": len(rows),
            "pending_datasets": pending,
            "nonempty_cells": sum(row["cells"] for row in rows),
            "rna_molecules": sum(row["rna_molecules"] for row in rows),
            "datasets": rows,
        },
        "experiment_plan": plan,
        "completed_results": None,
        "uploaded_revision": None,
        "verification_scope": "local_metadata_only",
        "evidence_sha256": evidence,
    }
