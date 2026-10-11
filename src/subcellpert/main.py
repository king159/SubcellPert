"""Configuration, command-line parsing, and benchmark run orchestration."""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from subcellpert import distributed
from subcellpert.data import load_fit_records
from subcellpert.data.features import (
    foundation_metadata_from_config,
    load_cell_foundation,
    load_foundation_configuration,
)
from subcellpert.data.intervention import artifact_path_token
from subcellpert.data.populations import iter_full_gene_populations
from subcellpert.data.sampling import iter_oracle_counts, sample_oracle_counts
from subcellpert.data.splits import resolve_fit_specs
from subcellpert.evaluate.reporting import format_run_summary, read_benchmark_status
from subcellpert.evaluate_one_step import (
    BestValidationCheckpoint,
    assert_independent_records,
    evaluate_one_step,
    evaluate_validation,
    freeze_generation_plan,
    load_joint_prediction,
    load_oracle_prediction,
    make_generation_plan,
    select_joint_donors,
    validate_oracle_count_identity,
)
from subcellpert.models.shared import ModelInput
from subcellpert.precision import check_precision_device
from subcellpert.train_one_step import (
    CELL_JOINT_OBJECTIVE,
    CELL_ORACLE_OBJECTIVE,
    CellObjective,
    configured_batch_profile,
    resolved_batch_configuration,
    train_model,
    train_one_step,
)
from subcellpert.utils import (
    CELL_CLASSICAL_SCHEMA,
    CELL_JOINT_CLASSICAL_SCHEMA,
    CLASSICAL_BASELINES,
    GEARS_JOINT_SCHEMA,
    GEARS_ORACLE_SCHEMA,
    SUPPORTED_BASELINES,
    SUPPORTED_FOUNDATION_BASELINES,
    _check_binding,
    _save_torch,
    atomic_write_json,
    cell_lineage,
    gears_implementation_binding,
    get_model,
    reset_run_output,
    seed_from_parts,
    source_gene_graph,
    validate_cell_configuration,
    validate_joint_configuration,
)


def _root_path(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path, overrides=(), *, _parents=()):
    """Merge relative `extends` files, then CLI dotlist values, and resolve YAML.

    Dataset/model/setting/seed files inherit a frozen protocol YAML. Relative
    inheritance is anchored to the declaring file, independent of shell cwd.
    """
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"Cyclic config inheritance: {path}")
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    bases = config.pop("extends", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) and (not OmegaConf.is_list(bases)):
        raise ValueError("extends must be a relative path or a list of relative paths")
    merged = OmegaConf.create({})
    for base in bases:
        merged = OmegaConf.merge(merged, load_config(path.parent / base, _parents=(*_parents, path)))
    config = OmegaConf.merge(merged, config, OmegaConf.from_dotlist(list(overrides)))
    return config


def main(argv=None, *, default_mode="oracle_count"):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "status":
        parser = argparse.ArgumentParser(description="Read local benchmark manifest status")
        parser.add_argument("--root", type=Path, default=Path.cwd())
        parser.add_argument("--json", action="store_true")
        args = parser.parse_args(argv[1:])
        print(json.dumps(read_benchmark_status(args.root), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--configuration", type=Path, help="Dataset × model × setting × seed YAML")
    parser.add_argument("--project-root", "--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset")
    parser.add_argument("--model", "--baseline", choices=SUPPORTED_BASELINES)
    parser.add_argument("--setting", "--mode", choices=("joint", "oracle", "oracle_count"))
    parser.add_argument("--seed", type=int, help="Frozen data-split seed (1, 2 or 3); no separate model-seed sweep")
    parser.add_argument("--fit-group", help="Select one frozen fit family within the split seed")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--feature-access", choices=("online",))
    parser.add_argument("--expected-lineage", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Bounded development check; not a scientific result")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration and jobs; do not execute")
    parser.add_argument(
        "--resume", action="store_true", help="Preserve outputs and resume a compatible complete-epoch checkpoint"
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. run.device=cpu")
    args = parser.parse_args(argv)
    if args.config is None and args.setting is None:
        args.setting = default_mode
    resolved = prepare_run(args)
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size > 1:
        if resolved["baseline"] not in SUPPORTED_FOUNDATION_BASELINES:
            raise ValueError("Distributed training is supported for the five trainable FM baselines")
        local = distributed.local_batch_size(resolved["configuration"]["training"]["batch_size"], size)
        resolved["distributed_execution"] = {
            "world_size": size,
            "global_batch_size": local * size,
            "local_batch_size": local,
        }
    if args.dry_run:
        if int(os.environ.get("RANK", "0")) == 0:
            print(OmegaConf.to_yaml(OmegaConf.create(resolved), resolve=True))
        return
    with distributed.launch(resolved["device"]) as device:
        resolved["device"] = device
        _execute_jobs(resolved)


def _execute_jobs(resolved):
    for job in resolved["jobs"]:
        output = Path(job["output"])
        if distributed.primary():
            if resolved.get("resume", False):
                if not any((output / name).exists() for name in ("cell_resume.pt", "cell_model.pt")):
                    raise ValueError(f"No checkpoint to resume in {output}")
            else:
                reset_run_output(output)
                atomic_write_json(output / "run_config.json", {**resolved, "world_size": distributed.world_size()})
        distributed.barrier()
        try:
            print(
                f"[run] {resolved['dataset']} | {resolved['baseline']} | {resolved['mode']} | "
                f"fit={job['fit_group']} | seed={resolved['seed']} | device={resolved['device']}\n"
                f"[run] Preparing training | output={output}",
                flush=True,
            )
            result = run_cell_job(
                root=resolved["root"],
                output=output,
                dataset=resolved["dataset"],
                baseline=resolved["baseline"],
                fit_id=job["fit_group"],
                seed=resolved["seed"],
                configuration=resolved["configuration"],
                device_name=resolved["device"],
                smoke=resolved["smoke"],
                gpu_memory_fraction=resolved["gpu_memory_fraction"],
                feature_access=resolved["feature_access"],
                expected_lineage=resolved["expected_lineage"],
                train_step=train_one_step,
                evaluate_step=evaluate_one_step,
            )
        except Exception as error:
            if distributed.primary():
                atomic_write_json(output / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
            raise
        distributed.barrier()
        if distributed.primary():
            print(format_run_summary(result))


def prepare_run(args):
    """Resolve a run without opening RNA, loading encoders or writing outputs."""
    root = args.project_root.resolve()
    if args.config is None:
        setting = args.setting or "oracle_count"
        filename = "cell_joint.yaml" if setting == "joint" else "cell_oracle.yaml"
        config_path = root / "configs/experiments" / filename
    else:
        config_path = args.config.resolve()
    config = load_config(config_path, args.overrides)
    explicit = {}
    for key, value in (
        ("dataset", args.dataset),
        ("model.name", args.model),
        ("seed", args.seed),
        ("setting", args.setting),
        ("run.device", args.device),
        ("run.output", str(args.output) if args.output else None),
        ("run.gpu_memory_fraction", args.gpu_memory_fraction),
        ("run.feature_access", args.feature_access),
    ):
        if value is not None:
            OmegaConf.update(config, key, value, merge=False)
    if args.smoke:
        explicit["smoke"] = True
    config = OmegaConf.merge(config, explicit)
    setting = config.get("setting", config.get("evaluation_mode", "oracle_count"))
    mode = "oracle_count" if setting == "oracle" else setting
    if mode not in {"joint", "oracle_count"} or config.get("evaluation_mode") != mode:
        raise ValueError("The selected setting must match the inherited protocol YAML")
    dataset = config.get("dataset")
    baseline = config.get("model", {}).get("name")
    seed = config.get("seed", 1)
    if not isinstance(dataset, str) or not dataset or baseline not in SUPPORTED_BASELINES:
        raise ValueError("Provide dataset and model.name in YAML, or --dataset and --model")
    if type(seed) is not int or seed not in {1, 2, 3}:
        raise ValueError("seed must be one of the frozen split seeds: 1, 2, 3")
    if args.config is None and baseline in SUPPORTED_FOUNDATION_BASELINES:
        config = OmegaConf.merge(load_foundation_configuration(root, dataset, baseline), config)
    run = config.get("run", {})
    feature_access = run.get("feature_access", "online")
    if feature_access != "online":
        raise ValueError("run.feature_access must be online for device tensor execution")
    fraction = run.get("gpu_memory_fraction", 1.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or (not 0 < fraction <= 1):
        raise ValueError("run.gpu_memory_fraction must lie in (0, 1]")
    smoke = config.get("smoke", False)
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be a YAML boolean")
    protocol = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for key in ("dataset", "setting", "seed", "run", "smoke"):
        protocol.pop(key, None)
    protocol["model"].pop("name", None)
    (validate_joint_configuration if mode == "joint" else validate_cell_configuration)(protocol)
    schedule = json.loads((root / "benchmark/schedules/cell_current.json").read_text())
    fits = list(
        dict.fromkeys(
            row["fit_group"]
            for row in schedule["jobs"]
            if row["dataset"] == dataset
            and row["baseline"] == baseline
            and row["seed"] == seed
            and row["mode"] == mode
        )
    )
    selected = [args.fit_group] if args.fit_group else list(run.get("fit_groups", fits))
    if not selected or len(selected) != len(set(selected)) or any(fit not in fits for fit in selected):
        raise ValueError(f"Invalid or unscheduled fits for {dataset}/{mode}/seed_{seed}: {selected}")
    if baseline in SUPPORTED_FOUNDATION_BASELINES:
        for spec in resolve_fit_specs(root, dataset):
            if spec.fit_id in selected:
                foundation_metadata_from_config(root, spec, baseline, protocol)
    if run.get("output") is not None:
        output = _root_path(root, run["output"])
        jobs = [{"fit_group": fit, "output": str(output if args.fit_group else output / fit)} for fit in selected]
    else:
        output = root / "outputs"
        if smoke:
            output /= "smoke"
        output = output / ("oracle" if mode == "oracle_count" else "joint") / dataset / baseline
        jobs = [{"fit_group": fit, "output": str(output / fit)} for fit in selected]
    expected_lineage = (
        json.loads(_root_path(root, args.expected_lineage).read_text()) if args.expected_lineage else None
    )
    return {
        "root": str(root),
        "config_path": str(config_path),
        "dataset": dataset,
        "baseline": baseline,
        "mode": mode,
        "seed": seed,
        "device": run.get("device", "cuda" if baseline in SUPPORTED_FOUNDATION_BASELINES else "cpu"),
        "smoke": smoke,
        "resume": getattr(args, "resume", False),
        "gpu_memory_fraction": fraction,
        "feature_access": feature_access,
        "configuration": protocol,
        "expected_lineage": expected_lineage,
        "jobs": jobs,
    }


def restore_foundation_batch(foundation, profile):
    """Restore fixed execution sizes without silently changing a saved encoder cap."""
    encoder = distributed.profile_encoder_batch(profile)
    foundation.set_training_batch_size(
        distributed.local_batch_size(profile["batch_size"], profile.get("world_size", 1))
    )
    if foundation.encoder_batch_size != encoder:
        raise ValueError("Saved encoder batch differs from the configured encoder batch; use a new output")


def run_cell_job(
    *,
    root,
    output,
    dataset,
    baseline,
    fit_id,
    seed,
    configuration,
    device_name="cpu",
    smoke=False,
    gpu_memory_fraction=1.0,
    expected_lineage=None,
    feature_access="online",
    train_step=None,
    evaluate_step=None,
):
    root, output = (Path(root).resolve(), Path(output).resolve())
    if feature_access != "online":
        raise ValueError("Unknown source feature access policy")
    joint = configuration.get("evaluation_mode") == "joint"
    mode = "joint" if joint else "oracle_count"
    configuration_values = (validate_joint_configuration if joint else validate_cell_configuration)(configuration)
    adapter = get_model(baseline, mode)
    if train_step is None:
        from subcellpert.train_one_step import train_one_step

        train_step = train_one_step
    if evaluate_step is None:
        from subcellpert.evaluate_one_step import evaluate_one_step

        evaluate_step = evaluate_one_step
    neural_schema = GEARS_JOINT_SCHEMA if joint else GEARS_ORACLE_SCHEMA
    classical_schema = CELL_JOINT_CLASSICAL_SCHEMA if joint else CELL_CLASSICAL_SCHEMA
    prediction_loader = load_joint_prediction if joint else load_oracle_prediction
    if baseline not in SUPPORTED_BASELINES:
        raise ValueError("Unknown complete-cell baseline")
    specs = resolve_fit_specs(root, dataset)
    spec = (
        next(s for s in specs if s.metadata.get("split_seed", seed) == seed)
        if fit_id is None
        else next(s for s in specs if s.fit_id == fit_id)
    )
    if spec.metadata.get("split_seed", seed) != seed:
        raise ValueError("Run seed must match the frozen split seed")
    validation_enabled = spec.validation_split.get("enabled", True)
    if not validation_enabled and dataset not in {"atm-rt", "ddx54-pd1"}:
        raise ValueError("Validation may only be disabled for ATM and DDX54")
    lineage = cell_lineage(root, spec, mode)
    if expected_lineage is not None and expected_lineage != lineage:
        raise ValueError("Frozen task, split, data or protocol changed")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    precision = configuration["training"].get("precision", "fp32")
    if baseline in SUPPORTED_FOUNDATION_BASELINES:
        check_precision_device(precision, device)
    matmul_precision = configuration["training"].get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
    foundation = (
        load_cell_foundation(root, spec, baseline, configuration=configuration)
        if baseline in SUPPORTED_FOUNDATION_BASELINES
        else None
    )
    if foundation is not None:
        print(
            f"[run] FM configuration | baseline={baseline} | "
            f"weights={foundation.metadata.get('model_directory', foundation.metadata.get('model_checkpoint'))} | "
            f"source_conditions={foundation.metadata['source_conditions']} | features=online",
            flush=True,
        )
        foundation.device = device
        options = configuration.get("foundation", {})
        compile_encoder = options.get("compile_encoder", False)
        if compile_encoder and baseline != "state_se":
            raise ValueError("foundation.compile_encoder currently supports STATE-SE")
        foundation.metadata["compile_encoder"] = compile_encoder
        foundation.metadata["scfoundation_gene_context"] = options.get("scfoundation_gene_context", "assay")
        if isinstance(configuration["training"]["batch_size"], int):
            foundation.set_training_batch_size(distributed.local_batch_size(configuration["training"]["batch_size"]))
        foundation.use_online_features()
    binding = {
        **lineage,
        "baseline": baseline,
        "model_seed": configuration["training"].get("model_seed", seed),
        "smoke": smoke,
        "configuration_values": configuration_values,
    }
    if foundation is not None:
        binding.update(gene_foundation=foundation.binding, gears_implementation=gears_implementation_binding())
    base = {
        **binding,
        "binding": binding,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "scientific_result": not smoke,
        "separately_trained_for_oracle": not joint,
        **(
            {
                "separately_trained_for_joint": True,
                "target_counts_in_prediction": False,
                "target_cell_count_in_prediction": True,
                "training_objective": CELL_JOINT_OBJECTIVE,
            }
            if joint
            else {}
        ),
        "target_geometry_in_prediction": False,
        "held_out_target_geometry_in_fit": False,
        "biological_cell_pairing": False,
        "rna_subsampling": False,
        "training_objective": (CELL_JOINT_OBJECTIVE if joint else CELL_ORACLE_OBJECTIVE)
        if baseline in SUPPORTED_FOUNDATION_BASELINES
        else "analytic_or_parameter_free",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json = atomic_write_json if distributed.primary() else lambda *args, **kwargs: None
    write_json(
        output / "feature_access_execution.json",
        {
            "feature_access_policy": "online_source_features_no_disk_cache_v1"
            if foundation is not None
            else "not_applicable_no_foundation",
            "feature_device": str(device) if foundation is not None else None,
            "encoder_batch_policy": (
                "fixed_encoder_microbatch"
                if isinstance(configuration["training"]["batch_size"], int)
                and foundation.encoder_batch_size < distributed.local_batch_size(configuration["training"]["batch_size"])
                else "match_training"
            )
            if foundation is not None
            else None,
            "matmul_precision": matmul_precision,
            "compute_precision": precision if foundation is not None else "not_applicable",
            "precision_transition": configuration.get("precision_transition"),
            "world_size": distributed.world_size(),
            "global_batch_size": configuration["training"]["batch_size"],
            "local_batch_size": distributed.local_batch_size(configuration["training"]["batch_size"])
            if foundation is not None and isinstance(configuration["training"]["batch_size"], int)
            else None,
            "encoder_batch_size": foundation.encoder_batch_size if foundation else None,
            "compile_encoder": configuration.get("foundation", {}).get("compile_encoder", False),
            "dataset": dataset,
            "baseline": baseline,
            "activated_at": time.time(),
            "whole_fit_cache_reserved": False,
            "persistent_features_written": False,
            "cell_feature_lru_enabled": False,
            "model_or_objective_changed": False,
            "foundation_configuration_source": "resolved_model_and_dataset_yaml" if foundation is not None else None,
        },
    )
    result_path = output / "result.json"
    checkpoint = output / "cell_model.pt"
    resume_path = output / "cell_resume.pt"
    best_path = output / "cell_best_validation.pt"
    execution_config_path = output / "resolved_config.json"
    if execution_config_path.exists() and foundation is not None:
        execution = json.loads(execution_config_path.read_text())
        _check_binding(execution, binding, "Resolved execution configuration")
        concrete = execution["configuration"]
        if (
            not checkpoint.exists()
            and concrete.get("distributed_execution", {}).get("world_size", 1) != distributed.world_size()
        ):
            raise ValueError("Resume WORLD_SIZE changed; use a new output or an explicit checkpoint migration")
        restore_foundation_batch(
            foundation,
            {
                "batch_size": concrete["training"]["batch_size"],
                "encoder_batch_size": concrete["foundation"]["encoder_batch_size"],
                "world_size": concrete.get("distributed_execution", {}).get("world_size", 1),
            },
        )
    # Completed fits and final inference are exclusively rank-zero work. Other
    # ranks wait at the outer job barrier, without entering training collectives.
    if not distributed.primary() and checkpoint.exists():
        return None
    if result_path.exists():
        result = json.loads(result_path.read_text())
        _check_binding(result, binding, "Completed result")
        if result.get("status") == "complete":
            if not checkpoint.exists():
                raise ValueError("Completed result checkpoint has changed")
            for path in sorted((output / "responses").glob("*/prediction_*.h5")):
                prediction_loader(path)
            if joint:
                from subcellpert.evaluate.reporting import audit_cell_result

                audit_cell_result(root, output, spec, baseline, seed, configuration, smoke=smoke)
            return result
    started = time.time()
    phase_seconds = {
        "source_feature_validation": 0.0,
        "training_and_validation": 0.0,
        "prediction": 0.0,
        "evaluation": 0.0,
    }
    prediction_batch_size = configuration.get("inference", {}).get("batch_size", 256)

    def status(phase, **values):
        if not distributed.primary():
            return
        atomic_write_json(output / "status.json", {**base, "status": phase, "updated_at": time.time(), **values})
        event = {"phase": phase, "timestamp": time.time(), **values}
        if isinstance(event.get("epoch"), dict):
            event["epoch"] = {k: v for k, v in event["epoch"].items() if not isinstance(v, (dict, list))}
        with (output / "events.jsonl").open("a") as log:
            log.write(json.dumps(event) + "\n")

    def validate_sources(populations):
        before = time.perf_counter()
        phase = "validating_online_source_features"
        status(phase)
        report = foundation.validate_sources(populations, on_progress=lambda row: status(phase, source_features=row))
        phase_seconds["source_feature_validation"] += time.perf_counter() - before
        write_json(output / "foundation_execution.json", report)

    train = configuration["training"]
    validation_config = configuration["validation"]
    history, profile, resume = ([], None, None)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _check_binding(saved, binding, "Checkpoint")
        schema = classical_schema if baseline in CLASSICAL_BASELINES else neural_schema
        if saved.get("schema") != schema:
            raise ValueError("Checkpoint does not implement the selected cell model")
        if baseline in CLASSICAL_BASELINES:
            model = adapter.build(saved["genes"], spec.descriptors.dimension, state=saved["classical_state"])
        else:
            foundation.set_statistics(saved["foundation_statistics"])
            model = adapter.build(
                saved["genes"],
                spec.descriptors.dimension,
                foundation_dimension=foundation.dimension,
                **saved["model_options"],
            ).to(device)
            model.load_state_dict(saved["model"], strict=True)
        genes, history, profile = (saved["genes"], saved["history"], saved.get("batch_profile"))
        if foundation is not None:
            if profile is None:
                raise ValueError("Checkpoint requires its batch execution profile")
            restore_foundation_batch(foundation, profile)
    else:
        status("loading_complete_development_cells")
        records = load_fit_records(spec, smoke=smoke)
        validation_records = (
            load_fit_records(spec, smoke=smoke, split="validation")
            if validation_enabled and distributed.primary()
            else []
        )
        if validation_enabled and distributed.primary():
            assert_independent_records(records, validation_records)
        write_json(output / "validation_split.json", spec.validation_split)
        validation_kwargs = {
            "mode": mode,
            "device": device,
            "foundation_store": foundation,
            "cell_cap": min(16, validation_config["cell_cap_per_response"])
            if smoke
            else validation_config["cell_cap_per_response"],
            "projections": validation_config["spatial_projections"],
            "scratch_directory": output,
            "gene_panel": validation_config["gene_panel"],
        }
        genes = list(records[0].target.genes)
        torch.manual_seed(seed)
        if baseline in CLASSICAL_BASELINES:
            print(f"[train] Starting analytic fit | baseline={baseline} | records={len(records)}", flush=True)
            training_started = time.perf_counter()
            model = adapter.fit_analytic(
                records,
                device=device,
                assignment_seed=train["assignment_seed"],
                control_descriptors=spec.descriptors.condition_vectors,
                ridge=train["classical_ridge"],
                on_progress=lambda row: status("fitting_cell_distribution_moments", **row),
            )
            print(f"[train] Analytic fit finished | elapsed={time.perf_counter() - training_started:.1f}s", flush=True)
            validation = (
                evaluate_validation(model, validation_records, **validation_kwargs)
                if validation_enabled
                else {"status": "disabled", "reason": "protocol_no_validation"}
            )
            saved = {
                "schema": classical_schema,
                "classical_state": model.state,
                "validation_selection": {"epoch": None, "validation": validation, "selection": "fixed_analytic"},
            }
        else:
            objective = (
                CellObjective(records, joint=joint, direction_lambda=train["direction_lambda"])
                if distributed.primary()
                else None
            )
            statistics = distributed.broadcast(objective.statistics if objective is not None else None)
            if objective is None:
                objective = CellObjective.from_statistics(
                    statistics, joint=joint, direction_lambda=train["direction_lambda"]
                )
            write_json(output / "training_objective.json", objective.metadata())
            if distributed.primary():
                foundation.prepare(records)
            foundation.set_statistics(distributed.broadcast(foundation.statistics))
            validate_sources(p for record in [*records, *validation_records] for p in record.source)
            torch.manual_seed(train.get("model_seed", seed))
            model_options = distributed.broadcast(
                {
                    "condition_vectors": [
                        spec.descriptors.condition_vectors[name].tolist()
                        for name in sorted(spec.descriptors.condition_vectors)
                    ],
                    "gene_graph": source_gene_graph(records).tolist(),
                    "hidden_dimension": configuration["model"]["hidden_dimension"],
                    "compute_precision": precision,
                }
                if distributed.primary()
                else None
            )
            model = adapter.build(
                genes, spec.descriptors.dimension, foundation_dimension=foundation.dimension, **model_options
            ).to(device)
            if resume_path.exists():
                restored = torch.load(resume_path, map_location="cpu", weights_only=False)
                _check_binding(restored, binding, "Resume checkpoint")
                model.load_state_dict(restored["model"], strict=True)
                resume, history, profile = (restored["resume"], restored["history"], restored["batch_profile"])
            profile = distributed.broadcast(profile)
            if profile is None:
                profile = distributed.broadcast(
                    json.loads((output / "batch_profile.json").read_text())
                    if distributed.primary() and execution_config_path.exists()
                    else None
                )
            if profile is None:
                batch_size = 2 if smoke else train["batch_size"]
                foundation.set_training_batch_size(distributed.local_batch_size(batch_size))
                profile = configured_batch_profile(
                    batch_size, distributed.world_size(), encoder_batch_size=foundation.encoder_batch_size
                )
                status(
                    "configured_complete_cell_batch",
                    batch_size=profile["batch_size"],
                    local_batch_size=profile["local_batch_size"],
                    encoder_batch_size=profile["encoder_batch_size"],
                )
                print(
                    f"[batch-config] fixed | {json.dumps({key: profile[key] for key in ('batch_size', 'local_batch_size', 'encoder_batch_size')})}",
                    flush=True,
                )
                write_json(output / "batch_profile.json", profile)
            if profile.get("world_size", 1) != distributed.world_size():
                raise ValueError("Resume WORLD_SIZE changed; use a new output or an explicit checkpoint migration")
            restore_foundation_batch(foundation, profile)
            execution_configuration = resolved_batch_configuration(
                configuration, profile, foundation.encoder_batch_size
            )
            write_json(execution_config_path, {"binding": binding, "configuration": execution_configuration})
            train = execution_configuration["training"]
            validate_sources(p for record in [*records, *validation_records] for p in record.source)
            smoke_batches = 2 if smoke else None
            best = (
                BestValidationCheckpoint(best_path, binding) if validation_enabled and distributed.primary() else None
            )
            if best is not None and best.selection is None and (resume is None):
                status("validating_initialization")
                print("[validation] Evaluating initialization before training...", flush=True)
                initial_validation = evaluate_validation(model, validation_records, **validation_kwargs)
                print(
                    f"[validation] Initialization | {initial_validation['metric']}={initial_validation['score']:.5g}",
                    flush=True,
                )
                best.consider(model, initial_validation, epoch=0, complete_epoch=True)
                atomic_write_json(output / "initial_validation.json", initial_validation)

            def on_epoch(row, current, optimizer, state):
                if validation_enabled:
                    validation_started = time.perf_counter()
                    status("validating_independent_cells", epoch=row["epoch"])
                    print(f"[validation] Epoch {row['epoch']}/{train['epochs']} started...", flush=True)
                    validation = evaluate_validation(current, validation_records, **validation_kwargs)
                    row["validation"] = validation
                    row["selected_as_best"] = best.consider(
                        current, validation, epoch=row["epoch"], complete_epoch=state["complete_epoch"]
                    )
                    row["validation_wall_seconds"] = time.perf_counter() - validation_started
                    print(
                        f"[validation] Epoch {row['epoch']}/{train['epochs']} | "
                        f"{validation['metric']}={validation['score']:.5g} | "
                        f"best={row['selected_as_best']} | elapsed={row['validation_wall_seconds']:.1f}s",
                        flush=True,
                    )
                history.append(row)
                status("training_complete_cells", epoch=row)
                atomic_write_json(output / "training_history.json", history)
                if state["complete_epoch"]:
                    _save_torch(
                        resume_path,
                        {
                            "binding": binding,
                            "schema": neural_schema,
                            "model": current.state_dict(),
                            "history": history,
                            "batch_profile": profile,
                            "resume": {**state, "optimizer": optimizer.state_dict()},
                        },
                    )

            status("training_complete_cells", batch_size=profile["batch_size"])
            training_started = time.perf_counter()
            train_model(
                model,
                records,
                device=device,
                seed=seed,
                epochs=train["epochs"],
                cells_per_batch=profile["batch_size"],
                accumulation_steps=train["accumulation_steps"],
                learning_rate=train["learning_rate"],
                weight_decay=train["weight_decay"],
                foundation_store=foundation,
                control_descriptors=spec.descriptors.condition_vectors,
                loss_function=objective,
                objective_id=objective.objective_id,
                assignment_seed=train["assignment_seed"],
                resume=resume,
                smoke_max_batches=smoke_batches,
                on_epoch=on_epoch,
                step_function=train_step,
            )
            if not distributed.primary():
                return None
            phase_seconds["training_and_validation"] += time.perf_counter() - training_started
            if not smoke and (
                not history or not history[-1]["complete_epoch"] or history[-1]["epoch"] != train["epochs"]
            ):
                raise ValueError(f"Formal checkpoint requires all {train['epochs']} complete training epochs")
            selection = (
                best.restore(model)
                if validation_enabled
                else {
                    "epoch": history[-1]["epoch"],
                    "selection": "final_epoch",
                    "validation": {"status": "disabled", "reason": "protocol_no_validation"},
                    "test_used_for_selection": False,
                }
            )
            saved = {
                "schema": neural_schema,
                "model": model.state_dict(),
                "model_options": model_options,
                "foundation_statistics": foundation.statistics,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "validation_selection": selection,
            }
        saved.update(
            initial_validation=json.loads((output / "initial_validation.json").read_text())
            if baseline in SUPPORTED_FOUNDATION_BASELINES and validation_enabled
            else None,
            binding=binding,
            genes=genes,
            history=history,
            batch_profile=profile,
            checkpoint_selection="fixed_analytic_train_only"
            if baseline in CLASSICAL_BASELINES
            else "best_validation"
            if validation_enabled
            else "final_epoch",
            configuration=configuration,
        )
        _save_torch(checkpoint, saved)
        print(f"[train] Checkpoint saved: {checkpoint}", flush=True)
        atomic_write_json(output / "validation_selection.json", saved["validation_selection"])
        best_path.unlink(missing_ok=True)
        del records, validation_records
        gc.collect()
    rows = []
    source_key, sources = (None, None)
    for response in spec.responses[:1] if smoke else spec.responses:
        directory = output / "responses" / artifact_path_token(response.response_id)
        directory.mkdir(parents=True, exist_ok=True)
        generation_cap = 64 if smoke else configuration["evaluation"]["cell_cap_per_group"]
        generation_plan = make_generation_plan(spec, response, cap=generation_cap)
        generation_hash = freeze_generation_plan(directory, generation_plan)
        status("predicting_source_only" if joint else "predicting_count_only", response_id=response.response_id)
        if source_key != (response.context, response.source_condition):
            sources = list(
                iter_full_gene_populations(
                    spec.index_dir,
                    conditions=[response.source_condition],
                    contexts=[response.context],
                    expected_dataset=spec.dataset,
                    expected_context_column=spec.context_column,
                    cell_cap=32 if smoke else None,
                )
            )
            source_key = (response.context, response.source_condition)
            if foundation is not None:
                validate_sources(sources)
        prediction_started = time.perf_counter()
        if joint:
            input_path = directory / "joint_source_input.json"
            donor_plan = select_joint_donors(
                sources, response_id=response.response_id, budget=generation_plan["output_cells"]
            )
            inputs = {
                "access_scope": "source_counts_and_coordinates_plus_target_population_size_budget",
                "target_counts_loaded": False,
                "target_geometry_loaded": False,
                "target_cell_count_loaded": True,
                "generation_plan_sha256": generation_hash,
                "generation_budget": generation_plan["output_cells"],
                "sources": [{"unit_id": p.unit_id, "cell_count": len(p.cell_ids)} for p in sources],
                "donors": donor_plan,
            }
            if input_path.exists() and json.loads(input_path.read_text()) != inputs:
                raise ValueError("Joint pre-generation source assignments changed")
            atomic_write_json(input_path, inputs)
            provenance = {
                "binding": binding,
                "response_id": response.response_id,
                "generation_plan_sha256": generation_hash,
            }
            paths = []
            source_by_unit = {p.unit_id: p for p in sources}
            for selected_donors in donor_plan:
                source = source_by_unit[selected_donors["unit_id"]]
                path = directory / f"prediction_{artifact_path_token(source.unit_id)}.h5"
                if path.exists() and path.with_suffix(".h5.json").exists():
                    prediction = load_joint_prediction(path)
                    if prediction.storage_manifest["provenance"] != provenance:
                        raise ValueError("Sealed Joint prediction belongs to another source input or checkpoint")
                else:
                    if path.exists():
                        path.rename(path.with_suffix(f".incomplete.{time.time_ns()}"))
                    if path.with_suffix(".h5.json").exists():
                        raise ValueError("Joint prediction seal exists without its payload")
                    geometry_seed = seed_from_parts(*[42, seed, spec.dataset, response.response_id, source.unit_id])
                    adapter.predict(
                        model,
                        ModelInput(
                            sources=(source,),
                            descriptor=spec.descriptors.condition_vectors[response.target_condition],
                            source_cell_ids=tuple(selected_donors["source_cell_ids"]),
                            generation_budget=generation_plan["output_cells"],
                        ),
                        path,
                        seed=geometry_seed,
                        device=device,
                        foundation_store=foundation,
                        provenance=provenance,
                        batch_size=prediction_batch_size,
                    )
                paths.append(path)
            phase_seconds["prediction"] += time.perf_counter() - prediction_started
            status("evaluating_sealed_joint_counts_and_coordinates", response_id=response.response_id)
            evaluation_started = time.perf_counter()
            rows.append(
                evaluate_step(
                    spec,
                    response,
                    paths,
                    directory,
                    setting=mode,
                    projections=configuration["evaluation"]["spatial_projections"],
                    cap=64 if smoke else configuration["evaluation"]["cell_cap_per_group"],
                    scientific_result=not smoke,
                    expected_provenance=provenance,
                    generation_plan=generation_plan,
                )
            )
            phase_seconds["evaluation"] += time.perf_counter() - evaluation_started
            continue
        oracles, sampling = sample_oracle_counts(
            list(iter_oracle_counts(spec.index_dir, condition=response.target_condition, context=response.context)),
            response_id=response.response_id,
            cap=64 if smoke else configuration["evaluation"]["cell_cap_per_group"],
        )
        if sampling != generation_plan["target"]:
            raise ValueError("Oracle target counts differ from the metadata-only pre-generation selection")
        input_path = directory / "oracle_count_input.json"
        atomic_write_json(
            input_path, {"access_scope": "target_counts_only", "target_geometry_loaded": False, "target": sampling}
        )
        paths = []
        for oracle in oracles:
            selected = sources
            if spec.source_pool == "same_unit_then_pooled":
                selected = [p for p in sources if p.unit_id == oracle.unit_id] or sources
            path = directory / f"prediction_{artifact_path_token(oracle.unit_id)}.h5"
            provenance = {
                "binding": binding,
                "response_id": response.response_id,
                "generation_plan_sha256": generation_hash,
            }
            if path.exists() and path.with_suffix(".h5.json").exists():
                prediction = load_oracle_prediction(path)
                actual = prediction.storage_manifest["provenance"]
                if any((actual.get(k) != v for k, v in provenance.items())):
                    raise ValueError("Sealed prediction belongs to another model or Oracle input")
                validate_oracle_count_identity(prediction, oracle)
            else:
                if path.exists():
                    path.rename(path.with_suffix(f".incomplete.{time.time_ns()}"))
                if path.with_suffix(".h5.json").exists():
                    raise ValueError("Prediction seal exists without its payload")
                geometry_seed = seed_from_parts(*[42, seed, spec.dataset, response.response_id, oracle.unit_id])
                adapter.predict(
                    model,
                    ModelInput(
                        sources=tuple(selected),
                        descriptor=spec.descriptors.condition_vectors[response.target_condition],
                        oracle_counts=oracle,
                    ),
                    path,
                    seed=geometry_seed,
                    device=device,
                    foundation_store=foundation,
                    provenance=provenance,
                    batch_size=prediction_batch_size,
                )
            paths.append(path)
        phase_seconds["prediction"] += time.perf_counter() - prediction_started
        status("evaluating_sealed_complete_cells", response_id=response.response_id)
        evaluation_started = time.perf_counter()
        rows.append(
            evaluate_step(
                spec,
                response,
                paths,
                directory,
                setting=mode,
                oracle_counts=oracles,
                projections=configuration["evaluation"]["spatial_projections"],
                scientific_result=not smoke,
                generation_plan=generation_plan,
            )
        )
        phase_seconds["evaluation"] += time.perf_counter() - evaluation_started
    steps = history[-1]["optimizer_steps"] if history else 0
    result = {
        **base,
        "status": "complete",
        "checkpoint": str(checkpoint),
        "gene_count": len(genes),
        "optimizer_steps": steps,
        "completed_epochs": history[-1]["epoch"] if history else None,
        "checkpoint_selection": saved["checkpoint_selection"],
        "validation_selection": saved["validation_selection"],
        "batch_profile": profile,
        "model_architecture": configuration["model"]["architecture"] if foundation is not None else "classical",
        "trainable_parameters": saved.get("trainable_parameters", 0),
        "completed_responses": len(rows),
        "responses": rows,
        "wall_seconds": time.time() - started,
        "phase_wall_seconds_this_invocation": phase_seconds,
    }
    atomic_write_json(result_path, result)
    status("complete", completed_responses=len(rows))
    return result


if __name__ == "__main__":
    main()
