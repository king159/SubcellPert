"""One optimizer update and the shared complete-cell training loop."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from subcellpert import distributed
from subcellpert.data import (
    SCFOUNDATION_ACCUMULATION_STEPS,
    SCFOUNDATION_BATCH_SIZE,
    SCFOUNDATION_EPOCHS,
    SCFOUNDATION_SAMPLING_SCHEMA,
    CompleteCellDataset,
    iter_training_batches,
)
from subcellpert.data.batching import CompleteCellCollator
from subcellpert.models.shared import TensorCells
from subcellpert.utils import cell_distribution_features, predict_cell_batch


@dataclass(frozen=True)
class TrainingStep:
    loss: float
    input_rna_molecules: int
    audit: list


def train_one_step(model, batch, optimizer, *, loss_function, rng, device, directions=None, audit=False):
    """Update on one batch; the supplied objective determines the setting.

    All RNA in each selected cell is retained. Epoch shuffling, scheduling,
    checkpointing and validation belong to the training loop.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, molecules, details = loss_function(model, batch, rng=rng, device=device, directions=directions, audit=audit)
    valid = torch.tensor(int(loss.ndim == 0 and torch.isfinite(loss)), device=device)
    parallel = isinstance(loss_function, distributed.DistributedObjective)
    if parallel:
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
    if not valid:
        raise FloatingPointError("Nonfinite or nonscalar complete-cell loss")
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if parallel:
        totals = torch.tensor([float(loss.detach()), molecules], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(totals)
        loss, molecules = totals[0] / distributed.world_size(), int(totals[1])
        if audit:
            details = [row for part in distributed.gather(details) for row in part]
    return TrainingStep(float(loss.detach()), int(molecules), details)


def configured_batch_profile(batch_size, world_size=1, *, encoder_batch_size=None):
    """Record a fixed configured batch without probing or automatic fallback."""
    local = distributed.local_batch_size(batch_size, world_size)
    profile = {
        "batch_size": batch_size,
        "encoder_batch_size": local if encoder_batch_size is None else encoder_batch_size,
        "local_batch_size": local,
        "world_size": world_size,
        "status": "configured",
        "performance_selection": False,
        "throughput": [],
        "selection_metric": "configured_batch_size",
        "memory_budget_bytes": None,
        "probe_selection": "disabled_for_fixed_batch",
        "probes": [],
    }
    distributed.profile_encoder_batch(profile)
    return profile


def resolved_batch_configuration(configuration, profile, encoder_batch_size):
    """Produce the concrete configuration saved before the first optimizer step."""
    size = profile.get("world_size", 1)
    local = distributed.local_batch_size(int(profile["batch_size"]), size)
    distributed.profile_encoder_batch({**profile, "encoder_batch_size": encoder_batch_size})
    resolved = deepcopy(configuration)
    resolved["training"]["batch_size"] = int(profile["batch_size"])
    resolved["training"].pop("batch_candidates", None)
    resolved["foundation"] = {**resolved.get("foundation", {}), "encoder_batch_size": int(encoder_batch_size)}
    resolved["foundation"].pop("encoder_batch_candidates", None)
    resolved["batch_execution"] = (
        "fixed_config_no_memory_probe_no_gradient_accumulation"
        if profile.get("status") == "configured"
        else "fixed_after_memory_probe_no_gradient_accumulation"
    )
    if size > 1:
        resolved["distributed_execution"] = {
            "world_size": size,
            "global_batch_size": int(profile["batch_size"]),
            "local_batch_size": local,
            "normalization": "global_batchnorm",
            "loss_weighting": "global_condition_groups",
        }
    if resolved["training"]["accumulation_steps"] != 1:
        raise ValueError("Gradient accumulation must be disabled")
    return resolved


def cell_microbatch_loss(
    model, examples, indices, read, *, rng, device, directions, foundation_store=None, audit=False, loss_function=None
):
    batch = CompleteCellCollator(examples, foundation_store=foundation_store, read=read)(indices)
    if loss_function is None:
        raise ValueError("Training requires the registered joint or oracle objective")
    return loss_function(model, batch, rng=rng, device=device, directions=directions, audit=audit)


def train_model(
    model,
    records,
    *,
    device,
    seed,
    epochs=SCFOUNDATION_EPOCHS,
    cells_per_batch=SCFOUNDATION_BATCH_SIZE,
    accumulation_steps=SCFOUNDATION_ACCUMULATION_STEPS,
    projections=64,
    learning_rate=0.0005,
    weight_decay=0.0005,
    foundation_store=None,
    control_descriptors=None,
    on_epoch=None,
    on_batch=None,
    resume=None,
    smoke_max_batches=None,
    assignment_seed=42,
    prefetch_batches=2,
    loss_function=None,
    objective_id=None,
    max_optimizer_steps=None,
    step_function=train_one_step,
):
    """Run shared training epochs, calling train_one_step for each batch.

    All trainable baselines use this loop. The supplied loss function selects
    the joint or oracle objective. This loop owns epoch shuffling, learning-rate
    scheduling, resume state, and progress callbacks; step_function owns the
    optimizer update. Frozen backbone features remain source-only and keyed by
    actual cell ID, with fixed computational donor assignments.
    """
    if loss_function is None:
        raise ValueError("Training requires the registered joint or oracle objective")
    if epochs < 1 or cells_per_batch < 1 or accumulation_steps < 1 or (projections < 1):
        raise ValueError("Training dimensions must be positive")
    if accumulation_steps != 1:
        raise ValueError("Gradient accumulation is disabled; use the measured fixed cell batch")
    if max_optimizer_steps is not None and max_optimizer_steps < 1:
        raise ValueError("Optimizer-step limit must be positive")
    if (model.foundation_dimension is None) != (foundation_store is None):
        raise ValueError("Model and cell foundation feature store disagree")
    if any(r.foundation_context is not None or r.source_foundation_contexts is not None for r in records):
        raise ValueError("Population-average foundation embeddings are invalid")
    if foundation_store is not None:
        foundation_store.set_training_batch_size(distributed.local_batch_size(cells_per_batch))
    dataset = CompleteCellDataset(
        records,
        assignment_seed=assignment_seed,
        control_descriptors=control_descriptors,
        foundation_store=foundation_store,
    )
    examples = dataset.examples
    if len(examples) < cells_per_batch:
        raise ValueError("drop_last=True would produce an empty training loader")
    binding = {
        "schema": SCFOUNDATION_SAMPLING_SCHEMA,
        "seed": seed,
        "assignment_seed": assignment_seed,
        "cells_per_batch": cells_per_batch,
        "accumulation_steps": accumulation_steps,
        "projections": projections,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "compute_precision": getattr(model, "compute_precision", "fp32"),
    }
    if loss_function is not None:
        if not objective_id:
            raise ValueError("The training objective needs an explicit checkpoint binding")
        binding["objective_id"] = objective_id
    size, rank = distributed.world_size(), distributed.rank()
    distributed.local_batch_size(cells_per_batch)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 71289] + ([rank] if size > 1 else [])))
    directions = None
    if getattr(loss_function, "requires_projections", True):
        projection_rng = torch.Generator(device=device).manual_seed(seed + 171)
        directions = F.normalize(
            torch.randn(model.n_genes * 30, projections, generator=projection_rng, device=device), dim=0
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    start = optimizer_steps = batches_seen = rna_seen = 0
    if resume is not None:
        if resume.get("binding") != binding or resume.get("complete_epoch") is not True:
            raise ValueError("Training sampler checkpoint bindings changed or epoch is incomplete")
        if resume.get("world_size", 1) != size:
            raise ValueError("Resume WORLD_SIZE changed; use a new output or an explicit checkpoint migration")
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rng.bit_generator.state = resume["numpy_rng_states"][rank] if size > 1 else resume["numpy_rng_state"]
        start, optimizer_steps = (resume["completed_epochs"], resume["optimizer_steps"])
        batches_seen, rna_seen = (resume["batches_seen"], resume["input_rna_molecules"])
    total_batches = len(examples) // cells_per_batch
    step_objective = distributed.DistributedObjective(model, loss_function, device) if size > 1 else loss_function
    print(
        f"[train] Starting epochs {start + 1}-{epochs} | device={device} | "
        f"examples={len(examples):,} | batch_size={cells_per_batch} | "
        f"batches_per_epoch={total_batches:,} | prefetch_batches={prefetch_batches} | "
        f"precision={getattr(model, 'compute_precision', 'fp32')} | "
        f"lr={optimizer.param_groups[0]['lr']:.3g}",
        flush=True,
    )
    training_started = time.perf_counter()
    history = []
    for epoch in range(start, epochs):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, observed = (0.0, 0)
        batches = iter_training_batches(
            dataset,
            batch_size=cells_per_batch,
            seed=seed,
            epoch=epoch,
            prefetch=prefetch_batches,
            rank=rank,
            world_size=size,
        )
        with (
            closing(batches),
            tqdm(
                total=total_batches,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=1.0,
                disable=not distributed.primary(),
            ) as progress,
        ):
            for batch_index, batch in enumerate(batches):
                step = step_function(
                    model,
                    batch,
                    optimizer,
                    loss_function=step_objective,
                    rng=rng,
                    device=device,
                    directions=directions,
                    audit=on_batch is not None,
                )
                rna_seen += step.input_rna_molecules
                batch_audit = step.audit
                batches_seen += 1
                observed += 1
                losses += step.loss
                smoke_stop = smoke_max_batches is not None and batches_seen >= smoke_max_batches
                optimizer_steps += 1
                optimizer_stop = max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps
                progress.set_postfix(
                    loss=f"{step.loss:.5g}",
                    avg_loss=f"{losses / observed:.5g}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3g}",
                    refresh=False,
                )
                progress.update(1)
                if on_batch is not None and distributed.primary():
                    on_batch(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "optimizer_steps": optimizer_steps,
                            "optimizer_updated": True,
                            "examples": batch_audit,
                        }
                    )
                if smoke_stop or optimizer_stop:
                    break
        complete = observed == total_batches
        if complete:
            scheduler.step()
        row = {
            "training_wall_seconds": time.perf_counter() - epoch_started,
            "compute_precision": getattr(model, "compute_precision", "fp32"),
            "epoch": epoch + 1,
            "complete_epoch": complete,
            "optimizer_steps": optimizer_steps,
            "batches_seen": batches_seen,
            "epoch_batches": observed,
            "cell_distribution_loss": losses / observed,
            "sampling_schema": SCFOUNDATION_SAMPLING_SCHEMA,
            "target_examples": examples.target_examples,
            "control_identity_examples": len(examples) - examples.target_examples,
            "training_examples": len(examples),
            "epoch_examples_consumed": observed * cells_per_batch,
            "dropped_last_examples": len(examples) % cells_per_batch if complete else None,
            "input_rna_molecules": rna_seen,
            "rna_subsampling": False,
            "biological_cell_pairing": False,
            "loader": "cell_major_batched_complete_RNA_v3",
            "forward_calls_per_microbatch": 1,
            "prefetch_batches": prefetch_batches,
        }
        if objective_id is not None:
            row["objective_id"] = objective_id
        if size > 1:
            row.update(world_size=size, global_batch_size=cells_per_batch, local_batch_size=cells_per_batch // size)
        history.append(row)
        # Every rank contributes its RNG state before rank zero validates/saves.
        rng_states = distributed.gather(deepcopy(rng.bit_generator.state)) if size > 1 else None
        if on_epoch is not None and distributed.primary():
            state = {
                "binding": binding,
                "complete_epoch": complete,
                "completed_epochs": epoch + 1 if complete else epoch,
                "optimizer_steps": optimizer_steps,
                "batches_seen": batches_seen,
                "input_rna_molecules": rna_seen,
                "numpy_rng_state": deepcopy(rng.bit_generator.state),
                "scheduler": deepcopy(scheduler.state_dict()),
            }
            if size > 1:
                state.update(world_size=size, numpy_rng_states=rng_states, rng_policy="independent_rank_streams")
            on_epoch(row, model, optimizer, state)
        distributed.barrier()
        if (
            smoke_max_batches is not None
            and batches_seen >= smoke_max_batches
            or (max_optimizer_steps is not None and optimizer_steps >= max_optimizer_steps)
        ):
            break
    if history:
        final = history[-1]
        print(
            f"[train] Finished | elapsed={time.perf_counter() - training_started:.1f}s | "
            f"epoch={final['epoch']}/{epochs} | complete_epoch={final['complete_epoch']} | "
            f"optimizer_steps={optimizer_steps:,} | loss={final['cell_distribution_loss']:.5g}",
            flush=True,
        )
    return history


CELL_JOINT_OBJECTIVE = "gears_gene_expression_L4_direction_plus_local_gene_spatial_L4_v1"
CELL_ORACLE_OBJECTIVE = "cell_supervision_local_gene_spatial_L4_v1"


class CellObjective:
    """Fixed random controls are computational training inputs, not biological pairs."""

    requires_projections = False

    @classmethod
    def from_statistics(cls, statistics, *, joint, direction_lambda=0.1):
        objective = cls.__new__(cls)
        objective.joint, objective.direction_lambda = joint, direction_lambda
        objective.objective_id = CELL_JOINT_OBJECTIVE if joint else CELL_ORACLE_OBJECTIVE
        objective.statistics, objective._device_statistics = statistics, {}
        return objective

    def __init__(self, records, *, joint, direction_lambda=0.1):
        self.joint, self.direction_lambda = (joint, direction_lambda)
        self.objective_id = CELL_JOINT_OBJECTIVE if joint else CELL_ORACLE_OBJECTIVE
        grouped = defaultdict(list)
        for record in records:
            grouped[record.context, record.condition].append(record)
        self.statistics = {}
        source_statistics = {}

        def source_mean(populations):
            unique = {(p.dataset, p.context, p.condition, p.unit_id): p for p in populations}
            key = tuple(sorted(unique))
            if key not in source_statistics:
                total, cells = (np.zeros(len(records[0].target.genes)), 0)
                for p in unique.values():
                    for first in range(0, len(p.cell_ids), 4096):
                        values = np.log1p(np.asarray(p.counts[first : first + 4096], dtype=np.float64))
                        total += values.sum(0)
                        cells += len(values)
                source_statistics[key] = (total / cells).astype(np.float32)
            return source_statistics[key]

        for key, group in grouped.items():
            mask = np.zeros(len(group[0].target.genes), dtype=bool)
            for record in group:
                for first in range(0, len(record.target.cell_ids), 4096):
                    mask |= np.any(record.target.counts[first : first + 4096] > 0, axis=0)
            self.statistics[key] = (source_mean(p for r in group for p in r.source), mask)
        controls = defaultdict(list)
        for record in records:
            for p in record.source:
                controls[p.context, p.condition].append(p)
        for key, populations in controls.items():
            self.statistics.setdefault(key, (source_mean(populations), np.ones(len(populations[0].genes), dtype=bool)))
        self._device_statistics = {}

    def metadata(self):
        return {
            "objective_id": self.objective_id,
            "reference": "scFoundation/397631c495eddf9ad6644fc00c6ea8139e651245/GEARS",
            "expression_scale": "raw_counts_log1p_local_RNA_extension",
            "direction_lambda": self.direction_lambda,
            "gene_support_scope": "training_responses_only; all_genes_for_control_identities",
            "control_mean_scope": "allowed_training_source_cells_only",
            "condition_weighting": "equal_groups_in_global_batch"
            if distributed.world_size() > 1
            else "equal_groups_in_microbatch",
            "spatial_extension": "cell_gene_complete_RNA_moment_Fourier_L4; no_cross_gene_projection",
            "biological_cell_pairing": False,
            "conditions": [
                {"context": k[0], "condition": k[1], "supported_genes": int(mask.sum())}
                for k, (_, mask) in sorted(self.statistics.items())
            ],
        }

    def __call__(self, model, batch, *, rng, device, directions=None, audit=False):
        foundation = None if batch.foundation is None else torch.as_tensor(batch.foundation, device=device)
        output = predict_cell_batch(
            model,
            batch.source,
            batch.target.counts,
            batch.descriptors,
            rng=rng,
            device=device,
            foundation=foundation,
            donor_rows=batch.donors,
            control_identity=batch.identity,
            joint_training=self.joint,
        )
        predicted, _, fallback = output[:3]
        target = TensorCells.from_batch(batch.target, device)
        b, g = target.counts.shape
        spatial_error = (
            (
                cell_distribution_features(predicted).reshape(b, g, 30)[..., 1:]
                - cell_distribution_features(target).reshape(b, g, 30)[..., 1:]
            )
            .pow(4)
            .mean(-1)
        )
        spatial, expression = self.reduce_losses(
            spatial_error, output[3] if self.joint else None, target.counts, batch.condition_keys, batch.identity
        )
        if fallback != sum(row["fallback_rna_molecules"] for row in batch.audit):
            raise ValueError("benchmark cell supervision lost complete-RNA donor accounting")
        if audit and getattr(self, "distributed", False) and distributed.world_size() > 1:
            logged = torch.stack((expression.detach(), spatial.detach()))
            torch.distributed.all_reduce(logged)
            logged = logged / distributed.world_size()
        else:
            logged = (expression.detach(), spatial.detach())
        details = (
            [dict(row, count_loss=float(logged[0]), spatial_loss=float(logged[1])) for row in batch.audit]
            if audit
            else []
        )
        return (expression + spatial, batch.logical_rna_molecules, details)

    def reduce_losses(self, spatial_error, predicted, counts, condition_keys, identity):
        """Average conditions equally with one batched reduction on the device."""
        device = spatial_error.device
        token = str(device)
        if token not in self._device_statistics:
            keys = list(self.statistics)
            controls, masks = zip(*(self.statistics[key] for key in keys), strict=True)
            masks = np.stack(masks)
            if not np.all(masks.any(1)):
                raise ValueError("Condition expression loss requires nonempty training gene support")
            self._device_statistics[token] = (
                {key: i for i, key in enumerate(keys)},
                torch.as_tensor(np.stack(controls), device=device),
                torch.as_tensor(masks, device=device),
            )
        lookup, controls, masks = self._device_statistics[token]
        groups = defaultdict(list)
        for row, key in enumerate(condition_keys):
            groups[key].append(row)
        sizes = np.empty(len(condition_keys), dtype=np.float32)
        identities = np.empty(len(condition_keys), dtype=bool)
        for rows in groups.values():
            sizes[rows] = len(rows)
            identities[rows] = np.all(identity[rows])
        indices = torch.tensor([lookup[key] for key in condition_keys], device=device)
        group_count = len(groups)
        scale = 1
        if getattr(self, "distributed", False) and distributed.world_size() > 1:
            counts_by_group = torch.bincount(indices, minlength=len(lookup))
            non_identity = torch.zeros_like(counts_by_group).index_add_(
                0, indices, torch.as_tensor(~np.asarray(identity, dtype=bool), device=device, dtype=torch.int64)
            )
            summary = torch.stack((counts_by_group, non_identity))
            torch.distributed.all_reduce(summary)
            sizes = summary[0, indices]
            identities = summary[1, indices] == 0
            group_count = (summary[0] > 0).sum()
            # DDP averages parameter gradients; compensate for the partitioned
            # global condition sum, including groups absent on this rank.
            scale = distributed.world_size()
        mask = masks[indices] | torch.as_tensor(identities, device=device)[:, None]
        denominator = mask.sum(1) * torch.as_tensor(sizes, device=device)
        spatial = (spatial_error.masked_fill(~mask, 0).sum(1) / denominator).sum() * scale / group_count
        expression = spatial_error.new_zeros(())
        if self.joint:
            truth, control = torch.log1p(counts), controls[indices]
            error = (predicted - truth).masked_fill(~mask, 0).pow(4)
            direction = (torch.sign(truth - control) - torch.sign(predicted - control)).masked_fill(~mask, 0)
            expression = ((error + self.direction_lambda * direction.square()).sum(1) / denominator).sum()
            expression = expression * scale / group_count
        return spatial, expression


def gene_expression_loss(predicted, target, control, mask, *, direction_lambda=0.1):
    """One condition's native GEARS loss; the caller averages conditions equally.

    The upstream sign term has zero derivative almost everywhere. Preserve
    that behavior instead of silently replacing it with a differentiable proxy.
    """
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("Expression supervision requires aligned cell/gene arrays")
    if mask.shape != (predicted.shape[1],) or not torch.any(mask):
        raise ValueError("Condition expression loss requires nonempty training gene support")
    pred, truth, ctrl = (predicted[:, mask], target[:, mask], control[mask])
    return (pred - truth).pow(4).mean() + direction_lambda * (
        torch.sign(truth - ctrl) - torch.sign(pred - ctrl)
    ).square().mean()
