"""radial_quantile_transport baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subcellpert.data import CellTrainingExamples
from subcellpert.data.batching import PopulationCellReader
from subcellpert.models.shared import Baseline, ClassicalModel, _Moments
from subcellpert.utils import (
    CELL_CLASSICAL_SCHEMA,
    CELL_JOINT_CLASSICAL_SCHEMA,
    MAX_LOG_COUNT_SHIFT,
    _masked_spatial_ridge,
    complete_cell_statistics,
    integer_counts,
)


@dataclass
class RadialQuantileOracle(ClassicalModel):
    baseline: str
    genes: tuple
    descriptor_dimension: int
    state: dict

    @classmethod
    def fit(
        cls,
        baseline,
        records,
        *,
        device="cpu",
        assignment_seed=42,
        control_descriptors=None,
        ridge=10.0,
        on_progress=None,
    ):
        genes = records[0].target.genes
        examples = CellTrainingExamples(records, assignment_seed, control_descriptors=control_descriptors)
        state = {
            "schema": CELL_CLASSICAL_SCHEMA,
            "training_examples": len(examples),
            "rna_subsampling": False,
            "biological_cell_pairing": False,
            "fit_objective": "condition_group_distribution_moments",
        }
        readers, cache, groups = ({}, {}, {})
        rna_seen = 0

        def read_stats(population, row, fallback):
            nonlocal rna_seen
            key = (id(population), row, fallback)
            if key not in cache:
                if id(population) not in readers:
                    readers[id(population)] = PopulationCellReader(population)
                batch = readers[id(population)].read(np.array([row]))
                rna_seen += len(batch.coordinates)
                result = complete_cell_statistics(batch, source_fallback=fallback, include_radial=True, device=device)
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
                cache[key] = result
            return cache[key]

        for i in range(len(examples)):
            example = examples.example(i)
            key = (*example.condition_key, tuple(example.descriptor.tolist()))
            if key not in groups:
                groups[key] = (_Moments(len(genes)), _Moments(len(genes)), example.descriptor.copy(), 0)
            source, target, descriptor, size = groups[key]
            source.add(read_stats(example.source, example.source_row, True))
            target.add(read_stats(example.target, example.target_row, False))
            groups[key] = (source, target, descriptor, size + 1)
            if on_progress is not None and (i + 1 == len(examples) or (i + 1) % 1000 == 0):
                on_progress(
                    {"examples_consumed": i + 1, "training_examples": len(examples), "input_rna_molecules": rna_seen}
                )
        features, outputs, masks, weights = ([], [], [], [])
        for source, target, descriptor, size in groups.values():
            sm, sc, sq, _ = source.result()
            tm, tc, tq, valid = target.result()
            output = np.column_stack((tm - sm, np.log(tq.clip(0.0001) / sq.clip(0.0001))))
            features.append(descriptor)
            outputs.append(output)
            masks.append(valid)
            weights.append(size)
        mean, scale, coefficients = _masked_spatial_ridge(
            np.stack(features), np.stack(outputs), np.stack(masks), np.asarray(weights), ridge, device
        )
        state.update(
            feature_mean=mean,
            feature_scale=scale,
            coefficients=coefficients,
            input_rna_molecules=rna_seen,
            fitted_condition_groups=len(groups),
            ridge=ridge,
        )
        return cls("radial_quantile_transport", genes, len(records[0].descriptor), state)

    def transform_cell(self, source, descriptor, *, device="cpu"):
        if source.genes != self.genes or len(source.cell_ids) != 1:
            raise ValueError("Classical generation consumes one complete, vocabulary-aligned source cell")
        x = np.r_[1.0, (np.asarray(descriptor) - self.state["feature_mean"]) / self.state["feature_scale"]]
        spatial = np.einsum("f,fgp->gp", x, self.state["coefficients"])
        mean, _, radial, _ = complete_cell_statistics(source, source_fallback=True, device=device)
        sq = radial[0].clip(0.0001)
        tq = np.maximum.accumulate(sq * np.exp(spatial[:, 2:].clip(-2.3, 2.3)), axis=1)
        return {
            "kind": "radial",
            "centers": mean[0],
            "shift": spatial[:, :2],
            "source_quantiles": sq,
            "target_quantiles": tq,
        }


@dataclass
class RadialQuantileJoint(RadialQuantileOracle):
    @classmethod
    def fit(
        cls,
        baseline,
        records,
        *,
        device="cpu",
        assignment_seed=42,
        control_descriptors=None,
        ridge=10.0,
        on_progress=None,
    ):
        spatial = RadialQuantileOracle.fit(
            "radial_quantile_transport",
            records,
            device=device,
            assignment_seed=assignment_seed,
            control_descriptors=control_descriptors,
            ridge=ridge,
            on_progress=on_progress,
        )
        state = dict(spatial.state, schema=CELL_JOINT_CLASSICAL_SCHEMA, count_policy="unpaired_log1p_group_shift")
        examples = CellTrainingExamples(records, assignment_seed, control_descriptors=control_descriptors)
        groups = {}
        for record, (units, donors) in zip(records, examples.assignments, strict=True):
            key = (record.context, record.condition, tuple(record.descriptor))
            if key not in groups:
                groups[key] = [record.descriptor, np.zeros(len(spatial.genes)), 0]
            for first in range(0, len(donors), 128):
                chosen_units, chosen_rows = (units[first : first + 128], donors[first : first + 128])
                source = np.zeros((len(chosen_rows), len(spatial.genes)), dtype=np.float64)
                for unit in np.unique(chosen_units):
                    keep = chosen_units == unit
                    source[keep] = record.source[int(unit)].counts[chosen_rows[keep]]
                delta = np.log1p(record.target.counts[first : first + 128]) - np.log1p(source)
                groups[key][1] += delta.sum(0)
                groups[key][2] += len(delta)
        for population, descriptor in examples.controls:
            key = (population.context, population.condition, tuple(descriptor))
            if key not in groups:
                groups[key] = [descriptor, np.zeros(len(spatial.genes)), 0]
            groups[key][2] += len(population.cell_ids)
        features = np.stack([v[0] for v in groups.values()])
        outputs = np.stack([v[1] / v[2] for v in groups.values()])[..., None]
        weights = np.asarray([v[2] for v in groups.values()])
        mean, scale, coefficients = _masked_spatial_ridge(
            features, outputs, np.ones(outputs.shape[:2], dtype=bool), weights, ridge, device
        )
        state["count_state"] = {"feature_mean": mean, "feature_scale": scale, "coefficients": coefficients}
        return cls("radial_quantile_transport", spatial.genes, spatial.descriptor_dimension, state)

    def predict_counts(self, source, descriptor):
        state = self.state["count_state"]
        x = np.r_[1.0, (np.asarray(descriptor) - state["feature_mean"]) / state["feature_scale"]]
        shift = np.einsum("f,fgp->gp", x, state["coefficients"])[:, 0].clip(-MAX_LOG_COUNT_SHIFT, MAX_LOG_COUNT_SHIFT)
        return integer_counts(np.maximum(np.log1p(source.counts) + shift, 0))


class Model(Baseline):
    name = "radial_quantile_transport"
    foundation = False
    oracle_class = RadialQuantileOracle
    joint_class = RadialQuantileJoint
