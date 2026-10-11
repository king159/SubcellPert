"""count_matched_resampling baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subcellpert.data import CellTrainingExamples
from subcellpert.models.shared import Baseline, ClassicalModel
from subcellpert.utils import CELL_CLASSICAL_SCHEMA, CELL_JOINT_CLASSICAL_SCHEMA


@dataclass
class ResamplingOracle(ClassicalModel):
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
        return cls("count_matched_resampling", genes, len(records[0].descriptor), state)

    def transform_cell(self, source, descriptor, *, device="cpu"):
        if source.genes != self.genes or len(source.cell_ids) != 1:
            raise ValueError("Classical generation consumes one complete, vocabulary-aligned source cell")
        return {"kind": "affine", "parameters": np.tile([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], (len(self.genes), 1))}


@dataclass
class ResamplingJoint(ResamplingOracle):
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
        spatial = ResamplingOracle.fit(
            "count_matched_resampling",
            records,
            device=device,
            assignment_seed=assignment_seed,
            control_descriptors=control_descriptors,
            ridge=ridge,
            on_progress=on_progress,
        )
        state = dict(spatial.state, schema=CELL_JOINT_CLASSICAL_SCHEMA, count_policy="source_resampling")
        return cls("count_matched_resampling", spatial.genes, spatial.descriptor_dimension, state)

    def predict_counts(self, source, descriptor):
        return source.counts.copy()


class Model(Baseline):
    name = "count_matched_resampling"
    foundation = False
    oracle_class = ResamplingOracle
    joint_class = ResamplingJoint
