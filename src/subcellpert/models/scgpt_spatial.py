"""scgpt_spatial baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

from subcellpert.models.scgpt import ScGPTGeneEncoder
from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel


class ScGPTSpatialGeneEncoder(ScGPTGeneEncoder):
    def __init__(self, root, metadata, device):
        if metadata["backbone"] != "scgpt_spatial":
            raise ValueError("scGPT-spatial requires its spatial checkpoint and source normalization")
        super().__init__(root, metadata, device)


class Model(Baseline):
    name = "scgpt_spatial"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = ScGPTSpatialGeneEncoder
