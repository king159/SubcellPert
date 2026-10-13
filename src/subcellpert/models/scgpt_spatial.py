"""scgpt_spatial baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from subcellpert.models.scgpt import ScGPTGeneEncoder, ScGPTNetwork
from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel
from subcellpert.utils import resolve_model_directory


class ScGPTSpatialNetwork(ScGPTNetwork):
    """Spatial checkpoint's non-generative continuous-value encoder.

    Its source-only encoding path shares scGPT's gene/value embeddings and
    Transformer. Spatial MoE, imputation and generation heads are unused here.
    See THIRD_PARTY_LICENSES.md for the original implementation's MIT terms.
    """

    def _encode(self, tokens, values, src_key_padding_mask):
        gene_states = self.encoder(tokens)
        expression_states = self.value_encoder(values)
        return self.transformer_encoder(gene_states + expression_states, src_key_padding_mask=src_key_padding_mask)


class ScGPTSpatialGeneEncoder(ScGPTGeneEncoder):
    network_class = ScGPTSpatialNetwork

    def __init__(self, root, metadata, device):
        if metadata["backbone"] != "scgpt_spatial":
            raise ValueError("scGPT-spatial requires its spatial checkpoint and source normalization")
        super().__init__(root, metadata, device)
        table = pd.read_csv(resolve_model_directory(root, metadata) / "all_dict_mean_std.csv", index_col=0)
        self.gene_means = {int(key): float(value) for key, value in table["mean"].items()}

    def normalize_values(self, values, genes, selected):
        """Normalize with checkpoint gene means and frozen training-source fallback."""

        if self.statistics is None:
            raise ValueError("scGPT-spatial needs frozen training-source normalization statistics")
        scale = self.statistics["nonzero_mean"]
        fallback = np.asarray(self.statistics["gene_nonzero_mean"]) / scale
        axis_ids = self.gene_ids(genes)
        means = np.asarray(
            [self.gene_means.get(int(token), value) for token, value in zip(axis_ids, fallback, strict=True)]
        )
        means = torch.as_tensor(means, dtype=torch.float64, device=self.device)
        means = torch.where(torch.isfinite(means) & (means > 0), means, 1.0)
        values = (values / scale / means[selected]).float()
        return values


class Model(Baseline):
    name = "scgpt_spatial"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = ScGPTSpatialGeneEncoder
