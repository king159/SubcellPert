"""scfoundation baseline: source encoder/analytic transport and joint/oracle prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from subcellpert.models.shared import Baseline, GearsCellJointModel, GearsCellOracleModel
from subcellpert.utils import align_gene_ids, freeze, install_transformer_sdpa


class CellIndependentAttention(torch.nn.Module):
    """Vectorize scFoundation's native singleton softmax Performer kernel.

    scFoundation's Performer key kernel subtracts a scalar maximum over the
    entire batch before adding epsilon. That epsilon makes a cell's output
    depend on its batch mates. Reduce over heads, tokens and random features
    separately for each cell. Keep the upstream epsilon, projection, scaling
    and linear attention equations, with no Python loop over cells.
    """

    def __init__(self, attention):
        super().__init__()
        if attention.causal or attention.generalized_attention or attention.no_projection:
            raise ValueError("Expected noncausal softmax Performer attention")
        self.attention = attention

    def _kernel(self, data, *, is_query):
        normalizer = data.shape[-1] ** (-0.25)
        projection = self.attention.projection_matrix.type_as(data)
        projection = projection.expand(data.shape[0], data.shape[1], -1, -1)
        projected = torch.einsum("...id,...jd->...ij", normalizer * data, projection)
        diagonal = ((data**2).sum(dim=-1) / 2.0 * normalizer**2).unsqueeze(-1)
        maximum = (
            projected.max(dim=-1, keepdim=True).values if is_query else projected.amax(dim=(1, 2, 3), keepdim=True)
        )
        return ((projected - diagonal - maximum).exp() + 0.0001) * projection.shape[-2] ** (-0.5)

    @staticmethod
    def _linear(q, k, v):
        inverse = 1.0 / torch.einsum("...nd,...d->...n", q, k.sum(dim=-2).type_as(q))
        context = torch.einsum("...nd,...ne->...de", k, v)
        return torch.einsum("...de,...nd,...n->...ne", context, q, inverse)

    def forward(self, q, k, v, output_attentions=False):
        q, k = (self._kernel(q, is_query=True), self._kernel(k, is_query=False))
        result = self._linear(q, k, v)
        if output_attentions:
            identity = torch.eye(v.shape[-2], device=v.device, dtype=torch.float16)
            identity = identity.expand(v.shape[0], v.shape[1], -1, -1)
            weights = self._linear(q.half(), k.half(), identity).detach().cpu() / q.shape[1]
            return (result, weights)
        return result


def isolate_attention_cells(module, attention_type):
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, attention_type):
            setattr(module, name, CellIndependentAttention(child))
            replaced += 1
        else:
            replaced += isolate_attention_cells(child, attention_type)
    return replaced


class ScFoundationGeneEncoder:
    def __init__(self, root, metadata, device):
        repository = root / "external/scFoundation"
        sys.path.insert(0, str(repository / "model"))
        from load import convertconfig, gatherData
        from pretrainmodels import select_model
        from pretrainmodels.performer import FastAttention

        checkpoint = Path(metadata["model_checkpoint"])
        checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
        payload = convertconfig(torch.load(checkpoint, map_location="cpu", weights_only=False)["gene"])
        self.config = payload["config"]
        self.config.setdefault("ppi_edge", None)
        self.config["device"] = str(device)
        self.model = select_model(self.config)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to_final = None
        install_transformer_sdpa(self.model)
        if not isolate_attention_cells(self.model, FastAttention):
            raise ValueError("Expected scFoundation Performer attention implementation was not found")
        freeze(self.model, device)
        self.gather = gatherData
        import pandas as pd

        names = pd.read_csv(repository / "OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"].tolist()
        if len(names) != 19264:
            raise ValueError("scFoundation requires its official 19,264-gene index")
        self.vocabulary = {name: index for index, name in enumerate(names)}
        self.dimension, self.device = (int(self.config["decoder"]["hidden_dim"]) + 1, device)
        self.gene_context = metadata.get("scfoundation_gene_context", "assay")
        if self.gene_context not in ("assay", "full"):
            raise ValueError("scFoundation gene context must be 'assay' or 'full'")

    def gene_ids(self, genes):
        return align_gene_ids(genes, self.vocabulary)

    @torch.no_grad()
    def encode(self, matrix, genes, condition):
        del condition
        matrix = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(self.gene_ids(genes), device=self.device)
        columns = torch.nonzero(ids >= 0, as_tuple=True)[0]
        columns = columns[ids[columns].argsort(stable=True)]
        result = torch.zeros((len(matrix), len(genes), self.dimension), dtype=torch.float32, device=self.device)
        nonempty = torch.nonzero(matrix[:, columns].sum(1) > 0, as_tuple=True)[0]
        if not len(nonempty):
            return result
        raw = matrix[nonempty[:, None], columns]
        gene_positions = ids[columns]
        output_positions = torch.arange(len(columns), device=self.device)
        if self.gene_context == "full":
            full = torch.zeros((len(nonempty), 19264), device=self.device)
            full[:, gene_positions] = raw
            raw, gene_positions = full, torch.arange(19264, device=self.device)
            output_positions = ids[columns]
        total = raw.sum(1)
        normalized = torch.log1p(raw / total[:, None] * 10000)
        depth = torch.log10(total)[:, None]
        decoder = torch.cat((normalized, depth, depth), 1)
        labels = decoder > 0
        labels[:, -2:] = True
        positions = torch.cat((gene_positions, torch.tensor([19264, 19265], device=self.device)))
        positions = positions.expand(len(nonempty), -1)
        encoded, padding = self.gather(decoder, labels, self.config["pad_token_id"])
        encoder_positions, _ = self.gather(positions, labels, self.config["pad_token_id"])
        encoder_positions[padding] = self.config["seq_len"]
        states = self.model(
            x=encoded,
            padding_label=padding,
            encoder_position_gene_ids=encoder_positions,
            encoder_labels=labels,
            decoder_data=decoder,
            mask_gene_name=False,
            mask_labels=None,
            decoder_position_gene_ids=positions,
            decoder_data_padding_labels=torch.zeros_like(decoder, dtype=torch.bool),
        )
        result[nonempty[:, None], columns, :-1] = states[:, output_positions].float()
        result[nonempty[:, None], columns, -1] = 1
        return result


class Model(Baseline):
    name = "scfoundation"
    foundation = True
    oracle_class = GearsCellOracleModel
    joint_class = GearsCellJointModel
    encoder_class = ScFoundationGeneEncoder
