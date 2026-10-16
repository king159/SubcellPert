"""Source-only foundation encoders and device-resident gene features."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from subcellpert.utils import (
    resolve_model_directory,
    sha256_file,
)


GENE_FOUNDATION_SCHEMA = "frozen_source_cell_gene_features_v1"


def foundation_config_paths(root, dataset_code, backbone):
    """Shared model and dataset configuration, never a feature-cache locator."""
    return (
        Path(root) / "configs/models" / f"{backbone}.yaml",
        Path(root) / "configs/datasets" / f"{dataset_code}.yaml",
    )


def load_foundation_configuration(root, dataset_code, backbone):
    """Defaults for diagnostic tools; the CLI passes its fully resolved config."""
    from omegaconf import OmegaConf

    paths = foundation_config_paths(root, dataset_code, backbone)
    return OmegaConf.to_container(OmegaConf.merge(*(OmegaConf.load(path) for path in paths)), resolve=True)


def foundation_metadata_from_config(root, spec, backbone, configuration):
    """Validate explicit weights and source tokens before any encoder is loaded."""
    root = Path(root)
    model, data = configuration.get("foundation", {}), configuration.get("data", {})
    if model.get("backbone") != backbone or data.get("dataset_id") != spec.dataset:
        raise ValueError(
            "FM configuration must declare matching foundation.backbone and data.dataset_id; inherit configs/models and configs/datasets"
        )
    dimension, length = model.get("embedding_dimension"), model.get("encoder_max_length")
    if type(dimension) is not int or dimension < 1 or type(length) is not int or length < 2:
        raise ValueError("foundation.embedding_dimension and encoder_max_length must be positive integers")
    sources = data.get("source_conditions")
    expected = {transition.source_condition for transition in spec.transitions}
    if (
        not isinstance(sources, list)
        or not all(isinstance(source, str) for source in sources)
        or len(set(sources)) != len(sources)
        or set(sources) != expected
    ):
        raise ValueError("data.source_conditions must match the frozen fit's source conditions exactly")
    if data.get("species") not in {"Human", "Mouse"}:
        raise ValueError("data.species must be Human or Mouse")
    metadata = {
        **deepcopy(model),
        "compute_precision": configuration.get("training", {}).get("precision", "fp32"),
        "dataset": spec.dataset,
        "task_id": spec.task_id,
        "conditioning": "source_only",
        "target_expression_used": False,
        "target_coordinates_used": False,
        "source_conditions": sorted(sources),
        "species": data["species"],
        "configuration_source": "resolved_model_and_dataset_yaml",
    }
    if backbone in {"scgpt", "scgpt_spatial", "state_se"}:
        key = "model_directory"
    elif backbone in {"scfoundation", "spatialformer"}:
        key = "model_checkpoint"
    else:
        raise ValueError(f"Unknown foundation backbone: {backbone}")
    if not isinstance(model.get(key), str) or not model[key]:
        raise ValueError(f"foundation.{key} must explicitly locate pretrained weights")
    location = root / model[key]
    if backbone in {"scgpt", "scgpt_spatial"}:
        required = [location / name for name in ("args.json", "vocab.json", "best_model.pt")]
        if backbone == "scgpt_spatial":
            required.append(location / "all_dict_mean_std.csv")
    elif backbone == "state_se":
        required = [
            location / name
            for name in (
                "model.safetensors",
                "protein_embeddings.pt",
                "config.yaml",
                "HOM_MouseHumanSequence.rpt",
            )
        ]
    elif backbone == "scfoundation":
        required = [location, location.parent / "OS_scRNA_gene_index.19264.tsv"]
    else:
        package = location.parent
        required = [
            location,
            *(
                package / name
                for name in (
                    "config/_config_train_large_pair.json",
                    "tokenizer/tokenv5.json",
                    "data/gene_median.pkl",
                )
            ),
            location.parent / "xenium_5k_pandavid_gene_embeddings.pkl",
        ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Configured pretrained FM asset is missing: {path}")
    if backbone in {"scgpt", "scgpt_spatial"}:
        expected_dimension = json.loads((location / "args.json").read_text())["embsize"]
    elif backbone == "spatialformer":
        expected_dimension = json.loads((package / "config/_config_train_large_pair.json").read_text())["dim_model"]
        tokens = deepcopy(data.get("spatialformer", {}))
        tokens.pop("token_mapping_note", None)
        tokens["species"] = data["species"]
        conditions = tokens.get("condition_by_source", {})
        if (
            not isinstance(conditions, dict)
            or set(conditions) != expected
            or not set(conditions.values()) <= {"Healthy", "Disease"}
        ):
            raise ValueError("data.spatialformer.condition_by_source must map every source to Healthy or Disease")
        vocabulary = json.loads((package / "tokenizer/tokenv5.json").read_text())
        if any(
            token not in vocabulary or not 3 <= vocabulary[token] <= 26
            for token in [tokens.get("tissue"), tokens.get("assay"), tokens["species"], *conditions.values()]
        ):
            raise ValueError(
                "SpatialFormer species/tissue/assay/condition tokens must exist in the pretrained prefix vocabulary"
            )
        metadata["spatialformer_tokens"] = tokens
    elif backbone == "state_se":
        from subcellpert.models.state_se import STATE_FEATURE_POLICY

        if model.get("state_feature_policy") != STATE_FEATURE_POLICY:
            raise ValueError("foundation.state_feature_policy must specify the supported STATE-SE policy")
        expected_dimension = yaml.safe_load((location / "config.yaml").read_text())["model"]["emsize"]
    else:
        expected_dimension = 512  # scFoundation gene states, rather than its pooled 3072-wide cell vector.
    if dimension != expected_dimension:
        raise ValueError(
            f"foundation.embedding_dimension={dimension} disagrees with pretrained gene width {expected_dimension}"
        )
    return metadata


def foundation_source_binding(root, backbone):
    """Bind integrated model code independently of the dataset root."""
    del root
    if backbone not in {"scgpt", "scgpt_spatial", "scfoundation", "spatialformer", "state_se"}:
        raise ValueError(f"Unknown foundation backbone: {backbone}")
    directory = Path(__file__).resolve().parents[1] / "models"
    names = [backbone, "shared"]
    if backbone == "scgpt_spatial":
        names.append("scgpt")
    return {f"models/{name}.py": sha256_file(directory / (name + ".py")) for name in names}


def source_statistics(records, genes, matched):
    sums, nonzero = (np.zeros(len(genes)), np.zeros(len(genes), dtype=np.int64))
    seen = set()
    for record in records:
        for source in record.source:
            if tuple(source.genes) != tuple(genes):
                raise ValueError("Source normalization gene order differs between populations")
            for first in range(0, len(source.cell_ids), 256):
                rows = []
                for row in range(first, min(first + 256, len(source.cell_ids))):
                    key = (source.unit_id, source.condition, str(source.cell_ids[row]))
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                if not rows:
                    continue
                values = np.asarray(source.counts[rows], dtype=np.float64)
                sums += values.sum(0)
                nonzero += (values > 0).sum(0)
    scale = float(sums[matched].sum() / max(int(nonzero[matched].sum()), 1))
    means = np.divide(sums, nonzero, out=np.ones(len(genes)), where=nonzero > 0)
    return {
        "genes": list(genes),
        "nonzero_mean": scale if scale > 0 else 1.0,
        "gene_nonzero_mean": means.tolist(),
        "source_cells": len(seen),
        "normalization_fit_scope": "training_source_cells_only",
    }


def load_gene_encoder(root, metadata, device):
    from subcellpert.utils import get_model

    adapter = get_model(metadata["backbone"], "oracle")
    if not adapter.foundation:
        raise ValueError("Analytic baselines do not load pretrained gene encoders")
    return adapter.encoder_class(Path(root), metadata, torch.device(device))


def load_cell_foundation(root, spec, baseline, *, configuration=None):
    if configuration is None:
        configuration = load_foundation_configuration(root, spec.code, baseline)
    metadata = foundation_metadata_from_config(root, spec, baseline, configuration)
    encoder_batch = configuration.get("foundation", {}).get("encoder_batch_size")
    return GeneFoundationStore(
        root,
        metadata,
        expected_dataset=spec.dataset,
        expected_backbone=baseline,
        expected_task_id=spec.task_id,
        encoder_batch_size=encoder_batch if type(encoder_batch) is int else None,
    )


class GeneFoundationStore:
    """Batch frozen encoders on the owner thread; never read pooled .npy files.

    Model paths and source tokens come from resolved model/dataset configuration.
    No dataset-specific cached embeddings or legacy metadata sidecars are read.
    """

    online = True

    def __init__(
        self,
        root,
        metadata,
        *,
        expected_dataset,
        expected_backbone,
        expected_task_id,
        device="cpu",
        encoder_batch_size=None,
        cache_cells=0,
        encoder_factory=load_gene_encoder,
    ):
        self.root = Path(root)
        if not isinstance(metadata, Mapping):
            raise TypeError("GeneFoundationStore requires a resolved FM configuration mapping, not a metadata file")
        self.metadata = deepcopy(dict(metadata))
        m = self.metadata
        if (
            m.get("dataset") != expected_dataset
            or m.get("backbone") != expected_backbone
            or m.get("task_id") != expected_task_id
            or (m.get("conditioning") != "source_only")
            or (m.get("target_expression_used") is not False)
            or (m.get("target_coordinates_used") is not False)
        ):
            raise ValueError("Gene foundation model locator has incompatible source-only bindings")
        if encoder_batch_size is not None and (type(encoder_batch_size) is not int or encoder_batch_size < 1):
            raise ValueError("Encoder batch size must be a positive integer")
        if cache_cells != 0:
            raise ValueError("Device feature execution does not retain a cell-feature cache")
        self.device, self.encoder_batch_size, self.cache_cells = (device, encoder_batch_size, cache_cells)
        self.encoder_batch_limit = encoder_batch_size
        self.training_batch_size = None
        self.factory, self._encoder, self._binding = (encoder_factory, None, None)
        self._lookups = {}
        self.statistics = None
        self.feature_access = "online"
        self.dimension = int(m["embedding_dimension"]) + 1

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = self.factory(self.root, self.metadata, self.device)
            if self._encoder.dimension != self.dimension:
                raise ValueError("Gene FM width disagrees with the configured pretrained checkpoint")
            self._encoder.statistics = self.statistics
        return self._encoder

    def feature_shape(self, genes):
        return (len(genes), self.dimension)

    def prepare(self, records):
        genes = records[0].source[0].genes
        self.set_statistics(source_statistics(records, genes, self.encoder.gene_ids(genes) >= 0))

    def set_statistics(self, statistics):
        if (
            statistics.get("normalization_fit_scope") != "training_source_cells_only"
            or not np.isfinite(statistics["nonzero_mean"])
            or statistics["nonzero_mean"] <= 0
            or (len(statistics["genes"]) != len(statistics["gene_nonzero_mean"]))
            or (not np.isfinite(statistics["gene_nonzero_mean"]).all())
        ):
            raise ValueError("Invalid frozen source normalization state")
        self.statistics = statistics
        self.online = True
        if self._encoder is not None:
            self._encoder.statistics = statistics

    @property
    def binding(self):
        if self._binding is None:
            m = self.metadata
            files = []
            if m["backbone"] in ("scgpt", "scgpt_spatial"):
                directory = resolve_model_directory(self.root, m)
                files = [directory / name for name in ("args.json", "vocab.json", "best_model.pt")]
                if m["backbone"] == "scgpt_spatial":
                    files.append(directory / "all_dict_mean_std.csv")
            elif m["backbone"] == "state_se":
                directory = resolve_model_directory(self.root, m)
                files.extend(
                    directory / name
                    for name in (
                        "model.safetensors",
                        "protein_embeddings.pt",
                        "config.yaml",
                        "HOM_MouseHumanSequence.rpt",
                    )
                )
                files.append(Path(__file__).resolve().parents[1] / "models/state_se.py")
            else:
                checkpoint = (self.root / m["model_checkpoint"]).resolve()
                files.append(checkpoint)
                if m["backbone"] == "scfoundation":
                    files.append(checkpoint.parent / "OS_scRNA_gene_index.19264.tsv")
                else:
                    files.extend(
                        checkpoint.parent / name
                        for name in (
                            "config/_config_train_large_pair.json",
                            "tokenizer/tokenv5.json",
                            "data/gene_median.pkl",
                        )
                    )
                    files.append(checkpoint.parent / "xenium_5k_pandavid_gene_embeddings.pkl")
            self._binding = {
                "schema": GENE_FOUNDATION_SCHEMA,
                "backbone": m["backbone"],
                "dimension": self.dimension,
                "features": "contextual_gene_states_plus_per_gene_availability",
                "frozen": True,
                "attention_execution": "torch_sdpa_preserve_upstream_scale_mask_qkv_v1",
                "attention_adapter_sha256": sha256_file(Path(__file__).resolve().parents[1] / "utils.py"),
                "source_conditions": sorted(m["source_conditions"]),
                "spatialformer_tokens": m.get("spatialformer_tokens"),
                "max_gene_tokens": int(m.get("encoder_max_length", 1200)),
                "weight_and_input_sha256": {
                    str(path): sha256_file(path if path.is_absolute() else self.root / path) for path in files
                },
                "upstream_source_sha256": foundation_source_binding(self.root, m["backbone"]),
                "normalization": "training_source_only; deterministic_lower_quantile_ties",
                "scfoundation_attention": "native_single_cell_kernel; vectorized_per_cell_max_v1"
                if m["backbone"] == "scfoundation"
                else None,
                "pooled_cache_consumed": False,
                "feature_execution": "device_tensor_encoder_microbatch_v1"
                if self.encoder_batch_policy == "fixed_encoder_microbatch"
                else "device_tensor_matched_training_batch_v1",
                "compute_precision": m.get("compute_precision", "fp32"),
            }
            if m["backbone"] == "state_se":
                from subcellpert.models.state_se import state_binding_details

                self._binding["state_se"] = state_binding_details(m)
                self._binding["normalization"] = self._binding["state_se"]["normalization"]
            if m["backbone"] == "scfoundation":
                self._binding["scfoundation_gene_context"] = m.get("scfoundation_gene_context", "assay")
                self._binding["scfoundation_gene_positions"] = (
                    "official_sorted_ids; retain_matched_zeros; two_depth_tokens"
                )
                self._binding["max_gene_tokens"] = None
        return self._binding

    def use_online_features(self):
        self.feature_access = "online"
        self.online, self.cache_cells = True, 0

    def validate_sources(self, populations, on_progress=None):
        self.use_online_features()
        cells = 0
        seen = set()
        for population in populations:
            self.validate_population(population)
            if id(population) not in seen:
                seen.add(id(population))
                cells += len(population.cell_ids)
        report = {
            "feature_access_policy": "online_source_features_no_disk_cache_v1",
            "online": True,
            "reserved_bytes": 0,
            "persistent_cache": False,
            "precomputed_cells": 0,
            "disk_cache_bytes_written": 0,
            "lru_cache_cells": 0,
            "source_cells_in_validated_views": cells,
            "source_population_objects_validated": len(seen),
            "encoder_batch_size": self.encoder_batch_size,
            "encoder_batch_policy": self.encoder_batch_policy,
            "compute_precision": self.metadata.get("compute_precision", "fp32"),
            "feature_device": str(self.device),
            "feature_transport": "device_tensor",
            "prefetch": "bounded_CPU_collation_with_encoder_on_owner_thread",
            "encoder_execution": "torch_compile_dynamic"
            if self.metadata.get("compile_encoder", False) and torch.device(self.device).type == "cuda"
            else "eager",
        }
        if on_progress:
            on_progress(report)
        return report

    def validate_features(self, values, cells, genes):
        if not isinstance(values, torch.Tensor):
            raise TypeError("Foundation features must remain device tensors")
        expected = torch.device(self.device)
        if values.device.type != expected.type or (
            expected.index is not None and values.device.index != expected.index
        ):
            raise ValueError("Foundation features left the encoder device")
        if (
            values.shape != (cells, genes, self.dimension)
            or values.dtype != torch.float32
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Gene FM returned invalid or incorrectly aligned states")
        if values.requires_grad or torch.any((values[..., -1] != 0) & (values[..., -1] != 1)):
            raise ValueError("Frozen gene features require binary availability and no autograd history")

    @property
    def encoder_batch_policy(self):
        return (
            "fixed_encoder_microbatch"
            if self.training_batch_size is not None and self.encoder_batch_size < self.training_batch_size
            else "match_training"
        )

    def set_training_batch_size(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Training/encoder batch size must be a positive integer")
        self.training_batch_size = batch_size
        self.encoder_batch_size = min(batch_size, self.encoder_batch_limit or batch_size)

    @torch.no_grad()
    def encode_counts(self, counts, genes, condition):
        """Chunk frozen encoding without changing the optimizer batch or leaving the device."""
        from subcellpert.precision import neural_autocast

        if self.encoder_batch_size is None:
            raise ValueError("Bind the training batch size before encoding source cells")
        counts = torch.as_tensor(counts, dtype=torch.float32, device=self.device)
        conditions = [condition] * len(counts) if isinstance(condition, str) else list(condition)
        if len(conditions) != len(counts) or any(c not in self.metadata["source_conditions"] for c in conditions):
            raise ValueError("Every encoded cell requires a registered source condition")
        if counts.ndim != 2 or counts.shape[1] != len(genes) or not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Source counts must be a finite nonnegative cell-by-gene matrix")
        chunks = []
        for first in range(0, len(counts), self.encoder_batch_size):
            block = counts[first : first + self.encoder_batch_size]
            with neural_autocast(self.metadata.get("compute_precision", "fp32"), self.device):
                values = self.encoder.encode(block, genes, conditions[first : first + len(block)])
            self.validate_features(values, len(block), len(genes))
            chunks.append(values)
        if not chunks:
            return torch.empty((0, len(genes), self.dimension), device=self.device)
        combined = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return combined

    def validate_population(self, population):
        if (
            population.dataset != self.metadata["dataset"]
            or population.condition not in self.metadata["source_conditions"]
        ):
            raise ValueError("Only registered source populations may enter a frozen gene FM")
        if self.statistics is None or tuple(population.genes) != tuple(self.statistics["genes"]):
            raise ValueError("Gene FM requires the frozen training-source gene axis and statistics")

    def _source_rows(self, population, cell_ids):
        self.validate_population(population)
        key = id(population)
        if key not in self._lookups or self._lookups[key][0] is not population:
            lookup = {str(cell): row for row, cell in enumerate(population.cell_ids)}
            if len(lookup) != len(population.cell_ids):
                raise ValueError("Duplicate source cell identities")
            if len(self._lookups) >= 8:
                self._lookups.clear()
            self._lookups[key] = population, lookup
        lookup = self._lookups[key][1]
        try:
            return [lookup[str(cell)] for cell in cell_ids]
        except KeyError as error:
            raise ValueError("Unknown source cell requested from gene FM") from error

    def get(self, population, cell_ids):
        rows = self._source_rows(population, cell_ids)
        return self.encode_counts(population.counts[rows], population.genes, population.condition)

    def get_batch(self, selections, genes):
        """Combine source populations by destination row, preserving each cell's condition."""
        size = sum(len(cell_ids) for _, cell_ids, _ in selections)
        counts = np.empty((size, len(genes)), dtype=np.float32)
        conditions = [None] * size
        seen = set()
        for population, cell_ids, destinations in selections:
            rows = self._source_rows(population, cell_ids)
            if tuple(population.genes) != tuple(genes) or len(rows) != len(destinations):
                raise ValueError("Source feature gene/cell axes differ across populations")
            for destination in destinations:
                if not 0 <= destination < size or destination in seen:
                    raise ValueError("Source feature destinations must form one permutation")
                seen.add(destination)
                conditions[destination] = population.condition
            counts[destinations] = population.counts[rows]
        if len(seen) != size:
            raise ValueError("Missing source feature destination")
        return self.encode_counts(counts, genes, conditions)
