#!/usr/bin/env python3
"""Download pretrained resources into the layout used by configs/models."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    "scgpt": "scgpt-human",
    "scgpt_spatial": "scgpt-spatial/scGPT_spatial_v1",
    "scfoundation": "scfoundation",
    "spatialformer": "spatialformer",
    "state_se": "state-se",
}
SPATIAL_CHECKPOINT = "stepstep=0176000-traintrain_total_loss=-2.7789-valval_total_loss=0.0000.ckpt"
SPATIAL_CONFIG = {
    "dim_model": 512,
    "nheads": 64,
    "nlayers": 10,
    "n_tokens": 6061,
    "n_atokens": 4,
    "spatial_embedding": True,
    "spatial_embedding_freeze": False,
}
FILES = {
    "scgpt": ("best_model.pt", "args.json", "vocab.json"),
    "scgpt_spatial": (
        "best_model.pt",
        "args.json",
        "vocab.json",
        "batch_mapping_dict.json",
        "all_dict_mean_std.csv",
    ),
    "scfoundation": ("models.ckpt", "OS_scRNA_gene_index.19264.tsv"),
    "spatialformer": (
        SPATIAL_CHECKPOINT,
        "xenium_5k_pandavid_gene_embeddings.pkl",
        "tokenizer/tokenv5.json",
        "data/gene_median.pkl",
    ),
    "state_se": ("model.safetensors", "protein_embeddings.pt", "config.yaml", "HOM_MouseHumanSequence.rpt"),
}
HF_MODELS = {
    "scgpt": ("wanglab/scGPT-human", "a24c237737a40f3720f75abb555489e9fe753be6", FILES["scgpt"]),
    "scfoundation": ("genbio-ai/scFoundation", "cb434153a1acfacd215eefc956ea445f7cc39cc3", ("models.ckpt",)),
    "state_se": (
        "arcinstitute/SE-600M",
        "5a9a80f44f7ce32ce57059933ef0d735d7c10ce5",
        (*FILES["state_se"][:3], "LICENSE.md", "MODEL_LICENSE.md", "MODEL_ACCEPTABLE_USE_POLICY.md"),
    ),
}
SPATIAL_BASE = (
    "https://raw.githubusercontent.com/TerminatorJ/Spatialformer/"
    "f02e7776e8a99c9abdc5fe5572a1a20369faf159/spatialformer"
)
MGI_URL = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
URLS = {
    "scfoundation": {
        "OS_scRNA_gene_index.19264.tsv": (
            "https://raw.githubusercontent.com/biomap-research/scFoundation/"
            "397631c495eddf9ad6644fc00c6ea8139e651245/OS_scRNA_gene_index.19264.tsv"
        ),
    },
    "spatialformer": {
        SPATIAL_CHECKPOINT: "https://ndownloader.figshare.com/files/61331527",
        "xenium_5k_pandavid_gene_embeddings.pkl": (
            f"{SPATIAL_BASE}/spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl"
        ),
        "tokenizer/tokenv5.json": f"{SPATIAL_BASE}/tokenizer/tokenv5.json",
        "data/gene_median.pkl": f"{SPATIAL_BASE}/data/gene_median.pkl",
    },
    "state_se": {"HOM_MouseHumanSequence.rpt": MGI_URL},
}
ARCHIVE_NAME = "scGPT_spatial_v1.zip"
ARCHIVE_URL = "https://ndownloader.figshare.com/files/52163879"


def install_file(destination, writer):
    """Keep existing files; publish new files only after writing succeeds."""
    if destination.is_file():
        print(f"Keep {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Prepared {destination}", flush=True)


def download_file(url, destination):
    def write(temporary):
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    install_file(destination, write)


def unpack_spatial(archive, directory):
    """Extract only the five expected files, retaining the archive's model directory."""
    with zipfile.ZipFile(archive) as bundle:
        for name in FILES["scgpt_spatial"]:
            member = f"scGPT_spatial_v1/{name}"
            bundle.getinfo(member)

            def write(temporary, member=member):
                with bundle.open(member) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)

            install_file(directory / name, write)


def prepare_model(model, weights_dir):
    directory = weights_dir / DIRECTORIES[model]
    if model in HF_MODELS:
        from huggingface_hub import hf_hub_download

        repository, revision, names = HF_MODELS[model]
        for name in names:
            if (directory / name).is_file():
                print(f"Keep {directory / name}", flush=True)
            else:
                hf_hub_download(repository, name, revision=revision, local_dir=directory)
    if model == "scgpt_spatial":
        if not all((directory / name).is_file() for name in FILES[model]):
            archive = weights_dir / "downloads" / ARCHIVE_NAME
            download_file(ARCHIVE_URL, archive)
            unpack_spatial(archive, directory)
    for name, url in URLS.get(model, {}).items():
        download_file(url, directory / name)

    if model == "spatialformer":
        # Keep the upstream fields used to construct the encoder.
        install_file(
            directory / "config/_config_train_large_pair.json",
            lambda temporary: temporary.write_text(json.dumps(SPATIAL_CONFIG, indent=2) + "\n"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(DIRECTORIES), default=list(DIRECTORIES))
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights", help="Destination weights directory")
    args = parser.parse_args()
    try:
        for model in dict.fromkeys(args.models):
            prepare_model(model, args.weights_dir)
    except (OSError, ValueError, zipfile.BadZipFile, KeyError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Resources prepared in {args.weights_dir}.")


if __name__ == "__main__":
    main()
