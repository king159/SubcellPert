# Pretrained models

The five FM networks are implemented directly in `src/subcellpert/models/`.
Keep each model's pretrained weights and auxiliary files in its own directory:

```text
weights/
├── scgpt-human/                       # best_model.pt, args.json, vocab.json
├── scgpt-spatial/scGPT_spatial_v1/     # weights, vocabulary, normalization
├── scfoundation/                     # models.ckpt, gene index
├── spatialformer/                    # checkpoint, gene embeddings, config/, tokenizer/, data/
└── state-se/                         # weights, protein embeddings, config, homology, assets.json
```

No upstream checkout or model package is required. The `weights/` directory is
ignored by Git and is not included in Python distributions. The commands below
prepare checkpoint files and the small resources needed by each encoder.
For a separate dataset root, link the complete `weights/` directory as described
in [reproduction.md](reproduction.md#data-and-model-resources).
Source revisions and original license texts are in
[third-party notices](../THIRD_PARTY_LICENSES.md).

Run these commands from the code repository root with your environment active.
Use fresh destinations; keep existing checkpoints used by other experiments.
These pointers and revisions are the recorded benchmark resources, not a claim
that all remote downloads were revalidated during code cleanup.

<details>
<summary><strong>scGPT</strong></summary>

Open the official
[whole-human folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)
and download `best_model.pt`, `args.json`, and `vocab.json` into
`weights/scgpt-human/`. Keep the three files from the same checkpoint
folder. Alternatively, with `gdown` installed in your active environment:

```bash
python -m gdown --folder \
  'https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y' \
  -O weights/scgpt-human
```

The files must be directly inside `scgpt-human/`, without an extra nested
download-directory level.

</details>

<details>
<summary><strong>scGPT-spatial</strong></summary>

The [official repository](https://github.com/bowang-lab/scGPT-spatial) links the
V1 archive. Download and extract it from the project root:

```bash
mkdir -p weights/scgpt-spatial
model_download_dir="$(mktemp -d)"
curl --fail --location --retry 3 \
  'https://ndownloader.figshare.com/files/52163879' \
  --output "$model_download_dir/scGPT_spatial_v1.zip"
unzip "$model_download_dir/scGPT_spatial_v1.zip" -d weights/scgpt-spatial
```

The final directory must be
`weights/scgpt-spatial/scGPT_spatial_v1/`, containing `best_model.pt`,
`args.json`, `vocab.json`, `batch_mapping_dict.json`, and `all_dict_mean_std.csv`.
Preserve this archive subdirectory; the YAML points to it. The archive's Figshare MD5 is
`470d40fb77535afc2b4e7cffc0d665ef`.

</details>

<details>
<summary><strong>scFoundation</strong></summary>

Follow the download pointer in the official repository's
[`model/models/download.txt`](https://github.com/biomap-research/scFoundation/blob/main/model/models/download.txt).
At the time these instructions were checked, it points to this
[BioMap SharePoint folder](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk).
Download `models.ckpt` and put it at
`weights/scfoundation/models.ckpt`.

Our current file is 1,432,587,886 bytes, with SHA-256
`9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9`.
The adapter loads its `gene` entry. Download its fixed 19,264-gene index beside
the checkpoint:

```bash
mkdir -p weights/scfoundation
curl --fail --location --retry 3 \
  'https://raw.githubusercontent.com/biomap-research/scFoundation/397631c495eddf9ad6644fc00c6ea8139e651245/OS_scRNA_gene_index.19264.tsv' \
  --output weights/scfoundation/OS_scRNA_gene_index.19264.tsv
```

</details>

<details>
<summary><strong>SpatialFormer</strong></summary>

The benchmark uses the **single-input 5k / 71-slide** checkpoint linked by the
[official repository](https://github.com/TerminatorJ/Spatialformer#get-the-embeddings).
Download the exact file below; the paired checkpoint has a similar filename
but different weights.

```bash
mkdir -p weights/spatialformer
curl --fail --location --retry 3 \
  'https://ndownloader.figshare.com/files/61331527' \
  --output 'weights/spatialformer/stepstep=0176000-traintrain_total_loss=-2.7789-valval_total_loss=0.0000.ckpt'
```

Its expected size is 720,940,445 bytes; Figshare MD5 is
`fb2c33044244bf21a962e106951769f5`. Prepare the tokenizer, gene medians and
pretrained gene embeddings in the same model directory:

```bash
mkdir -p weights/spatialformer/{config,tokenizer,data}
spatialformer_source='https://raw.githubusercontent.com/TerminatorJ/Spatialformer/f02e7776e8a99c9abdc5fe5572a1a20369faf159/spatialformer'
curl --fail --location --retry 3 \
  "$spatialformer_source/tokenizer/tokenv5.json" \
  --output weights/spatialformer/tokenizer/tokenv5.json
curl --fail --location --retry 3 \
  "$spatialformer_source/data/gene_median.pkl" \
  --output weights/spatialformer/data/gene_median.pkl
curl --fail --location --retry 3 \
  "$spatialformer_source/spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl" \
  --output weights/spatialformer/xenium_5k_pandavid_gene_embeddings.pkl
```

Write the inference configuration used by this checkpoint. Only the network
settings consumed by the integrated encoder are needed:

```bash
cat > weights/spatialformer/config/_config_train_large_pair.json <<'JSON'
{
  "dim_model": 512,
  "nheads": 64,
  "nlayers": 10,
  "n_tokens": 6061,
  "n_atokens": 4,
  "spatial_embedding": true,
  "spatial_embedding_freeze": false
}
JSON
```

</details>

<details>
<summary><strong>STATE-SE</strong></summary>

Download the inference assets at the pinned revision into the configured model
directory:

```bash
hf download arcinstitute/SE-600M \
  model.safetensors protein_embeddings.pt config.yaml \
  LICENSE.md MODEL_LICENSE.md MODEL_ACCEPTABLE_USE_POLICY.md \
  --revision 5a9a80f44f7ce32ce57059933ef0d735d7c10ce5 \
  --local-dir weights/state-se

curl --fail --location \
  https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt \
  --output weights/state-se/HOM_MouseHumanSequence.rpt
```

The second command retrieves the mouse/human mapping report from
[MGI](https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt).
The report changes over time: compare it with the recorded checksum to identify
the mapping version used in our runs. The large `se600m_epoch*.ckpt` training
snapshots are not needed for this encoder.

Create the local asset identity file from the downloaded files. This records
the actual mapping report and weight hashes used by the encoder:

```bash
python - <<'PYTHON'
import hashlib
import json
from pathlib import Path

directory = Path("weights/state-se")
names = [
    "model.safetensors", "protein_embeddings.pt", "config.yaml",
    "LICENSE.md", "MODEL_LICENSE.md", "MODEL_ACCEPTABLE_USE_POLICY.md",
    "HOM_MouseHumanSequence.rpt",
]
hashes = {}
for name in names:
    with (directory / name).open("rb") as stream:
        hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
report = {
    "model_revision": "5a9a80f44f7ce32ce57059933ef0d735d7c10ce5",
    "source_revision": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
    "sha256": hashes,
    "homology_source": "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
    "purpose": "STATE-SE frozen gene features; not native STATE-ST",
    "training_checkpoints_downloaded": False,
}
(directory / "assets.json").write_text(json.dumps(report, indent=2) + "\n")
PYTHON
```

</details>

The [checkpoint checksum list](../configs/models/checkpoints.sha256) identifies the current weights, vocabularies, and auxiliary files. From the repository root:

```bash
sha256sum --check configs/models/checkpoints.sha256
```
