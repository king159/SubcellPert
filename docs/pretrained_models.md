# Pretrained models

The five FM baselines use separately downloaded pretrained assets in `artifacts/upstream/`.
Run these commands from the code repository root with your environment active.
Use fresh destinations; keep existing checkpoints used by other experiments.
These pointers and revisions are the recorded benchmark resources, not a claim
that all remote downloads were revalidated during code cleanup.

<details>
<summary><strong>scGPT</strong></summary>

Open the official
[whole-human folder](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)
and download `best_model.pt`, `args.json`, and `vocab.json` into
`artifacts/upstream/scgpt-human/`. Keep the three files from the same checkpoint
folder. Alternatively, with `gdown` installed in your active environment:

```bash
python -m gdown --folder \
  'https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y' \
  -O artifacts/upstream/scgpt-human
```

The files must be directly inside `scgpt-human/`, without an extra nested
download-directory level.

For a fresh setup, obtain the upstream implementation at the revision used here:

```bash
git clone https://github.com/bowang-lab/scGPT.git external/scGPT
git -C external/scGPT checkout cebd6fae655b9c585a4807daa3ac31bb764f06b4
```

</details>

<details>
<summary><strong>scGPT-spatial</strong></summary>

The [official repository](https://github.com/bowang-lab/scGPT-spatial) links the
V1 archive. Download and extract it from the project root:

```bash
mkdir -p artifacts/upstream/scgpt-spatial
model_download_dir="$(mktemp -d)"
curl --fail --location --retry 3 \
  'https://ndownloader.figshare.com/files/52163879' \
  --output "$model_download_dir/scGPT_spatial_v1.zip"
unzip "$model_download_dir/scGPT_spatial_v1.zip" -d artifacts/upstream/scgpt-spatial
```

The final directory must be
`artifacts/upstream/scgpt-spatial/scGPT_spatial_v1/`, containing `best_model.pt`,
`args.json`, `vocab.json`, `batch_mapping_dict.json`, and `all_dict_mean_std.csv`.
Preserve this archive subdirectory; the YAML points to it. The archive's Figshare MD5 is
`470d40fb77535afc2b4e7cffc0d665ef`.

For a fresh setup, obtain the upstream implementation at the revision used here:

```bash
git clone https://github.com/bowang-lab/scGPT-spatial.git external/scGPT-spatial
git -C external/scGPT-spatial checkout f9442d777ba47cd3dcac8253833a233ff5bf7262
```

</details>

<details>
<summary><strong>scFoundation</strong></summary>

Follow the download pointer in the official repository's
[`model/models/download.txt`](https://github.com/biomap-research/scFoundation/blob/main/model/models/download.txt).
At the time these instructions were checked, it points to this
[BioMap SharePoint folder](https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk).
Download `models.ckpt` and put it at
`artifacts/upstream/scfoundation/models.ckpt`.

Our current file is 1,432,587,886 bytes, with SHA-256
`9f40bf324d3d0084c4b288d06f5af4fddd12206e2a3f022551d12e89e33a0ea9`.
The adapter loads its `gene` entry. It also needs the source repository's
`model/` implementation and `OS_scRNA_gene_index.19264.tsv`.

For a fresh setup, obtain the upstream implementation at the revision used here:

```bash
git clone https://github.com/biomap-research/scFoundation.git external/scFoundation
git -C external/scFoundation checkout 397631c495eddf9ad6644fc00c6ea8139e651245
```

</details>

<details>
<summary><strong>SpatialFormer</strong></summary>

The benchmark uses the **single-input 5k / 71-slide** checkpoint linked by the
[official repository](https://github.com/TerminatorJ/Spatialformer#get-the-embeddings).
Download the exact file below; the paired checkpoint has a similar filename
but different weights.

```bash
mkdir -p artifacts/upstream/spatialformer
curl --fail --location --retry 3 \
  'https://ndownloader.figshare.com/files/61331527' \
  --output 'artifacts/upstream/spatialformer/stepstep=0176000-traintrain_total_loss=-2.7789-valval_total_loss=0.0000.ckpt'
```

Its expected size is 720,940,445 bytes; Figshare MD5 is
`fb2c33044244bf21a962e106951769f5`. The adapter also reads these files from
`external/Spatialformer/spatialformer/`:

- `config/_config_train_large_pair.json`
- `tokenizer/tokenv5.json`
- `data/gene_median.pkl`
- `spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl`

The configuration filename is the one used by the current adapter; the
checkpoint selection above identifies the actual pretrained weights.

For a fresh setup, obtain the upstream implementation at the revision used here:

```bash
git clone https://github.com/TerminatorJ/Spatialformer.git external/Spatialformer
git -C external/Spatialformer checkout f02e7776e8a99c9abdc5fe5572a1a20369faf159
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
  --local-dir artifacts/upstream/state-se

curl --fail --location \
  https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt \
  --output artifacts/upstream/state-se/HOM_MouseHumanSequence.rpt
```

The second command retrieves the mouse/human mapping report from
[MGI](https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt).
The report changes over time: compare it with the recorded checksum to identify
the mapping version used in our runs. The large `se600m_epoch*.ckpt` training
snapshots are not needed for this encoder.

For a fresh setup, obtain the upstream implementation at the revision used here:

```bash
git clone https://github.com/arcinstitute/state.git external/state
git -C external/state checkout 9bbfe78a434a55205e4de834e1ea99f85f7a3add
```

</details>

The [checkpoint checksum list](../configs/models/checkpoints.sha256) identifies the current weights, vocabularies, and auxiliary files. From the repository root:

```bash
sha256sum --check configs/models/checkpoints.sha256
```
