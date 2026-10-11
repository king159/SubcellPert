<h1 align="center">🧬 SubcellPert</h1>
<p align="center"><strong>Benchmarking Subcellular Perturbation Prediction</strong></p>
<p align="center">Train and evaluate existing baselines on changes in RNA abundance and subcellular localization.</p>

<p align="center">
  <a href="https://huggingface.co/datasets/king159/SubcellPert">🤗 Dataset repository</a> ·
  <a href="docs/reproduction.md">Reproduction guide</a> ·
  <a href="docs/pretrained_models.md">Pretrained models</a> ·
  <a href="configs/README.md">Configurations</a>
</p>

## 🧬 About SubcellPert

SubcellPert studies how perturbations change both **how much RNA a cell contains**
and **where that RNA is located**. This repository provides data loading, baseline
adapters, training, validation, and evaluation for **14 datasets and eight baselines**.

The benchmark has two independently fitted settings:

| Setting | Prediction task | Main evaluation |
| --- | --- | --- |
| **Joint** | Predict gene counts and RNA coordinates from source cells and a condition change | Expression MSE and PCC-delta |
| **Oracle** | Predict RNA coordinates given target per-cell gene counts | Spatial W1 |

There are **224 run configurations**: 14 datasets × eight baselines × two settings.
The frozen schedule contains 38 dataset/split instances and 608 planned fits.
Most datasets have split seeds 1–3; ATM-RT and DDX54-PD1 have seed 1 only.
These are data-split seeds, not an additional model-seed sweep.

## 🗂️ Installation

Use **Linux and Python 3.12**. Dependencies, including the pinned PyTorch and
torchvision versions, are specified in [pyproject.toml](pyproject.toml).
GPU runs require a compatible NVIDIA driver. On a shared cluster, use an
allocated compute node for environment checks and experiments.

Run these commands from the code repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m subcellpert.main --help
```

`subcellpert-benchmark` is also available as an installed entrypoint.
Data and pretrained weights are downloaded separately.

## 🤗 Dataset

The dataset repository is [king159/SubcellPert](https://huggingface.co/datasets/king159/SubcellPert).

> **Dataset status (2026-10-11):** The [population-min50 snapshot](https://huggingface.co/datasets/king159/SubcellPert/tree/74196fa91563138c2f1e72acc9f2bec6c0c637a6)
> is uploaded and verified, with **14 datasets, 13,880,723 cells, and
> 3,653,986,724 RNA events**. The repository remains private; authorized Hugging
> Face login is required. This code release targets that dataset revision.

The release tag is `population-min50-20260924`. The small Hugging Face catalog
is an inventory, not the training dataset. Download the complete RNA observation,
frozen split, task metadata, and intervention descriptor bundle. Pin its **full
commit SHA** when reporting results. The manifest preserves historical code
provenance, including a dirty-worktree flag; that commit alone does not identify
an installable public code snapshot.

Authenticate with `hf auth login` or `HF_TOKEN`. Use a new, empty destination:
the Hub downloader does not remove leftover files from older snapshots. On
clusters without compute-node internet access, download on a networked login
node to shared storage, then validate and run experiments in a compute allocation.

```bash
export SUBCELLPERT_CODE="$PWD"
export SUBCELLPERT_ROOT="$PWD/data/subcellpert_population_min50_20260924"
export SUBCELLPERT_DATA_REVISION="74196fa91563138c2f1e72acc9f2bec6c0c637a6"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="king159/SubcellPert",
    repo_type="dataset",
    revision=os.environ["SUBCELLPERT_DATA_REVISION"],
    local_dir=os.environ["SUBCELLPERT_ROOT"],
    ignore_patterns=[".gitattributes"],
    token=True,
)
PY
```

`.gitattributes` is Hub repository metadata and is excluded from the download.
The two upstream model weights that were outside the dataset inventory have
been removed from this revision. Follow the
[dataset validation instructions](src/subcellpert/data/RELEASE.md#validate-a-downloaded-snapshot)
to check the inventory and content hashes. Keep the bundle together so its
relative paths remain valid.

Training inputs changed for Adipose, Cardiac-MI, GBM-oHSV, Gut-Microbiota,
Sickness, Spinal-KO, and UC-VDZ. Historical model results and checkpoints were
not recomputed for this release; use a compatible code checkout and new fits
when reporting results for these inputs.

See the [reproduction guide](docs/reproduction.md) for data-root setup and the
[dataset format](src/subcellpert/data/RELEASE.md) for the bundle layout.

## 🧠 Baselines

| Baseline | Configuration name | Pretrained weights |
| --- | --- | --- |
| Count-matched resampling | `count_matched_resampling` | Not required |
| Gaussian affine transport | `gaussian_affine_transport` | Not required |
| Radial quantile transport | `radial_quantile_transport` | Not required |
| scGPT | `scgpt` | Required |
| scGPT-spatial | `scgpt_spatial` | Required |
| scFoundation | `scfoundation` | Required |
| SpatialFormer | `spatialformer` | Required |
| STATE-SE | `state_se` | Required |

The five foundation-model baselines use frozen encoders with trainable
prediction heads. Follow the [pretrained-model instructions](docs/pretrained_models.md)
for upstream source revisions, weight downloads, and checksums. When code and
data live in separate directories, follow the
[resource-path setup](docs/reproduction.md#data-and-model-resources) before
running these baselines.

## ☄️ Training and evaluation

Run from the code repository root with `SUBCELLPERT_ROOT` pointing to the
prepared data snapshot. Each command fits a baseline, performs validation
where the frozen split provides it, and evaluates held-out responses.

### Start with a smoke run

This small Oracle example uses a classical baseline and needs no pretrained weights:

```bash
python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/oracle/pf-astro/count_matched_resampling.yaml \
  --seed 1 --fit-group seed1 --device cpu --smoke \
  --output "$SUBCELLPERT_CODE/outputs/smoke/oracle/pf-astro/resampling/seed1"
```

`--smoke` bounds the workload for a diagnostic run; its outputs are not benchmark
results. Use `--dry-run` to inspect the resolved configuration and selected jobs
without training.

### Run a full fit

After the smoke run, remove `--smoke` and choose a fresh output directory:

```bash
python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/oracle/pf-astro/count_matched_resampling.yaml \
  --seed 1 --fit-group seed1 --device cuda \
  --output "$SUBCELLPERT_CODE/outputs/reproduction/oracle/pf-astro/resampling/seed1"
```

Classical baselines support `--device cpu` as well. To run Joint, select the
corresponding YAML under `configs/joint/` and use a separate output directory.
All available dataset, baseline, setting, and split-seed choices are listed in
[configs/README.md](configs/README.md).

For example, after preparing its pretrained resources, train the Joint scGPT adapter:

```bash
python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/joint/pf-astro/scgpt.yaml \
  --seed 1 --fit-group seed1 --device cuda \
  --output "$SUBCELLPERT_CODE/outputs/reproduction/joint/pf-astro/scgpt/seed1"
```

The five foundation-model configurations specify five epochs, BF16, and a global
training batch of 256. Encoder microbatches may be smaller without changing the
optimizer batch. Scientific settings are defined in the YAML files.

> Use a fresh output directory for every new fit. The CLI replaces generated
> files unless `--resume` is supplied. Resume a stopped fit with its original
> command plus `--resume`, keeping the code, data, weights, and training settings
> unchanged.

For multiple GPUs, resume details, and resource checks, see the
[reproduction guide](docs/reproduction.md).

## 📊 Validation and results

Foundation-model fits select checkpoints using the validation split. ATM-RT
and DDX54-PD1 have no validation split and use the final epoch. Test responses
are reserved for evaluation. Joint and Oracle are fitted independently.

Each fit saves its resolved configuration in `run_config.json`, its checkpoint
in `cell_model.pt`, and evaluation results in `result.json`. Per-response
predictions and metrics are stored under `responses/`. A completed benchmark
fit has `status: "complete"` and `scientific_result: true` in `result.json`.

Report **Joint expression MSE and PCC-delta** and **Oracle spatial W1** separately.
The all-gene panel is primary; the top-100 DEG panel is supplementary. Shared-gene
spatial scores and target-gene coverage are diagnostics: a missing predicted
gene does not have zero spatial error. Compare spatial scores only over matching
gene support.

## 📑 Citation and acknowledgements

Public paper links and BibTeX will be added once the citation metadata is confirmed.
Please also cite the original datasets and the baseline models used in your
experiments. We thank their authors for providing the data, implementations,
and pretrained weights that support this benchmark.

## License

Original SubcellPert code is released under the [MIT License](LICENSE).
Third-party source code, pretrained weights, and datasets retain their
respective terms and attribution requirements.
