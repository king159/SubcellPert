# Reproducing the benchmark

Run from the code repository root with the Python 3.12 environment from the
[installation instructions](../README.md) activated.
On a managed cluster, perform validation and experiments inside a compute
allocation.

[The validation environment](../release/environment-reference.json) records
the installed versions of the direct runtime dependencies, Python, and the
PyTorch CUDA build. All recorded versions satisfy `pyproject.toml`. It is a
reference environment, not a cross-platform lockfile or proof of a fresh
dependency installation.

## Data and model resources

Use a complete dataset snapshot in a directory separate from the code. The
[dataset repository](https://huggingface.co/datasets/king159/SubcellPert)
remains private; authenticate with `hf auth login` or `HF_TOKEN` before download.
The [verified snapshot](https://huggingface.co/datasets/king159/SubcellPert/tree/b6db87c2023de68344e2c4a4a8d45a03eb86fdb7) is
`b6db87c2023de68344e2c4a4a8d45a03eb86fdb7`, with release ID
`subcellpert_14datasets_population_min50_20260924`. It contains **13,880,723
cells and 3,653,986,724 RNA events across 14 datasets**.

The release contains 112 frozen views for 38 dataset/seed instances and three
descriptor pairs. The snapshot contains 3,104 payload files, including the
data card and teaser image. Its inventory and content hashes have been verified
against remote LFS SHA-256 / Git object IDs and file sizes.
This public code release targets that dataset revision. The manifest preserves
historical code provenance and a modified-worktree flag, so its Git commit alone
does not identify the complete source used for export.

Download into a new, empty directory. `snapshot_download` does not delete files
left over from an older snapshot. On a cluster whose compute nodes lack network
access, download on a networked login node into shared storage, then perform
validation and experiments in a compute allocation.

```bash
export SUBCELLPERT_CODE="$PWD"
export SUBCELLPERT_ROOT="$PWD/data/subcellpert_population_min50_20260924"
export SUBCELLPERT_DATA_REVISION="b6db87c2023de68344e2c4a4a8d45a03eb86fdb7"

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

The download excludes `.gitattributes`, which is Hub repository metadata outside
the benchmark inventory. The validator also excludes the client's local
`.cache/huggingface/` download metadata; unexpected payload files still fail.
Follow the [dataset validation instructions](../src/subcellpert/data/RELEASE.md#validate-a-downloaded-snapshot)
to check the inventory, relative references, and optional content hashes.

Keep the entire bundle together. See the
[data release specification](../src/subcellpert/data/RELEASE.md) for its layout.

For FM experiments, follow [pretrained models](pretrained_models.md) from the
code repository root. Models and their upstream implementations must also be
visible relative to the benchmark root. For a new downloaded root:

```bash
mkdir -p "$SUBCELLPERT_ROOT/artifacts"
ln -s "$SUBCELLPERT_CODE/external" "$SUBCELLPERT_ROOT/external"
ln -s "$SUBCELLPERT_CODE/artifacts/upstream" "$SUBCELLPERT_ROOT/artifacts/upstream"
```

Create these links only if the destination paths do not exist. They are local
runtime conveniences, not part of the portable dataset release. A classical
run needs neither pretrained weights nor these links.

Use `--dry-run` with the selected FM configuration to check its resource paths.
A smoke run exercises encoding and evaluation with the prepared environment
and checkpoint.

## One fit

The following commands use the prepared scGPT resources. For a classical
baseline that needs no pretrained weights, use
`configs/oracle/pf-astro/count_matched_resampling.yaml`; it also supports
`--device cpu`.

```bash
python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/joint/pf-astro/scgpt.yaml \
  --seed 1 --fit-group seed1 --device cuda --smoke \
  --output "$SUBCELLPERT_CODE/outputs/smoke/joint/pf-astro/scgpt/seed1"

python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/joint/pf-astro/scgpt.yaml \
  --seed 1 --fit-group seed1 --device cuda \
  --output "$SUBCELLPERT_CODE/outputs/reproduction/joint/pf-astro/scgpt/seed1"
```

`--config` is resolved relative to the current working directory.
`--project-root` points to the data and benchmark inputs. Relative `--output`
paths are resolved under that data root, so these examples use absolute output
paths under `SUBCELLPERT_CODE` to keep generated results outside the snapshot.
Always pass `--config` when code and data have separate roots.

Use `configs/oracle/...` for Oracle; do not turn a Joint YAML into Oracle by
changing only the setting flag. Choose registered split seeds from
[configs/README.md](../configs/README.md). Lung-Drug jointly fits its two drug
responses; its fit IDs are `seed1`, `seed2`, and `seed3`.
Most datasets have split seeds 1–3; ATM-RT and DDX54-PD1 have seed 1 only and no
validation split. Foundation-model fits select checkpoints by validation where
available, and use the final epoch for those two datasets. The seed controls the
data split, not an additional independent model-seed sweep.

All five FMs use the configured global training batch of 256, five complete
epochs, and BF16. STATE-SE on DDX54-PD1 and Lung-Drug uses encoder microbatches
of 64. Encoder chunking does not change the optimizer batch. There is no
automatic batch profiling, OOM batch fallback, or gradient accumulation in
the formal configuration. `batch_profile.json` records the configured batch;
its presence does not mean that candidate batches were benchmarked.

## Multiple GPUs

The five trainable foundation-model baselines support distributed training.
For example, launch the same scGPT configuration on eight allocated GPUs:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  --module subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/joint/pf-astro/scgpt.yaml \
  --seed 1 --fit-group seed1 --device cuda \
  --output "$SUBCELLPERT_CODE/outputs/reproduction-ddp8/joint/pf-astro/scgpt/seed1"
```

Keep the same batch configuration: global 256 becomes 32 cells
per GPU. Use only allocated GPUs and a process count that divides the global
batch. Do not manually change the global batch to the local per-GPU value.
The effective encoder batch is the smaller of its configured cap and the
per-GPU training batch.
Multi-GPU and single-GPU runs are not promised to be bitwise identical.

## Resume and output safety

The low-level CLI clears generated files in the selected output directory
unless `--resume` is supplied. Use a fresh directory for a new experiment.
For a stopped run, repeat its original command with `--resume`; this requires
a compatible `cell_resume.pt` or `cell_model.pt`. Complete-epoch resume restores
optimizer, scheduler, and sampling state. Incomplete epochs are replayed.
Keep code, data, pretrained weights, world size, and global batch unchanged.
Never run two workers against the same output directory.

Checkpoints record the runtime source hashes. This baseline release removes
private experiment and data-export code, so checkpoints created with the
private development tree are not automatically resumable here. Keep the
original checkout for those runs, or start a new fit with this release; the
source-identity checks remain enabled.

Do not silently reinterpret historical fits as runs under new settings.
Execution changes and non-default training budgets must be reported with the
results; changing a YAML does not migrate an existing checkpoint.

`result.json` contains per-response metrics and both gene panels. Check its
completion status and `scientific_result`, not just the presence of a log or
`status.json`. Smoke results are never formal results. Predictions and their
integrity metadata are under `responses/`; configuration and checkpoint
selection records stay with the fit.
