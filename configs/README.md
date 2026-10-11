# Benchmark run configurations

The run matrix contains **224 YAML files: 14 datasets × 8 baselines × 2 settings**.
Run configs are grouped by setting first, then dataset. Each setting contains
14 dataset directories, each with eight `<baseline>.yaml` files:

```text
configs/
├── oracle/
│   ├── adipose/
│   │   ├── count_matched_resampling.yaml
│   │   ├── scgpt.yaml
│   │   └── ...              # 8 baselines per dataset
│   └── ...                  # 14 datasets, 112 run configs
├── joint/
│   ├── adipose/
│   │   ├── count_matched_resampling.yaml
│   │   ├── scgpt.yaml
│   │   └── ...              # 8 baselines per dataset
│   └── ...                  # 14 datasets, 112 run configs
├── experiments/
│   ├── cell_joint.yaml      # shared joint protocol
│   └── cell_oracle.yaml     # shared oracle protocol
├── models/                 # 5 shared FM weight/encoder configurations
└── datasets/               # 14 shared dataset/source/token configurations
```

The eight baselines are `count_matched_resampling`, `gaussian_affine_transport`,
`radial_quantile_transport`, `scgpt`, `scgpt_spatial`, `scfoundation`,
`spatialformer`, and `state_se`. Oracle is named `oracle` in directories and config
selectors; the runtime resolves it to the protocol mode `oracle_count`.

Dataset folders use the runtime dataset codes. Their identifiers match the
dataset inventory in the data snapshot's `manifest.json`:

| Directory | Manifest dataset ID | Available split seeds |
| --- | --- | --- |
| `adipose` | `gse295862_adipose_weightloss_xenium` | 1, 2, 3 |
| `atm-rt` | `gse246584_atm_radiation_xenium` | 1 |
| `breast-fus` | `gse296517_breast_fus_cosmx` | 1, 2, 3 |
| `cardiac-mi` | `gse280376_cardiac_repair_xenium` | 1, 2, 3 |
| `ddx54-pd1` | `gse285341_ddx54_pd1_xenium` | 1 |
| `dss-colitis` | `dss_colitis_merfish` | 1, 2, 3 |
| `gbm-ohsv` | `gse296577_gbm_ohsv_xenium` | 1, 2, 3 |
| `gut-microbiota` | `gut_microbiota_merfish` | 1, 2, 3 |
| `kidney-iri` | `gse269719_kidney_iri_xenium` | 1, 2, 3 |
| `lung-drug` | `gse325253_lung_drug_xenium` | 1, 2, 3 |
| `pf-astro` | `perturbfish_astrocytes` | 1, 2, 3 |
| `sickness` | `merfish_sickness` | 1, 2, 3 |
| `spinal-ko` | `gse291460_spinal_ko_xenium` | 1, 2, 3 |
| `uc-vdz` | `uc_vedolizumab_cosmx` | 1, 2, 3 |

## Run a configuration

From the code repository root, activate the Python environment described in
the [installation instructions](../README.md). Set the data root
to a complete snapshot prepared using the
[reproduction guide](../docs/reproduction.md#data-and-model-resources):

```bash
export SUBCELLPERT_CODE="$PWD"
export SUBCELLPERT_ROOT="$PWD/data/subcellpert"

python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/oracle/pf-astro/count_matched_resampling.yaml \
  --seed 1 --fit-group seed1 --device cpu --dry-run \
  --output "$SUBCELLPERT_CODE/outputs/oracle/pf-astro/resampling/seed1"

python -m subcellpert.main \
  --project-root "$SUBCELLPERT_ROOT" \
  --config configs/joint/lung-drug/scgpt.yaml \
  --seed 2 --fit-group seed2 --device cuda --dry-run \
  --output "$SUBCELLPERT_CODE/outputs/joint/lung-drug/scgpt/seed2"
```

Adjust `SUBCELLPERT_ROOT` if the data live elsewhere. The scGPT example also
requires the [pretrained resources](../docs/pretrained_models.md) and the
resource-path setup in the reproduction guide.

Remove `--dry-run` to train and evaluate. For a bounded diagnostic, replace it
with `--smoke` and choose a separate output directory.
Every config defaults to seed 1; use `--seed 2` or `--seed 3` where the frozen
schedule permits. Seeds are selected at runtime, so there is one file per
dataset/baseline/setting combination. `lung-drug` jointly fits its two drug responses for the selected seed.
`seed` means the data-split seed; model randomness is not an independent experiment dimension.
`--config` is resolved relative to the current working directory. Relative
`--output` paths and the default `outputs/` directory are resolved relative to
`--project-root`, so the examples use absolute output paths under the code
directory. Without `--fit-group`, the CLI appends each selected fit ID to the
specified output path. Use fresh output directories; the CLI replaces generated
files unless `--resume` is supplied.

`main.py` uses OmegaConf to load each YAML and its `extends` entry, which is
relative to that YAML. Run configs inherit `../../experiments/cell_<setting>.yaml`.
Preserve these shared settings when reproducing the benchmark. The 140 FM
run configs also inherit their shared model and dataset YAMLs, in this order:

```yaml
extends:
  - ../../experiments/cell_joint.yaml
  - ../../models/scgpt.yaml
  - ../../datasets/spinal-ko.yaml
```

`models/<baseline>.yaml` declares `foundation.backbone`, the pretrained
`model_directory` or `model_checkpoint`, the per-gene `embedding_dimension`, and
`encoder_max_length`. scFoundation also declares its gene context; STATE-SE
declares its feature policy. `datasets/<dataset>.yaml` declares `data.dataset_id`,
`species`, `source_conditions`, and SpatialFormer tissue/assay/condition tokens.
These are shared inputs to both settings. An availability flag adds one channel
to the declared gene width: 513 channels for the four 512-wide encoders, and
2049 for STATE-SE. scFoundation's pooled cell-vector width is not used here.

SpatialFormer tissue/assay approximations are recorded as `token_mapping_note`
in the relevant data YAMLs. The existing Adipose→Skin and Perturb-FISH→Merfish
choices are retained; Spinal→Brain and Gut→Colon are explicit vocabulary
approximations. The notes are not encoder inputs.

These settings describe SpatialFormer's metadata prefix, separate from gene
vocabulary alignment. For example, Lung-Drug encodes its source cells with
`<CLS>, Disease, Lung, Mouse, Xenium, gene tokens..., <SEP>`: `Bleomycin` maps
to the existing `Disease` token. This source-state prefix does not replace the
perturbation descriptor used by the prediction head.

All five foundation models align assay genes with their pretrained inputs.
scGPT and scGPT-spatial use their own `vocab.json`; scFoundation uses its fixed
19,264-gene index; SpatialFormer uses `tokenv5.json`. STATE-SE uses gene-associated
protein embeddings, with one-to-one mouse-to-human homology mapping for mouse
datasets. The current adapters for the other four foundation models do not use
SpatialFormer's condition/tissue/species/assay prefix, so they do not need this
dataset-specific prefix configuration. Classical baselines use the dataset's
gene axis directly.

Training loads this resolved configuration and the upstream weights directly.
Official checkpoint download links and exact local destinations are listed in
[pretrained models](../docs/pretrained_models.md).
Use an explicit `--config` path when code and data have separate roots.
Without it, the CLI looks for shared configurations under `--project-root`.
`--dry-run` checks configured assets and source/token identities before outputs
can be overwritten, without loading model weights or RNA. New runs save the
resolved configuration in `run_config.json` and the online source report in
`foundation_execution.json`.

CLI flags override YAML and dotlist values. For example,
`run.device=cpu` is a dotlist override. Frozen protocol and split checks run
before execution.

Foundation encoder batches follow `training.batch_size` through
`foundation.encoder_batch_size: ${training.batch_size}` in both shared protocols.
The formal run matrix keeps the training batch at 256 for all five FMs.
STATE-SE on `ddx54-pd1` and `lung-drug` explicitly uses
`foundation.encoder_batch_size: 64` in both joint/oracle configs. This only
chunks frozen encoder calls; the training loader, optimizer batch, loss grouping,
and accumulation setting remain unchanged. Other encoder caps stay at 256.
The effective encoder batch is the smaller of its configured cap and the
per-rank training batch. There is no startup profiling or automatic OOM batch reduction; all five
use `training.matmul_precision: high`. The configured sizes stay fixed throughout training.
Foundation runs default to CUDA; `run.device=cpu` remains available for development.
Smoke checks cap both batches at 2. See the
[reproduction guide](../docs/reproduction.md#multiple-gpus) for global and
per-rank batch semantics.

All eight implemented baselines, including `state_se`, are in the formal schedule.
The 38 frozen data-split instances produce 608 planned fits across both settings.
A configuration defines a planned fit. Check `result.json` for
`status: "complete"` and `scientific_result: true` to identify a completed
benchmark fit; smoke outputs are diagnostics.
