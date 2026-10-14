# Dataset layout and validation

The baseline runtime reads a complete, versioned data bundle containing processed
RNA observations, frozen train/validation/test memberships, and condition
metadata. Obtain the bundle using the [data preparation instructions](../../../README.md#-data-preparation).
The code repository provides the loaders, configurations, training, and metrics;
RNA observations and pretrained model weights are separate downloads.

Pin the full Hugging Face revision and use its matching code version. A code
commit accompanied by `working_tree_modified: true` does not uniquely identify
the source used to produce a bundle; it also needs the corresponding source
snapshot.

## Bundle layout

```text
<data-root>/
├── README.md
├── manifest.json
├── checksums.sha256
├── assets/teaser.png       # Dataset card illustration
├── catalog/datasets.parquet
├── datasets/<dataset_id>/
│   └── population/...
└── benchmark/
    ├── registry/...
    ├── protocols/...
    ├── tasks/...
    ├── splits/...
    ├── schedules/cell_current.json
    ├── split_views/<dataset_id>/seed<N>/<policy>/<train|validation|test>/...
    └── intervention_descriptors/<task>/<dataset_id>.{npz,json}
```

Keep the whole bundle together. The small Parquet catalog is a dataset inventory
for browsing; it is not the training dataset. `manifest.json` identifies the
included files, datasets, observation totals, and frozen benchmark inputs.
Use the observation totals recorded in the chosen snapshot.

## Frozen split views

The current protocol uses 14 datasets, 38 dataset/seed instances, and 112
train/validation/test view manifests. ATM-RT and DDX54-PD1 have seed 1 and no
validation view. Joint, Oracle, and the eight baselines share these memberships.
The schedule defines 608 fits; the presence of their inputs does not mean those
fits have completed.

`benchmark/splits/cell_seed_schedule_dual_min50.json` and
`benchmark/splits/cell_validation_dual_min50.json` define the current fit families
and validation policy. Each frozen view supplies its own cell and group tables.
Do not replace those tables with the base population tables, redraw the splits,
or apply a different filtering policy when reproducing the benchmark.

View manifests refer to shared observation arrays under
`datasets/<dataset_id>/population/` using relative paths. These paths may contain
`..` to reach the arrays, but must stay within the same dataset's population
inside the downloaded bundle. The portable layout does not require links to
the authors' filesystem. Moving only one view or one dataset out of the bundle
can invalidate the references.

## Intervention descriptors

Three intervention datasets require frozen descriptor arrays and their JSON
metadata in addition to observations:

| Dataset | Representation | Vector width |
| --- | --- | ---: |
| Perturb-FISH astrocytes | GEARS GO/source-coexpression component descriptors | 612 |
| ATM-RT | Component indicators for ATM loss and radiation | 2 |
| DDX54-PD1 | Component indicators for Ddx54 knockdown and anti-PD1 | 2 |

The NPZ files store `components` and `vectors`; preserve their values and order.
The JSON metadata describe identities, dimensions, normalization, and component
composition. Task metadata point to the matching NPZ and JSON files. Other
current context tasks construct known-condition one-hot vectors from the frozen
task and split metadata.

These descriptors are separate from the cellular foundation-model encoders.
The five encoders are implemented in `src/subcellpert/models/`; resource paths
and checksums are listed in [configs/models/](../../../configs/models/).

## Validate a downloaded snapshot

From the installed code repository, with `SUBCELLPERT_ROOT` set to the data root:

```bash
python - <<'PY'
import os
from subcellpert.data.release import validate_public_release

print(validate_public_release(os.environ["SUBCELLPERT_ROOT"], verify_hashes=False))
PY
```

The default check validates the file inventory, relative references, dataset
layout, frozen splits, and descriptor presence. It does not reread every payload
to verify its content hash. Set `verify_hashes=True` in the call above to check
every declared file against the SHA-256 inventory.

The full hash check reads all declared payloads. The checksum inventory excludes
its own checksum file. Hugging Face's local `.cache/huggingface/` download metadata
is excluded from the file-inventory comparison; unrelated payload files are not.
Exclude Hub repository metadata with `ignore_patterns=[".gitattributes"]`
when downloading, as shown in the reproduction guide. Use a new, empty
destination so files from older snapshots cannot remain in the inventory.
Store training outputs outside the downloaded bundle so later validation sees
the original snapshot.

## Read the deposited memberships

```python
import os
from pathlib import Path
from subcellpert.data import load_fit_records
from subcellpert.data.splits import load_seed_specs

root = Path(os.environ["SUBCELLPERT_ROOT"]).resolve()
spec = next(s for s in load_seed_specs(root, "lung-drug") if s.fit_id == "seed1")
train_records = load_fit_records(spec, split="train")
validation_records = load_fit_records(spec, split="validation")
test_records = load_fit_records(spec, split="test")
```

The loader reads existing memberships. To train and evaluate a baseline, use a
configuration from [configs](../../../configs/README.md) with the same data root.
