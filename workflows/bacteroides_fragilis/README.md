# Bacteroides fragilis — end-to-end CP-HMM example

A focused, reproducible run of the CP-HMM recombination pipeline on one species,
`Bacteroides_fragilis_54507`, using the published Liu & Good 2024 quasi-phaseable (QP)
SNV catalog. It loads the raw SNV catalog, builds the per-pair SNV-difference profiles
over 4-fold-degenerate core sites, runs inference over the published close pairs, and
compares the detected transfers against the published ground truth.

## Quick start

From an editable install with the workflow extras
(`pip install -e ".[workflows]"`):

```bash
# quick smoke test (1 pair), then the full published set:
python workflows/bacteroides_fragilis/infer.py --max-pairs 1
python workflows/bacteroides_fragilis/infer.py
```

The SNV catalog ships in the repo as `data/bf_data.tar.gz` (~15 MB); the first run
extracts it into `data/extracted/` (gitignored) and caches it — no network needed.
Results are written to `results/` (gitignored); compact reference outputs for validation
are committed under `reference_outputs/`.

If you already have the catalog extracted elsewhere, point at it with
`export CPHMM_BF_DATA_DIR=/path/to/bf_data` (the dir with `snvs/` and `reference/`).

## What it does

- `datahelper.py` — `DataHelper_Bf`, a thin subclass of
  `cphmm.datahelper.BaseClosePairDataHelper`. It reads the catalog through the in-package
  reader `cphmm.io.liugood2024_qp.SNVHelper` (real 4D annotation from the reference
  genome) and implements `get_pair_snp_info`; the prior-sampling logic is inherited.
- `infer.py` — the entry point: build the datahelper → `cphmm.infer_pipelines.infer_pairs`
  → write summaries → `compare_to_ground_truth`.
- `compare_to_ground_truth.py` — event-count and interval-overlap comparison vs. the
  committed published-transfer subset (`data/..._ground_truth_transfers.csv`).
- `generate_prior.py` — *(optional)* regenerate `priors/Bacteroides_fragilis_54507.csv`
  from the same catalog. The committed prior was made with the default arguments.

## Data

| What | Where | In git? |
| --- | --- | --- |
| SNV catalog (`snv_catalog`/`coverage`/`biallelic_snvs` feather) + reference genome | `data/bf_data.tar.gz` (~15 MB) | yes |
| Transfer-divergence prior | `priors/Bacteroides_fragilis_54507.csv` | yes (4 KB) |
| Published ground-truth transfers (this species) | `data/..._ground_truth_transfers.csv` | yes (185 KB) |
| Reference inference/comparison outputs | `reference_outputs/` | yes (~7 KB) |

The SNV catalog is a subset of the published Liu & Good 2024 QP catalog — see the main
[README](../../README.md#where-to-get-example-data-and-reproduce-the-paper) for the
full [LiuGood-2024-SNVs](https://github.com/zhiru-liu/LiuGood-2024-SNVs) repo and Zenodo.

### Maintainer: regenerating the committed catalog

`export_tarball.py` rebuilds `data/bf_data.tar.gz` from the full catalog (needs the
`/Volumes` data) and prints its size/sha256; commit the regenerated file:

```bash
python workflows/bacteroides_fragilis/export_tarball.py
git add workflows/bacteroides_fragilis/data/bf_data.tar.gz
```
