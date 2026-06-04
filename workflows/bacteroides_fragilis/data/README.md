# Bacteroides fragilis example data

## `Bacteroides_fragilis_54507_ground_truth_transfers.csv`

The "ground truth" transfer events that `infer.py` compares against. This is the subset
of rows for `Bacteroides_fragilis_54507` from the **Liu & Good 2024 supplementary table
`pbio.3002472.s003.csv`** — the published CP-HMM transfer calls accompanying:

> Liu Z, Good BH. Dynamics of bacterial recombination in the human gut microbiome.
> PLoS Biol. 2024;22(2):e3002472. https://doi.org/10.1371/journal.pbio.3002472

The full supplementary table (all species) is distributed with the paper and via the
[LiuGood-2024-SNVs](https://github.com/zhiru-liu/LiuGood-2024-SNVs) resources; only the
B. fragilis rows are committed here. The `Sample 1`/`Sample 2` columns also define the
published close pairs this example runs inference on (see `../datahelper.py`).

## `bf_data.tar.gz`

A compact subset of the published Liu & Good 2024 quasi-phaseable SNV catalog for this
species (`snv_catalog` / `coverage` / `biallelic_snvs` feather tables + the reference
genome for 4D annotation). Extracted into `extracted/` (gitignored) on first run. See
the workflow [README](../README.md) for details.
