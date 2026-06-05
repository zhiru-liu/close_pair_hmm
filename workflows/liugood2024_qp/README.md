# Liu & Good 2024 QP reproduction

Reproduce the quasi-phaseable (QP) recombination results across the published species and
verify the CP-HMM calls against the paper's supplementary transfer table. This is the
multi-species generalization of [`../bacteroides_fragilis/`](../bacteroides_fragilis/).

Only the workflow is committed (scripts + the 29 cached per-species priors). The QP SNV
catalog (~6.8 GB) and the published table (~44 MB) are **not** committed — you point the
driver at local copies.

## Inputs (local, not committed)

| Input | How to provide |
| --- | --- |
| QP SNV catalog: `<dir>/<species>/{snv_catalog,coverage,biallelic_snvs}.feather` | `--data-dir` or `$CPHMM_QP_DATA_DIR` |
| Reference genomes: `<dir>/<species>/{genome.fna.gz,genome.features.gz}` | `--reference-dir` or `$CPHMM_QP_REFERENCE_DIR` |
| Published table `gut_microbiome_transfers.csv` (pbio.3002472.s003) | `--ground-truth` or `$CPHMM_QP_GROUND_TRUTH` |

The QP catalog + table are distributed with the paper / the
[LiuGood-2024-SNVs](https://github.com/zhiru-liu/LiuGood-2024-SNVs) Zenodo deposit. Species
names are uniform across the catalog, reference, prior, and table (`<genus>_<species>_<midasID>`).

## Run

```bash
# small representative subset (default), verifying against the published table:
python workflows/liugood2024_qp/reproduce.py \
    --data-dir <catalog> --reference-dir <refs> --ground-truth <table.csv>

# all 29 species (heavy), 8 species in parallel, with iterative transfer length:
python workflows/liugood2024_qp/reproduce.py --all --jobs 8 --transfer-length iterative \
    --data-dir … --reference-dir … --ground-truth …
```

Per species the driver loads the catalog, takes the published close pairs, runs inference
with the cached prior, writes `results/<species>__{inference,transfer}_summary.csv` and
comparison CSVs, and appends a row to `results/verification_summary.csv` (predicted vs
published transfer/pair counts, clonal-divergence correlation, transfer-interval overlap).
`results/` is gitignored.

### Options

- `--species A B C` — explicit species list.
- `--pair-source select` — re-derive close pairs from the catalog by clonal fraction
  (> 0.5 over 1000 bp 4D-core blocks) instead of using the published pairs. Approximate:
  the published pipeline first kept one sample per host, and that host map is not in the
  public data (so same-host pairs can leak in).
- `--regenerate-prior` — rebuild the prior from the catalog instead of using the cached one
  (may drift slightly from the published prior).
- `--iterative {auto,on,off}` — clonal-emission refinement; `auto` enables it for the
  two-clade species (`Bacteroides_vulgatus_57955`, `Alistipes_shahii_62199`), which also use
  clade separation (detected from the 80-bin prior shape).
- `--transfer-length` — expected transfer length (bp) for the HMM (default 1000), or
  `iterative` to re-estimate it per species the way the paper did (see below).
- `--jobs N` / `-j N` — run N species in parallel (process pool). Each worker loads its
  own species catalog, so peak memory scales with N; size it to your RAM.
- `--max-pairs N` — cap pairs per species (quick smoke tests).

## Note on the transfer-length parameter

The published QP analysis did **not** use a fixed transfer length. Its final pipeline
(`close_pair_stage2-3_iterative.py` in `microbiome_evolution`) determined the transfer
length **per species, iteratively, starting from 1000 bp**: decode, recompute the mean
length of the detected transfers, re-decode, until convergence (per-clade for the two-clade
species). So `--transfer-length 1000` here matches the paper's *initialization* and gives
close agreement (~96–98% transfer-interval overlap); it is not a fixed published value.

Pass `--transfer-length iterative` to reproduce that procedure: the driver decodes all
pairs, recomputes the mean detected transfer length, and re-decodes until the fractional
change is < 0.1 (per-clade for the two-clade species), up to 3 passes, starting from 1000.
This is implemented in the workflow (`infer_iterative_transfer_length` in `reproduce.py`),
not in cphmm core. One caveat: the paper averaged over *merged + filtered* transfers (drop
< 5 blocks); that filter is **not** ported yet, so the estimate runs slightly short of the
published one. See [../../docs/transfer_length_iteration.md](../../docs/transfer_length_iteration.md).

Two distinct refinements are easy to conflate: cphmm's `--iterative` flag refines the
**clonal emission** rate, which is *not* the paper's **transfer-length** iteration. They are
orthogonal and compose (the two-clade species use both).

## Priors

`priors/<species>_<midasID>.csv` are the cached QP priors used in the paper (copied from the
original analysis). 27 are single-clade (40 bins); `Bacteroides_vulgatus_57955` and
`Alistipes_shahii_62199` are two-clade (80 bins). Use `--regenerate-prior` to rebuild them.
