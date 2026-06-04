# Bundled reference priors

This directory holds the **bundled reference priors** for the CP-HMM transfer-
divergence distribution. There is one `<species_name>.csv` per species. Each file
has two rows produced by `cphmm.prior.save_prior`:

- row 0 — the bin-center divergences (`divs`)
- row 1 — the empirical counts in each bin (`counts`)

By default `cphmm.config.HMM_PRIOR_PATH` points here, so `ClosePairHMM(species_name=...)`
loads the matching file. At load time the per-block "probability of a SNP" is derived
from the divergence with `1 - (1 - div) ** block_size` (see
`ClosePairHMM._get_empirical_emissions`).

## How these were derived

These bundled priors were computed from the **quasi-phaseable (QP) SNV catalogues of
Liu & Good 2024** (PLoS Biol 22(2):e3002472, https://doi.org/10.1371/journal.pbio.3002472).
For each species, the pipeline:

1. samples blocks of length `block_size` (default 1000) from random close pairs of
   genomes and records each block's mean divergence plus the pair's genome-wide
   divergence — `cphmm.prior.sample_blocks` (default 5000 samples);
2. bins the block divergences into `HMM_PRIOR_BINS` (40) bins to form the empirical
   transfer-divergence distribution — `cphmm.prior.compute_div_histogram`
   (optionally split into within-/between-clade at a `clade_cutoff`);
3. writes the result with `cphmm.prior.save_prior`.

See `workflows/bacteroides_fragilis/generate_prior.py` for a worked prior-generation
script (it reproduces the bundled prior for that species).

## Using a run-specific prior instead of the bundled set

The bundled priors are specific to the LiuGood2024 QP catalogue. For a new dataset
you should generate your own prior into a separate work folder rather than reusing
these. The prior directory is fully overridable — nothing is hardcoded to this path:

```python
import cphmm.prior as prior
import cphmm.model as hmm

# 1. generate a prior for your dataset into your own run folder
local_divs, genome_divs = prior.sample_blocks(my_datahelper)
divs, counts = prior.compute_div_histogram(local_divs, genome_divs)
prior.save_prior(divs, counts, species_name, prior_path="/path/to/run_work_folder/priors")

# 2. point the model at that folder
model = hmm.ClosePairHMM(species_name=species_name,
                         prior_path="/path/to/run_work_folder/priors")
```

In the pipeline (`infer_pipelines.infer_pairs`), set the `hmm_prior_path` attribute on
your datahelper to route inference at a run-specific prior directory.
