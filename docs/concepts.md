# CP-HMM concepts

This document explains how the Close-Pair HMM works and how `cphmm` computes its
outputs. For the full statistical treatment see Liu & Good 2024
([10.1371/journal.pbio.3002472](https://doi.org/10.1371/journal.pbio.3002472)); this is
a practical companion to the code.

## The problem

Two conspecific bacterial samples that share a recent common ancestor inherit most of
their genome **clonally**, accumulating point mutations at a roughly uniform low rate.
Homologous **recombination** overwrites local stretches with sequence from a more
divergent donor, producing localized spikes in their SNV-difference profile. The CP-HMM
takes the SNV-difference profile of one such pair and segments it into clonal vs.
recombined regions, then summarizes the result.

## Observations: blocks, not sites

The genome is coarse-grained into non-overlapping **blocks** of `HMM_BLOCK_SIZE` core
sites (`config.HMM_BLOCK_SIZE`, default 10). Each block becomes one observation:

- `1` — the block contains at least one SNV difference between the pair,
- `0` — the block is identical between the pair.

Working in blocks and using a *presence/absence* indicator (rather than raw SNV counts)
reduces the influence of clustered mutations and locally correlated sites. Blocks are
processed **per contig**, so the HMM never transitions across a contig boundary; contigs
shorter than `HMM_MIN_SEQ_LEN` blocks (default 100) are skipped as too short to inform
inference.

## Hidden states

The hidden state of each block is one of:

- **State 0 — clonal.** Emits a `1` with probability `clonal_emission` (the per-block
  clonal divergence). Low, and shared across the whole genome of the pair.
- **States 1…N — transfer / recombined.** Each corresponds to a donor **transfer
  divergence** taken from an empirical grid. State *k* emits a `1` with probability
  `1 - (1 - div_k) ** block_size`, i.e. the chance a block of that size carries a SNV at
  divergence `div_k` (`ClosePairHMM._get_empirical_emissions`).

### Sparse transition topology

The transition matrix is deliberately sparse (`ClosePairHMM._init_transitions`):

- From **clonal**: stay clonal with probability `1 - transfer_rate`, or enter transfer
  state *k* with probability `transfer_rate * transition_prior[k]`.
- From a **transfer** state: stay with probability `1 - exit_rate`, or return directly to
  clonal with probability `exit_rate = 1 / transfer_length`. There are **no**
  transitions between different transfer states.

This "star" topology (clonal hub + self-looping transfer spokes) makes each
forward/backward/Viterbi step `O(n_components)` instead of `O(n_components²)`, which the
Numba kernels in `_cphmm_kernels.py` exploit directly.

## The transfer-divergence prior

`transition_prior[k]` — how likely a transfer is to come from a donor at divergence
`div_k` — is an **empirical, per-species** input, stored as `cphmm/priors/<species>.csv`
(two rows: bin-center divergences, and counts). It is estimated by sampling blocks across
many pairs of genomes and histogramming their local divergences
(`prior.sample_blocks` → `prior.compute_div_histogram`). See
[../cphmm/priors/README.md](../cphmm/priors/README.md) for the file format and
derivation.

### Within- vs between-clade transfers

When `compute_div_histogram(..., separate_clades=True)` is used, the prior is the
within-clade divergence histogram **concatenated** with the between-clade histogram
(split at a genome-wide divergence `clade_cutoff`). The number of bins per half is
`HMM_PRIOR_BINS` (default 40). At inference time, passing `clade_cutoff_bin=40` tells
`recomb_inference` to report transfers in two groups — states `1…clade_cutoff_bin`
(within-clade) and states `clade_cutoff_bin…N` (between-clade) — recorded in the `types`
column of the output (0 = within, 1 = between).

## Fitting and decoding (per pair, per contig)

For each contig (`recomb_inference._single_pass`):

1. **Fit** (`ClosePairHMM.fit`): a few Baum–Welch / EM iterations (`n_iter`, default 5)
   re-estimate the **transfer rate** (from the expected number of clonal→transfer
   transitions) and the **clonal emission** (from the forward–backward posterior over the
   clonal state). The clonal emission is floored at `min_clonal_emission` to avoid a
   degenerate zero.
2. **Decode** (`ClosePairHMM.decode`): Viterbi assigns each block a state; maximal runs of
   transfer states become candidate transfer segments (`find_segments`), and the blocks
   left in state 0 form the clonal sequence.
3. The model's emission/transfer rates are reset before the next contig
   (`reinit_emission_and_transfer_rates`).

### Iterative clonal-emission refinement

With `iterative=True`, `infer` runs `n_iter` outer passes (default 3). After each pass it
pools all blocks the HMM decoded as clonal **across contigs**, re-estimates the clonal
emission as the fraction of those blocks carrying a SNV
(`_bernoulli_clonal_emission_from_seq`), updates `init_clonal_emission`, and refits. This
mirrors the legacy `_fit_and_count_transfers_iterative` from `microbiome_evolution` but
pools per-pair rather than per-contig. It helps when abundant transfers would otherwise
bias the clonal-divergence estimate upward. The original `init_clonal_emission` is always
restored before `infer` returns.

## Outputs

`recomb_inference.infer` returns `(clonal_div, genome_len, clonal_len, transfer_df)`:

- **`clonal_div`** = `(naive_div, est_div)`. `naive_div` is the SNV fraction over decoded
  clonal blocks; `est_div` additionally coarse-grains the clonal region into ~1 kb blocks
  and drops blocks with > 2 SNVs, suppressing the contribution of short, undetected
  transfers (`estimate_clonal_divergence`).
- **`genome_len`** — number of analyzed core sites (length of `snp_vec`).
- **`clonal_len`** — number of core sites decoded as clonal.
- **`transfer_df`** — one row per detected transfer with `block_start/block_end` (block
  coordinates), `snp_vec_start/snp_vec_end` (site coordinates; start inclusive, end
  exclusive), and `types` (0 = within-clade, 1 = between-clade when `clade_cutoff_bin` is
  set).

`infer_pipelines.infer_pairs` additionally annotates transfers with inclusive reference
genome coordinates (`start_site`, `end_site`, `contig`) via
`annotate_transfer_reference_coordinates`.
