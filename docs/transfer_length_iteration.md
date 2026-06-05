# Spec: per-species iterative transfer-length estimation

Implementation spec for porting the Liu & Good 2024 transfer-length determination into the
QP reproduction. **Implemented** as `--transfer-length iterative`
(`infer_iterative_transfer_length` in `workflows/liugood2024_qp/reproduce.py`); the default
remains a fixed `transfer_length` of 1000 (the paper's initialization). **One piece is
deliberately deferred:** the merge/filter of detected transfers before averaging (see the
"Important" note below) — the current mean is over raw decoded segments, so it runs slightly
short of the published estimate. This note remains the reference for that follow-up.

## What the paper actually did

The published QP table came from `microbiome_evolution` (branch **`analysis`**),
`HGT_scripts/close_pair_stage2-3_iterative.py` (functions `decode_one_pass` /
`process_one_species`); its `converged_pass/` output feeds
`close_pair_stage3_merge_and_filter.py`. The transfer length is **not fixed** — it is
re-estimated per species, iterating to convergence. (The `transfer_length = 2810` seen on
some branches is in the *non-iterative* stage2 script and did **not** produce the table.)

## Algorithm (from `decode_one_pass`)

Initialize `transfer_length = 1000` bp (single-clade) or `(1000, 1000)` for the two-clade
species `Bacteroides_vulgatus_57955` / `Alistipes_shahii_62199` (within, between). Then up to
**3 passes**:

1. Decode all close pairs with the current `transfer_length`.
2. Compute the **mean detected transfer length** (in bp = length-in-blocks ×
   `second_pass_block_size`=10). Single-clade: scalar mean over all transfers. Two-clade:
   separate means for within-clade (`types==0`) and between-clade (`types==1`) transfers.
3. `frac_change = |new - old| / old`. If `< 0.1` (both clades, for two-clade) → **converged,
   stop**.
4. Otherwise set `transfer_length = new` and repeat. For two-clade build a **per-state
   array** of length `2 * HMM_PRIOR_BINS` (=80): first 40 entries = within-mean, last 40 =
   between-mean.

**Important:** the original computes the mean over **merged + filtered** transfers —
`close_pair_utils.merge_and_filter_transfers(merge_threshold=0, filter_threshold=5)`, i.e.
transfers shorter than 5 blocks are dropped. `cphmm` has **no** such step today
(`cphmm/recomb_inference.py` returns raw decoded segments). Replicate the length filter
(and decide whether to merge adjacent segments) when computing the mean, or the estimate
will be biased short.

## cphmm integration points (already in place)

- **Per-clade transfer length is supported by the model.** `cphmm/model.py`
  `ClosePairHMM.__init__` accepts `transfer_length` as an ndarray of length
  `n_components - 1` and sets per-state `exit_rate = 1/transfer_length` (model.py ~lines
  35–39, used in `_init_transitions`). So passing an 80-length array for two-clade species
  works directly.
- **`transfer_length` is already plumbed** through `cphmm/infer_pipelines.py`
  `init_hmm(..., transfer_length=...)` and `infer_pairs(..., transfer_length=...)`; the
  `transfer_length / block_size` scaling is elementwise, so scalars and arrays both work.
- **Getting the mean length from output:** `infer_pairs` returns `transfer_dat` with
  `block_start` / `block_end` (inclusive) and `types`. Length in blocks =
  `block_end - block_start + 1`; multiply by `cphmm.config.HMM_BLOCK_SIZE` (10) for bp. Split
  by `types` for the two-clade case.
- **Two-clade detection:** prior CSV has `2 * HMM_PRIOR_BINS` (80) columns; use
  `clade_cutoff_bin = HMM_PRIOR_BINS` (40). Single-clade priors have 40 columns →
  `clade_cutoff_bin = None`. (`reproduce.py::prior_clade_settings` already does this.)

## Placement

Implemented as a new mode in `workflows/liugood2024_qp/reproduce.py`
(`infer_iterative_transfer_length`, a per-species outer loop that re-calls `infer_pairs`
until convergence), selected with `--transfer-length iterative`. It lives in the workflow,
not in `cphmm` core. It is **orthogonal to** `cphmm`'s existing `iterative` flag, which
refines the **clonal emission** rate — a different quantity; the two compose. Covered by
`tests/test_qp_iterative_transfer_length.py`.

## Verification

Run `reproduce.py` for one single-clade species (e.g. `Bacteroides_fragilis_54507`) and one
two-clade species (`Bacteroides_vulgatus_57955`) with the new mode; confirm convergence in
≤3 passes and that `verification_summary.csv` (clonal-div correlation, transfer-interval
overlap) matches or improves on the fixed-1000 run.
