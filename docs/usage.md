# Usage guide

How to run `cphmm` on your own data: write a `DataHelper`, run single-pair or batch
inference, generate a prior, and read the output tables. See
[concepts.md](concepts.md) for what the model is doing under the hood.

## 1. The `DataHelper` interface

The batch pipeline (`cphmm.infer_pipelines.infer_pairs`) and the prior builder
(`cphmm.prior`) are written against a small **duck-typed** `DataHelper` object that you
provide. It abstracts away how your SNV catalogue is stored. The expected surface:

| Attribute / method | Purpose |
| --- | --- |
| `species` | Species name; also selects the prior file `<species>.csv`. |
| `genome_len` | Number of core sites; used to initialize the per-block transfer rate. |
| `get_close_pairs()` | Returns the list of **closely related** sample pairs to analyze (see below). |
| `get_pair_snp_info(pair)` | Returns `(snp_vec, contigs, locs)` for one pair. |
| `hmm_prior_path` *(optional)* | If set, inference loads the prior from this directory instead of the bundled default. |
| `sample_prior_blocks(...)` *(optional)* | Fast path for prior estimation; otherwise `prior.sample_blocks` falls back to `get_random_pair` + `get_snp_vector`. |

`get_pair_snp_info(pair)` must return, over the **shared, well-covered core sites** of
the pair:

- `snp_vec` — a boolean array; `True` where the two samples differ.
- `contigs` — same length; the contig label of each site (keeps the HMM from
  transitioning across contig boundaries).
- `locs` — same length; the reference position of each site (used to annotate transfer
  coordinates).

> **Screen for close pairs first.** `get_close_pairs()` should return *only* pairs that
> share a recent clonal ancestor, pre-filtered by a **pairwise identical fraction**
> threshold (see [README best practices](../README.md#using-it-on-real-data-best-practices)).
> Running the CP-HMM on distantly related pairs produces meaningless segmentation.

A minimal sketch:

The easiest way to implement this is to subclass
`cphmm.datahelper.BaseClosePairDataHelper`, which provides `get_snp_vector`,
`get_random_pair`, and the vectorized `sample_prior_blocks` for you. You then only
supply `species`, `genome_len`, `get_close_pairs`, and `get_pair_snp_info` (set
`self.sample_names` so the default prior sampler can draw random pairs, or override
`_prior_pair_pool()` to restrict sampling to a diverged-pair pool):

```python
from cphmm.datahelper import BaseClosePairDataHelper

class MyDataHelper(BaseClosePairDataHelper):
    def __init__(self, species):
        self.species = species
        self.genome_len = ...                 # number of core sites
        self.sample_names = [...]             # for prior block sampling
        self._pairs = self._screen_close_pairs()   # apply identical-fraction cutoff here

    def get_close_pairs(self):
        return self._pairs

    def get_pair_snp_info(self, pair):
        s1, s2 = pair
        snp_vec = ...    # bool array over shared core sites
        contigs = ...    # contig label per site
        locs    = ...    # reference position per site
        return snp_vec, contigs, locs
```

You can also implement the [`cphmm.datahelper.ClosePairDataHelper`](../cphmm/datahelper.py)
protocol structurally without inheriting. See
[`workflows/bacteroides_fragilis/datahelper.py`](../workflows/bacteroides_fragilis/datahelper.py)
for a worked adapter (over the `cphmm.io.liugood2024_qp` catalog reader).

## 2. Batch inference over many pairs

```python
import cphmm.infer_pipelines as infer_pipelines

dh = MyDataHelper(species="Bacteroides_vulgatus_57955")
pairs = dh.get_close_pairs()

pair_dat, transfer_dat = infer_pipelines.infer_pairs(
    dh, pairs,
    clade_cutoff_bin=40,   # split within- vs between-clade transfers (= HMM_PRIOR_BINS)
    iterative=False,       # set True for iterative clonal-emission refinement
    n_iter=3,
)

pair_dat.to_csv("inference_summary.csv")
transfer_dat.to_csv("transfer_summary.csv")
```

`infer_pairs` initializes the model once per species (loading the prior, optionally from
`dh.hmm_prior_path`) and loops over pairs.

### Output tables

**`pair_dat`** — one row per pair:

| column | meaning |
| --- | --- |
| `genome1`, `genome2` | the two sample names |
| `naive_div` | clonal divergence over decoded clonal blocks |
| `est_div` | recombination-corrected clonal divergence |
| `genome_len` | analyzed core sites |
| `clonal_len` | core sites decoded as clonal |

**`transfer_dat`** — one row per detected transfer:

| column | meaning |
| --- | --- |
| `genome1`, `genome2` | the pair the transfer belongs to |
| `block_start`, `block_end` | transfer extent in block coordinates (inclusive) |
| `snp_vec_start`, `snp_vec_end` | extent in core-site coordinates (start inclusive, end exclusive) |
| `types` | `0` = within-clade, `1` = between-clade (when `clade_cutoff_bin` set) |
| `start_site`, `end_site`, `contig` | inclusive reference-genome coordinates |

## 3. Single-pair inference (lower level)

```python
import cphmm.model as hmm
import cphmm.recomb_inference as ri
import cphmm.config as config

model = hmm.ClosePairHMM(species_name=species, block_size=config.HMM_BLOCK_SIZE)
snp_vec, contigs, locs = dh.get_pair_snp_info(pair)

clonal_div, genome_len, clonal_len, transfers = ri.infer(
    snp_vec, contigs, model, config.HMM_BLOCK_SIZE, clade_cutoff_bin=40,
)
```

`ClosePairHMM` can also be constructed without a species prior by passing
`transfer_emissions`, `transition_prior`, `transfer_rate`, `clonal_emission`, and
`transfer_length` directly — see its docstring.

## 4. Generating a prior for a new dataset

The bundled priors are specific to the Liu & Good 2024 QP catalogue. For a new dataset,
generate your own into a separate work folder and point inference at it:

```python
import cphmm.prior as prior

# Sample blocks across close pairs, histogram their local divergences
local_divs, genome_divs = prior.sample_blocks(dh, num_samples=5000, block_size=1000)
divs, counts = prior.compute_div_histogram(
    local_divs, genome_divs,
    separate_clades=True, clade_cutoff=0.03,   # enables within/between-clade split
)
prior.save_prior(divs, counts, dh.species, prior_path="/path/to/run_work_folder/priors")
```

Then either set `dh.hmm_prior_path = "/path/to/run_work_folder/priors"` (batch pipeline)
or pass `prior_path=...` to `ClosePairHMM`. See
[../cphmm/priors/README.md](../cphmm/priors/README.md) for the file format.

`workflows/bacteroides_fragilis/generate_prior.py` is a complete prior-preparation script.

## 5. Tuning knobs

Defaults live in `cphmm/config.py`:

- `HMM_BLOCK_SIZE` (10) — core sites per block. Larger blocks = coarser resolution but
  more robust per-block statistics.
- `HMM_MIN_SEQ_LEN` (100) — contigs shorter than this many blocks are skipped.
- `HMM_PRIOR_BINS` (40) — bins per clade in the prior; also the natural `clade_cutoff_bin`.

Per-model knobs (constructor / `infer`): `n_iter` (EM iterations per fit),
`transfer_length` (expected transfer length, sets the transfer-state exit rate),
`min_clonal_emission` (floor on clonal divergence), and `iterative` / `n_iter`
(outer clonal-emission refinement passes).

## 6. Performance notes

The forward/backward/Viterbi kernels are Numba-JIT compiled and cached. The first call in
a fresh process incurs a one-time compile cost (~1–2 s); subsequent runs reuse the cached
bitcode. A 200 kb sequence with the default settings decodes in well under a second once
the kernels are warm.
