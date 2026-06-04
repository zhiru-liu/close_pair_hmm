# cphmm — Close-Pair HMM for bacterial recombination

`cphmm` is a Python 3 implementation of the **Close-Pair Hidden Markov Model (CP-HMM)**
for detecting homologous recombination (horizontal transfer) between closely related
bacterial genomes, as described in:

> Liu Z, Good BH. *Dynamics of bacterial recombination in the human gut microbiome.*
> PLoS Biol. 2024 Feb 8;22(2):e3002472.
> doi: [10.1371/journal.pbio.3002472](https://doi.org/10.1371/journal.pbio.3002472).
> PMID: 38329938; PMCID: PMC10852326.

Given the pattern of SNV differences along the genome between a **pair** of conspecific
samples, the CP-HMM segments the genome into **clonally inherited** regions (low,
roughly uniform divergence reflecting their shared ancestry) and **recombined / transferred**
regions (locally elevated divergence from a donor lineage). From those segments it
estimates the pair's clonal divergence and enumerates transfer events and their
divergence.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Key concepts](#key-concepts)
- [Using it on real data (best practices)](#using-it-on-real-data-best-practices)
- [Where to get example data and reproduce the paper](#where-to-get-example-data-and-reproduce-the-paper)
- [Repository layout](#repository-layout)
- [Further documentation](#further-documentation)
- [Citation](#citation)

---

## Installation

Create the conda environment and install the package in editable mode:

```bash
conda env create -f requirements.yml
conda activate cphmm
pip install -e .
```

To also pull in the dependencies used by the bundled example workflows (`pyarrow`,
`biopython`):

```bash
pip install -e ".[workflows]"
```

The CP-HMM uses an internal NumPy/SciPy log-space HMM backend with
[Numba](https://numba.pydata.org/)-JIT'd forward/backward/Viterbi kernels — it does
**not** require `hmmlearn`. The first inference call in a fresh process pays a
one-time JIT-compilation cost (~1–2 s); compiled kernels are cached to disk and reused
afterwards.

Core runtime dependencies: `numpy`, `scipy`, `pandas`, `numba`.

---

## Quick start

```python
import numpy as np
import cphmm.model as hmm
import cphmm.recomb_inference as ri
import cphmm.config as config

# A binary SNV-difference vector along the core genome of one *pair* of samples:
#   snp_vec[i] = 1 if the two samples differ at core site i, else 0
snp_vec = np.zeros(200_000, dtype=bool)
snp_vec[5_000:10_000] = 1          # a locally divergent (recombined) stretch
contigs = np.zeros(len(snp_vec))   # contig label per site (single contig here)

# Build a model whose transfer-divergence prior is the bundled prior for this species
model = hmm.ClosePairHMM(species_name="Bacteroides_vulgatus_57955",
                         block_size=config.HMM_BLOCK_SIZE)

clonal_div, genome_len, clonal_len, transfers = ri.infer(
    snp_vec, contigs, model, config.HMM_BLOCK_SIZE,
    clade_cutoff_bin=config.HMM_PRIOR_BINS,   # split within- vs between-clade transfers
)

naive_div, est_div = clonal_div   # clonal divergence (naive and recombination-corrected)
print(est_div, transfers)          # `transfers` is a DataFrame of detected segments
```

For batch inference over many pairs (and reference-coordinate annotation of
transfers), wrap your data in a small **DataHelper** object and call
`cphmm.infer_pipelines.infer_pairs`. See [docs/usage.md](docs/usage.md) and the
`workflows/` examples.

---

## Key concepts

A CP-HMM is fit **independently to each pair of samples**. The genome is coarse-grained
into blocks of `HMM_BLOCK_SIZE` core sites, and each block is an observation: does it
contain a SNV difference (1) or not (0)?

- **Clonal state (state 0).** The two samples' shared, vertically inherited backbone.
  Block SNVs here occur at a low per-pair rate (the **clonal divergence**, ∝ time since
  their most recent common ancestor).
- **Transfer / recombined states (states 1…N).** A locally introgressed segment whose
  donor diverged from the recipient by some amount. The states tile a grid of donor
  **transfer divergences**; the per-state emission rate is the probability that a block
  carries a SNV at that divergence.
- **Transfer-divergence prior.** The distribution over donor divergences is supplied
  empirically, per species, as a histogram (`cphmm/priors/<species>.csv`). It is
  estimated by sampling genome blocks across many pairs — see
  [cphmm/priors/README.md](cphmm/priors/README.md). The prior directory is fully
  overridable so each analysis can use its own (see best practices below).
- **Within- vs between-clade transfers (`clade_cutoff_bin`).** When the prior is built
  with clade separation, its bins are the within-clade divergences concatenated with the
  between-clade divergences. Passing `clade_cutoff_bin` (e.g. `HMM_PRIOR_BINS = 40`)
  lets inference label each detected transfer as within- or between-clade.
- **Fitting.** Per pair, the transfer rate and clonal emission are re-estimated by a few
  Baum–Welch (EM) iterations, then the genome is Viterbi-decoded into clonal/transfer
  segments. The optional **iterative mode** (`iterative=True`) re-estimates the clonal
  emission from the blocks the HMM itself called clonal and refits — useful when many
  transfers would otherwise inflate the apparent clonal divergence.
- **Recombination-corrected clonal divergence.** Clonal divergence is reported both
  "naively" and after coarse-graining the decoded clonal region to suppress the
  contribution of short, undetected transfers (`estimate_clonal_divergence`).

See [docs/concepts.md](docs/concepts.md) for the model in more depth.

---

## Using it on real data (best practices)

**Only apply the CP-HMM to closely related ("close") pairs.** The model assumes the two
samples share a recent clonal ancestor, so that most of the genome is clonally inherited
and recombined segments stand out as local divergence spikes. For distantly related
pairs the whole genome looks divergent, there is no clonal backbone to anchor the HMM,
and the segmentation is meaningless.

**Diagnose closeness with the pairwise identical fraction before inference.** In
practice, screen candidate pairs by the fraction of the genome (or of coarse-grained
blocks) that is **identical** between the two samples. Closely related pairs share long
runs of identical sequence and have a high identical fraction; unrelated pairs do not.
Keep only pairs above a closeness threshold and discard the rest — this is what a
`DataHelper.get_close_pairs()` method is expected to return. (In the paper, candidate
pairs were pre-screened this way; the example datahelpers reproduce that selection.)

Other practical notes:

- **Inputs are per-pair SNV-difference vectors over shared, well-covered core sites.**
  Restrict to core genome positions covered in *both* samples (the example datahelpers
  use 4-fold-degenerate core sites). Carry contig labels so the HMM does not run
  transitions across contig boundaries; very short contigs (`< HMM_MIN_SEQ_LEN` blocks)
  are skipped.
- **Use a species-appropriate transfer-divergence prior.** The bundled priors are
  specific to the Liu & Good 2024 quasi-phaseable (QP) SNV catalogue. For a new dataset,
  generate your own prior into a separate work folder and pass `prior_path=...` (or set
  `datahelper.hmm_prior_path`) rather than reusing the bundled set — see
  [cphmm/priors/README.md](cphmm/priors/README.md).
- **Interpret transfer divergence relative to the clade cutoff.** Within-clade transfers
  (low donor divergence) are harder to detect than between-clade ones; treat counts near
  the detection limit with care.

---

## Where to get example data and reproduce the paper

### Example SNV catalogues (Liu & Good 2024)

The quasi-phaseable SNV catalogues used in the paper — the inputs the example workflows
and bundled priors are built from — are released here:

- **SNV data + access code:** <https://github.com/zhiru-liu/LiuGood-2024-SNVs>
- The processed SNV tables are archived on **Zenodo** (linked from that repository's
  README); follow that link to download the per-species catalogues.

These let you reproduce the close-pair inference end to end and regenerate priors for the
paper's species.

### Figure-generation scripts (original Python 2 analysis code)

The original analysis and figure-generation code accompanying the paper (Python 2) lives
in a separate repository:

- **Original analysis / figures:** <https://github.com/zhiru-liu/microbiome_evolution>

`cphmm` is a refactor of the CP-HMM portion of that project into an installable Python 3
package; use `microbiome_evolution` if you need to reproduce the exact paper figures or
the upstream data-processing pipeline.

---

## Repository layout

```
cphmm/                  Installable package
  model.py              ClosePairHMM: states, transitions, EM fitting, decoding
  recomb_inference.py   infer(): per-pair segmentation + clonal-divergence estimation
  prior.py              Build/save transfer-divergence priors from a DataHelper
  hmm_backend.py        Log-space forward/backward/Viterbi (dispatch + numpy fallback)
  _cphmm_kernels.py     Numba-JIT kernels exploiting the sparse transition topology
  seq_manip.py          Block coarse-graining + block<->genome coordinate conversion
  infer_pipelines.py    Batch inference over many pairs via a DataHelper
  config.py             Block size, min sequence length, default prior path, # bins
  priors/               Bundled LiuGood2024 reference priors (+ README on derivation)

workflows/              Worked examples (not part of the installed package)
  example1/             Reproduce one species from the PLoS Biol 2024 paper
  example2/             Tsimane/Hadza cohort inference + prior preparation
  test/                 Performance/profiling script

tests/                  Unit tests (run: python tests/test_recomb_coordinates.py)
docs/                   Concepts and usage guides
```

---

## Further documentation

- [docs/concepts.md](docs/concepts.md) — the model: states, prior, EM fitting, decoding,
  iterative refinement, and how outputs are computed.
- [docs/usage.md](docs/usage.md) — writing a `DataHelper`, running single-pair and batch
  inference, generating priors, and interpreting the output tables.
- [cphmm/priors/README.md](cphmm/priors/README.md) — prior file format, how the bundled
  priors were derived, and generating run-specific priors.

---

## Citation

If you use this software, please cite the paper:

> Liu Z, Good BH. Dynamics of bacterial recombination in the human gut microbiome.
> PLoS Biol. 2024;22(2):e3002472. https://doi.org/10.1371/journal.pbio.3002472

## License

Released under the [MIT License](LICENSE).
