# Architecture & extensibility

`cphmm` is designed so that new analysis campaigns (different SNV-catalog formats,
different ways of building catalogs) reuse the inference core without copying adapter
code. This document describes the layering, the contract that ties it together, and the
extensions that are planned but not yet built.

## Layers

```
catalog construction  →  canonical SNV tables  →  SNV-table reader  →  DataHelper  →  CP-HMM
   (per campaign)          (Contig, Location)      (cphmm.io.*)      (protocol)     (inference)
```

1. **CP-HMM core** — `cphmm.model`, `cphmm.recomb_inference`, `cphmm.infer_pipelines`,
   `cphmm.prior`. Format-agnostic: they only ever touch a *DataHelper*.
2. **The DataHelper contract** — `cphmm.datahelper`:
   - `ClosePairDataHelper` (a `typing.Protocol`) is the exact interface the core consumes
     (`species`, `genome_len`, `get_close_pairs`, `get_pair_snp_info`, and prior hooks).
   - `BaseClosePairDataHelper` is an optional mixin implementing the campaign-agnostic
     boilerplate once — `get_snp_vector`, `get_random_pair`, and the vectorized
     `sample_prior_blocks`. A campaign adapter subclasses it and supplies only
     `get_close_pairs` + `get_pair_snp_info` (and a sample pool for prior sampling).
3. **SNV-catalog readers** — `cphmm.io.*` (optional, `[workflows]` extra). Each reads one
   on-disk catalog format and exposes the matrices/masks an adapter needs. Currently
   `cphmm.io.liugood2024_qp` (the LiuGood2024 QP feather catalog).
4. **Workflows** — `workflows/<campaign>/` glue: a thin DataHelper subclass + an `infer.py`
   entry point. See `workflows/bacteroides_fragilis/`.

## The canonical SNV-table schema (integration contract)

Independently built catalogs converge on the same tabular schema, indexed by
`(Contig, Location)` with one column per sample:

- `coverage` — boolean, all reference sites × samples.
- `biallelic_snvs` — `0/1/255` (major / alt / missing) at biallelic sites, with
  `Ref`/`Major`/`Alt` columns.
- `site_annotations` — per-site `Site Type` (`1D`/`2D`/`3D`/`4D`), gene, and per-base
  mutation effects (used to build the 4D-core mask).
- `identical_fraction` — long-form per-pair `(sample_1, sample_2, identical_fraction,
  num_blocks, num_identical_blocks)`; the basis for close-pair / diverged-pair selection.

The LiuGood2024 QP catalog (feather) and the UHGG/NCBI isolate catalogs (parquet) are both
members of this family, differing mainly in format and whether the 4D annotation is
precomputed (`site_annotations`) or computed on the fly from a reference genome. Treating
this schema as the contract is what lets one reader serve multiple campaigns.

## Planned extensions (not yet built)

- **`cphmm.io.snv_table`** — a single generic reader for the canonical schema
  (parquet + feather; precomputed-or-on-the-fly annotation). It would subsume
  `cphmm.io.liugood2024_qp` as a thin format/path preset, so most campaigns need no bespoke
  reader at all.
- **`cphmm.catalog`** (or a sibling package) — "specify a species → download NCBI genomes →
  align to a reference (nucmer/MUMmer) → emit canonical SNV tables," mirroring the existing
  `dNdS_dynamics_revision/pipeline/` build. This depends on external CLI tools (MUMmer, the
  NCBI `datasets` CLI), so it would live behind a `[catalog]` extra with those tools
  documented as system requirements. This "build a catalog for any species" path is expected
  to be the most broadly useful entry point.
- **Shared base for existing campaigns** — the isolate adapter
  (`dNdS_dynamics_revision`'s `IsolateCPHMMDataHelper`) already implements this exact
  protocol with a duplicated `sample_prior_blocks`; it can inherit
  `BaseClosePairDataHelper` to drop that duplication.
