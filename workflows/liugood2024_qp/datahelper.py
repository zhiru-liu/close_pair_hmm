"""DataHelper for the Liu & Good 2024 QP recombination reproduction.

A species-parameterized adapter over the QP SNV catalog, reusing the in-package reader
``cphmm.io.liugood2024_qp.SNVHelper`` and the shared
``cphmm.datahelper.BaseClosePairDataHelper`` (prior sampling). It is the multi-species
generalization of ``workflows/bacteroides_fragilis/datahelper.py``.

Close pairs come from one of two sources:

- ``pair_source="published"`` (default): the distinct (Sample 1, Sample 2) pairs for this
  species in the published supplementary table -- reproduce inference on exactly the pairs
  the paper analyzed.
- ``pair_source="select"``: re-derive close pairs from the catalog by clonal fraction
  (fraction of 1000bp 4D-core blocks with zero SNPs) > cutoff. Approximate vs the paper,
  which de-duplicated same-host samples first (host metadata is not in the public data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cphmm.datahelper import BaseClosePairDataHelper
from cphmm.io.liugood2024_qp import SNVHelper
from cphmm.seq_manip import to_block

DEFAULT_CLONAL_FRAC_CUTOFF = 0.5
CLONAL_FRAC_BLOCK_SIZE = 1000  # first-pass block size used for close-pair selection


class DataHelper_QP(BaseClosePairDataHelper):
    """QP-catalog adapter consumed by ``infer_pipelines`` and ``cphmm.prior``."""

    def __init__(self, species, *, data_dir, reference_dir, prior_dir,
                 ground_truth_path=None, pair_source="published", max_pairs=None,
                 clonal_frac_cutoff=DEFAULT_CLONAL_FRAC_CUTOFF):
        self.species = species
        self.hmm_prior_path = str(prior_dir)
        self.ground_truth_path = Path(ground_truth_path) if ground_truth_path else None
        self.pair_source = pair_source
        self.clonal_frac_cutoff = clonal_frac_cutoff

        self.snv_data = SNVHelper(
            species,
            data_dir=data_dir,
            reference_dir=reference_dir,
            snv_format="feather",
            compute_bi_snvs=False,
            save_bi_snvs=False,
            mask_multi_sites=True,
            annotate=True,
        )

        # Dense arrays for fast per-pair lookups (sample columns by position).
        self.sample_names = self.snv_data.samples.astype(str).to_numpy()
        self.sample_to_index = {s: i for i, s in enumerate(self.sample_names)}
        self.coverage_values = self.snv_data.coverage.loc[
            :, self.snv_data.samples
        ].to_numpy(dtype=bool, copy=False)
        self.snv_values = self.snv_data.snvs.loc[
            :, self.snv_data.samples
        ].to_numpy(dtype=np.uint8, copy=False)
        self.core_to_snvs = np.asarray(self.snv_data.core_to_snvs, dtype=bool)
        self.core_4D = np.asarray(self.snv_data.core_4D, dtype=bool)
        self.core_1D = np.asarray(self.snv_data.core_1D, dtype=bool)
        self.genome_len = int(self.snv_data.core_4D.sum())

        self.ground_truth = self._load_ground_truth()
        self.missing_pairs: list[tuple[str, str]] = []
        self.close_pairs = self._resolve_close_pairs(max_pairs=max_pairs)

    # -- ground truth + close pairs -------------------------------------------

    def _load_ground_truth(self):
        if self.ground_truth_path is None:
            return None
        truth = pd.read_csv(self.ground_truth_path, low_memory=False)
        truth = truth[truth["Species name"] == self.species].copy()
        truth["Sample 1"] = truth["Sample 1"].astype(str)
        truth["Sample 2"] = truth["Sample 2"].astype(str)
        return truth

    def _resolve_close_pairs(self, *, max_pairs=None):
        if self.pair_source == "published":
            pairs = self._published_pairs()
        elif self.pair_source == "select":
            pairs = self._selected_pairs()
        else:
            raise ValueError(f"unknown pair_source {self.pair_source!r}")
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return pairs

    def _published_pairs(self):
        if self.ground_truth is None:
            raise ValueError(
                "pair_source='published' requires a ground-truth table (--ground-truth)."
            )
        unique = self.ground_truth[["Sample 1", "Sample 2"]].drop_duplicates()
        pairs = list(zip(unique["Sample 1"], unique["Sample 2"]))
        kept = []
        for pair in pairs:
            if pair[0] in self.sample_to_index and pair[1] in self.sample_to_index:
                kept.append(pair)
            else:
                self.missing_pairs.append(pair)
        return kept

    def _selected_pairs(self):
        """Re-derive close pairs by clonal fraction over all sample pairs.

        NOTE: the published pipeline first kept one sample per host; that host map is
        not in the public data, so this may include same-host pairs (approximate).
        """
        n = len(self.sample_names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                cf = self._clonal_fraction(self.sample_names[i], self.sample_names[j])
                if self.clonal_frac_cutoff < cf < 1.0:
                    pairs.append((self.sample_names[i], self.sample_names[j]))
        return pairs

    def _clonal_fraction(self, sample1, sample2):
        snp_vec = self.get_snp_vector((sample1, sample2))
        if len(snp_vec) < CLONAL_FRAC_BLOCK_SIZE:
            return 0.0
        blocks = to_block(snp_vec, CLONAL_FRAC_BLOCK_SIZE)
        return float(np.mean(blocks == 0))

    def get_close_pairs(self):
        return self.close_pairs

    # -- CP-HMM inference input (same 4D-core logic as the Bf helper) ----------

    def get_pair_snp_info(self, pair, site_class="4D"):
        """Return CP-HMM input: SNP vector, contig names, contig locations.

        ``site_class`` selects the covered core sites used: ``'4D'`` (default,
        the synonymous sites the HMM runs on), ``'1D'`` (nonsynonymous, used by
        the 1D tract-extension add-on), or ``'all'`` (every covered core site).
        Matches the ``IsolateSNVHelper.get_pair_snp_info`` contract.
        """
        sample1, sample2 = str(pair[0]), str(pair[1])
        idx1 = self.sample_to_index[sample1]
        idx2 = self.sample_to_index[sample2]

        s1 = self.snv_values[:, idx1]
        s2 = self.snv_values[:, idx2]
        snv_diffs = (s1 != s2) & (s1 != 255) & (s2 != 255)

        coverage = self.coverage_values[:, idx1] & self.coverage_values[:, idx2]
        snp_vec = np.zeros(self.core_to_snvs.shape[0], dtype=bool)
        snp_vec[self.core_to_snvs] = snv_diffs

        if site_class == "4D":
            site_mask = self.core_4D
        elif site_class == "1D":
            site_mask = self.core_1D
        elif site_class in ("all", "covered"):
            site_mask = np.ones_like(self.core_4D, dtype=bool)
        else:
            raise ValueError("site_class must be one of '4D', '1D', or 'all'")

        mask = site_mask & coverage
        snp_vec = snp_vec[mask]

        indices = self.snv_data.coverage.index[mask]
        contigs = indices.get_level_values(0).values.astype(str)
        locs = indices.get_level_values(1).values.astype(int)
        return snp_vec, contigs, locs
