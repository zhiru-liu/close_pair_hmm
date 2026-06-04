"""Self-contained DataHelper for the Bacteroides fragilis CP-HMM example.

Reads the LiuGood2024 QP SNV catalog through the vendored, in-package reader
:class:`cphmm.io.liugood2024_qp.SNVHelper`, and exposes the close-pair DataHelper
interface by subclassing :class:`cphmm.datahelper.BaseClosePairDataHelper` -- so all
the prior-sampling boilerplate lives in the library, not here. The catalog itself is
downloaded on first use (see ``download_data.py``); the close pairs and ground truth
come from the committed SI subset. No sibling-repo imports, no hardcoded paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cphmm.datahelper import BaseClosePairDataHelper
from cphmm.io.liugood2024_qp import SNVHelper

import download_data

SPECIES = "Bacteroides_fragilis_54507"
THIS_DIR = Path(__file__).resolve().parent
PRIOR_DIR = THIS_DIR / "priors"
GROUND_TRUTH_PATH = THIS_DIR / "data" / f"{SPECIES}_ground_truth_transfers.csv"


class DataHelper_Bf(BaseClosePairDataHelper):
    """Bacteroides fragilis adapter consumed by ``infer_pipelines`` and ``cphmm.prior``."""

    def __init__(self, data_root=None, *, max_pairs=None, drop_missing_pairs=True,
                 ground_truth_path=GROUND_TRUTH_PATH):
        self.species = SPECIES
        self.hmm_prior_path = str(PRIOR_DIR)
        self.ground_truth_path = Path(ground_truth_path)

        if data_root is None:
            data_root = download_data.resolve_data_root()
        snvs_dir, reference_dir = download_data.snv_and_reference_dirs(Path(data_root))

        self.snv_data = SNVHelper(
            SPECIES,
            data_dir=snvs_dir,
            reference_dir=reference_dir,
            snv_format="feather",
            compute_bi_snvs=False,
            save_bi_snvs=False,
            mask_multi_sites=True,
            annotate=True,
        )

        # Precompute dense arrays for fast per-pair lookups (mirrors the reference
        # implementation; coverage/snvs columns are indexed by sample position).
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
        self.genome_len = int(self.snv_data.core_4D.sum())

        self.ground_truth = self._load_ground_truth()
        self.missing_pairs: list[tuple[str, str]] = []
        self.close_pairs = self._load_close_pairs(
            max_pairs=max_pairs, drop_missing_pairs=drop_missing_pairs
        )

    # -- close pairs + ground truth (from the committed SI subset) -------------

    def _load_ground_truth(self) -> pd.DataFrame:
        truth = pd.read_csv(self.ground_truth_path, low_memory=False)
        truth = truth[truth["Species name"] == self.species].copy()
        truth["Sample 1"] = truth["Sample 1"].astype(str)
        truth["Sample 2"] = truth["Sample 2"].astype(str)
        return truth

    def _load_close_pairs(self, *, max_pairs=None, drop_missing_pairs=True):
        unique_pairs = self.ground_truth[["Sample 1", "Sample 2"]].drop_duplicates()
        pairs = list(zip(unique_pairs["Sample 1"], unique_pairs["Sample 2"]))
        if drop_missing_pairs:
            valid = []
            for pair in pairs:
                if pair[0] in self.sample_to_index and pair[1] in self.sample_to_index:
                    valid.append(pair)
                else:
                    self.missing_pairs.append(pair)
            pairs = valid
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return pairs

    def get_close_pairs(self):
        return self.close_pairs

    def write_ground_truth_subset(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_csv(path, index=False)
        return path

    # -- CP-HMM inference input ------------------------------------------------

    def get_pair_snp_info(self, pair):
        sample1, sample2 = str(pair[0]), str(pair[1])
        idx1 = self.sample_to_index[sample1]
        idx2 = self.sample_to_index[sample2]

        s1 = self.snv_values[:, idx1]
        s2 = self.snv_values[:, idx2]
        snv_diffs = (s1 != s2) & (s1 != 255) & (s2 != 255)

        coverage = self.coverage_values[:, idx1] & self.coverage_values[:, idx2]
        snp_vec = np.zeros(self.core_to_snvs.shape[0], dtype=bool)
        snp_vec[self.core_to_snvs] = snv_diffs
        mask = self.core_4D & coverage
        snp_vec = snp_vec[mask]

        indices = self.snv_data.coverage.index[mask]
        contigs = indices.get_level_values(0).values.astype(str)
        locs = indices.get_level_values(1).values.astype(int)
        return snp_vec, contigs, locs
