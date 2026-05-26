"""
Data helper for the Bacteroides fragilis CP-HMM test.

This intentionally wraps the published LiuGood2024 SNV handling code instead
of duplicating the SNV parsing logic here.
"""
from pathlib import Path
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd


SNVUTILS_REPO = Path("/Users/Device6/Documents/Research/bgoodlab/LiuGood2024_data")
SNV_SPECIES = "Bacteroides_fragilis_54507"
HMM_SPECIES = SNV_SPECIES
HMM_PRIOR_PATH = Path(__file__).resolve().parent / "priors"
GROUND_TRUTH_PATH = Path(
    "/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics/data/"
    "gut_microbiome_transfers.csv"
)


def _safe_load_simple_config(stream):
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = str(stream)

    config = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        config[key.strip()] = value
    return config


def install_yaml_fallback():
    """
    dNdS_310 does not currently have PyYAML, but LiuGood2024_data only needs
    yaml.safe_load for a tiny config.yml. Provide that narrow fallback locally.
    """
    try:
        import yaml  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = _safe_load_simple_config
    sys.modules["yaml"] = yaml_module


def load_snv_utils():
    """
    Import LiuGood2024_data/snv_utils.py.

    That module reads config.yml at import time using a relative path, so import
    it with its repository as the temporary working directory.
    """
    repo = str(SNVUTILS_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    install_yaml_fallback()

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        return importlib.import_module("snv_utils")
    finally:
        os.chdir(cwd)


class DataHelper_Bf:
    """
    Adapter expected by infer_pipelines.infer_pairs.

    Attributes used by infer_pipelines:
    - species: CP-HMM prior species name.
    - genome_len: number of usable 4D core sites.
    - get_close_pairs()
    - get_pair_snp_info(pair)
    """

    def __init__(
        self,
        snv_species=SNV_SPECIES,
        hmm_species=HMM_SPECIES,
        ground_truth_path=GROUND_TRUTH_PATH,
        max_pairs=None,
        drop_missing_pairs=True,
    ):
        self.snv_species = snv_species
        self.species = hmm_species
        self.hmm_prior_path = str(HMM_PRIOR_PATH)
        self.ground_truth_path = Path(ground_truth_path)
        self.snv_utils = load_snv_utils()

        self.snv_data = self.snv_utils.SNVHelper(
            snv_species,
            compute_bi_snvs=False,
            save_bi_snvs=False,
            mask_multi_sites=True,
            annotate=True,
            snv_format="feather",
        )
        self.sample_names = self.snv_data.samples.astype(str).to_numpy()
        self.sample_to_index = {
            sample: idx for idx, sample in enumerate(self.sample_names)
        }
        self.coverage_values = self.snv_data.coverage.loc[
            :, self.snv_data.samples
        ].to_numpy(dtype=bool, copy=False)
        self.snv_values = self.snv_data.snvs.loc[
            :, self.snv_data.samples
        ].to_numpy(dtype=np.uint8, copy=False)
        self.core_to_snvs = np.asarray(self.snv_data.core_to_snvs, dtype=bool)
        self.core_4D = np.asarray(self.snv_data.core_4D, dtype=bool)
        self.genome_len = int(self.snv_data.core_4D.sum())
        self.ground_truth = self.load_ground_truth()
        self.missing_pairs = []
        self.close_pairs = self.load_close_pairs(
            max_pairs=max_pairs,
            drop_missing_pairs=drop_missing_pairs,
        )

    def load_ground_truth(self):
        truth = pd.read_csv(self.ground_truth_path, low_memory=False)
        truth = truth[truth["Species name"] == self.snv_species].copy()
        truth["Sample 1"] = truth["Sample 1"].astype(str)
        truth["Sample 2"] = truth["Sample 2"].astype(str)
        return truth

    def load_close_pairs(self, max_pairs=None, drop_missing_pairs=True):
        unique_pairs = self.ground_truth[["Sample 1", "Sample 2"]].drop_duplicates()
        pairs = list(zip(unique_pairs["Sample 1"], unique_pairs["Sample 2"]))

        if drop_missing_pairs:
            samples = set(self.snv_data.samples.astype(str))
            valid_pairs = []
            for pair in pairs:
                if pair[0] in samples and pair[1] in samples:
                    valid_pairs.append(pair)
                else:
                    self.missing_pairs.append(pair)
            pairs = valid_pairs

        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        return pairs

    def get_close_pairs(self):
        return self.close_pairs

    def get_random_pair(self):
        idxs = np.random.choice(len(self.sample_names), size=2, replace=False)
        return self.sample_names[idxs[0]], self.sample_names[idxs[1]]

    def _get_pair_snp_vector_and_mask(self, pair):
        sample1, sample2 = str(pair[0]), str(pair[1])
        idx1 = self.sample_to_index[sample1]
        idx2 = self.sample_to_index[sample2]

        sample1_snvs = self.snv_values[:, idx1]
        sample2_snvs = self.snv_values[:, idx2]
        snv_diffs = (
            (sample1_snvs != sample2_snvs)
            & (sample1_snvs != 255)
            & (sample2_snvs != 255)
        )

        coverage = self.coverage_values[:, idx1] & self.coverage_values[:, idx2]
        snp_vec = np.zeros(shape=self.core_to_snvs.shape[0], dtype=bool)
        snp_vec[self.core_to_snvs] = snv_diffs
        mask = self.core_4D & coverage
        snp_vec = snp_vec[mask]

        return snp_vec, mask

    def get_snp_vector(self, pair):
        snp_vec, _ = self._get_pair_snp_vector_and_mask(pair)
        return snp_vec

    def get_pair_snp_info(self, pair):
        snp_vec, mask = self._get_pair_snp_vector_and_mask(pair)
        indices = self.snv_data.coverage.index[mask]
        contigs = indices.get_level_values(0).values.astype(str)
        locs = indices.get_level_values(1).values.astype(int)

        return snp_vec, contigs, locs

    def sample_prior_blocks(self, num_samples=5000, block_size=1000, random_state=None):
        rng = np.random.default_rng(random_state)
        local_divs = np.empty(num_samples)
        genome_divs = np.empty(num_samples)

        num_samples_available = len(self.sample_names)
        idx1s = rng.integers(0, num_samples_available, size=num_samples)
        idx2s = rng.integers(0, num_samples_available - 1, size=num_samples)
        idx2s += idx2s >= idx1s

        grouped_pairs = {}
        for out_idx, pair_idxs in enumerate(zip(idx1s, idx2s)):
            grouped_pairs.setdefault(tuple(sorted(pair_idxs)), []).append(out_idx)

        pending = []
        for pair_idxs, out_idxs in grouped_pairs.items():
            pair = (self.sample_names[pair_idxs[0]], self.sample_names[pair_idxs[1]])
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.extend(out_idxs)
                continue

            genome_div = np.mean(snp_vec)
            starts = rng.integers(0, len(snp_vec) - block_size + 1, size=len(out_idxs))
            for out_idx, start_idx in zip(out_idxs, starts):
                local_divs[out_idx] = np.mean(snp_vec[start_idx:start_idx + block_size])
                genome_divs[out_idx] = genome_div

        attempts = 0
        max_attempts = max(len(pending) * 100, 1000)
        while pending:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    "Unable to sample enough covered genome blocks for prior generation"
                )

            out_idx = pending.pop()
            idx1, idx2 = rng.choice(num_samples_available, size=2, replace=False)
            pair = (self.sample_names[idx1], self.sample_names[idx2])
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.append(out_idx)
                continue

            start_idx = rng.integers(0, len(snp_vec) - block_size + 1)
            local_divs[out_idx] = np.mean(snp_vec[start_idx:start_idx + block_size])
            genome_divs[out_idx] = np.mean(snp_vec)

        return local_divs, genome_divs

    def write_ground_truth_subset(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ground_truth.to_csv(path, index=False)
        return path
