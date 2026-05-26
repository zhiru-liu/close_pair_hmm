"""
Data helper for UHGG isolate CP-HMM reruns.

Thin wrapper around ``dnds_revision.isolate_snv_helper.IsolateSNVHelper``.

The helper already produces ``(snp_vec, contigs, locs)`` over covered 4D sites
in the CP-HMM-expected shape, and exposes ``get_close_pairs(cutoff=0.5)`` that
reads the cached identical-fraction artifact under
``/Volumes/Botein/uhgg/isolate_snvs/<accession>/identical_fraction.parquet``.
This wrapper exists only to adapt those attributes/methods onto the names
``infer_pipelines.infer_pairs`` and ``cphmm.prior.sample_blocks`` expect.

No DH-format files are read.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


DNDS_REVISION_ROOT = Path(
    "/Users/Device6/Documents/Research/bgoodlab/dNdS/dNdS_dynamics_revision"
)
HMM_PRIOR_PATH = Path(__file__).resolve().parent / "priors"
CLONAL_FRAC_CUTOFF = 0.5
IDENTICAL_FRACTION_BLOCK_SIZE = 1000

ACCESSIONS = (
    "MGYG-HGUT-01337",
    "MGYG-HGUT-01346",
    "MGYG-HGUT-01378",
    "MGYG-HGUT-02331",
    "MGYG-HGUT-02366",
    "MGYG-HGUT-02422",
    "MGYG-HGUT-02438",
    "MGYG-HGUT-02478",
    "MGYG-HGUT-02538",
)


def _ensure_dnds_revision_on_path() -> None:
    repo = str(DNDS_REVISION_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_isolate_snv_helper():
    """Import ``IsolateSNVHelper`` from dNdS_dynamics_revision."""
    _ensure_dnds_revision_on_path()
    from dnds_revision.isolate_snv_helper import IsolateSNVHelper  # type: ignore

    return IsolateSNVHelper


class DataHelper_Isolate:
    """
    Adapter expected by ``infer_pipelines.infer_pairs`` and ``cphmm.prior``.

    Attributes/methods consumed downstream:
    - ``species``: CP-HMM prior species name (here: accession).
    - ``genome_len``: number of annotated 4D core sites.
    - ``hmm_prior_path``: directory containing ``<species>.csv``.
    - ``get_close_pairs()``: list of sample-name tuples.
    - ``get_pair_snp_info(pair)``: ``(snp_vec, contigs, locs)`` over covered 4D.
    - ``get_random_pair()``, ``get_snp_vector(pair)``, ``sample_prior_blocks(...)``
      for prior generation.
    """

    def __init__(
        self,
        accession: str,
        *,
        clonal_frac_cutoff: float = CLONAL_FRAC_CUTOFF,
        identical_fraction_block_size: int = IDENTICAL_FRACTION_BLOCK_SIZE,
        max_pairs: int | None = None,
        hmm_prior_path: Path | str = HMM_PRIOR_PATH,
    ) -> None:
        if accession not in ACCESSIONS:
            print(
                f"[isolate_datahelper] note: accession {accession!r} is not in "
                "the canonical 9-accession list; proceeding anyway."
            )
        self.accession = accession
        self.species = accession
        self.species_name = accession
        self.clonal_frac_cutoff = float(clonal_frac_cutoff)
        self.identical_fraction_block_size = int(identical_fraction_block_size)
        self.hmm_prior_path = str(hmm_prior_path)

        IsolateSNVHelper = load_isolate_snv_helper()
        self.helper = IsolateSNVHelper(
            accession,
            source="tables",
            annotate=True,
            compute_bi_snvs=False,
            save_bi_snvs=False,
            mask_multi_sites=True,
        )
        if not self.helper.has_site_annotations:
            raise RuntimeError(
                f"{accession}: site annotations missing on the SNV table. "
                "CP-HMM needs core_4D; regenerate site_annotations.parquet first."
            )

        self.sample_names = np.asarray(
            [str(s) for s in self.helper.samples], dtype=object
        )
        self.sample_to_index = {s: i for i, s in enumerate(self.sample_names)}
        self.genome_len = int(self.helper.genome_len)

        self.missing_pairs: list[tuple[str, str]] = []
        self.close_pairs = self._load_close_pairs(max_pairs=max_pairs)

    # ---- close-pair selection ------------------------------------------------

    def _load_close_pairs(
        self, *, max_pairs: int | None
    ) -> list[tuple[str, str]]:
        pairs = self.helper.get_close_pairs(
            cutoff=self.clonal_frac_cutoff,
            block_size=self.identical_fraction_block_size,
            site_class="4D",
        )
        kept: list[tuple[str, str]] = []
        for a, b in pairs:
            a_str, b_str = str(a), str(b)
            if a_str in self.sample_to_index and b_str in self.sample_to_index:
                kept.append((a_str, b_str))
            else:
                self.missing_pairs.append((a_str, b_str))
        if max_pairs is not None:
            kept = kept[:max_pairs]
        return kept

    def get_close_pairs(self) -> list[tuple[str, str]]:
        return self.close_pairs

    # ---- CP-HMM inference inputs --------------------------------------------

    def get_pair_snp_info(self, pair):
        snp_vec, contigs, locs = self.helper.get_pair_snp_info(
            (str(pair[0]), str(pair[1])), site_class="4D"
        )
        return (
            np.asarray(snp_vec, dtype=bool),
            np.asarray(contigs, dtype=str),
            np.asarray(locs, dtype=int),
        )

    # ---- Prior-generation interface (cphmm.prior.sample_blocks) -------------

    def get_snp_vector(self, pair):
        snp_vec, _, _ = self.get_pair_snp_info(pair)
        return snp_vec

    def get_random_pair(self):
        idxs = np.random.choice(len(self.sample_names), size=2, replace=False)
        return self.sample_names[idxs[0]], self.sample_names[idxs[1]]

    def sample_prior_blocks(
        self,
        num_samples: int = 5000,
        block_size: int = 1000,
        random_state=None,
    ):
        """
        Vectorized prior-block sampler.

        Mirrors ``Bf_test.bf_datahelper.DataHelper_Bf.sample_prior_blocks``.
        Groups random pairs so each pair's SNP vector is only built once.
        """
        rng = np.random.default_rng(random_state)
        local_divs = np.empty(num_samples)
        genome_divs = np.empty(num_samples)

        n_samples_available = len(self.sample_names)
        if n_samples_available < 2:
            raise ValueError(
                f"{self.accession}: need at least 2 samples for prior sampling; "
                f"helper.samples has {n_samples_available}."
            )

        idx1s = rng.integers(0, n_samples_available, size=num_samples)
        idx2s = rng.integers(0, n_samples_available - 1, size=num_samples)
        idx2s += idx2s >= idx1s

        grouped: dict[tuple[int, int], list[int]] = {}
        for out_idx, pair_idxs in enumerate(zip(idx1s, idx2s)):
            grouped.setdefault(tuple(sorted(pair_idxs)), []).append(out_idx)

        pending: list[int] = []
        for pair_idxs, out_idxs in grouped.items():
            pair = (
                self.sample_names[pair_idxs[0]],
                self.sample_names[pair_idxs[1]],
            )
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.extend(out_idxs)
                continue

            genome_div = float(np.mean(snp_vec))
            starts = rng.integers(
                0, len(snp_vec) - block_size + 1, size=len(out_idxs)
            )
            for out_idx, start_idx in zip(out_idxs, starts):
                local_divs[out_idx] = float(
                    np.mean(snp_vec[start_idx : start_idx + block_size])
                )
                genome_divs[out_idx] = genome_div

        attempts = 0
        max_attempts = max(len(pending) * 100, 1000)
        while pending:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    f"{self.accession}: unable to sample enough covered genome "
                    f"blocks for prior generation (block_size={block_size})."
                )
            out_idx = pending.pop()
            idx1, idx2 = rng.choice(
                n_samples_available, size=2, replace=False
            )
            pair = (self.sample_names[idx1], self.sample_names[idx2])
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.append(out_idx)
                continue
            start_idx = rng.integers(0, len(snp_vec) - block_size + 1)
            local_divs[out_idx] = float(
                np.mean(snp_vec[start_idx : start_idx + block_size])
            )
            genome_divs[out_idx] = float(np.mean(snp_vec))

        return local_divs, genome_divs
