"""
The close-pair DataHelper contract that CP-HMM inference consumes.

``cphmm`` is deliberately agnostic about how SNV data is stored. Every analysis
campaign (a QP metagenomic catalog, an NCBI-isolate alignment catalog, ...) wraps
its own SNV reader in a small adapter that exposes the interface defined here, and
then feeds that adapter to :func:`cphmm.infer_pipelines.infer_pairs` and
:func:`cphmm.prior.sample_blocks`.

Two things live here:

- :class:`ClosePairDataHelper` -- a :class:`typing.Protocol` documenting exactly
  what the inference/prior code reads off a datahelper. Implement it (structurally;
  no inheritance required) for a new campaign.
- :class:`BaseClosePairDataHelper` -- an optional concrete mixin that implements the
  campaign-agnostic boilerplate once: ``get_snp_vector``, ``get_random_pair``, and the
  vectorized ``sample_prior_blocks``. A subclass then only has to provide ``species``,
  ``genome_len``, ``get_close_pairs`` and ``get_pair_snp_info`` (plus, optionally, a
  diverged-pair pool for prior sampling). This removes the per-campaign duplication of
  the prior-block sampler.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

Pair = Tuple[str, str]
SNPInfo = Tuple[np.ndarray, np.ndarray, np.ndarray]


@runtime_checkable
class ClosePairDataHelper(Protocol):
    """Structural contract consumed by CP-HMM inference and prior estimation.

    Required by :func:`cphmm.infer_pipelines.infer_pairs`:

    - ``species``: name used to locate the species' transfer-divergence prior.
    - ``genome_len``: number of analyzable core sites (initializes the transfer rate).
    - ``get_close_pairs()``: the closely related pairs to run inference on.
    - ``get_pair_snp_info(pair)``: ``(snp_vec, contigs, locs)`` over the pair's covered
      core sites -- ``snp_vec`` is a bool array (True where the two samples differ),
      ``contigs``/``locs`` are the contig label and reference position of each site.

    Required by :func:`cphmm.prior.sample_blocks` (only when building a prior):

    - ``get_snp_vector(pair)``: just the ``snp_vec`` for a pair.
    - ``get_random_pair()``: a random pair to sample prior blocks from.
    - ``sample_prior_blocks(...)``: optional fast path; if present it is used directly.

    Optional:

    - ``hmm_prior_path``: directory of per-species prior CSVs (overrides the bundled default).
    - ``get_diverged_pairs()``: pool of fully recombined pairs to restrict prior sampling to.
    """

    species: str
    genome_len: int

    def get_close_pairs(self) -> Sequence[Pair]: ...

    def get_pair_snp_info(self, pair: Pair) -> SNPInfo: ...

    def get_snp_vector(self, pair: Pair) -> np.ndarray: ...

    def get_random_pair(self) -> Pair: ...


class BaseClosePairDataHelper:
    """Concrete mixin implementing the campaign-agnostic datahelper boilerplate.

    Subclasses must define:

    - ``species`` and ``genome_len`` (attributes),
    - ``get_close_pairs(self) -> Sequence[Pair]``,
    - ``get_pair_snp_info(self, pair) -> (snp_vec, contigs, locs)``.

    For prior sampling, a subclass provides the candidate-pair pool in one of two ways:

    - set ``self.sample_names`` (any sequence of sample names) and the default draws
      uniform random distinct pairs across all samples; or
    - override :meth:`_prior_pair_pool` to return an explicit list of pairs (e.g. the
      "fully recombined" / diverged pairs), which is the recommended pool for species
      with a near-clonal subcluster (uniform sampling would oversample clonal regions
      and bias the prior toward zero).
    """

    species: str
    genome_len: int
    sample_names: Optional[Sequence[str]] = None

    # -- inference inputs (subclass must implement get_pair_snp_info) ----------

    def get_close_pairs(self) -> Sequence[Pair]:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_pair_snp_info(self, pair: Pair) -> SNPInfo:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_snp_vector(self, pair: Pair) -> np.ndarray:
        snp_vec, _, _ = self.get_pair_snp_info(pair)
        return np.asarray(snp_vec, dtype=bool)

    # -- prior-sampling pool ---------------------------------------------------

    def _prior_pair_pool(self) -> Optional[Sequence[Pair]]:
        """Override to restrict prior sampling to an explicit pool of pairs.

        Return ``None`` (default) to sample uniform random distinct pairs across
        ``self.sample_names`` instead.
        """
        return None

    def _draw_candidate_pairs(self, n: int, rng: np.random.Generator) -> List[Pair]:
        pool = self._prior_pair_pool()
        if pool is not None:
            if len(pool) == 0:
                raise ValueError(
                    "Prior-sampling pair pool is empty; cannot sample prior blocks."
                )
            idxs = rng.integers(0, len(pool), size=n)
            return [tuple(pool[i]) for i in idxs]

        if self.sample_names is None:
            raise NotImplementedError(
                "Provide self.sample_names or override _prior_pair_pool()/get_random_pair() "
                "to sample prior blocks."
            )
        names = np.asarray(self.sample_names, dtype=object)
        n_avail = len(names)
        if n_avail < 2:
            raise ValueError("Need at least two samples to sample prior pairs.")
        idx1 = rng.integers(0, n_avail, size=n)
        idx2 = rng.integers(0, n_avail - 1, size=n)
        idx2 += idx2 >= idx1  # ensure idx2 != idx1, uniform over the other samples
        return [(str(names[a]), str(names[b])) for a, b in zip(idx1, idx2)]

    def get_random_pair(self) -> Pair:
        """A single random pair from the prior-sampling pool (uses global RNG)."""
        return self._draw_candidate_pairs(1, np.random.default_rng())[0]

    # -- prior block sampling (shared, vectorized) -----------------------------

    def sample_prior_blocks(
        self,
        num_samples: int = 5000,
        block_size: int = 1000,
        random_state=None,
    ):
        """Sample ``num_samples`` block divergences for the transfer-divergence prior.

        For each output draw a candidate pair (from :meth:`_draw_candidate_pairs`),
        then a uniformly random block of length ``block_size`` within that pair's
        covered SNP vector. Pairs are grouped so each pair's SNP vector is built at
        most once. Returns ``(local_divs, genome_divs)`` -- the per-block mean
        divergence and the pair's genome-wide divergence.
        """
        rng = np.random.default_rng(random_state)
        local_divs = np.empty(num_samples)
        genome_divs = np.empty(num_samples)

        candidates = self._draw_candidate_pairs(num_samples, rng)
        grouped: dict[Pair, List[int]] = {}
        for out_idx, pair in enumerate(candidates):
            grouped.setdefault(tuple(sorted(pair)), []).append(out_idx)

        pending: List[int] = []
        for pair, out_idxs in grouped.items():
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.extend(out_idxs)
                continue
            genome_div = float(np.mean(snp_vec))
            starts = rng.integers(0, len(snp_vec) - block_size + 1, size=len(out_idxs))
            for out_idx, start in zip(out_idxs, starts):
                local_divs[out_idx] = float(np.mean(snp_vec[start:start + block_size]))
                genome_divs[out_idx] = genome_div

        attempts = 0
        max_attempts = max(len(pending) * 100, 1000)
        while pending:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(
                    "Unable to sample enough covered genome blocks for prior "
                    f"generation (block_size={block_size}); the candidate pairs may be "
                    "too short or too poorly covered."
                )
            out_idx = pending.pop()
            pair = self._draw_candidate_pairs(1, rng)[0]
            snp_vec = self.get_snp_vector(pair)
            if len(snp_vec) < block_size:
                pending.append(out_idx)
                continue
            start = int(rng.integers(0, len(snp_vec) - block_size + 1))
            local_divs[out_idx] = float(np.mean(snp_vec[start:start + block_size]))
            genome_divs[out_idx] = float(np.mean(snp_vec))

        return local_divs, genome_divs
