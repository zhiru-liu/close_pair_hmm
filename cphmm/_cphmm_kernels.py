"""
Numba-JIT kernels for the CP-HMM topology forward/backward/Viterbi.

These functions are imported by :mod:`cphmm.hmm_backend`. If numba is not
installed the module exposes ``HAS_NUMBA = False`` and the public backend
falls back to the pure-numpy implementations.

The CP-HMM transition topology is sparse: state 0 (clonal) is fully connected,
and every nonzero state (transfer-divergence bin) has only the self-loop and
the transition back to state 0. That sparsity is exploited explicitly here:
each forward/backward step is O(n_components) instead of O(n_components**2).

All kernels are decorated ``@njit(cache=True)``. The first call
in a fresh process pays a one-time JIT-compilation cost (~1-2 s per kernel,
amortized across all pairs); subsequent runs read the cached bitcode.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit  # type: ignore

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - fallback path
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """No-op decorator so the module can still import without numba."""
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def wrap(fn):
            return fn

        return wrap


@njit(cache=True)
def forward_log_cphmm(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    fwdlattice = np.empty((n_samples, n_components))

    for j in range(n_components):
        fwdlattice[0, j] = log_startprob[j] + framelogprob[0, j]

    for t in range(1, n_samples):
        # State 0: logsumexp over (prev[j] + log_transmat[j, 0]) for all j.
        m = -math.inf
        for j in range(n_components):
            v = fwdlattice[t - 1, j] + log_transmat[j, 0]
            if v > m:
                m = v
        if not math.isfinite(m):
            fwdlattice[t, 0] = m + framelogprob[t, 0]
        else:
            s = 0.0
            for j in range(n_components):
                s += math.exp(
                    fwdlattice[t - 1, j] + log_transmat[j, 0] - m
                )
            fwdlattice[t, 0] = m + math.log(s) + framelogprob[t, 0]

        # Recomb states j>=1: logaddexp(prev[0]+from_clonal[j], prev[j]+self_j).
        prev0 = fwdlattice[t - 1, 0]
        for j in range(1, n_components):
            a = prev0 + log_transmat[0, j]
            b = fwdlattice[t - 1, j] + log_transmat[j, j]
            if a > b:
                if math.isfinite(a):
                    fwdlattice[t, j] = (
                        a + math.log1p(math.exp(b - a)) + framelogprob[t, j]
                    )
                else:
                    fwdlattice[t, j] = a + framelogprob[t, j]
            else:
                if math.isfinite(b):
                    fwdlattice[t, j] = (
                        b + math.log1p(math.exp(a - b)) + framelogprob[t, j]
                    )
                else:
                    fwdlattice[t, j] = b + framelogprob[t, j]

    # Final logprob = logsumexp(fwdlattice[-1]).
    m = -math.inf
    for j in range(n_components):
        if fwdlattice[n_samples - 1, j] > m:
            m = fwdlattice[n_samples - 1, j]
    if not math.isfinite(m):
        logprob = m
    else:
        s = 0.0
        for j in range(n_components):
            s += math.exp(fwdlattice[n_samples - 1, j] - m)
        logprob = m + math.log(s)

    return logprob, fwdlattice


@njit(cache=True)
def backward_log_cphmm(log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    bwdlattice = np.empty((n_samples, n_components))
    for j in range(n_components):
        bwdlattice[n_samples - 1, j] = 0.0

    for t in range(n_samples - 2, -1, -1):
        # State 0: logsumexp over (from_clonal[k] + framelogprob[t+1, k] + bwdlattice[t+1, k]).
        m = -math.inf
        for k in range(n_components):
            v = log_transmat[0, k] + framelogprob[t + 1, k] + bwdlattice[t + 1, k]
            if v > m:
                m = v
        if not math.isfinite(m):
            bwdlattice[t, 0] = m
        else:
            s = 0.0
            for k in range(n_components):
                s += math.exp(
                    log_transmat[0, k]
                    + framelogprob[t + 1, k]
                    + bwdlattice[t + 1, k]
                    - m
                )
            bwdlattice[t, 0] = m + math.log(s)

        # Recomb states j>=1: logaddexp(to_clonal[j]+next[0], self_j+next[j]).
        next0 = framelogprob[t + 1, 0] + bwdlattice[t + 1, 0]
        for j in range(1, n_components):
            a = log_transmat[j, 0] + next0
            b = (
                log_transmat[j, j]
                + framelogprob[t + 1, j]
                + bwdlattice[t + 1, j]
            )
            if a > b:
                if math.isfinite(a):
                    bwdlattice[t, j] = a + math.log1p(math.exp(b - a))
                else:
                    bwdlattice[t, j] = a
            else:
                if math.isfinite(b):
                    bwdlattice[t, j] = b + math.log1p(math.exp(a - b))
                else:
                    bwdlattice[t, j] = b

    return bwdlattice


@njit(cache=True)
def viterbi_log_cphmm(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    viterbi_lattice = np.empty((n_samples, n_components))
    backpointers = np.zeros((n_samples, n_components), dtype=np.int64)

    for j in range(n_components):
        viterbi_lattice[0, j] = log_startprob[j] + framelogprob[0, j]

    for t in range(1, n_samples):
        # State 0: argmax over (prev[j] + to_clonal[j]).
        best = -math.inf
        best_j = 0
        for j in range(n_components):
            v = viterbi_lattice[t - 1, j] + log_transmat[j, 0]
            if v > best:
                best = v
                best_j = j
        backpointers[t, 0] = best_j
        viterbi_lattice[t, 0] = best + framelogprob[t, 0]

        # Recomb states j>=1: max over {clonal->j, self-loop}; ties go to clonal.
        prev0 = viterbi_lattice[t - 1, 0]
        for j in range(1, n_components):
            from_clonal = prev0 + log_transmat[0, j]
            self_loop = viterbi_lattice[t - 1, j] + log_transmat[j, j]
            if from_clonal >= self_loop:
                backpointers[t, j] = 0
                viterbi_lattice[t, j] = from_clonal + framelogprob[t, j]
            else:
                backpointers[t, j] = j
                viterbi_lattice[t, j] = self_loop + framelogprob[t, j]

    # Backtrace.
    state_sequence = np.empty(n_samples, dtype=np.int64)
    best_last = 0
    best_val = -math.inf
    for j in range(n_components):
        if viterbi_lattice[n_samples - 1, j] > best_val:
            best_val = viterbi_lattice[n_samples - 1, j]
            best_last = j
    state_sequence[n_samples - 1] = best_last
    logprob = best_val
    for t in range(n_samples - 2, -1, -1):
        state_sequence[t] = backpointers[t + 1, state_sequence[t + 1]]

    return logprob, state_sequence


@njit(cache=True)
def log_xi_sum_cphmm(fwdlattice, log_transmat, bwdlattice, framelogprob):
    n_samples, n_components = framelogprob.shape
    log_xi_sum = np.full((n_components, n_components), -math.inf)

    for t in range(n_samples - 1):
        next0 = framelogprob[t + 1, 0] + bwdlattice[t + 1, 0]
        # Column 0: every state j -> clonal.
        for j in range(n_components):
            contrib = fwdlattice[t, j] + log_transmat[j, 0] + next0
            current = log_xi_sum[j, 0]
            if contrib > current:
                if math.isfinite(contrib):
                    log_xi_sum[j, 0] = (
                        contrib + math.log1p(math.exp(current - contrib))
                    )
                else:
                    log_xi_sum[j, 0] = contrib
            else:
                if math.isfinite(current):
                    log_xi_sum[j, 0] = (
                        current + math.log1p(math.exp(contrib - current))
                    )
                else:
                    log_xi_sum[j, 0] = current

        # Row 0 (recomb cols): clonal -> j for j>=1.
        f0 = fwdlattice[t, 0]
        for j in range(1, n_components):
            next_j = framelogprob[t + 1, j] + bwdlattice[t + 1, j]
            contrib = f0 + log_transmat[0, j] + next_j
            current = log_xi_sum[0, j]
            if contrib > current:
                if math.isfinite(contrib):
                    log_xi_sum[0, j] = (
                        contrib + math.log1p(math.exp(current - contrib))
                    )
                else:
                    log_xi_sum[0, j] = contrib
            else:
                if math.isfinite(current):
                    log_xi_sum[0, j] = (
                        current + math.log1p(math.exp(contrib - current))
                    )
                else:
                    log_xi_sum[0, j] = current

        # Recomb self-loops: j -> j for j>=1.
        for j in range(1, n_components):
            next_j = framelogprob[t + 1, j] + bwdlattice[t + 1, j]
            contrib = fwdlattice[t, j] + log_transmat[j, j] + next_j
            current = log_xi_sum[j, j]
            if contrib > current:
                if math.isfinite(contrib):
                    log_xi_sum[j, j] = (
                        contrib + math.log1p(math.exp(current - contrib))
                    )
                else:
                    log_xi_sum[j, j] = contrib
            else:
                if math.isfinite(current):
                    log_xi_sum[j, j] = (
                        current + math.log1p(math.exp(contrib - current))
                    )
                else:
                    log_xi_sum[j, j] = current

    return log_xi_sum
