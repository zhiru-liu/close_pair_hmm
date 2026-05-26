import numpy as np
from scipy import special


def _logsumexp_1d(x):
    """Tight log-sum-exp for a 1-D float array.

    ``scipy.special.logsumexp`` has ~25-30 us of per-call dispatch overhead
    (array_api shimming, validation, edge cases) that dominates CP-HMM runtime
    because the function is called once per (block, EM iter, fwd/bwd pass) on a
    ~40-element state vector. This inline version is ~5x faster on those
    inputs while remaining correct for the values we ever pass: at least one
    finite entry. If every entry is -inf the result is -inf (matching scipy).
    """
    m = x.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.exp(x - m).sum())


def _logsumexp_axis0(x):
    """Vectorized log-sum-exp along axis 0 of a 2-D array."""
    m = x.max(axis=0)
    finite = np.isfinite(m)
    out = m.copy()
    if finite.any():
        out[finite] = m[finite] + np.log(
            np.exp(x[:, finite] - m[finite][np.newaxis, :]).sum(axis=0)
        )
    return out


def _has_cphmm_transition_topology(log_transmat):
    """
    Check for CP-HMM topology: state 0 can transition anywhere, while every
    nonzero state can only stay put or exit to state 0.
    """
    n_components = log_transmat.shape[0]
    if n_components <= 2:
        return True
    off_diagonal = log_transmat[1:, 1:][~np.eye(n_components - 1, dtype=bool)]
    return np.all(np.isneginf(off_diagonal))


def forward_log(log_startprob, log_transmat, framelogprob):
    """
    Run the HMM forward pass in log space.

    Parameters are log start probabilities, log transition probabilities, and
    frame log likelihoods with shape ``(n_samples, n_components)``.
    """
    if _has_cphmm_transition_topology(log_transmat):
        return _forward_log_cphmm(log_startprob, log_transmat, framelogprob)

    n_samples, n_components = framelogprob.shape
    fwdlattice = np.empty((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        fwdlattice[t] = (
            special.logsumexp(fwdlattice[t - 1][:, np.newaxis] + log_transmat, axis=0)
            + framelogprob[t]
        )

    with np.errstate(under="ignore"):
        logprob = special.logsumexp(fwdlattice[-1])
    return logprob, fwdlattice


def _forward_log_cphmm(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    fwdlattice = np.empty((n_samples, n_components))
    fwdlattice[0] = log_startprob + framelogprob[0]

    to_clonal = log_transmat[:, 0]
    from_clonal = log_transmat[0]
    recomb_self = np.diag(log_transmat)[1:]

    for t in range(1, n_samples):
        prev = fwdlattice[t - 1]
        fwdlattice[t, 0] = _logsumexp_1d(prev + to_clonal) + framelogprob[t, 0]
        fwdlattice[t, 1:] = (
            np.logaddexp(prev[0] + from_clonal[1:], prev[1:] + recomb_self)
            + framelogprob[t, 1:]
        )

    with np.errstate(under="ignore"):
        logprob = _logsumexp_1d(fwdlattice[-1])
    return logprob, fwdlattice


def backward_log(log_transmat, framelogprob):
    """
    Run the HMM backward pass in log space.
    """
    if _has_cphmm_transition_topology(log_transmat):
        return _backward_log_cphmm(log_transmat, framelogprob)

    n_samples, n_components = framelogprob.shape
    bwdlattice = np.empty((n_samples, n_components))
    bwdlattice[-1] = 0.0

    for t in range(n_samples - 2, -1, -1):
        bwdlattice[t] = special.logsumexp(
            log_transmat
            + framelogprob[t + 1][np.newaxis, :]
            + bwdlattice[t + 1][np.newaxis, :],
            axis=1,
        )

    return bwdlattice


def _backward_log_cphmm(log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    bwdlattice = np.empty((n_samples, n_components))
    bwdlattice[-1] = 0.0

    to_clonal = log_transmat[:, 0]
    from_clonal = log_transmat[0]
    recomb_self = np.diag(log_transmat)[1:]

    for t in range(n_samples - 2, -1, -1):
        next_score = framelogprob[t + 1] + bwdlattice[t + 1]
        bwdlattice[t, 0] = _logsumexp_1d(from_clonal + next_score)
        bwdlattice[t, 1:] = np.logaddexp(
            to_clonal[1:] + next_score[0],
            recomb_self + next_score[1:],
        )

    return bwdlattice


def viterbi_log(log_startprob, log_transmat, framelogprob):
    """
    Run Viterbi decoding in log space.
    """
    if _has_cphmm_transition_topology(log_transmat):
        return _viterbi_log_cphmm(log_startprob, log_transmat, framelogprob)

    n_samples, n_components = framelogprob.shape
    viterbi_lattice = np.empty((n_samples, n_components))
    backpointers = np.zeros((n_samples, n_components), dtype=int)
    viterbi_lattice[0] = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        scores = viterbi_lattice[t - 1][:, np.newaxis] + log_transmat
        backpointers[t] = np.argmax(scores, axis=0)
        viterbi_lattice[t] = scores[backpointers[t], np.arange(n_components)] + framelogprob[t]

    state_sequence = np.empty(n_samples, dtype=int)
    state_sequence[-1] = np.argmax(viterbi_lattice[-1])
    logprob = viterbi_lattice[-1, state_sequence[-1]]

    for t in range(n_samples - 2, -1, -1):
        state_sequence[t] = backpointers[t + 1, state_sequence[t + 1]]

    return logprob, state_sequence


def _viterbi_log_cphmm(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    viterbi_lattice = np.empty((n_samples, n_components))
    backpointers = np.zeros((n_samples, n_components), dtype=int)
    viterbi_lattice[0] = log_startprob + framelogprob[0]

    to_clonal = log_transmat[:, 0]
    from_clonal = log_transmat[0]
    recomb_self = np.diag(log_transmat)[1:]
    recomb_indices = np.arange(1, n_components)

    for t in range(1, n_samples):
        prev = viterbi_lattice[t - 1]

        scores_to_clonal = prev + to_clonal
        backpointers[t, 0] = np.argmax(scores_to_clonal)
        viterbi_lattice[t, 0] = scores_to_clonal[backpointers[t, 0]] + framelogprob[t, 0]

        from_clonal_scores = prev[0] + from_clonal[1:]
        self_scores = prev[1:] + recomb_self
        from_clonal_wins = from_clonal_scores >= self_scores
        backpointers[t, 1:] = np.where(from_clonal_wins, 0, recomb_indices)
        viterbi_lattice[t, 1:] = (
            np.maximum(from_clonal_scores, self_scores)
            + framelogprob[t, 1:]
        )

    state_sequence = np.empty(n_samples, dtype=int)
    state_sequence[-1] = np.argmax(viterbi_lattice[-1])
    logprob = viterbi_lattice[-1, state_sequence[-1]]

    for t in range(n_samples - 2, -1, -1):
        state_sequence[t] = backpointers[t + 1, state_sequence[t + 1]]

    return logprob, state_sequence


def compute_log_xi_sum(fwdlattice, log_transmat, bwdlattice, framelogprob):
    """
    Sum expected transition probabilities over adjacent frames in log space.

    The returned values are unnormalized by the sequence log likelihood. Current
    CP-HMM parameter updates only use row-wise ratios, so the shared
    normalization constant cancels out.
    """
    if _has_cphmm_transition_topology(log_transmat):
        return _compute_log_xi_sum_cphmm(
            fwdlattice,
            log_transmat,
            bwdlattice,
            framelogprob,
        )

    n_samples, n_components = framelogprob.shape
    log_xi_sum = np.full((n_components, n_components), -np.inf)

    for t in range(n_samples - 1):
        log_xi_t = (
            fwdlattice[t][:, np.newaxis]
            + log_transmat
            + framelogprob[t + 1][np.newaxis, :]
            + bwdlattice[t + 1][np.newaxis, :]
        )
        log_xi_sum = np.logaddexp(log_xi_sum, log_xi_t)

    return log_xi_sum


def _compute_log_xi_sum_cphmm(fwdlattice, log_transmat, bwdlattice, framelogprob):
    """Sum expected transitions in log space for the CP-HMM topology.

    Vectorized across t: instead of a Python loop with three ``np.logaddexp``
    calls per step, we stack per-step log-contributions into 2-D arrays and
    call ``logsumexp`` once along the time axis. Bit-exactly matches the
    previous loop implementation up to floating-point associativity.
    """
    n_samples, n_components = framelogprob.shape
    log_xi_sum = np.full((n_components, n_components), -np.inf)

    to_clonal = log_transmat[:, 0]
    from_clonal = log_transmat[0]
    recomb_self = np.diag(log_transmat)[1:]
    recomb_indices = np.arange(1, n_components)

    next_score = framelogprob[1:] + bwdlattice[1:]  # (T-1, n_components)

    # Column 0: every state -> clonal.
    # contribs[t, j] = fwdlattice[t, j] + to_clonal[j] + next_score[t, 0]
    col0_contribs = (
        fwdlattice[:-1] + to_clonal[np.newaxis, :] + next_score[:, 0:1]
    )
    log_xi_sum[:, 0] = _logsumexp_axis0(col0_contribs)

    # Row 0, recomb states: clonal -> recomb_j.
    row0_contribs = (
        fwdlattice[:-1, 0:1]
        + from_clonal[np.newaxis, 1:]
        + next_score[:, 1:]
    )
    log_xi_sum[0, 1:] = _logsumexp_axis0(row0_contribs)

    # Recomb self-loops: recomb_j -> recomb_j.
    self_contribs = (
        fwdlattice[:-1, 1:] + recomb_self[np.newaxis, :] + next_score[:, 1:]
    )
    log_xi_sum[recomb_indices, recomb_indices] = _logsumexp_axis0(self_contribs)

    return log_xi_sum
