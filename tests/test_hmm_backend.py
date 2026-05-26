import itertools
import sys
import unittest

import numpy as np
from scipy import special

from cphmm.hmm_backend import (
    backward_log,
    compute_log_xi_sum,
    forward_log,
    viterbi_log,
)
from cphmm.model import ClosePairHMM
from cphmm.utils import log_mask_zero


class HMMBackendTest(unittest.TestCase):
    def setUp(self):
        self.startprob = np.array([0.6, 0.4])
        self.transmat = np.array([[0.7, 0.3], [0.2, 0.8]])
        self.frameprob = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
            ]
        )
        self.log_startprob = log_mask_zero(self.startprob)
        self.log_transmat = log_mask_zero(self.transmat)
        self.framelogprob = np.log(self.frameprob)

    def _path_logprob(self, path):
        logp = self.log_startprob[path[0]] + self.framelogprob[0, path[0]]
        for t in range(1, len(path)):
            logp += self.log_transmat[path[t - 1], path[t]]
            logp += self.framelogprob[t, path[t]]
        return logp

    def _all_path_logprobs(self):
        n_components = self.framelogprob.shape[1]
        paths = list(itertools.product(range(n_components), repeat=self.framelogprob.shape[0]))
        logprobs = np.array([self._path_logprob(path) for path in paths])
        return paths, logprobs

    def test_forward_and_backward_match_bruteforce_likelihood(self):
        paths, logprobs = self._all_path_logprobs()
        expected_logprob = special.logsumexp(logprobs)

        forward_logprob, fwdlattice = forward_log(
            self.log_startprob,
            self.log_transmat,
            self.framelogprob,
        )
        bwdlattice = backward_log(self.log_transmat, self.framelogprob)
        backward_logprob = special.logsumexp(
            self.log_startprob + self.framelogprob[0] + bwdlattice[0]
        )

        self.assertEqual(len(paths), 8)
        self.assertAlmostEqual(forward_logprob, expected_logprob)
        self.assertAlmostEqual(backward_logprob, expected_logprob)
        self.assertEqual(fwdlattice.shape, self.framelogprob.shape)
        self.assertEqual(bwdlattice.shape, self.framelogprob.shape)

    def test_viterbi_matches_bruteforce_best_path(self):
        paths, logprobs = self._all_path_logprobs()
        expected_path = paths[np.argmax(logprobs)]
        expected_logprob = np.max(logprobs)

        logprob, state_sequence = viterbi_log(
            self.log_startprob,
            self.log_transmat,
            self.framelogprob,
        )

        self.assertAlmostEqual(logprob, expected_logprob)
        self.assertEqual(tuple(state_sequence), expected_path)

    def test_log_xi_sum_matches_bruteforce_transition_totals(self):
        _, fwdlattice = forward_log(
            self.log_startprob,
            self.log_transmat,
            self.framelogprob,
        )
        bwdlattice = backward_log(self.log_transmat, self.framelogprob)

        observed = compute_log_xi_sum(
            fwdlattice,
            self.log_transmat,
            bwdlattice,
            self.framelogprob,
        )

        paths, logprobs = self._all_path_logprobs()
        expected = np.full(self.transmat.shape, -np.inf)
        for path, logprob in zip(paths, logprobs):
            for t in range(len(path) - 1):
                i, j = path[t], path[t + 1]
                expected[i, j] = np.logaddexp(expected[i, j], logprob)

        np.testing.assert_allclose(observed, expected)

    def test_close_pair_hmm_fits_and_decodes_without_hmmlearn(self):
        self.assertNotIn("hmmlearn", sys.modules)

        model = ClosePairHMM(
            species_name=None,
            transfer_emissions=np.array([0.35, 0.8]),
            transition_prior=np.array([0.5, 0.5]),
            transfer_rate=0.1,
            clonal_emission=0.02,
            transfer_length=4,
            n_iter=2,
        )
        sequence = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=float).reshape((-1, 1))

        model.fit(sequence)
        logprob, state_sequence = model.decode(sequence)

        self.assertTrue(np.isfinite(logprob))
        self.assertEqual(len(state_sequence), len(sequence))
        self.assertNotIn("hmmlearn", sys.modules)

    def test_sparse_cphmm_topology_with_three_states(self):
        self.startprob = np.array([1.0, 0.0, 0.0])
        self.transmat = np.array(
            [
                [0.75, 0.10, 0.15],
                [0.25, 0.75, 0.00],
                [0.20, 0.00, 0.80],
            ]
        )
        self.frameprob = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.4, 0.7, 0.3],
                [0.2, 0.8, 0.6],
                [0.8, 0.2, 0.4],
            ]
        )
        self.log_startprob = log_mask_zero(self.startprob)
        self.log_transmat = log_mask_zero(self.transmat)
        self.framelogprob = np.log(self.frameprob)

        paths, logprobs = self._all_path_logprobs()
        expected_logprob = special.logsumexp(logprobs)
        expected_path = paths[np.argmax(logprobs)]

        forward_logprob, fwdlattice = forward_log(
            self.log_startprob,
            self.log_transmat,
            self.framelogprob,
        )
        bwdlattice = backward_log(self.log_transmat, self.framelogprob)
        viterbi_logprob, state_sequence = viterbi_log(
            self.log_startprob,
            self.log_transmat,
            self.framelogprob,
        )

        self.assertAlmostEqual(forward_logprob, expected_logprob)
        self.assertAlmostEqual(
            special.logsumexp(self.log_startprob + self.framelogprob[0] + bwdlattice[0]),
            expected_logprob,
        )
        self.assertAlmostEqual(viterbi_logprob, np.max(logprobs))
        self.assertEqual(tuple(state_sequence), expected_path)
        np.testing.assert_allclose(
            compute_log_xi_sum(
                fwdlattice,
                self.log_transmat,
                bwdlattice,
                self.framelogprob,
            )[1, 2],
            -np.inf,
        )


if __name__ == "__main__":
    unittest.main()
