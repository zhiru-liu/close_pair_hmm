"""Tests for the per-species iterative transfer-length estimation in the QP workflow.

These exercise ``infer_iterative_transfer_length`` / ``_mean_transfer_bp`` in
``workflows/liugood2024_qp/reproduce.py`` with a stubbed ``infer_pairs`` (no catalog
needed), checking the convergence loop, the bp mean, and the per-state array handed to
the HMM for two-clade species.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows" / "liugood2024_qp"
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

import reproduce  # noqa: E402
import cphmm.config  # noqa: E402


def _transfer_dat(specs):
    """Build a transfer_dat from (type, length_in_blocks) specs.

    A length of ``n`` blocks is encoded as block_start=0, block_end=n-1 (both
    inclusive), i.e. ``n * HMM_BLOCK_SIZE`` bp.
    """
    rows = [{"block_start": 0, "block_end": n - 1, "types": t} for t, n in specs]
    return pd.DataFrame(rows, columns=["block_start", "block_end", "types"])


class FakeInferPairs:
    """Stub for infer_pipelines.infer_pairs: records transfer_length, returns scripted data."""

    def __init__(self, scripted):
        self.scripted = scripted          # list of transfer_dat per call
        self.transfer_lengths = []        # transfer_length seen on each call

    def __call__(self, dh, pairs, clade_cutoff_bin=None, iterative=False,
                 transfer_length=None):
        self.transfer_lengths.append(transfer_length)
        td = self.scripted[len(self.transfer_lengths) - 1]
        return pd.DataFrame(), td


class MeanTransferBpTest(unittest.TestCase):
    def test_inclusive_length_times_block_size(self):
        # 150 blocks -> 1500 bp; mean of {1500, 1000} = 1250.
        td = _transfer_dat([(0, 150), (0, 100)])
        self.assertEqual(reproduce._mean_transfer_bp(td), 1250.0)

    def test_split_by_clade_type(self):
        td = _transfer_dat([(0, 150), (1, 200)])
        self.assertEqual(reproduce._mean_transfer_bp(td, type_label=0), 1500.0)
        self.assertEqual(reproduce._mean_transfer_bp(td, type_label=1), 2000.0)

    def test_none_when_empty(self):
        td = _transfer_dat([(0, 150)])
        self.assertIsNone(reproduce._mean_transfer_bp(td, type_label=1))


class IterativeSingleCladeTest(unittest.TestCase):
    def test_converges_and_tracks_lengths(self):
        # pass1: tl=1000 -> mean 1500 (frac 0.5); pass2: tl=1500 -> mean 1450 (frac 0.033 < 0.1).
        fake = FakeInferPairs([_transfer_dat([(0, 150)]), _transfer_dat([(0, 145)])])
        orig = reproduce.infer_pipelines.infer_pairs
        reproduce.infer_pipelines.infer_pairs = fake
        try:
            _, td, final = reproduce.infer_iterative_transfer_length(
                dh=None, pairs=[("a", "b")], species="Test_sp",
                clade_cutoff_bin=None, iterative=False, two_clade=False)
        finally:
            reproduce.infer_pipelines.infer_pairs = orig

        self.assertEqual(fake.transfer_lengths, [1000.0, 1500.0])  # init, then pass1 mean
        self.assertEqual(final, 1450.0)

    def test_stops_at_max_passes(self):
        # Never converges (frac stays large): should run exactly TL_MAX_PASSES.
        scripted = [_transfer_dat([(0, n)]) for n in (300, 100, 300, 100)]
        fake = FakeInferPairs(scripted)
        orig = reproduce.infer_pipelines.infer_pairs
        reproduce.infer_pipelines.infer_pairs = fake
        try:
            reproduce.infer_iterative_transfer_length(
                dh=None, pairs=[("a", "b")], species="Test_sp",
                clade_cutoff_bin=None, iterative=False, two_clade=False)
        finally:
            reproduce.infer_pipelines.infer_pairs = orig
        self.assertEqual(len(fake.transfer_lengths), reproduce.TL_MAX_PASSES)


class IterativeTwoCladeTest(unittest.TestCase):
    def test_per_state_array_and_convergence(self):
        bins = cphmm.config.HMM_PRIOR_BINS
        # pass1: within 1500, between 2000 (frac max 1.0); pass2: within 1480, between 1980 (conv).
        fake = FakeInferPairs([
            _transfer_dat([(0, 150), (1, 200)]),
            _transfer_dat([(0, 148), (1, 198)]),
        ])
        orig = reproduce.infer_pipelines.infer_pairs
        reproduce.infer_pipelines.infer_pairs = fake
        try:
            _, td, final = reproduce.infer_iterative_transfer_length(
                dh=None, pairs=[("a", "b")], species="Two_clade_sp",
                clade_cutoff_bin=bins, iterative=True, two_clade=True)
        finally:
            reproduce.infer_pipelines.infer_pairs = orig

        # First call: 80-length array all at the 1000 init.
        first = fake.transfer_lengths[0]
        self.assertEqual(first.shape, (2 * bins,))
        self.assertTrue(np.all(first == reproduce.TL_INIT))
        # Second call: first 40 = within mean (1500), last 40 = between mean (2000).
        second = fake.transfer_lengths[1]
        self.assertTrue(np.all(second[:bins] == 1500.0))
        self.assertTrue(np.all(second[bins:] == 2000.0))
        self.assertEqual(final, (1480.0, 1980.0))


if __name__ == "__main__":
    unittest.main()
