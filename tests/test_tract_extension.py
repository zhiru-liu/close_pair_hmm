import unittest

import numpy as np
import pandas as pd

from cphmm.tract_extension import (
    ExtensionParams,
    extend_tracts_by_1d_density,
    extend_tracts_by_density,
)


def _cov_grid(stop=10000, step=30):
    """Densely covered 1D sites so coverage never gates the walk."""
    return np.arange(0, stop + 1, step)


def _snp_info(diff_positions, contig="c1", cov_stop=10000):
    """Build a (snp_vec, contigs, locs) tuple from a set of difference positions.

    ``locs`` is the covered-site grid (plus the difference positions, so every
    difference sits on a covered site); ``snp_vec`` is True at the differences.
    """
    cov = _cov_grid(cov_stop)
    locs = np.unique(np.concatenate([cov, np.asarray(diff_positions, dtype=int)]))
    snp_vec = np.isin(locs, np.asarray(diff_positions, dtype=int))
    contigs = np.array([contig] * len(locs))
    return snp_vec, contigs, locs


def _tract(start, end, contig="c1", types=0):
    return {
        "genome1": "A", "genome2": "B", "contig": contig,
        "types": types, "start_site": start, "end_site": end,
        "snp_vec_start": 0, "snp_vec_end": 0,
    }


class TractExtensionTest(unittest.TestCase):
    def test_absorbs_dense_right_flank(self):
        # Detected tract [2000, 4000]; dense 1D flank every 50 bp out to 5000.
        flank = np.arange(4050, 5001, 50)  # 20 differences, 20/kb
        background = np.array([200, 600, 8000, 9000])  # far from the tract
        snp_1d = _snp_info(np.concatenate([flank, background]))
        df = pd.DataFrame([_tract(2000, 4000)])

        out = extend_tracts_by_1d_density(df, snp_1d)

        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(int(row["start_site"]), 2000)        # left flank sparse: no move
        self.assertEqual(int(row["end_site"]), 5000)          # right flank absorbed
        self.assertEqual(int(row["orig_end_site"]), 4000)
        self.assertEqual(int(row["extension_1d_snvs"]), len(flank))
        self.assertEqual(int(row["extension_bp"]), 1000)

    def test_sparse_flank_not_extended(self):
        # Only scattered background differences -> no abnormal density -> no move.
        background = np.array([200, 600, 4500, 8000, 9000])
        snp_1d = _snp_info(background)
        df = pd.DataFrame([_tract(2000, 4000)])

        out = extend_tracts_by_1d_density(df, snp_1d)

        row = out.iloc[0]
        self.assertEqual(int(row["start_site"]), 2000)
        self.assertEqual(int(row["end_site"]), 4000)
        self.assertEqual(int(row["extension_1d_snvs"]), 0)
        self.assertEqual(int(row["extension_bp"]), 0)

    def test_gap_stops_extension(self):
        # Dense run stops at 4500; the next difference is 1500 bp away (> gap_bp).
        flank = np.arange(4050, 4501, 50)  # absorbed
        far = np.array([6000, 6050, 6100])  # beyond the gap, must NOT be absorbed
        snp_1d = _snp_info(np.concatenate([flank, far]))
        df = pd.DataFrame([_tract(2000, 4000)])

        out = extend_tracts_by_1d_density(df, snp_1d)

        row = out.iloc[0]
        self.assertEqual(int(row["end_site"]), 4500)
        self.assertEqual(int(row["extension_1d_snvs"]), len(flank))

    def test_max_extension_cap(self):
        # A very long dense run is capped at max_extension_bp.
        flank = np.arange(4050, 8001, 50)
        snp_1d = _snp_info(np.concatenate([flank, [200, 600]]), cov_stop=12000)
        df = pd.DataFrame([_tract(2000, 4000)])
        params = ExtensionParams(max_extension_bp=1000)

        out = extend_tracts_by_1d_density(df, snp_1d, params=params)

        row = out.iloc[0]
        self.assertLessEqual(int(row["end_site"]) - 4000, 1000)

    def test_merges_tracts_that_meet_after_extension(self):
        # Two tracts whose extensions collide into one continuous dense region.
        # Sparse genome-wide background keeps lambda0 low (realistic).
        between = np.arange(4050, 6001, 50)  # bridges [.,4000] and [6000,.]
        sparse = np.arange(10000, 100001, 5000)
        snp_1d = _snp_info(np.concatenate([between, sparse]), cov_stop=100000)
        df = pd.DataFrame([_tract(2000, 4000), _tract(6000, 8000)])

        out = extend_tracts_by_1d_density(df, snp_1d)

        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(int(row["start_site"]), 2000)
        self.assertEqual(int(row["end_site"]), 8000)

    def test_counts_4d_snvs_in_extension(self):
        flank = np.arange(4050, 5001, 50)
        snp_1d = _snp_info(np.concatenate([flank, [200, 9000]]))
        # 4D differences: one inside the original core (not counted), one in flank.
        snp_4d = _snp_info(np.array([3000, 4500]))
        df = pd.DataFrame([_tract(2000, 4000)])

        out = extend_tracts_by_1d_density(df, snp_1d, snp_info_4d=snp_4d)

        self.assertEqual(int(out.iloc[0]["extension_4d_snvs"]), 1)

    def test_generic_density_driver_absorbs_flank(self):
        # The generic entry point driving on "all" SNVs absorbs a mixed flank
        # (e.g. 2D/3D sites a 1D-only driver would miss) and tallies them.
        flank = np.arange(4050, 5001, 50)
        snp_all = _snp_info(np.concatenate([flank, [200, 9000]]))
        snp_4d = _snp_info(np.array([4500]))  # one 4D diff inside the flank
        df = pd.DataFrame([_tract(2000, 4000)])

        out = extend_tracts_by_density(df, snp_all, count_infos={"4d": snp_4d})

        row = out.iloc[0]
        self.assertEqual(int(row["end_site"]), 5000)
        self.assertEqual(int(row["extension_snvs"]), len(flank))
        self.assertEqual(int(row["extension_4d_snvs"]), 1)
        self.assertIn("extension_snvs", out.columns)

    def test_empty_transfer_df(self):
        snp_1d = _snp_info(np.array([100, 200]))
        out = extend_tracts_by_1d_density(pd.DataFrame(), snp_1d)
        self.assertEqual(len(out), 0)
        self.assertIn("extension_bp", out.columns)

    def test_per_contig_isolation(self):
        # A dense flank on c2 must not influence a tract on c1.
        cov = _cov_grid()
        flank_c2 = np.arange(4050, 5001, 50)
        locs = np.unique(np.concatenate([cov, flank_c2]))
        snp_vec = np.isin(locs, flank_c2)
        # c1: covered grid, no differences; c2: the dense flank
        locs_all = np.concatenate([locs, locs])
        contigs_all = np.array(["c1"] * len(locs) + ["c2"] * len(locs))
        snp_all = np.concatenate([np.zeros(len(locs), bool), snp_vec])
        df = pd.DataFrame([_tract(2000, 4000, contig="c1")])

        out = extend_tracts_by_1d_density(df, (snp_all, contigs_all, locs_all))

        self.assertEqual(int(out.iloc[0]["end_site"]), 4000)  # c2 flank ignored


if __name__ == "__main__":
    unittest.main()
