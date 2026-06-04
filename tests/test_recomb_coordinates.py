import unittest

import numpy as np
import pandas as pd

import cphmm.infer_pipelines as infer_pipelines
from cphmm.recomb_inference import prepare_transfer_df


class RecombinationCoordinateTest(unittest.TestCase):
    def _single_transfer(self, starts, ends, contig_lengths, block_size=10):
        return prepare_transfer_df(
            [np.asarray(starts)],
            [np.asarray(ends)],
            contig_lengths,
            block_size,
        )

    def test_inclusive_block_end_maps_to_right_exclusive_snp_end(self):
        transfers = self._single_transfer([2], [4], [100])

        self.assertEqual(int(transfers.loc[0, "snp_vec_start"]), 20)
        self.assertEqual(int(transfers.loc[0, "snp_vec_end"]), 50)

    def test_snp_end_is_clipped_at_contig_end(self):
        transfers = self._single_transfer([8], [9], [95])

        self.assertEqual(int(transfers.loc[0, "snp_vec_start"]), 80)
        self.assertEqual(int(transfers.loc[0, "snp_vec_end"]), 95)

    def test_second_contig_block_coordinates_include_cumulative_offset(self):
        transfers = self._single_transfer([4], [5], [25, 40])

        self.assertEqual(int(transfers.loc[0, "snp_vec_start"]), 35)
        self.assertEqual(int(transfers.loc[0, "snp_vec_end"]), 55)

    def test_infer_pair_reference_annotation_uses_inclusive_end_site(self):
        transfers = pd.DataFrame(
            {
                "snp_vec_start": [1, 1],
                "snp_vec_end": [4, 5],
            }
        )
        contigs = np.array(["ctg"] * 5)
        locs = np.array([10, 20, 30, 40, 50])

        annotated = infer_pipelines.annotate_transfer_reference_coordinates(
            transfers,
            contigs,
            locs,
        )

        self.assertEqual(int(annotated.loc[0, "start_site"]), 20)
        self.assertEqual(int(annotated.loc[0, "end_site"]), 40)
        self.assertEqual(int(annotated.loc[1, "start_site"]), 20)
        self.assertEqual(int(annotated.loc[1, "end_site"]), 50)
        self.assertEqual(annotated.loc[1, "contig"], "ctg")


if __name__ == "__main__":
    unittest.main()
