"""
Goal is to reproduce the results of the Liu and Good plos bio 2024 results for
one species.
"""
import os
import numpy as np
import pandas as pd
import dNdS_analysis.utils.snv_utils as snv_utils

class DataHelper_plosbio24:
    def __init__(self, species):
        self.snv_data = snv_utils.SNVHelper(species, compute_bi_snvs=False, 
                                            mask_multi_sites=True, annotate=True)
        self.species = species
        self.genome_len = self.snv_data.core_4D.sum()
        self.close_pairs = self.load_close_pairs()

    def get_example_pairs(self):
         # choose only two example pairs highlighted in fig2
         pair1_idx = [54, 238]
         pair1 = self.snv_data.samples[pair1_idx].values
         pair2_idx = [39, 74]
         pair2 = self.snv_data.samples[pair2_idx].values
         return [pair1, pair2]
    
    def load_close_pairs(self):
        # load plos bio paper si table
        # get current folder path
        folder_path = os.path.dirname(os.path.abspath(__file__))
        # load the table
        infered_transfers = pd.read_csv(os.path.join(folder_path, 'journal.pbio.3002472.s003.csv'), low_memory=False)

        infered_transfers = infered_transfers[infered_transfers['Species name']==self.species]
        # get the analyzed pairs of samples
        unique_pairs = infered_transfers[['Sample 1', 'Sample 2']].drop_duplicates()
        # zip into pairs
        pairs = list(zip(unique_pairs['Sample 1'], unique_pairs['Sample 2']))
        return pairs
    
    def get_close_pairs(self):
        return self.close_pairs

    def get_pair_snp_info(self, pair):
        snv_diffs = self.snv_data.compute_pairwise_snvs(pair[0], pair[1])
        coverage = self.snv_data.compute_pairwise_coverage(pair[0], pair[1])

        # some index gymnastics to get the snp vector
        snp_vec = np.zeros(shape=self.snv_data.core_to_snvs.shape[0], dtype=bool)
        # first, set the snp differences to sites that have SNVs in the population
        snp_vec[self.snv_data.core_to_snvs] = snv_diffs
        # then, keep only 4D core sites that have coverage in both samples
        snp_vec = snp_vec[self.snv_data.core_4D & coverage]

        # get the indices of these sites in the contigs
        indices = coverage[self.snv_data.core_4D & coverage].index
        # contigs are the first level of the index
        contigs = indices.get_level_values(0).values.astype(str)
        locs = indices.get_level_values(1).values.astype(int)

        return snp_vec, contigs, locs 