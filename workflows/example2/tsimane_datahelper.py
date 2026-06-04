"""
Goal is to reproduce the results of the Liu and Good plos bio 2024 results for
one species.
"""
import os
import numpy as np
import pandas as pd
import sys

# ugly workaround for now
sys.path.append('/Users/Device6/Documents/Research/bgoodlab/microbiome_codiv/comigration_metagenomics/')
from utils import snv_utils, metadata_utils
import config

class DataHelper_Hadza_Tsimane:
    def __init__(self, species, drep_summary):
        self.species = species
        self.snv_data = snv_utils.PairwiseSNVHelper(species_name=species, data_batch=config.databatch)
        self.drep_summary = drep_summary.copy()
        self.drep_summary['genome1'] = self.drep_summary['genome1'].str.replace('.fa', '')
        self.drep_summary['genome2'] = self.drep_summary['genome2'].str.replace('.fa', '')
        self.all_pairs = self.load_close_pairs(perc_id_threshold=-1)

        self.genome_len = self.snv_data.get_4D_core_genome_length()

    def load_close_pairs(self, perc_id_threshold=0.5):
        close_ones = self.drep_summary[self.drep_summary['perc_id']>perc_id_threshold]
        unique_pairs = close_ones[['genome1', 'genome2']].drop_duplicates()
        pairs = list(zip(unique_pairs['genome1'], unique_pairs['genome2']))
        return pairs
    
    def get_close_pairs(self, perc_id_threshold=0.5):
        return self.load_close_pairs(perc_id_threshold=perc_id_threshold)
    
    def get_random_pair(self):
        idx = np.random.randint(0, len(self.all_pairs))
        return self.all_pairs[idx]

    def get_pair_snp_info(self, pair):
        snv_locs, cover_vec = self.snv_data.get_snv_vector(pair[0], pair[1])
        snv_vec = np.zeros(cover_vec.shape)
        snv_vec[snv_locs] = 1
        snv_vec = snv_vec[cover_vec]

        indices = self.snv_data.core_4D_coverage.index[cover_vec].copy()
        contigs = indices.get_level_values(0).values.astype(str)
        locs = indices.get_level_values(1).values.astype(int)

        return snv_vec, contigs, locs 
    
    def get_snp_vector(self, pair):
        snv_vec, _, _ = self.get_pair_snp_info(pair)
        return snv_vec