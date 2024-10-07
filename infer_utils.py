import sys
import os
import logging
import json
import pickle
import numpy as np
import pandas as pd
import traceback

import cphmm.cphmm as hmm
import cphmm.recomb_inference as ri
import cphmm.config


def init_hmm(species_name, genome_len, block_size):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    num_blocks = genome_len / block_size
    transfer_counts = 20.
    clonal_div = 5e-5
    # transfer_length = 10000.  # default 1000
    transfer_length = 2810.  # default 1000

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    model = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5)
    return model 


def infer_pairs(datahelper):
    model = init_hmm(datahelper.species_name, datahelper.genome_len, 
                     cphmm.config.HMM_BLOCK_SIZE)
    good_pairs = datahelper.get_close_pairs()

    pair_dat = pd.DataFrame(columns=['genome1', 'genome2', 'naive_div', 'est_div', 'genome_len', 'clonal_len'])
    transfer_dats = []
    for i, pair in enumerate(good_pairs):
        transfer_dat = pd.DataFrame(columns=['genome1', 'genome2', 'start_block', 'end_block', 'start_site', 'end_site'])
        snp_vec = datahelper.get_snp_vector(pair)
        contigs = datahelper.get_contigs(pair)
        
        try:
            clonal_div, genome_len, clonal_len, transfer_dat = \
                ri.infer(snp_vec, contigs, model, cphmm.config.HMM_BLOCK_SIZE)
        except:
            e = sys.exc_info()[0]
            tb = traceback.format_exc()
            print(pair)
            print(tb)
            raise e
        naive_div, est_div = clonal_div
        pair_dat.loc[i] = [pair[0], pair[1], naive_div, est_div, genome_len, clonal_len]
        transfer_dat['genome1'] = pair[0]
        transfer_dat['genome2'] = pair[1]
        transfer_dats.append(transfer_dat)
        # TODO: annotate the reference genome coordinates
        # transfer_dat['start_site'] = datahelper.snp_vec_to_genome_loc(pair, transfer_dat['snp_vec_start'])
        # transfer_dat['end_site'] = datahelper.snp_vec_to_genome_loc(pair, transfer_dat['snp_vec_end'])

        processed_count += 1
        if processed_count % 100 == 0:
            logging.info("Finished %d out of %d pairs" % (processed_count, len(good_pairs)))

    transfer_dat = pd.concat(transfer_dats)
    return pair_dat, transfer_dat