import sys
import logging
import pandas as pd
import traceback
import time

import cphmm.model as hmm
import cphmm.recomb_inference as ri
import cphmm.config


def init_hmm(species_name, genome_len, block_size):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    num_blocks = genome_len / block_size
    transfer_counts = 20.
    clonal_div = 5e-5
    transfer_length = 1000.  # default 1000

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    model = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5)
    return model 


def infer_pairs(datahelper, pairs, clade_cutoff_bin=None):
    """
    Infer the clonal divergence and transfer events for all close pairs in the datahelper

    The datahelper should have the following methods / attributes:
    - species_name: for loading the relevant species-specific data
    - genome_len: the length of the core reference genome; for initializing the 
    HMM
    - get_pair_snp_info: a method that returns the SNP vector and location indices 
    for a given pair. The SNP vector is a bool array for all the SNV differences
    between a given pair. The contig names are the same length as the SNP vector
    and indicate which contig each site belongs to. The locs are the indices of
    the SNV sites in the contigs.

    :param datahelper: an object with the above methods / attributes
    :param pairs: a list of pairs of sample names to infer
    """
    model = init_hmm(datahelper.species, datahelper.genome_len, 
                     cphmm.config.HMM_BLOCK_SIZE)

    pair_dat = pd.DataFrame(columns=['genome1', 'genome2', 'naive_div', 
                                     'est_div', 'genome_len', 'clonal_len'])
    transfer_dats = []
    processed_count = 0
    for i, pair in enumerate(pairs):
        snp_vec, contigs, locs = datahelper.get_pair_snp_info(pair)
        
        try:
            clonal_div, genome_len, clonal_len, transfer_dat = \
                ri.infer(snp_vec, contigs, model, cphmm.config.HMM_BLOCK_SIZE, clade_cutoff_bin=clade_cutoff_bin)
        except:
            e = sys.exc_info()[0]
            tb = traceback.format_exc()
            print(pair)
            print(tb)
            raise e
        print("Inferred pair %s" % str(pair))
        naive_div, est_div = clonal_div
        pair_dat.loc[i] = [pair[0], pair[1], naive_div, est_div, genome_len, clonal_len]
        transfer_dat['genome1'] = pair[0]
        transfer_dat['genome2'] = pair[1]
        transfer_dats.append(transfer_dat)

        # annotate the reference genome coordinates
        transfer_dat['start_site'] = locs[transfer_dat['snp_vec_start'].astype(int).values]
        transfer_dat['end_site'] = locs[transfer_dat['snp_vec_end'].astype(int).values]
        transfer_dat['contig'] = contigs[transfer_dat['snp_vec_start'].astype(int).values]

        processed_count += 1
        if processed_count % 100 == 0:
            print("Finished {} out of {} pairs at {}".format(processed_count, len(pairs), time.ctime()))
    if len(transfer_dats) == 0:
        print("No transfer events detected")
        transfer_dat = pd.DataFrame(columns=['genome1', 'genome2', 'snp_vec_start', 'snp_vec_end'])
    else:
        transfer_dat = pd.concat(transfer_dats).reset_index(drop=True)
    return pair_dat, transfer_dat