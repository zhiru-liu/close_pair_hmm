"""
Codes for computing the prior for transfer divergence distribution

The prior is computed by sampling blocks of length l from a random pair of 
genomes and computing the mean divergence in the block.

Requires a DataHelper object dh that has the following methods:
- get_random_pair: returns a random pair of genomes
- get_snp_vector(pair): returns the SNP vector for the pair

"""
import os
import numpy as np
import cphmm.config as config


def get_transfer(datahelper, l):
    """
    Sample a block of length l from a random pair
    """
    # good_idxs = dh.get_single_subject_idxs()
    # pair = random.sample(good_idxs, 2)
    pair = datahelper.get_random_pair()
    snp_vec, _ = datahelper.get_snp_vector(pair)
    div = np.mean(snp_vec)
    start_idx = np.random.randint(0, len(snp_vec) - l)
    return snp_vec[start_idx:start_idx + l], div


def sample_blocks(datahelper, num_samples=5000, block_size=1000):
    """
    Samples num_samples blocks of length block_size from random pairs of genomes
    and computes the mean divergence in each block.
    Returns two arrays: local_divs and genome_divs. local_divs contains the mean
    divergence in each block, and genome_divs contains the divergence of the 
    genome.
    """
    local_divs = []
    genome_divs = []
    for i in range(num_samples):
        seq, genome_div = get_transfer(datahelper, block_size)
        local_div = np.mean(seq)
        local_divs.append(local_div)
        genome_divs.append(genome_div)
    local_divs = np.array(local_divs)
    genome_divs = np.array(genome_divs)
    return local_divs, genome_divs


def compute_div_histogram(local_divs, genome_divs, num_bins=config.HMM_PRIOR_BINS, 
                          separate_clades=True, clade_cutoff=0.03):
    """
    Bin the local_divs and genome_divs into num_bins bins and return the 
    empirical distribution of local_divs. The center of the bin is returned.

    If separate_clades is True, the distribution is split into two parts: 
    within-clade and between-clade. 
    The clade_cutoff parameter determines the genome_div threshold for 
    separating the clades.
    """
    bins = np.linspace(0, max(local_divs), num_bins + 1)
    divs = (bins[:-1] + bins[1:]) / 2
    if separate_clades:
        # For B. vulgatus and others that need to classify the transfer
        # Dists of within-clade and between-clade will be concat together
        within_counts, _ = np.histogram(local_divs[genome_divs <= clade_cutoff],
                                         bins=bins)
        between_counts, _ = np.histogram(local_divs[genome_divs > clade_cutoff],
                                          bins=bins)
        divs = np.concatenate([divs, divs])
        counts = np.concatenate([within_counts, between_counts])
    else:
        counts, _ = np.histogram(local_divs, bins=bins)
    return divs, counts


def save_prior(divs, counts, name):
    """
    Save the prior to a csv file
    """
    save_path = os.path.join(config.HMM_PRIOR_PATH, name + '.csv')
    np.savetxt(save_path, np.vstack([divs, counts]))