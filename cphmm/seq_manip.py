import numpy as np
import pandas as pd
import cphmm.config as config

def length_to_num_blocks(seq_len, block_size):
    """
    Compute the number of blocks needed to cover a sequence of length seq_len
    """
    # Magical formula that works for all edge cases
    return (seq_len + block_size - 1) // block_size


def to_block(bool_array, block_size):
    
    """
    Converting a boolean array into blocks of True counts. The last
    block could be shorter than block_size
    :param bool_array:
    :param block_size:
    :return: An array of counts of Trues in blocks
    """
    # coarse-graining the bool array (snp array) into blocks
    num_blocks = length_to_num_blocks(len(bool_array), block_size)
    bins = np.arange(0, num_blocks * block_size + 1, block_size)
    counts, _ = np.histogram(np.nonzero(bool_array), bins)
    return counts


def to_block_seq_all_contigs(bool_array, contigs, block_size):
    """
    For sequence with multiple contigs / chromosomes, each sub sequence is converted to blocks separately
    and concatenated together. Main usage is to use block sequence index compatible saved in
    _decode_and_count_transfers
    :param bool_array:
    :param contigs: array of chromosome names. Same length as bool_array
    :param block_size:
    :return: An array of concatenated block sequences
    """
    all_seqs = []
    for chromo in pd.unique(contigs):
        # iterate over contigs; similar to run length dist calculation
        subvec = bool_array[contigs==chromo]
        all_seqs.append(to_block(subvec, block_size).reshape((-1, 1)))
    return np.concatenate(all_seqs)


def compute_clonal_fraction(snp_array, block_size):
    """
    Compute the fraction of non-zero blocks in the snp array
    """
    snp_blocks = to_block(snp_array, block_size)
    nonzeros = np.sum(snp_blocks == 0)
    return float(nonzeros) / len(snp_blocks)


def block_loc_to_genome_loc(block_loc, contig_lengths, block_size, left=True):
    """
    Hacky function to translate a location coordinate in blocks to the correct genome location
    :param block_loc: location in block coordinate
    :param contig_lengths: a list of contig lengths, can be computed by relevant function in snp_data_utils
    :param block_size:
    :param left: Whether returning the location of the left end of the block or the right end
    :return:
    """
    contig_blk_lens = [length_to_num_blocks(ctg_len, block_size) for ctg_len in contig_lengths]
    cum_blk = np.insert(np.cumsum(contig_blk_lens), 0, 0)
    cum_genome = np.insert(np.cumsum(contig_lengths), 0, 0)
    contig_id = np.nonzero(block_loc < cum_blk)[0][0] - 1
    blk_loc_in_ctg = block_loc - cum_blk[contig_id]
    if left:
        return cum_genome[contig_id] + blk_loc_in_ctg * block_size
    else:
        # right end of the block, exclusive
        return min(cum_genome[contig_id] + (blk_loc_in_ctg + 1) * block_size, cum_genome[contig_id + 1])