import numpy as np
import pandas as pd

import cphmm.seq_manip as seq_manip
import cphmm.config as config

def find_segments(states, target_state=None, target_range=None):
    """
    Find the continuous segments of a target state. By default, all
    nonzero segments will be found.
    :param states: array of non-negative integers, output of HMM
    :param target_state: the state of the segments, if None and no target_range,
    find nonzero segments
    :param target_range: the range of accepted states, left inclusive
    :return: start and end indices of segments, *inclusive*
    """
    if target_state is not None:
        states = states == target_state
        states = states.astype(int)
    elif target_range is not None:
        if len(target_range) != 2:
            raise ValueError("Please supply the desired range of states as [min, max]")
        states = (states >= target_range[0]) & (states < target_range[1])
        states = states.astype(int)
    else:
        # find all non zero segments
        for n in np.unique(states):
            if n not in [0, 1]:
                import warnings
                warnings.warn(
                    "Treating all nonzero states as recombined regions", RuntimeWarning)
                states = states.copy()
                states[states != 0] = 1
                break
    # padding to take care of end points
    padded = np.empty(len(states) + 2)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = states
    diff = padded[1:] - padded[:-1]
    ups = np.nonzero(diff == 1)[0]
    downs = np.nonzero(diff == -1)[0]
    return ups, downs - 1


def decode_and_count_transfers(sequence, model, sequence_with_snps=None, need_fit=True, clade_cutoff_bin=None,
                                index_offset=0):
    """
    Use a HMM to decode the sequence and eventually compute the number of runs, as well as
    and estimate for wall clock T. Can distinguish different types of transfers using clade_cutoff_bin.
    Does not fit every sequence!
    :param sequence: The sequence in blocks
    :param model: The hidden markov model
    :param sequence_with_snps: The sequence with actual snp counts. Could be different from the 0/1 sequence for fitting
    :param clade_cutoff_bin: For determining whether transfer is within clade or between clade
    based on the inferred bin in the empirical distribution. Within clade range from state 1 to
    state clade_cutoff_bin, inclusive.
    :return: triplet of start and end indices (inclusive) as well as the clonal sequence
    """
    if need_fit:
        model.fit(sequence)
    _, states = model.decode(sequence)
    if sequence_with_snps is not None:
        clonal_seq = sequence_with_snps[states == 0]
    else:
        clonal_seq = sequence[states == 0]
    # clonal_len = len(clonal_seq)

    if clade_cutoff_bin is not None:
        # compute segments lengths for desire states
        starts = []
        ends = []
        for limits in [[1, clade_cutoff_bin], [clade_cutoff_bin, np.inf]]:
            tmp_starts, tmp_ends = find_segments(states, target_state=None, target_range=limits)
            if len(tmp_starts) != 0:
                # for taking care of contigs
                tmp_starts += index_offset
                tmp_ends += index_offset
            starts.append(tmp_starts)
            ends.append(tmp_ends)
    else:
        tmp_starts, tmp_ends = find_segments(states)
        if len(tmp_starts) != 0:
            tmp_starts += index_offset
            tmp_ends += index_offset
        # put in [] for compatibility with the above case
        starts = [tmp_starts]
        ends = [tmp_ends]
    return starts, ends, clonal_seq


def estimate_clonal_divergence(clonal_sequence, seq_blk_size=config.HMM_BLOCK_SIZE, est_blk_size=1000):
    """
    Correcting for missed recombination events by coarse-graining the clonal sequence into blocks
    Then count the fraction of blocks with snps
    Undetected recombination events will not inflate clonal divergence in this case
    :param clonal_sequence: Sequence of the clonal region of a genome after HMM decoding
    :param seq_blk_size: the size for each of the block in clonal_sequence, default the same as HMM
    :return: estimated clonal divergence
    """
    naive_div = np.sum(clonal_sequence) / float(len(clonal_sequence) * seq_blk_size)

    new_blk_size = int(est_blk_size / seq_blk_size)
    blk_seq = seq_manip.to_block(clonal_sequence, new_blk_size)
    est_div = np.sum(blk_seq[blk_seq<=2]) / float(np.sum(blk_seq<=2) * new_blk_size * seq_blk_size)
    return naive_div, est_div


def _get_contig_lengths(contigs):
    unique_contigs = pd.unique(contigs)
    contig_lengths = [np.sum(contigs==contig) for contig in unique_contigs]
    return contig_lengths


def prepare_transfer_df(starts, ends, contig_lengths, block_size):
    """
    Prepare a DataFrame for transfer results
    :param starts: list of arrays of start indices, produced in after aggregating
    over decode_and_count_transfers
    :param ends: list of arrays of end indices
    :param contig_lengths: list of contig lengths
    :param block_size: size of the block
    :return: DataFrame with columns starts, ends, types
    """
    df_transfers = pd.DataFrame(
        columns=['block_start', 'block_end', 'snp_vec_start', 
                 'snp_vector_end', 'types'])
    df_transfers['block_start'] = np.concatenate(starts)
    df_transfers['block_end'] = np.concatenate(ends)
    df_transfers['types'] = np.concatenate(
        [np.repeat(i, len(x)) for i, x in enumerate(starts)])

    for i, row in df_transfers.iterrows():
        snp_vec_start = seq_manip.block_loc_to_genome_loc(row['starts'], contig_lengths, block_size, left=True)
        snp_vec_end = seq_manip.block_loc_to_genome_loc(row['ends'], contig_lengths, block_size, left=False)
        df_transfers.loc[i, 'snp_vec_start'] = snp_vec_start
        df_transfers.loc[i, 'snp_vec_end'] = snp_vec_end
    return df_transfers
    

def infer(snp_vec, contigs, model, block_size, clade_cutoff_bin=None):
    """
    Accumulate the results of above function for all contigs
    :param snp_vec: The full snp vector for a given pair of QP samples
    :param contigs: Array of same length as snp_vec, containing the contig name of each site
    :param model: CP-HMM model
    :param block_size: size of the block
    :param clade_cutoff_bin: For determining whether transfer is within clade or between clade; see prior estimation scripts
    :return: tuple of starts and ends of transfers (in blocks), and # transferred snps, # clonal snps, genome length and
    clonal region length
    """
    all_starts = []
    all_ends = []
    clonal_seqs = []
    index_offset = 0
    for contig in pd.unique(contigs):
        # iterate over contigs; similar to run length dist calculation
        subvec = snp_vec[contigs==contig]
        blk_seq = seq_manip.to_block(subvec, block_size).reshape((-1, 1))
        # to reduce effect of correlated mutation over short distances
        blk_seq_fit = (blk_seq > 0).astype(float)
        if np.sum(blk_seq) == 0:
            # some time will have an identical contig
            # have to skip otherwise will mess up hmm
            starts = [np.array([])]
            ends = [np.array([])]
            clonal_seq = blk_seq  # full sequence is clonal
        else:
            starts, ends, clonal_seq = decode_and_count_transfers(
                blk_seq_fit, model, sequence_with_snps=blk_seq, index_offset=index_offset,
                clade_cutoff_bin=clade_cutoff_bin)
        all_starts.append(starts)
        all_ends.append(ends)
        clonal_seqs.append(clonal_seq)
        model.reinit_emission_and_transfer_rates()
        index_offset += len(blk_seq)

    # group transfers of the same type together over contigs
    num_types = len(all_starts[0])
    starts = []
    ends = []
    for i in range(num_types):
        starts.append(np.concatenate([s[i] for s in all_starts]))
        ends.append(np.concatenate([s[i] for s in all_ends]))

    full_clonal_seq = np.concatenate(clonal_seqs)
    clonal_div = estimate_clonal_divergence(full_clonal_seq, block_size)
    total_clonal_len = len(full_clonal_seq) * block_size

    contig_lengths = _get_contig_lengths(contigs)
    transfer_df = prepare_transfer_df(starts, ends, contig_lengths, block_size)

    return clonal_div, len(snp_vec), total_clonal_len, transfer_df