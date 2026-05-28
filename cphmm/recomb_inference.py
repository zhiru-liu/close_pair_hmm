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
                # import warnings
                # warnings.warn(
                #     "Treating all nonzero states as recombined regions", RuntimeWarning)
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
                 'snp_vec_end', 'types'], dtype=int)
    df_transfers['block_start'] = np.concatenate(starts)
    df_transfers['block_end'] = np.concatenate(ends)
    df_transfers['types'] = np.concatenate(
        [np.repeat(i, len(x)) for i, x in enumerate(starts)])

    for i, row in df_transfers.iterrows():
        snp_vec_start = seq_manip.block_loc_to_genome_loc(row['block_start'], contig_lengths, block_size, left=True)
        # right end is exclusive
        snp_vec_end = seq_manip.block_loc_to_genome_loc(row['block_end'], contig_lengths, block_size, left=False)
        df_transfers.loc[i, 'snp_vec_start'] = snp_vec_start
        df_transfers.loc[i, 'snp_vec_end'] = snp_vec_end
    return df_transfers
    

def _single_pass(snp_vec, contigs, model, block_size, clade_cutoff_bin=None):
    """One full pass of per-contig fit+decode across all contigs.

    Returns the raw per-contig segment lists plus the pooled clonal block
    sequence, which the iterative path needs to re-estimate the clonal
    emission rate before the next pass.
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
        if (np.sum(blk_seq) == 0):
            # some time will have an identical contig
            # have to skip otherwise will mess up hmm
            num_types = 2 if clade_cutoff_bin is not None else 1
            starts = [np.array([], dtype=int) for _ in range(num_types)]
            ends = [np.array([], dtype=int) for _ in range(num_types)]
            clonal_seq = blk_seq  # full sequence is clonal
        elif len(blk_seq) < config.HMM_MIN_SEQ_LEN:
            # too short to make any inference
            # add the length of the contig to the offset
            # but not including the sequence as clonal
            index_offset += len(blk_seq)
            continue
        else:
            starts, ends, clonal_seq = decode_and_count_transfers(
                blk_seq_fit, model, sequence_with_snps=blk_seq, index_offset=index_offset,
                clade_cutoff_bin=clade_cutoff_bin)
        all_starts.append(starts)
        all_ends.append(ends)
        clonal_seqs.append(clonal_seq)
        model.reinit_emission_and_transfer_rates()
        index_offset += len(blk_seq)

    full_clonal_seq = np.concatenate(clonal_seqs) if clonal_seqs else np.array([])
    return all_starts, all_ends, full_clonal_seq


def _bernoulli_clonal_emission_from_seq(clonal_seq, min_emission):
    """Bernoulli ML estimator for the clonal-state emission.

    ``clonal_seq`` is the pooled block-sum SNP counts in regions the HMM just
    decoded as clonal. The model treats clonal emission as the probability
    that a block has any SNP (matches ``blk_seq_fit = (blk_seq > 0)``), so the
    natural estimator is the mean of that binary indicator over clonal blocks.

    Floored at ``min_emission`` so the rate never collapses to zero on an
    accidentally SNP-free clonal call (legacy used the same safety net).
    """
    if len(clonal_seq) == 0:
        return min_emission
    indicators = (np.asarray(clonal_seq).reshape(-1) > 0).astype(float)
    rate = float(indicators.mean())
    if rate < min_emission:
        rate = min_emission
    return rate


def infer(snp_vec, contigs, model, block_size, clade_cutoff_bin=None,
          iterative=False, n_iter=3):
    """
    Accumulate per-contig HMM inference into a per-pair recombination summary.

    :param snp_vec: The full snp vector for a given pair of QP samples
    :param contigs: Array of same length as snp_vec, containing the contig name of each site
    :param model: CP-HMM model
    :param block_size: size of the block
    :param clade_cutoff_bin: For determining whether transfer is within clade or between clade; see prior estimation scripts
    :param iterative: When True, run ``n_iter`` outer passes. After each pass,
        re-estimate the clonal-state emission from the pooled clonal blocks
        that the HMM itself called clonal, then refit. Mirrors the legacy
        ``_fit_and_count_transfers_iterative`` from microbiome_evolution but
        pools across contigs (per-pair) rather than iterating per-contig.
        The model's ``init_clonal_emission`` is restored before returning.
    :param n_iter: Number of outer iterations when ``iterative=True``. Legacy
        default was 3.
    :return: tuple of starts and ends of transfers (in blocks), and # transferred snps, # clonal snps, genome length and
    clonal region length
    """
    saved_init = model.init_clonal_emission
    try:
        if iterative:
            for _ in range(n_iter):
                all_starts, all_ends, full_clonal_seq = _single_pass(
                    snp_vec, contigs, model, block_size,
                    clade_cutoff_bin=clade_cutoff_bin,
                )
                new_emission = _bernoulli_clonal_emission_from_seq(
                    full_clonal_seq, model.min_clonal_emissions
                )
                # Update the value reinit_emission_and_transfer_rates() resets
                # to, so the next outer pass starts EM from the new emission.
                model.init_clonal_emission = new_emission
                model.reinit_emission_and_transfer_rates()
        else:
            all_starts, all_ends, full_clonal_seq = _single_pass(
                snp_vec, contigs, model, block_size,
                clade_cutoff_bin=clade_cutoff_bin,
            )
    finally:
        model.init_clonal_emission = saved_init
        model.reinit_emission_and_transfer_rates()

    # group transfers of the same type together over contigs
    num_types = len(all_starts[0]) if all_starts else (
        2 if clade_cutoff_bin is not None else 1
    )
    starts = []
    ends = []
    for i in range(num_types):
        if all_starts:
            starts.append(np.concatenate([s[i] for s in all_starts]))
            ends.append(np.concatenate([s[i] for s in all_ends]))
        else:
            starts.append(np.array([], dtype=int))
            ends.append(np.array([], dtype=int))

    clonal_div = estimate_clonal_divergence(full_clonal_seq, block_size) if len(full_clonal_seq) else (0.0, 0.0)
    total_clonal_len = len(full_clonal_seq) * block_size

    contig_lengths = _get_contig_lengths(contigs)
    transfer_df = prepare_transfer_df(starts, ends, contig_lengths, block_size)

    return clonal_div, len(snp_vec), total_clonal_len, transfer_df
