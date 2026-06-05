import sys
import logging
import pandas as pd
import traceback
import time

import cphmm.model as hmm
import cphmm.recomb_inference as ri
import cphmm.config
import cphmm.tract_extension as tract_extension


def init_hmm(species_name, genome_len, block_size, prior_path=None,
             transfer_counts=20., clonal_div=5e-5, transfer_length=1000.):
    # initialize the hmm with default params
    # clonal emission and transfer rate will be fitted per sequence later in the pipeline
    # transfer_length is the expected transferred-segment length in base pairs.
    num_blocks = genome_len / block_size

    transfer_rate = transfer_counts / num_blocks
    transfer_length = transfer_length / block_size
    clonal_emission = clonal_div * block_size
    model = hmm.ClosePairHMM(species_name=species_name, block_size=block_size,
                             transfer_rate=transfer_rate, clonal_emission=clonal_emission,
                             transfer_length=transfer_length, n_iter=5,
                             prior_path=prior_path)
    return model


def annotate_transfer_reference_coordinates(transfer_dat, contigs, locs):
    """
    Add inclusive reference coordinates to transfer rows.

    ``snp_vec_start`` is left-inclusive and ``snp_vec_end`` is right-exclusive.
    Reference start/end sites in the output are both inclusive.
    """
    start_idx = transfer_dat['snp_vec_start'].astype(int)
    end_idx = transfer_dat['snp_vec_end'].astype(int).clip(upper=len(locs)) - 1

    transfer_dat['start_site'] = locs[start_idx.values]
    transfer_dat['end_site'] = locs[end_idx.values]
    transfer_dat['contig'] = contigs[start_idx.values]
    return transfer_dat


def infer_pairs(datahelper, pairs, clade_cutoff_bin=None, iterative=False, n_iter=3,
                transfer_length=1000., extend_with=None, extension_params=None):
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
    :param extend_with: site class to drive 1D-style tract extension, or None
        (default) to disable it and leave the 4D-only behaviour unchanged. When
        set, each pair's detected tracts are post-processed with
        :mod:`cphmm.tract_extension`, widening boundaries into adjacent regions of
        abnormally high SNV density in that site class. ``'1D'`` uses the narrow,
        validated :func:`~cphmm.tract_extension.extend_tracts_by_1d_density`
        (nonsynonymous-driven, emits ``extension_1d_snvs``/``extension_4d_snvs``);
        any other class (e.g. ``'all'``) uses
        :func:`~cphmm.tract_extension.extend_tracts_by_density` so 2D/3D-rich
        flanks are also absorbed (emits ``extension_snvs``/``extension_4d_snvs``).
        Requires the datahelper to support ``get_pair_snp_info(pair,
        site_class=...)``. When enabled, the returned transfer table uses the
        extended schema (reference coordinates + provenance columns) rather than
        the raw block/snp_vec coordinates.
    :param extension_params: optional :class:`cphmm.tract_extension.ExtensionParams`
        overriding the default extension thresholds.
    """
    model = init_hmm(datahelper.species, datahelper.genome_len,
                     cphmm.config.HMM_BLOCK_SIZE,
                     prior_path=getattr(datahelper, 'hmm_prior_path', None),
                     transfer_length=transfer_length)

    pair_dat = pd.DataFrame(columns=['genome1', 'genome2', 'naive_div', 
                                     'est_div', 'genome_len', 'clonal_len'])
    transfer_dats = []
    processed_count = 0
    for i, pair in enumerate(pairs):
        snp_vec, contigs, locs = datahelper.get_pair_snp_info(pair)
        
        try:
            clonal_div, genome_len, clonal_len, transfer_dat = \
                ri.infer(snp_vec, contigs, model, cphmm.config.HMM_BLOCK_SIZE,
                         clade_cutoff_bin=clade_cutoff_bin,
                         iterative=iterative, n_iter=n_iter)
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

        # annotate the reference genome coordinates
        annotate_transfer_reference_coordinates(transfer_dat, contigs, locs)

        if extend_with:
            # Post-process: widen boundaries into adjacent SNV-dense flanks the
            # 4D HMM cannot see. The main snp_vec is already the 4D info.
            driver = datahelper.get_pair_snp_info(pair, site_class=extend_with)
            snp_info_4d = (snp_vec, contigs, locs)
            if extend_with == '1D':
                transfer_dat = tract_extension.extend_tracts_by_1d_density(
                    transfer_dat, driver, params=extension_params,
                    snp_info_4d=snp_info_4d,
                )
            else:
                transfer_dat = tract_extension.extend_tracts_by_density(
                    transfer_dat, driver, params=extension_params,
                    count_infos={'4d': snp_info_4d},
                )

        transfer_dats.append(transfer_dat)

        processed_count += 1
        if processed_count % 100 == 0:
            print("Finished {} out of {} pairs at {}".format(processed_count, len(pairs), time.ctime()))
    if len(transfer_dats) == 0:
        print("No transfer events detected")
        transfer_dat = pd.DataFrame(columns=['genome1', 'genome2', 'snp_vec_start', 'snp_vec_end'])
    else:
        transfer_dat = pd.concat(transfer_dats).reset_index(drop=True)
    return pair_dat, transfer_dat
