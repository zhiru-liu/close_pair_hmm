"""Codon site-type / mutation-effect annotation.

Vendored from the LiuGood-2024-SNVs repository
(https://github.com/zhiru-liu/LiuGood-2024-SNVs, commit 3d9949f), the author's
own MIT-licensed code, so the CP-HMM LiuGood2024 workflows are self-contained.
Unmodified except for this header.
"""
import numpy as np
import os
import pandas as pd
from pathlib import Path

def generate_alt_codons(codon, loc, alt):
    if loc not in [0, 1, 2]:
        raise ValueError("Location must be 0, 1, or 2")
    prefix = codon[:loc]
    suffix = codon[loc + 1:]

    new_codon = prefix + alt + suffix
    return new_codon


def generate_all_alt_codons(codon, loc):
    basepairs = ['A', 'T', 'C', 'G']
    if loc not in [0, 1, 2]:
        raise ValueError("Location must be 0, 1, or 2")
    prefix = codon[:loc]
    suffix = codon[loc+1:]
    res = []
    for i in range(4):
        new_codon = prefix + basepairs[i] + suffix
        res.append(new_codon)
    return res


def compute_mut_type_dicts():
    basepairs = ['A', 'T', 'C', 'G']
    site_type_dict = {1: '1D', 2:'2D', 3:'3D', 4:'4D'}
    codon_type_dict = {}
    codon_mut_dict = {}
    for codon in CODON_DICT:
        types = []
        muts = []
        for i in range(3):
            aa = CODON_DICT[codon]
            alt_codons = generate_all_alt_codons(codon, i)
            alt_type = {}
            for j, alt in enumerate(alt_codons):
                alt_aa = CODON_DICT[alt]
                if alt_aa==aa:
                    alt_type[basepairs[j]] = 's'
                elif (alt_aa=='*') and (aa!='*'):
                    alt_type[basepairs[j]] = 'n'
                elif (aa=='*') and (alt_aa!='*'):
                    # stop codon got replaced by aa
                    alt_type[basepairs[j]] = 'nn'
                else:
                    alt_type[basepairs[j]] = 'm'
            alt_aas = [aa==CODON_DICT[alt] for alt in alt_codons]
            num_same = np.sum(alt_aas)
            types.append(site_type_dict[num_same])
            muts.append(alt_type)
        codon_type_dict[codon] = types
        codon_mut_dict[codon] = muts
    return codon_type_dict, codon_mut_dict


def codon_mut_dict_to_array(codon_mut_dict):
    basepairs = ['A', 'T', 'C', 'G']
    codon_mut_array = {}
    for codon in codon_mut_dict:
        res = np.empty((3, 4)).astype(str)
        for j in range(3):
            for k in range(4):
                res[j, k] = codon_mut_dict[codon][j][basepairs[k]]
        codon_mut_array[codon] = res
    return codon_mut_array


def annotate_site_types(seq, strand, return_full_mut=False):
    # works for one gene
    assert(len(seq)%3 == 0) # has to be multiple of codon length
    assert(strand in ['+', '-'])
    types = []
    muts = []
    if strand == '-':
        seq = seq.reverse_complement()
    for i in range(len(seq) // 3):
        codon = str(seq[i*3:i*3+3])
        # check if the sequence contain ambiguous nucleotide
        bad_nn = False
        for nn in codon:
            if nn not in ALLOWED_BASEPAIRS:
                bad_nn = True
        if bad_nn:
            types.append(['NA', 'NA', 'NA'])
            muts.append(np.full((3, 4), 'NA'))
        else:
            types.append(CODON_TYPE_DICT[codon])
            muts.append(CODON_MUT_ARRAY[codon])
    res = np.hstack(types)
    mut_array = np.vstack(muts)
    if strand == '-':
        res = res[::-1]
        mut_array = np.flip(mut_array, axis=0)
        mut_array = mut_array[:, [1, 0, 3, 2]]  # swap A and T, C and G
    if return_full_mut:
        return res, mut_array
    else:
        return res


def annotate_sequence_site_types(sequence, gene_df):
    """
    Requirements for gene_df: has columns Type, Start, End, Strand, Gene ID
    Only genes whose lengths are multiples of 3 are considered
    For overlapping genes, the last gene's result is recorded
    :param sequence: Sequence of a single contig
    :param gene_df: Gene annotation dataframe for the contig
    :return: an array of site types, an array of gene names, an array of mutation types and an array of gene counts
    """

    # initialize all the data arrays
    contig_len = len(sequence)
    variants = np.full(contig_len, 'NA')
    gene_counts = np.zeros(contig_len)
    gene_names = np.full(contig_len, 'NA')
    muts = np.full((contig_len, 4), 'NA')

    # work with one contig at a time
    cds_genes = gene_df[gene_df['Type'] == 'CDS']
    # annotate the ref genome contig
    for idx, row in cds_genes.iterrows():
        start = row['Start']
        end = row['End']
        if ((end - start + 1) % 3) != 0:
            continue # require all coding genes to have the whole reading frames; can improve later
        subseq = sequence[start - 1:end]
        types, mut_array = annotate_site_types(subseq, str(row['Strand']), return_full_mut=True)
        gene_name = row['Gene ID']
        for i, var_type in enumerate(types):
            variants[i + start - 1] = var_type
            gene_names[i + start - 1] = gene_name
            muts[i + start - 1, :] = mut_array[i, :]
        gene_counts[start - 1:end] += 1
    return variants, gene_names, muts, gene_counts

def annotate_sequence_site_types_to_df(sequence, gene_df):
    """
    Wrapper function for annotate_sequence_site_types
    """
    variants, gene_names, muts, gene_counts = annotate_sequence_site_types(sequence, gene_df)
    res = pd.DataFrame({'Site Type': variants, 'Gene Name': gene_names, 'Gene Count': gene_counts})
    for i in range(4):
        base = ['A', 'T', 'C', 'G'][i]
        res['{} Mut'.format(base)] = muts[:, i]
    res['Location'] = np.arange(1, len(sequence) + 1)
    res['Ref Base'] = list(sequence)
    return res


ALLOWED_BASEPAIRS = {'A', 'T', 'C', 'G'}

# Load codon definitions from file
CODON_FILE = Path(__file__).resolve().parent / 'codons.csv'
CODONS = pd.read_csv(CODON_FILE)
CODON_DICT = {row['Codon']: row['AA'] for _, row in CODONS.iterrows()}

# Compute mutation dictionaries/arrays
CODON_TYPE_DICT, CODON_MUT_DICT = compute_mut_type_dicts()
CODON_MUT_ARRAY = codon_mut_dict_to_array(CODON_MUT_DICT)