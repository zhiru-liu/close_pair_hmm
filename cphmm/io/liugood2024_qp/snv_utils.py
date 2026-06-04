"""SNV catalog reader for the LiuGood2024 quasi-phaseable (QP) catalog format.

Vendored from the LiuGood-2024-SNVs repository
(https://github.com/zhiru-liu/LiuGood-2024-SNVs, commit 3d9949f), the author's
own MIT-licensed code, so the CP-HMM LiuGood2024 workflows are self-contained.

Refactored from the original: the upstream module read a ``config.yml`` at import
time and resolved data/reference paths from module-level globals. Here those paths
are explicit constructor arguments (``data_dir``, ``reference_dir``) so the reader
works against any extracted catalog without an ambient working directory or config
file.

The catalog for one species ``<data_dir>/<species_name>/`` is expected to contain
``snv_catalog.<fmt>``, ``alleles.<fmt>``, ``coverage.<fmt>`` (and optionally
``biallelic_snvs.<fmt>``), each a table with a ``(Contig, Location)`` index and one
column per sample. When ``annotate=True``, ``<reference_dir>/<species_name>/`` must
contain ``genome.fna.gz`` and ``genome.features.gz`` for the 4D/1D core annotation.
"""
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

from . import annotation_utils

# A pair sharing <= this fraction of identical blocks has essentially no clonal
# region left ("fully recombined"); used only by sample_random_fully_recombined_pair.
DEFAULT_FULLY_RECOMBINED_THRESHOLD = 0.05


def compute_biallelic_snvs(snvs, alleles, coverage):
    """
    Compute biallelic SNVs from the three input dataframes
    The input dataframes should have index (Contig, Location)
    snvs and alleles should have the same index

    Important: the input dataframes will be modified in place

    Returns:
    --------
    bi_snvs: pd.DataFrame
        Biallelic SNVs with columns: 'Ref', 'Major', 'Alt' and samples names
        0=Major, 1=Alt, 255=missing
    multi_sites: pd.Series
        Boolean series indicating sites with more than two segregating alleles
    """
    refs = alleles.pop('Ref')

    # alt alleles at non SNV sites are not well defined, so set to reference allele
    snv_mask = snvs & coverage.loc[snvs.index]
    alleles.where(snv_mask, refs, axis=0, inplace=True)
    # also mask the uncovered sites with nan
    alleles.where(coverage.loc[alleles.index], inplace=True)

    # count alleles at each site (A, C, G, T)
    allele_counts = alleles.apply(lambda row: row.value_counts(), axis=1)

    # some sites are monomorphic in the samples, but different from the reference allele
    # we will save these sites as "biallelic as well"
    mono_sites = allele_counts.notna().sum(axis=1)==1
    bi_sites = allele_counts.notna().sum(axis=1)==2
    # we will filter sites with more than two segregating alleles
    multi_sites = allele_counts.notna().sum(axis=1)>2

    # identify major and alt allele at mono and bi sites
    alt_alleles = allele_counts.idxmin(axis=1)
    major_alleles = allele_counts.idxmax(axis=1)

    # the above does not always work, because of ties and mono sites
    # the below two holds the true major and alt alleles
    true_major_alleles = refs.copy()
    true_alt_alleles = alt_alleles.copy()

    # first set mono sites, these sites the reference allele is the alt allele
    true_major_alleles[mono_sites] = major_alleles[mono_sites]
    true_alt_alleles[mono_sites] = refs[mono_sites]

    # now set bi sites
    true_major_alleles[bi_sites] = major_alleles[bi_sites]
    # major and alt are not well defined for ties (e.g. alt and major counts are the same)
    # in these cases we set the major allele to the reference allele, if possible
    tie_mask = bi_sites & ((allele_counts.min(axis=1))==(allele_counts.max(axis=1)))
    # true_major_alleles[tie_mask] = refs[tie_mask]

    # we still need to identify the alt allele at the tied sites
    # since there are not that many of these sites, we can loop through them
    for loc in tie_mask[tie_mask].index:
        # find nonzero column names, these are the two segregating alleles
        tie_alleles = allele_counts.loc[loc].dropna().index.tolist()
        # find the one different from reference
        if refs[loc] in tie_alleles:
            tie_alleles.remove(refs[loc])
            true_major_alleles[loc] = refs[loc]
            true_alt_alleles[loc] = tie_alleles[0]
        else:
            # if reference is not one of the two segregating alleles, just set the two tied alleles randomly
            true_major_alleles[loc] = tie_alleles[1]
            true_alt_alleles[loc] = tie_alleles[0]

    # lastly, group all relevant information for the biallelic sites into a dataframe
    bi_mask = ~multi_sites
    bi_snvs = snvs[bi_mask].copy().astype(int)

    bi_ref = refs[bi_mask]
    bi_major = true_major_alleles[bi_mask]
    bi_alt = true_alt_alleles[bi_mask]

    polarize_mask = bi_ref!=bi_major
    bi_snvs[polarize_mask] = 1 - bi_snvs[polarize_mask]

    # finally fill the missing sites with 255
    bi_snvs.where(coverage.loc[bi_snvs.index], other=255, inplace=True)

    bi_snvs['Ref'] = bi_ref
    bi_snvs['Major'] = bi_major
    bi_snvs['Alt'] = bi_alt
    return bi_snvs, multi_sites


def polarize_reference_seq(seq_records, snv_df):
    """
    Polarize the reference sequence based on the information in the SNV df
    Input:
    ------
    seq_records: list of Bio.SeqRecord
        List of SeqRecord objects when loading a fasta file with Bio.SeqIO
    snv_df: pd.DataFrame
        SNV dataframe with columns: 'Ref', 'Major', 'Alt' and multiindex (Contig, Location)

    Returns:
    --------
    polarized_seq_records: list of Bio.SeqRecord
    """

    contig_to_process = snv_df.index.get_level_values(0).unique()
    for record in seq_records:
        contig = record.id
        seq = np.array(record.seq)
        if contig not in contig_to_process:
            continue
        locations = snv_df.loc[contig].index
        # make sure seq is the same as the reference sequence
        subseq = seq[locations-1]
        assert(np.all(subseq==snv_df.loc[contig, 'Ref'].values))
        # polarize the reference sequence
        seq[locations-1] = snv_df.loc[contig, 'Major'].values

        new_seq = Seq(''.join(seq))
        record.seq = new_seq
    return seq_records


class SNVHelper:
    """
    Helper class for working with the SNV and allele tables of one species.

    Parameters
    ----------
    species_name : str
        Subdirectory name under ``data_dir`` (and ``reference_dir``) for this species.
    data_dir : str | Path
        Directory containing ``<species_name>/{snv_catalog,alleles,coverage}.<fmt>``.
    reference_dir : str | Path, optional
        Directory containing ``<species_name>/{genome.fna.gz,genome.features.gz}``.
        Required when ``annotate=True``.
    snv_format : str
        ``'feather'`` (default) or ``'parquet'``.
    compute_bi_snvs, save_bi_snvs, annotate, mask_multi_sites : bool
        See the original LiuGood2024 documentation. ``annotate=True`` computes the
        4D/1D core-site masks from the reference genome.
    """

    def __init__(self, species_name, *, data_dir, reference_dir=None,
                 snv_format='feather', compute_bi_snvs=True, save_bi_snvs=False,
                 annotate=False, mask_multi_sites=True,
                 fully_recombined_threshold=DEFAULT_FULLY_RECOMBINED_THRESHOLD):
        self.species_name = species_name
        self.format = snv_format
        self.data_dir = Path(data_dir)
        self.reference_dir = Path(reference_dir) if reference_dir is not None else None
        self.fully_recombined_threshold = fully_recombined_threshold

        species_dir = self.data_dir / species_name
        self.bi_snvs_path = species_dir / f'biallelic_snvs.{self.format}'
        self.allele_path = species_dir / f'alleles.{self.format}'
        self.coverage_path = species_dir / f'coverage.{self.format}'
        self.all_snvs_path = species_dir / f'snv_catalog.{self.format}'

        logging.info(f'Loading data for {species_name}')
        logging.info(f'Configuration: compute_bi_snvs={compute_bi_snvs}, save_bi_snvs={save_bi_snvs}, format={self.format}, annotate={annotate}')

        self.full_snvs = self.load_df(self.all_snvs_path, format=self.format)
        self.coverage = self.load_df(self.coverage_path, format=self.format)
        self.samples = self.coverage.columns

        if compute_bi_snvs or not self.bi_snvs_path.exists():
            self.snvs, self.multi_sites = compute_biallelic_snvs(self.full_snvs, self.load_df(self.allele_path, format=self.format), self.coverage)
            if save_bi_snvs:
                logging.info('Saving biallelic SNVs')
                self.save_snvs(self.bi_snvs_path)
        else:
            self.snvs = self.load_df(self.bi_snvs_path, format=self.format)
            # multi_sites are the sites in full_snvs but not in snvs
            multi_mask = pd.Series(~self.full_snvs.index.isin(self.snvs.index), self.full_snvs.index)
            self.multi_sites = multi_mask
        self.core_to_snvs = self.coverage.index.isin(self.snvs.index)
        if mask_multi_sites:
            # ignore all sites with more than two segregating alleles
            multi_site_index = self.multi_sites[self.multi_sites].index
            self.coverage.loc[multi_site_index] = False
        if annotate:
            self.annotate_snvs()
        else:
            self.annotated=False

    # static helper for loading dataframes
    @staticmethod
    def load_df(path, format):
        if format not in ['feather', 'parquet']:
            raise ValueError('format should be either feather or parquet')
        if format=='feather':
            df = pd.read_feather(path)
        elif format=='parquet':
            df = pd.read_parquet(path)
        for col in df.columns:
            if df[col].dtype == 'O':
                if isinstance(df[col][0], str):
                    continue
                elif isinstance(df[col][0], bytes):
                    # some of string series (from python2) were saved as bytes
                    df[col] = df[col].str.decode('utf-8')
        df.set_index(['Contig', 'Location'], inplace=True)
        return df

    def save_snvs(self, path):
        if self.format not in ['feather', 'parquet']:
            raise ValueError('format should be either feather or parquet')
        if self.format=='feather':
            self.snvs.reset_index().to_feather(path)
        elif self.format=='parquet':
            self.snvs.reset_index().to_parquet(path)

    def annotate_snvs(self):
        if self.reference_dir is None:
            raise ValueError(
                "annotate=True requires reference_dir pointing at "
                "<reference_dir>/<species_name>/{genome.fna.gz,genome.features.gz}"
            )
        ref_path = self.reference_dir / self.species_name / 'genome.fna.gz'
        gene_feature_path = self.reference_dir / self.species_name / 'genome.features.gz'

        with gzip.open(ref_path, 'rt') as file:
            records = list(SeqIO.parse(file, "fasta"))
        with gzip.open(gene_feature_path, 'rt') as file:
            gene_df = pd.read_csv(file, sep='\t')

        gene_df.columns = ['Gene ID', 'Contig', 'Start', 'End', 'Strand', 'Type', 'Info']
        # polarize the reference sequence by the major alleles
        polarized_records = polarize_reference_seq(records, snv_df=self.snvs)

        res_dfs = []
        for record in polarized_records:
            # annotate one contig
            sample_seq = record.seq
            contig = record.id
            contig_gene_df = gene_df[gene_df['Contig']==contig]
            res = annotation_utils.annotate_sequence_site_types_to_df(sample_seq, contig_gene_df)
            res['Contig'] = contig
            res_dfs.append(res)

        mut_df = pd.concat(res_dfs)
        mut_df.set_index(['Contig', 'Location'], inplace=True)
        mut_df = mut_df.loc[self.coverage.index]
        # rename A mut to A, etc
        mut_df = mut_df.rename(columns={'{} Mut'.format(x): x for x in ['A', 'C', 'G', 'T']})
        self.mut_df = mut_df

        # count mutation opportunities
        self.mut_df['s'] = (self.mut_df.loc[:, ['A', 'T', 'C', 'G']]=='s').sum(axis=1)-1
        self.mut_df['n'] = (self.mut_df.loc[:, ['A', 'T', 'C', 'G']]=='n').sum(axis=1)
        self.mut_df['m'] = (self.mut_df.loc[:, ['A', 'T', 'C', 'G']]=='m').sum(axis=1)
        self.mut_df['nn'] = (self.mut_df.loc[:, ['A', 'T', 'C', 'G']]=='nn').sum(axis=1)

        # save a few useful boolean masks
        self.core_1D = self.mut_df['Site Type']=='1D'
        self.core_4D = self.mut_df['Site Type']=='4D'
        self.snv_1D = self.core_1D.loc[self.snvs.index]
        self.snv_4D = self.core_4D.loc[self.snvs.index]

        snv_types = [self.mut_df.at[idx, col] for idx, col in zip(self.snvs.index, self.snvs['Alt'])]
        self.snv_types = pd.Series(snv_types, index=self.snvs.index)
        self.annotated = True
        logging.info('Done annotating SNVs')

    def compute_pairwise_snvs(self, sample1, sample2):
        """
        Compute pairwise SNVs between two samples
        Both sites need to be covered
        Return:
        -------
        diff_sites: pd.Series
            Boolean series indicating sites with different alleles between the genomes
            Length is the number of bi-allelic sites
        """
        sample1_alleles = self.snvs[sample1]
        sample2_alleles = self.snvs[sample2]
        diff_sites = (sample1_alleles!=sample2_alleles) & (sample1_alleles!=255) & (sample2_alleles!=255)
        return diff_sites

    def compute_pairwise_coverage(self, sample1, sample2):
        """
        Compute pairwise coverage between two samples
        Return:
        -------
        covered_mask: pd.Series
            Boolean series indicating sites covered in both genomes
        """
        sample1_cov = self.coverage[sample1]
        sample2_cov = self.coverage[sample2]
        covered_mask = (sample1_cov & sample2_cov)
        return covered_mask

    def sample_random_pair(self):
        """
        Sample a random pair of samples
        """
        sample1, sample2 = np.random.choice(self.samples, 2, replace=False)
        return sample1, sample2
