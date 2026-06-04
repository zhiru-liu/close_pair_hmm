"""Species covered by the Liu & Good 2024 QP recombination analysis.

These are the 29 species with published transfer events in the supplementary table
(``gut_microbiome_transfers.csv`` / pbio.3002472.s003). Names match the QP catalog
directory, the reference-genome directory, and the cached prior filename
(``<genus>_<species>_<midasID>``), so no name bridging is needed.
"""

ALL_SPECIES = [
    "Akkermansia_muciniphila_55290",
    "Alistipes_finegoldii_56071",
    "Alistipes_onderdonkii_55464",
    "Alistipes_putredinis_61533",
    "Alistipes_shahii_62199",
    "Alistipes_sp_60764",
    "Bacteroidales_bacterium_58650",
    "Bacteroides_caccae_53434",
    "Bacteroides_cellulosilyticus_58046",
    "Bacteroides_coprocola_61586",
    "Bacteroides_eggerthii_54457",
    "Bacteroides_finegoldii_57739",
    "Bacteroides_fragilis_54507",
    "Bacteroides_massiliensis_44749",
    "Bacteroides_ovatus_58035",
    "Bacteroides_stercoris_56735",
    "Bacteroides_thetaiotaomicron_56941",
    "Bacteroides_uniformis_57318",
    "Bacteroides_vulgatus_57955",
    "Barnesiella_intestinihominis_62208",
    "Dialister_invisus_61905",
    "Eubacterium_rectale_56927",
    "Eubacterium_siraeum_57634",
    "Oscillibacter_sp_60799",
    "Parabacteroides_distasonis_56985",
    "Parabacteroides_merdae_56972",
    "Phascolarctobacterium_sp_59817",
    "Ruminococcus_bicirculans_59300",
    "Ruminococcus_bromii_62047",
]

# A small, quick-to-run default subset (one validated species + two modest ones).
DEFAULT_SPECIES = [
    "Bacteroides_fragilis_54507",
    "Bacteroides_uniformis_57318",
    "Dialister_invisus_61905",
]

# Two-clade species that the paper analyzed with clade separation + iterative
# refinement. Detectable independently from the prior shape (80 = 2 x 40 bins),
# but listed here for clarity.
TWO_CLADE_SPECIES = [
    "Alistipes_shahii_62199",
    "Bacteroides_vulgatus_57955",
]
