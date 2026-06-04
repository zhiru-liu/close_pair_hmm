"""Optional SNV-catalog readers for CP-HMM workflows.

Each submodule here is a *campaign-specific* adapter that reads one SNV-catalog
format and exposes the data a :class:`cphmm.datahelper.ClosePairDataHelper` needs.
These readers depend on the ``[workflows]`` extra (pandas/pyarrow/biopython) and are
kept out of the core inference modules so ``cphmm`` itself stays format-agnostic.

Currently available:

- :mod:`cphmm.io.liugood2024_qp` -- the LiuGood2024 quasi-phaseable feather catalog.
"""
