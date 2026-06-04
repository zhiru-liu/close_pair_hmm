"""Reader for the LiuGood2024 quasi-phaseable (QP) SNV catalog format.

Vendored (with a small path-handling refactor) from the author's published
LiuGood-2024-SNVs repository so the LiuGood2024 CP-HMM workflows are self-contained.
"""
from .snv_utils import SNVHelper, compute_biallelic_snvs, polarize_reference_seq

__all__ = ["SNVHelper", "compute_biallelic_snvs", "polarize_reference_seq"]
