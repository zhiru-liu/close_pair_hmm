"""1D-density tract extension: a post-processing add-on for the CP-HMM.

The CP-HMM detects recombination from **4D (synonymous)** SNV density only. A
transfer imported from a donor that differs mainly at **1D (nonsynonymous)** sites
(e.g. a diversifying-selection surface locus) carries few 4D differences, so the
HMM finds the synonymous-rich core of the transfer but truncates the boundary
where the 4D signal fades, leaving a 1D-rich / 4D-poor flank misclassified as
"clonal". This module widens already-detected tract boundaries into immediately
adjacent regions of abnormally high 1D-difference density.

This is **not** a re-inference: the 4D HMM still does the primary detection. We
only extend boundaries afterward, gated behind a flag so default behaviour is
unchanged.

Everything operates **per pair, in reference coordinates** (so extension can cross
4D-poor stretches that have no 4D sites to index). Two public entry points share
identical machinery and differ only in which differences drive the boundary walk:

- :func:`extend_tracts_by_1d_density` -- the narrow, validated path; extension is
  driven by 1D (nonsynonymous) differences only.
- :func:`extend_tracts_by_density` -- the general path; extension is driven by an
  arbitrary site class (e.g. all SNVs), so 2D/3D-rich flanks are also absorbed,
  not just strictly-1D ones.

:func:`cphmm.infer_pipelines.infer_pairs` calls one of them right after
``annotate_transfer_reference_coordinates`` when ``extend_with`` is set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SNPInfo = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class ExtensionParams:
    """Thresholds controlling 1D-density boundary extension.

    A boundary is walked outward over successive 1D **difference** sites. A step
    onto the next difference at position ``d`` is accepted only if all hold:

    - the gap from the current frontier to ``d`` is at most ``gap_bp`` (a longer
      contiguous run with no 1D difference is treated as background -> stop);
    - the flank absorbed so far has at least ``threshold`` 1D differences per kb
      of reference, where ``threshold = max(min_abs_per_kb, rate_multiple * λ0)``
      and ``λ0`` is the pair's background clonal 1D-difference density per kb. The
      ``min_abs_per_kb`` floor matches the empirical missed-tract definition; the
      ``rate_multiple * λ0`` term raises the bar for species whose clonal
      background is high enough that the floor alone would absorb ordinary
      clustering;
    - the total extension stays within ``max_extension_bp`` of the original
      boundary.
    """

    gap_bp: int = 1000
    min_abs_per_kb: float = 3.0
    rate_multiple: float = 5.0
    max_extension_bp: int = 5000


# ---------------------------------------------------------------------------- #
# small geometry helpers (all in reference coordinates, per contig)


def _extend_one_side(
    boundary: int,
    diff_locs: np.ndarray,
    params: ExtensionParams,
    threshold_per_kb: float,
    direction: int,
) -> int:
    """Walk one boundary outward; return the new boundary position.

    ``direction`` is +1 to extend the right boundary toward larger coordinates,
    -1 to extend the left boundary toward smaller coordinates. ``diff_locs`` is
    the contig's sorted 1D difference positions.
    """
    # Candidate difference sites strictly outside the boundary, ordered outward.
    if direction > 0:
        cand = diff_locs[diff_locs > boundary]
    else:
        cand = diff_locs[diff_locs < boundary][::-1]

    frontier = boundary
    n_absorbed = 0
    for d in cand:
        d = int(d)
        gap = abs(d - frontier)
        if gap > params.gap_bp:
            break
        ext_len = abs(d - boundary)
        if ext_len > params.max_extension_bp:
            break

        n_obs = n_absorbed + 1
        dens_per_kb = n_obs / (ext_len / 1000.0)
        if dens_per_kb < threshold_per_kb:
            break

        frontier = d
        n_absorbed += 1
    return frontier


def _background_lambda0_per_kb(
    diff_locs: np.ndarray,
    diff_contigs: np.ndarray,
    cov_locs: np.ndarray,
    cov_contigs: np.ndarray,
    tracts: pd.DataFrame,
) -> float:
    """Background clonal 1D-difference density (per kb of reference).

    Clonal = the covered reference span outside every detected tract. The span
    per contig is approximated by the covered-1D extent (min..max covered
    position); summed across contigs and reduced by the detected-tract spans.
    Returns 0.0 when there is no clonal span (degenerate; the absolute floor then
    solely gates extension).
    """
    in_tract = np.zeros(len(diff_locs), dtype=bool)
    for _, t in tracts.iterrows():
        c = t["contig"]
        s, e = int(t["start_site"]), int(t["end_site"])
        in_tract |= (diff_contigs == c) & (diff_locs >= s) & (diff_locs <= e)
    n_clonal_diff = int(np.sum(~in_tract))

    total_ref_bp = 0
    for c in np.unique(cov_contigs):
        cl = cov_locs[cov_contigs == c]
        if len(cl):
            total_ref_bp += int(cl.max() - cl.min() + 1)
    tract_ref_bp = int((tracts["end_site"].astype(int)
                        - tracts["start_site"].astype(int) + 1).clip(lower=0).sum())
    clonal_ref_bp = max(total_ref_bp - tract_ref_bp, 1)
    return n_clonal_diff / (clonal_ref_bp / 1000.0)


def _merge_intervals(rows: list[dict]) -> list[dict]:
    """Merge overlapping/abutting extended intervals within each contig.

    Each row carries the extended interval (``start``/``end``) plus the original
    core interval (``orig_start``/``orig_end``) and carry-through metadata. Merged
    rows union the extended span and keep the tightest original-core span seen
    (min orig_start, max orig_end) for provenance.
    """
    merged: list[dict] = []
    by_contig: dict[str, list[dict]] = {}
    for r in rows:
        by_contig.setdefault(r["contig"], []).append(r)
    for contig, group in by_contig.items():
        group.sort(key=lambda r: r["start"])
        cur = dict(group[0])
        for r in group[1:]:
            if r["start"] <= cur["end"]:  # overlapping or abutting (inclusive)
                cur["end"] = max(cur["end"], r["end"])
                cur["orig_start"] = min(cur["orig_start"], r["orig_start"])
                cur["orig_end"] = max(cur["orig_end"], r["orig_end"])
                cur["types"] = min(cur["types"], r["types"])
            else:
                merged.append(cur)
                cur = dict(r)
        merged.append(cur)
    return merged


# ---------------------------------------------------------------------------- #
# core + public entry points


def _diffs(snp_info: SNPInfo) -> Tuple[np.ndarray, np.ndarray]:
    """(difference contigs, difference locs) from a ``(snp_vec, contigs, locs)``."""
    snp_vec, contigs, locs = snp_info
    snp_vec = np.asarray(snp_vec, dtype=bool)
    contigs = np.asarray(contigs).astype(str)
    locs = np.asarray(locs).astype(int)
    return contigs[snp_vec], locs[snp_vec]


def _extend_tracts_core(
    transfer_df: pd.DataFrame,
    driver_info: SNPInfo,
    params: ExtensionParams,
    count_specs: Sequence[Tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Shared boundary-walk + merge + provenance, driven by one site class.

    ``driver_info`` is the ``(snp_vec, contigs, locs)`` whose differences drive
    the boundary walk and set the background rate. ``count_specs`` is a sequence
    of ``(column_name, diff_contigs, diff_locs)`` whose difference counts in the
    extension flanks become provenance columns.
    """
    out_cols = [
        "genome1", "genome2", "contig", "types",
        "start_site", "end_site",
        "orig_start_site", "orig_end_site", "extension_bp",
    ] + [name for name, _, _ in count_specs]

    if transfer_df is None or len(transfer_df) == 0:
        return pd.DataFrame(columns=out_cols)

    _, contigs_d, locs_d = driver_info
    contigs_d = np.asarray(contigs_d).astype(str)
    locs_d = np.asarray(locs_d).astype(int)
    diff_contigs, diff_locs = _diffs(driver_info)

    lambda0 = _background_lambda0_per_kb(
        diff_locs, diff_contigs, locs_d, contigs_d, transfer_df
    )
    threshold_per_kb = max(params.min_abs_per_kb, params.rate_multiple * lambda0)

    by_contig_diff: dict[str, np.ndarray] = {}
    for c in np.unique(contigs_d):
        by_contig_diff[c] = np.sort(diff_locs[diff_contigs == c])

    meta_cols = [col for col in ("genome1", "genome2") if col in transfer_df.columns]

    rows: list[dict] = []
    for _, t in transfer_df.iterrows():
        contig = str(t["contig"])
        orig_l, orig_r = int(t["start_site"]), int(t["end_site"])
        diff_c = by_contig_diff.get(contig, np.array([], dtype=int))

        new_l = _extend_one_side(orig_l, diff_c, params, threshold_per_kb, direction=-1)
        new_r = _extend_one_side(orig_r, diff_c, params, threshold_per_kb, direction=+1)

        row = {
            "contig": contig,
            "start": new_l, "end": new_r,
            "orig_start": orig_l, "orig_end": orig_r,
            "types": int(t["types"]) if "types" in transfer_df.columns and not pd.isna(t["types"]) else 0,
        }
        for col in meta_cols:
            row[col] = t[col]
        rows.append(row)

    merged = _merge_intervals(rows)

    records = []
    for m in merged:
        c = m["contig"]
        s, e = int(m["start"]), int(m["end"])
        os_, oe = int(m["orig_start"]), int(m["orig_end"])
        rec = {
            "genome1": m.get("genome1"),
            "genome2": m.get("genome2"),
            "contig": c,
            "types": m["types"],
            "start_site": s,
            "end_site": e,
            "orig_start_site": os_,
            "orig_end_site": oe,
            "extension_bp": (os_ - s) + (e - oe),
        }
        for name, cc, ll in count_specs:
            # Extension flank = extended interval minus the original-core span.
            mask = (cc == c) & (ll >= s) & (ll <= e) & ~((ll >= os_) & (ll <= oe))
            rec[name] = int(np.sum(mask))
        records.append(rec)

    return pd.DataFrame(records, columns=out_cols)


def extend_tracts_by_1d_density(
    transfer_df: pd.DataFrame,
    snp_info_1d: SNPInfo,
    params: Optional[ExtensionParams] = None,
    snp_info_4d: Optional[SNPInfo] = None,
) -> pd.DataFrame:
    """Extend detected tract boundaries into adjacent 1D-dense flanks.

    The narrow, validated entry point: extension is driven by 1D (nonsynonymous)
    differences only. Use :func:`extend_tracts_by_density` to drive on all SNVs
    (which also captures 2D/3D-rich flanks).

    Parameters
    ----------
    transfer_df : pd.DataFrame
        One pair's detected tracts, already carrying ``contig``/``start_site``/
        ``end_site`` reference coordinates (i.e. after
        ``annotate_transfer_reference_coordinates``). ``genome1``/``genome2`` and
        ``types`` are carried through when present.
    snp_info_1d : (snp_vec, contigs, locs)
        The pair's covered 1D sites, in the ``get_pair_snp_info(pair,
        site_class='1D')`` form: ``snp_vec`` is a bool array (True where the two
        samples differ), ``contigs``/``locs`` the reference contig/position of
        each covered 1D site.
    params : ExtensionParams, optional
        Extension thresholds; defaults to :class:`ExtensionParams`.
    snp_info_4d : (snp_vec, contigs, locs), optional
        The pair's covered 4D sites, only used to populate ``extension_4d_snvs``.

    Returns
    -------
    pd.DataFrame
        Extended tracts with updated ``start_site``/``end_site`` plus provenance
        columns: ``orig_start_site``, ``orig_end_site``, ``extension_bp``,
        ``extension_1d_snvs``, ``extension_4d_snvs``. Carry-through metadata
        (``genome1``/``genome2``/``types``/``contig``) is preserved. Rows that
        overlap or abut after extension are merged.
    """
    params = params or ExtensionParams()
    count_specs = [("extension_1d_snvs", *_diffs(snp_info_1d))]
    if snp_info_4d is not None:
        count_specs.append(("extension_4d_snvs", *_diffs(snp_info_4d)))
    else:
        count_specs.append(("extension_4d_snvs", np.array([], dtype=str), np.array([], dtype=int)))
    return _extend_tracts_core(transfer_df, snp_info_1d, params, count_specs)


def extend_tracts_by_density(
    transfer_df: pd.DataFrame,
    snp_info: SNPInfo,
    params: Optional[ExtensionParams] = None,
    count_infos: Optional[dict] = None,
) -> pd.DataFrame:
    """Extend detected tract boundaries on an arbitrary site class.

    The general entry point: extension is driven by the differences in
    ``snp_info`` -- pass ``get_pair_snp_info(pair, site_class='all')`` to extend
    on every covered SNV (so 2D/3D-rich flanks, not just 1D, are absorbed), or any
    other site class. Identical machinery (background rate, gap stop, max cap,
    merge) to :func:`extend_tracts_by_1d_density`; only the driving differences
    differ.

    Parameters
    ----------
    transfer_df, params
        As in :func:`extend_tracts_by_1d_density`.
    snp_info : (snp_vec, contigs, locs)
        Covered sites of the driving site class; their differences drive the
        boundary walk and set the background rate.
    count_infos : dict[str, (snp_vec, contigs, locs)], optional
        Extra site classes to tally in the extension flanks. Each key ``k`` adds
        an ``extension_{k}_snvs`` column (e.g. ``{'4d': snp_info_4d}`` ->
        ``extension_4d_snvs``).

    Returns
    -------
    pd.DataFrame
        Extended tracts with the same schema as
        :func:`extend_tracts_by_1d_density`, except the driver tally column is
        ``extension_snvs`` plus one ``extension_{k}_snvs`` per ``count_infos`` key.
    """
    params = params or ExtensionParams()
    count_specs = [("extension_snvs", *_diffs(snp_info))]
    for label, info in (count_infos or {}).items():
        count_specs.append((f"extension_{label}_snvs", *_diffs(info)))
    return _extend_tracts_core(transfer_df, snp_info, params, count_specs)
