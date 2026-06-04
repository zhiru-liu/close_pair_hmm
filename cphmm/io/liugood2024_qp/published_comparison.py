"""Compare CP-HMM transfer calls against the LiuGood2024 published table.

The published supplementary table (``gut_microbiome_transfers.csv`` / pbio.3002472.s003)
has one row per transfer with columns ``Species name``, ``Sample 1``, ``Sample 2``,
``Clonal divergence``, ``Reference contig``, ``Reference genome start/end loc``, etc. CP-HMM
inference (``cphmm.infer_pipelines.infer_pairs``) emits a per-pair inference summary
(``genome1``/``genome2``/``est_div``/...) and a per-transfer table
(``genome1``/``genome2``/``contig``/``start_site``/``end_site``/...).

These helpers compare the two -- per-pair transfer counts, transfer interval overlap, and
per-species aggregate agreement. Shared by the Bf and full-QP reproduction workflows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _pair_key(sample1, sample2):
    return tuple(sorted((str(sample1), str(sample2))))


def _add_pair_key(df, sample1_col, sample2_col):
    df = df.copy()
    df["pair_key"] = [
        _pair_key(a, b) for a, b in zip(df[sample1_col], df[sample2_col])
    ]
    return df


def compare_pair_event_counts(predicted_transfers, ground_truth):
    """Per-pair predicted vs published transfer-event counts."""
    predicted = _add_pair_key(predicted_transfers, "genome1", "genome2")
    truth = _add_pair_key(ground_truth, "Sample 1", "Sample 2")

    pred_counts = predicted.groupby("pair_key").size().rename("predicted_events")
    truth_counts = truth.groupby("pair_key").size().rename("ground_truth_events")
    comparison = pd.concat([truth_counts, pred_counts], axis=1).fillna(0).astype(int)
    comparison = comparison.reset_index()
    comparison[["sample1", "sample2"]] = pd.DataFrame(
        comparison["pair_key"].tolist(), index=comparison.index
    )
    return comparison[["sample1", "sample2", "ground_truth_events", "predicted_events"]]


def annotate_interval_overlaps(predicted_transfers, ground_truth):
    """Flag predicted transfers overlapping a published event on the same pair + contig."""
    predicted = _add_pair_key(predicted_transfers, "genome1", "genome2")
    truth = _add_pair_key(ground_truth, "Sample 1", "Sample 2")

    overlap_flags = []
    for _, pred in predicted.iterrows():
        same_pair = truth["pair_key"] == pred["pair_key"]
        same_contig = truth["Reference contig"].astype(str) == str(pred["contig"])
        candidates = truth[same_pair & same_contig]
        pred_start = min(int(pred["start_site"]), int(pred["end_site"]))
        pred_end = max(int(pred["start_site"]), int(pred["end_site"]))
        overlaps = (
            (candidates["Reference genome start loc"] <= pred_end)
            & (candidates["Reference genome end loc"] >= pred_start)
        )
        overlap_flags.append(bool(overlaps.any()))

    annotated = predicted_transfers.copy()
    annotated["overlaps_ground_truth"] = overlap_flags
    return annotated


def write_comparison_files(predicted_transfers_path, ground_truth_path, output_dir,
                           output_prefix=None):
    """Write the per-pair event-count and interval-overlap comparison CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted = pd.read_csv(predicted_transfers_path)
    truth = pd.read_csv(ground_truth_path, low_memory=False)
    truth["Sample 1"] = truth["Sample 1"].astype(str)
    truth["Sample 2"] = truth["Sample 2"].astype(str)

    pair_counts = compare_pair_event_counts(predicted, truth)
    name = "pair_event_count_comparison.csv"
    if output_prefix is not None:
        name = f"{output_prefix}__{name}"
    pair_counts_path = output_dir / name
    pair_counts.to_csv(pair_counts_path, index=False)

    overlap_path = None
    if len(predicted) and {"contig", "start_site", "end_site"}.issubset(predicted.columns):
        annotated = annotate_interval_overlaps(predicted, truth)
        oname = "transfer_interval_overlap_comparison.csv"
        if output_prefix is not None:
            oname = f"{output_prefix}__{oname}"
        overlap_path = output_dir / oname
        annotated.to_csv(overlap_path, index=False)

    return pair_counts_path, overlap_path


def aggregate_species_stats(species, inference_summary, transfer_summary, ground_truth):
    """Return a one-row per-species verification record.

    Compares predicted vs published transfer/pair counts, clonal-divergence agreement on
    shared pairs, and the fraction of predicted transfers overlapping a published event.

    Parameters
    ----------
    inference_summary : DataFrame with columns genome1, genome2, est_div (per pair).
    transfer_summary  : DataFrame of predicted transfers (per-transfer rows).
    ground_truth      : the published rows for this species (Sample 1/2, Clonal divergence,
                        Reference contig, Reference genome start/end loc).
    """
    truth = _add_pair_key(ground_truth, "Sample 1", "Sample 2")
    pred_pairs = _add_pair_key(inference_summary, "genome1", "genome2")

    truth_pairs = set(truth["pair_key"])
    pred_pair_set = set(pred_pairs["pair_key"])
    shared = truth_pairs & pred_pair_set

    # clonal-divergence agreement on shared pairs
    clonal_corr = np.nan
    clonal_median_absdiff = np.nan
    if shared and "est_div" in inference_summary.columns and "Clonal divergence" in truth.columns:
        truth_cd = truth.groupby("pair_key")["Clonal divergence"].first()
        pred_cd = pred_pairs.groupby("pair_key")["est_div"].first()
        merged = pd.concat([truth_cd, pred_cd], axis=1, join="inner").dropna()
        if len(merged) >= 2:
            clonal_corr = float(np.corrcoef(merged["Clonal divergence"], merged["est_div"])[0, 1])
        if len(merged):
            clonal_median_absdiff = float(
                np.median(np.abs(merged["Clonal divergence"] - merged["est_div"]))
            )

    # transfer interval overlap fraction
    overlap_frac = np.nan
    if len(transfer_summary) and {"contig", "start_site", "end_site"}.issubset(transfer_summary.columns):
        annotated = annotate_interval_overlaps(transfer_summary, ground_truth)
        overlap_frac = float(np.mean(annotated["overlaps_ground_truth"]))

    return {
        "species": species,
        "pairs_predicted": len(pred_pair_set),
        "pairs_published": len(truth_pairs),
        "pairs_shared": len(shared),
        "transfers_predicted": int(len(transfer_summary)),
        "transfers_published": int(len(ground_truth)),
        "clonal_div_corr": clonal_corr,
        "clonal_div_median_absdiff": clonal_median_absdiff,
        "transfer_overlap_frac": overlap_frac,
    }
