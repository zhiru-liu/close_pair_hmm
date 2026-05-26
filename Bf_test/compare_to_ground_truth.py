from pathlib import Path

import pandas as pd


def _pair_key(sample1, sample2):
    sample1 = str(sample1)
    sample2 = str(sample2)
    return tuple(sorted((sample1, sample2)))


def _add_pair_key(df, sample1_col, sample2_col):
    df = df.copy()
    df["pair_key"] = [
        _pair_key(sample1, sample2)
        for sample1, sample2 in zip(df[sample1_col], df[sample2_col])
    ]
    return df


def compare_pair_event_counts(predicted_transfers, ground_truth):
    predicted = _add_pair_key(predicted_transfers, "genome1", "genome2")
    truth = _add_pair_key(ground_truth, "Sample 1", "Sample 2")

    pred_counts = predicted.groupby("pair_key").size().rename("predicted_events")
    truth_counts = truth.groupby("pair_key").size().rename("ground_truth_events")
    comparison = pd.concat([truth_counts, pred_counts], axis=1).fillna(0).astype(int)
    comparison = comparison.reset_index()
    comparison[["sample1", "sample2"]] = pd.DataFrame(
        comparison["pair_key"].tolist(),
        index=comparison.index,
    )
    return comparison[["sample1", "sample2", "ground_truth_events", "predicted_events"]]


def annotate_interval_overlaps(predicted_transfers, ground_truth):
    """
    Mark predicted transfers whose contig interval overlaps a published event.
    This is intentionally conservative and only compares events on the same
    unordered sample pair and reference contig.
    """
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


def write_comparison_files(
    predicted_transfers_path,
    ground_truth_path,
    output_dir,
    output_prefix=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted = pd.read_csv(predicted_transfers_path)
    truth = pd.read_csv(ground_truth_path, low_memory=False)
    truth["Sample 1"] = truth["Sample 1"].astype(str)
    truth["Sample 2"] = truth["Sample 2"].astype(str)

    pair_counts = compare_pair_event_counts(predicted, truth)
    pair_count_name = "pair_event_count_comparison.csv"
    if output_prefix is not None:
        pair_count_name = f"{output_prefix}__{pair_count_name}"
    pair_counts_path = output_dir / pair_count_name
    pair_counts.to_csv(pair_counts_path, index=False)

    overlap_path = None
    if len(predicted) and {"contig", "start_site", "end_site"}.issubset(predicted.columns):
        annotated = annotate_interval_overlaps(predicted, truth)
        overlap_name = "transfer_interval_overlap_comparison.csv"
        if output_prefix is not None:
            overlap_name = f"{output_prefix}__{overlap_name}"
        overlap_path = output_dir / overlap_name
        annotated.to_csv(overlap_path, index=False)

    return pair_counts_path, overlap_path
