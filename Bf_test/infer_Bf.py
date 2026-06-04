from pathlib import Path
import argparse
import os
import sys
import time


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import cphmm.infer_pipelines as infer_pipelines
from bf_datahelper import DataHelper_Bf, HMM_PRIOR_PATH, SNV_SPECIES
from compare_to_ground_truth import write_comparison_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CP-HMM inference for Bacteroides_fragilis_54507."
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit the number of published close pairs to process.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(THIS_DIR / "results"),
        help="Directory for inference and comparison outputs.",
    )
    parser.add_argument(
        "--clade-cutoff-bin",
        type=int,
        default=40,
        help="CP-HMM clade cutoff bin passed through to infer_pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SNV data for {SNV_SPECIES} at {time.ctime()}")
    datahelper = DataHelper_Bf(max_pairs=args.max_pairs)
    close_pairs = datahelper.get_close_pairs()
    print(
        f"{len(close_pairs)} close pairs loaded; "
        f"{len(datahelper.missing_pairs)} pairs skipped because samples were absent."
    )
    print(
        f"Using CP-HMM prior species {datahelper.species}; "
        f"{datahelper.genome_len} annotated 4D core sites."
    )
    prior_path = HMM_PRIOR_PATH / f"{datahelper.species}.csv"
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Missing species-specific prior {prior_path}. "
            "Run Bf_test/generate_prior.py before inference."
        )

    ground_truth_path = datahelper.write_ground_truth_subset(
        results_dir / "Bacteroides_fragilis_54507_ground_truth_transfers.csv"
    )

    start_time = time.time()
    print(f"Starting inference at {time.ctime()}")
    pair_dat, transfer_dat = infer_pipelines.infer_pairs(
        datahelper,
        close_pairs,
        clade_cutoff_bin=args.clade_cutoff_bin,
    )
    elapsed = time.time() - start_time
    print(f"Inference complete at {time.ctime()}, took {elapsed:.1f} secs")

    suffix = "all_pairs" if args.max_pairs is None else f"first_{args.max_pairs}_pairs"
    pair_path = results_dir / f"Bacteroides_fragilis_54507__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"Bacteroides_fragilis_54507__{suffix}__transfer_summary.csv"
    pair_dat.to_csv(pair_path, index=False)
    transfer_dat.to_csv(transfer_path, index=False)

    pair_counts_path, overlap_path = write_comparison_files(
        transfer_path,
        ground_truth_path,
        results_dir,
        output_prefix=f"Bacteroides_fragilis_54507__{suffix}",
    )

    print(f"Wrote inference summary: {pair_path}")
    print(f"Wrote transfer summary: {transfer_path}")
    print(f"Wrote ground truth subset: {ground_truth_path}")
    print(f"Wrote pair event comparison: {pair_counts_path}")
    if overlap_path is not None:
        print(f"Wrote interval overlap comparison: {overlap_path}")


if __name__ == "__main__":
    main()
