"""End-to-end Bacteroides fragilis CP-HMM example.

Downloads the SNV catalog (first run), runs CP-HMM inference over the published
close pairs, writes the inference + transfer summaries, and compares the detected
transfers against the published LiuGood2024 ground truth.

    python workflows/bacteroides_fragilis/infer.py --max-pairs 1   # quick smoke test
    python workflows/bacteroides_fragilis/infer.py                 # full published set

Outputs go to ``results/`` (gitignored). Small reference outputs for validation are
committed under ``reference_outputs/``.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import cphmm.infer_pipelines as infer_pipelines
from cphmm.io.liugood2024_qp.published_comparison import write_comparison_files
from datahelper import DataHelper_Bf, SPECIES


def parse_args():
    parser = argparse.ArgumentParser(description=f"CP-HMM inference for {SPECIES}.")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit the number of published close pairs to process.")
    parser.add_argument("--results-dir", default=str(THIS_DIR / "results"),
                        help="Directory for inference and comparison outputs.")
    parser.add_argument("--clade-cutoff-bin", type=int, default=40,
                        help="CP-HMM clade cutoff bin passed to infer_pairs.")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SNV data for {SPECIES} at {time.ctime()}")
    dh = DataHelper_Bf(max_pairs=args.max_pairs)
    close_pairs = dh.get_close_pairs()
    print(f"{len(close_pairs)} close pairs loaded; "
          f"{len(dh.missing_pairs)} skipped (samples absent). "
          f"{dh.genome_len} annotated 4D core sites.")

    start = time.time()
    print(f"Starting inference at {time.ctime()}")
    pair_dat, transfer_dat = infer_pipelines.infer_pairs(
        dh, close_pairs, clade_cutoff_bin=args.clade_cutoff_bin)
    print(f"Inference complete at {time.ctime()}, took {time.time() - start:.1f} secs")

    suffix = "all_pairs" if args.max_pairs is None else f"first_{args.max_pairs}_pairs"
    pair_path = results_dir / f"{SPECIES}__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"{SPECIES}__{suffix}__transfer_summary.csv"
    pair_dat.to_csv(pair_path, index=False)
    transfer_dat.to_csv(transfer_path, index=False)

    pair_counts_path, overlap_path = write_comparison_files(
        transfer_path, dh.ground_truth_path, results_dir,
        output_prefix=f"{SPECIES}__{suffix}")

    print(f"Wrote inference summary:    {pair_path}")
    print(f"Wrote transfer summary:     {transfer_path}")
    print(f"Wrote pair event comparison: {pair_counts_path}")
    if overlap_path is not None:
        print(f"Wrote interval overlap:      {overlap_path}")


if __name__ == "__main__":
    main()
