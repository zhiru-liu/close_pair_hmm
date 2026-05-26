from pathlib import Path
import argparse
import sys
import time


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import pandas as pd

import cphmm.config
import cphmm.prior
from bf_datahelper import DataHelper_Bf, HMM_PRIOR_PATH, SNV_SPECIES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CP-HMM prior for Bacteroides_fragilis_54507."
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-bins", type=int, default=cphmm.config.HMM_PRIOR_BINS)
    parser.add_argument("--clade-cutoff", type=float, default=0.03)
    parser.add_argument("--prior-dir", default=str(HMM_PRIOR_PATH))
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Also write sampled local/genome divergence values.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prior_dir = Path(args.prior_dir)
    prior_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SNV data for {SNV_SPECIES} at {time.ctime()}")
    datahelper = DataHelper_Bf()

    print(
        f"Sampling {args.num_samples} prior blocks "
        f"of {args.block_size} covered 4D sites at {time.ctime()}"
    )
    local_divs, genome_divs = cphmm.prior.sample_blocks(
        datahelper,
        num_samples=args.num_samples,
        block_size=args.block_size,
        random_state=args.seed,
    )
    divs, counts = cphmm.prior.compute_div_histogram(
        local_divs,
        genome_divs,
        num_bins=args.num_bins,
        separate_clades=True,
        clade_cutoff=args.clade_cutoff,
    )
    cphmm.prior.save_prior(
        divs,
        counts,
        datahelper.species,
        prior_path=str(prior_dir),
    )
    prior_path = cphmm.prior.get_prior_filename(
        datahelper.species,
        prior_path=str(prior_dir),
    )
    print(f"Wrote CP-HMM prior: {prior_path}")

    if args.save_samples:
        sample_path = prior_dir / f"{datahelper.species}__prior_samples.csv"
        pd.DataFrame(
            {
                "local_divergence": local_divs,
                "genome_divergence": genome_divs,
            }
        ).to_csv(sample_path, index=False)
        print(f"Wrote sampled divergences: {sample_path}")


if __name__ == "__main__":
    main()
