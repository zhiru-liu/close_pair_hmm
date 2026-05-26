"""
Generate per-accession CP-HMM transfer-divergence priors for UHGG isolates.

Usage:
  python generate_prior.py MGYG-HGUT-01337
  python generate_prior.py --all
  python generate_prior.py --all --save-samples

The prior file is written to Isolate_test/priors/<accession>.csv via
``cphmm.prior.save_prior``. With ``--save-samples``, the raw sampled
local/genome divergences are also written next to it.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (REPO_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

import cphmm.config
import cphmm.prior
from isolate_datahelper import (
    ACCESSIONS,
    HMM_PRIOR_PATH,
    DataHelper_Isolate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CP-HMM prior(s) for UHGG isolate accessions.",
    )
    parser.add_argument(
        "accessions",
        nargs="*",
        help="One or more accession IDs (e.g. MGYG-HGUT-01337).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run for all nine canonical isolate accessions.",
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-bins", type=int, default=cphmm.config.HMM_PRIOR_BINS
    )
    parser.add_argument("--clade-cutoff", type=float, default=0.03)
    parser.add_argument("--prior-dir", default=str(HMM_PRIOR_PATH))
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Also write raw sampled local/genome divergences alongside the prior.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if <accession>.csv already exists.",
    )
    return parser.parse_args()


def _resolve_accessions(args: argparse.Namespace) -> list[str]:
    if args.all:
        if args.accessions:
            raise SystemExit(
                "Pass either explicit accessions or --all, not both."
            )
        return list(ACCESSIONS)
    if not args.accessions:
        raise SystemExit(
            "No accession given. Pass one or more accession IDs, or --all."
        )
    return list(args.accessions)


def generate_one(
    accession: str,
    *,
    prior_dir: Path,
    num_samples: int,
    block_size: int,
    num_bins: int,
    clade_cutoff: float,
    seed: int,
    save_samples: bool,
    overwrite: bool,
) -> Path:
    prior_path = cphmm.prior.get_prior_filename(
        accession, prior_path=str(prior_dir)
    )
    if Path(prior_path).exists() and not overwrite:
        print(f"[{accession}] prior already exists at {prior_path}; skipping.")
        return Path(prior_path)

    print(f"[{accession}] loading SNV helper at {time.ctime()}")
    datahelper = DataHelper_Isolate(accession)
    print(
        f"[{accession}] {len(datahelper.sample_names)} samples, "
        f"{datahelper.genome_len} 4D core sites"
    )

    print(
        f"[{accession}] sampling {num_samples} prior blocks "
        f"of {block_size} covered 4D sites at {time.ctime()}"
    )
    local_divs, genome_divs = cphmm.prior.sample_blocks(
        datahelper,
        num_samples=num_samples,
        block_size=block_size,
        random_state=seed,
    )
    divs, counts = cphmm.prior.compute_div_histogram(
        local_divs,
        genome_divs,
        num_bins=num_bins,
        separate_clades=True,
        clade_cutoff=clade_cutoff,
    )
    cphmm.prior.save_prior(
        divs, counts, datahelper.species, prior_path=str(prior_dir)
    )
    print(f"[{accession}] wrote CP-HMM prior: {prior_path}")

    if save_samples:
        sample_path = (
            Path(prior_dir) / f"{datahelper.species}__prior_samples.csv"
        )
        pd.DataFrame(
            {
                "local_divergence": local_divs,
                "genome_divergence": genome_divs,
            }
        ).to_csv(sample_path, index=False)
        print(f"[{accession}] wrote sampled divergences: {sample_path}")

    return Path(prior_path)


def main() -> None:
    args = parse_args()
    accessions = _resolve_accessions(args)
    prior_dir = Path(args.prior_dir)
    prior_dir.mkdir(parents=True, exist_ok=True)

    for accession in accessions:
        generate_one(
            accession,
            prior_dir=prior_dir,
            num_samples=args.num_samples,
            block_size=args.block_size,
            num_bins=args.num_bins,
            clade_cutoff=args.clade_cutoff,
            seed=args.seed,
            save_samples=args.save_samples,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
