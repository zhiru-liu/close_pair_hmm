"""Reproduce the Liu & Good 2024 QP recombination results across species.

For each requested species: load the QP catalog, take the published close pairs (or
re-select from the catalog), run CP-HMM inference with the cached prior, and compare the
calls against the published supplementary table. Writes per-species result CSVs and an
overall ``verification_summary.csv``.

    python workflows/liugood2024_qp/reproduce.py \
        --data-dir <snv-catalog-dir> --reference-dir <reference-genome-dir> \
        --ground-truth <gut_microbiome_transfers.csv>

Data locations may instead come from $CPHMM_QP_DATA_DIR / $CPHMM_QP_REFERENCE_DIR /
$CPHMM_QP_GROUND_TRUTH. The 6.8 GB catalog and 44 MB table are not committed.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import cphmm.config
import cphmm.prior
import cphmm.infer_pipelines as infer_pipelines
from cphmm.io.liugood2024_qp.published_comparison import (
    write_comparison_files,
    aggregate_species_stats,
)

from datahelper import DataHelper_QP
from species import ALL_SPECIES, DEFAULT_SPECIES, TWO_CLADE_SPECIES

PRIOR_DIR = THIS_DIR / "priors"
# Clade-divergence cutoffs used when regenerating two-clade priors (paper values).
REGEN_CLADE_CUTOFFS = {
    "Bacteroides_vulgatus_57955": 0.03,
    "Alistipes_shahii_62199": 0.04,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--species", nargs="+", default=None,
                   help="Species to run (default: a small representative subset).")
    p.add_argument("--all", action="store_true", help="Run all 29 published species.")
    p.add_argument("--data-dir", default=os.environ.get("CPHMM_QP_DATA_DIR"),
                   help="Dir with <species>/<*.feather> (or $CPHMM_QP_DATA_DIR).")
    p.add_argument("--reference-dir", default=os.environ.get("CPHMM_QP_REFERENCE_DIR"),
                   help="Dir with <species>/{genome.fna.gz,genome.features.gz}.")
    p.add_argument("--ground-truth", default=os.environ.get("CPHMM_QP_GROUND_TRUTH"),
                   help="Published transfer table (gut_microbiome_transfers.csv).")
    p.add_argument("--pair-source", choices=["published", "select"], default="published")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--transfer-length", type=float, default=1000.,
                   help="Expected transfer length (bp) for the HMM. The paper refined this "
                        "per species iteratively starting from 1000 (not a fixed value); "
                        "1000 matches that initialization. See README.")
    p.add_argument("--iterative", choices=["auto", "on", "off"], default="auto",
                   help="Iterative clonal-emission refinement (auto: on for two-clade species).")
    p.add_argument("--regenerate-prior", action="store_true",
                   help="Regenerate the prior from the catalog instead of using the cached one.")
    p.add_argument("--results-dir", default=str(THIS_DIR / "results"))
    return p.parse_args()


def resolve_species(args):
    if args.all:
        return ALL_SPECIES
    if args.species:
        return args.species
    return DEFAULT_SPECIES


def prior_clade_settings(species):
    """Return (clade_cutoff_bin, is_two_clade) from the cached prior shape."""
    prior = np.loadtxt(PRIOR_DIR / f"{species}.csv")
    n_states = prior.shape[1]
    two_clade = n_states == 2 * cphmm.config.HMM_PRIOR_BINS
    clade_cutoff_bin = cphmm.config.HMM_PRIOR_BINS if two_clade else None
    return clade_cutoff_bin, two_clade


def maybe_regenerate_prior(dh, species, two_clade):
    cutoff = REGEN_CLADE_CUTOFFS.get(species, 0.03)
    local_divs, genome_divs = cphmm.prior.sample_blocks(dh)
    divs, counts = cphmm.prior.compute_div_histogram(
        local_divs, genome_divs, separate_clades=two_clade, clade_cutoff=cutoff)
    cphmm.prior.save_prior(divs, counts, species, prior_path=str(PRIOR_DIR))
    print(f"  regenerated prior for {species} (two_clade={two_clade}, cutoff={cutoff})")


def run_species(species, args, results_dir):
    clade_cutoff_bin, two_clade = prior_clade_settings(species)
    if args.iterative == "on":
        iterative = True
    elif args.iterative == "off":
        iterative = False
    else:
        iterative = species in TWO_CLADE_SPECIES

    print(f"[{species}] loading catalog at {time.ctime()}")
    dh = DataHelper_QP(
        species, data_dir=args.data_dir, reference_dir=args.reference_dir,
        prior_dir=PRIOR_DIR, ground_truth_path=args.ground_truth,
        pair_source=args.pair_source, max_pairs=args.max_pairs)

    if args.regenerate_prior:
        maybe_regenerate_prior(dh, species, two_clade)

    pairs = dh.get_close_pairs()
    print(f"[{species}] {len(pairs)} close pairs ({args.pair_source}); "
          f"{len(dh.missing_pairs)} skipped; {dh.genome_len} 4D core sites; "
          f"clade_cutoff_bin={clade_cutoff_bin}, iterative={iterative}")
    if not pairs:
        print(f"[{species}] no pairs to infer; skipping")
        return None

    start = time.time()
    pair_dat, transfer_dat = infer_pipelines.infer_pairs(
        dh, pairs, clade_cutoff_bin=clade_cutoff_bin, iterative=iterative,
        transfer_length=args.transfer_length)
    print(f"[{species}] inference done in {time.time() - start:.1f}s")

    pair_path = results_dir / f"{species}__inference_summary.csv"
    transfer_path = results_dir / f"{species}__transfer_summary.csv"
    pair_dat.to_csv(pair_path, index=False)
    transfer_dat.to_csv(transfer_path, index=False)

    record = None
    if dh.ground_truth is not None:
        write_comparison_files(transfer_path, args.ground_truth, results_dir,
                               output_prefix=species)
        record = aggregate_species_stats(species, pair_dat, transfer_dat, dh.ground_truth)
        print(f"[{species}] predicted/published transfers="
              f"{record['transfers_predicted']}/{record['transfers_published']}; "
              f"clonal-div corr={record['clonal_div_corr']:.3f}; "
              f"overlap_frac={record['transfer_overlap_frac']:.3f}")
    return record


def main():
    args = parse_args()
    if not args.data_dir or not args.reference_dir:
        sys.exit("Provide --data-dir and --reference-dir (or the CPHMM_QP_* env vars).")
    if args.pair_source == "published" and not args.ground_truth:
        sys.exit("pair_source='published' needs --ground-truth (the published table).")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    species_list = resolve_species(args)
    print(f"Reproducing {len(species_list)} species at {time.ctime()}")

    records = []
    for species in species_list:
        try:
            rec = run_species(species, args, results_dir)
        except Exception as exc:  # keep going across species
            print(f"[{species}] ERROR: {exc}")
            continue
        if rec is not None:
            records.append(rec)

    if records:
        summary = pd.DataFrame(records)
        summary_path = results_dir / "verification_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nWrote verification summary: {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
