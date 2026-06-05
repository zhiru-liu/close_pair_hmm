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
    p.add_argument("--transfer-length", default="1000",
                   help="Expected transfer length (bp) for the HMM, or 'iterative' to "
                        "re-estimate it per species (the paper's procedure: decode, recompute "
                        "the mean detected transfer length, re-decode until convergence, "
                        "starting from 1000). A fixed 1000 matches the paper's initialization. "
                        "See README.")
    p.add_argument("--iterative", choices=["auto", "on", "off"], default="auto",
                   help="Iterative clonal-emission refinement (auto: on for two-clade species). "
                        "Orthogonal to --transfer-length iterative; the two compose.")
    p.add_argument("--regenerate-prior", action="store_true",
                   help="Regenerate the prior from the catalog instead of using the cached one.")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Number of species to run in parallel (process pool). Each worker "
                        "loads its own species catalog, so memory scales with --jobs.")
    p.add_argument("--results-dir", default=str(THIS_DIR / "results"))
    return p.parse_args()


# Iterative transfer-length estimation parameters (Liu & Good 2024 decode_one_pass).
TL_INIT = 1000.        # bp; initialization, single- and two-clade alike
TL_MAX_PASSES = 3      # up to 3 decode passes
TL_FRAC_TOL = 0.1      # converged when |new - old| / old < 0.1 (both clades, two-clade)


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


def _mean_transfer_bp(transfer_dat, type_label=None):
    """Mean detected transfer length in bp, optionally restricted to one clade type.

    Length in blocks = ``block_end - block_start + 1`` (both inclusive), times the
    HMM block size. Returns None when there are no transfers to average. NB: this is
    the *raw* decoded mean -- the paper averaged over merged+filtered transfers
    (drop < 5 blocks), which is deliberately not ported yet, so the estimate runs a
    little short of the published one.
    """
    df = transfer_dat
    if type_label is not None:
        df = df[df['types'] == type_label]
    if len(df) == 0:
        return None
    lengths = ((df['block_end'].astype(int) - df['block_start'].astype(int) + 1)
               * cphmm.config.HMM_BLOCK_SIZE)
    return float(lengths.mean())


def infer_iterative_transfer_length(dh, pairs, species, clade_cutoff_bin, iterative,
                                    two_clade):
    """Re-estimate the transfer length per species, iterating to convergence.

    Ports ``decode_one_pass`` from the paper's ``close_pair_stage2-3_iterative.py``:
    decode all pairs, recompute the mean detected transfer length, re-decode, up to
    ``TL_MAX_PASSES`` passes, stopping when the fractional change drops below
    ``TL_FRAC_TOL``. Two-clade species track within- (``types==0``) and between-clade
    (``types==1``) means separately and pass an 80-length per-state array to the HMM
    (first 40 = within, last 40 = between).

    Returns ``(pair_dat, transfer_dat, final_length)`` where the data are the decode
    that triggered convergence and ``final_length`` is a scalar (single-clade) or a
    ``(within, between)`` tuple (two-clade).
    """
    bins = cphmm.config.HMM_PRIOR_BINS

    def _length_arg(cur):
        if two_clade:
            arr = np.empty(2 * bins)
            arr[:bins] = cur[0]
            arr[bins:] = cur[1]
            return arr
        return cur

    cur = [TL_INIT, TL_INIT] if two_clade else TL_INIT
    pair_dat = transfer_dat = None
    for p in range(1, TL_MAX_PASSES + 1):
        pair_dat, transfer_dat = infer_pipelines.infer_pairs(
            dh, pairs, clade_cutoff_bin=clade_cutoff_bin, iterative=iterative,
            transfer_length=_length_arg(cur))

        if two_clade:
            new_within = _mean_transfer_bp(transfer_dat, type_label=0)
            new_between = _mean_transfer_bp(transfer_dat, type_label=1)
            # Hold a clade's length fixed when it produced no transfers this pass.
            new = [new_within if new_within is not None else cur[0],
                   new_between if new_between is not None else cur[1]]
            frac = max(abs(new[k] - cur[k]) / cur[k] for k in range(2))
            print(f"[{species}]   pass {p}: transfer_length within/between = "
                  f"{new[0]:.0f}/{new[1]:.0f} bp (frac_change={frac:.3f})")
        else:
            new_scalar = _mean_transfer_bp(transfer_dat)
            if new_scalar is None:
                print(f"[{species}]   pass {p}: no transfers detected; keeping "
                      f"transfer_length={cur:.0f} bp")
                break
            new = new_scalar
            frac = abs(new - cur) / cur
            print(f"[{species}]   pass {p}: transfer_length = {new:.0f} bp "
                  f"(frac_change={frac:.3f})")

        converged = frac < TL_FRAC_TOL
        cur = new
        if converged:
            print(f"[{species}]   converged after {p} pass(es)")
            break

    final = tuple(cur) if two_clade else cur
    return pair_dat, transfer_dat, final


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
    if args.transfer_length == "iterative":
        pair_dat, transfer_dat, final_tl = infer_iterative_transfer_length(
            dh, pairs, species, clade_cutoff_bin, iterative, two_clade)
        if two_clade:
            print(f"[{species}] final transfer_length within/between = "
                  f"{final_tl[0]:.0f}/{final_tl[1]:.0f} bp")
        else:
            print(f"[{species}] final transfer_length = {final_tl:.0f} bp")
    else:
        pair_dat, transfer_dat = infer_pipelines.infer_pairs(
            dh, pairs, clade_cutoff_bin=clade_cutoff_bin, iterative=iterative,
            transfer_length=float(args.transfer_length))
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


def _run_one(species, args, results_dir):
    """Run one species, swallowing errors so a single failure can't abort the batch."""
    try:
        return run_species(species, args, results_dir)
    except Exception as exc:  # keep going across species
        print(f"[{species}] ERROR: {exc}")
        return None


def main():
    args = parse_args()
    if not args.data_dir or not args.reference_dir:
        sys.exit("Provide --data-dir and --reference-dir (or the CPHMM_QP_* env vars).")
    if args.pair_source == "published" and not args.ground_truth:
        sys.exit("pair_source='published' needs --ground-truth (the published table).")
    if args.transfer_length != "iterative":
        try:
            float(args.transfer_length)
        except ValueError:
            sys.exit("--transfer-length must be a number (bp) or 'iterative'.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    species_list = resolve_species(args)
    print(f"Reproducing {len(species_list)} species at {time.ctime()} "
          f"(jobs={args.jobs})")

    records = []
    if args.jobs > 1 and len(species_list) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_run_one, sp, args, results_dir)
                       for sp in species_list]
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
    else:
        for species in species_list:
            rec = _run_one(species, args, results_dir)
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
