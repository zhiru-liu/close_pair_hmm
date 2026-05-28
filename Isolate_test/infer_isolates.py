"""
Run CP-HMM recombination inference for UHGG isolate accessions.

Usage:
  python infer_isolates.py MGYG-HGUT-01337
  python infer_isolates.py --all
  python infer_isolates.py MGYG-HGUT-01337 --max-pairs 5

Writes two CSVs per accession into ``Isolate_test/results/``:
  <accession>__all_pairs__inference_summary.csv
  <accession>__all_pairs__transfer_summary.csv

When ``--max-pairs N`` is used, the suffix becomes ``first_N_pairs`` instead.

Each accession requires its per-species prior at
``Isolate_test/priors/<accession>.csv``. Generate priors first with
``generate_prior.py`` (or pass ``--ensure-prior`` to auto-generate any
missing prior with default settings).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import multiprocessing as mp
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (REPO_ROOT, THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

import cphmm.config
import infer_pipelines
from isolate_datahelper import (
    ACCESSIONS,
    HMM_PRIOR_PATH,
    DataHelper_Isolate,
)


# ---- worker-side state (one helper + one model per process) ----------------

_WORKER_STATE: dict = {}


def _worker_init(accession: str, prior_dir: str, clade_cutoff_bin: int,
                 iterative: bool, n_iter: int) -> None:
    """Initializer for ``multiprocessing.Pool`` workers.

    Each worker loads the SNV helper once and constructs an HMM model with the
    species-specific prior. Both objects are kept in module-global state and
    reused across the worker's pair workload.
    """
    # Make sure the repo + Isolate_test dirs are importable inside the worker.
    for p in (str(REPO_ROOT), str(THIS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from isolate_datahelper import DataHelper_Isolate as _DH  # noqa: WPS433
    import infer_pipelines as _ip  # noqa: WPS433

    dh = _DH(accession, hmm_prior_path=prior_dir)
    model = _ip.init_hmm(
        dh.species,
        dh.genome_len,
        cphmm.config.HMM_BLOCK_SIZE,
        prior_path=prior_dir,
    )
    _WORKER_STATE["dh"] = dh
    _WORKER_STATE["model"] = model
    _WORKER_STATE["clade_cutoff_bin"] = clade_cutoff_bin
    _WORKER_STATE["iterative"] = iterative
    _WORKER_STATE["n_iter"] = n_iter


def _worker_infer_pair(pair):
    """Run CP-HMM inference for one pair in a worker process."""
    import cphmm.recomb_inference as ri  # noqa: WPS433
    from infer_pipelines import annotate_transfer_reference_coordinates  # noqa: WPS433

    dh = _WORKER_STATE["dh"]
    model = _WORKER_STATE["model"]
    clade_cutoff_bin = _WORKER_STATE["clade_cutoff_bin"]

    snp_vec, contigs, locs = dh.get_pair_snp_info(pair)
    clonal_div, genome_len, clonal_len, transfer_dat = ri.infer(
        snp_vec,
        contigs,
        model,
        cphmm.config.HMM_BLOCK_SIZE,
        clade_cutoff_bin=clade_cutoff_bin,
        iterative=_WORKER_STATE["iterative"],
        n_iter=_WORKER_STATE["n_iter"],
    )
    naive_div, est_div = clonal_div
    transfer_dat = transfer_dat.copy()
    transfer_dat["genome1"] = pair[0]
    transfer_dat["genome2"] = pair[1]
    annotate_transfer_reference_coordinates(transfer_dat, contigs, locs)
    return {
        "genome1": pair[0],
        "genome2": pair[1],
        "naive_div": naive_div,
        "est_div": est_div,
        "genome_len": genome_len,
        "clonal_len": clonal_len,
        "transfer_dat": transfer_dat,
    }


def _infer_pairs_parallel(accession, close_pairs, prior_dir, clade_cutoff_bin,
                          workers, iterative=False, n_iter=3):
    """Run ``infer_pairs`` across ``workers`` processes.

    Returns ``(pair_dat, transfer_dat)`` matching the serial path's schema.
    """
    pair_rows: list[dict] = []
    transfer_dats: list[pd.DataFrame] = []

    # Cap workers at the number of pairs (no point spawning more).
    workers = max(1, min(workers, len(close_pairs)))
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers,
        initializer=_worker_init,
        initargs=(accession, str(prior_dir), clade_cutoff_bin, iterative, n_iter),
    ) as pool:
        chunksize = max(1, len(close_pairs) // (workers * 8))
        completed = 0
        for result in pool.imap_unordered(
            _worker_infer_pair, close_pairs, chunksize=chunksize
        ):
            completed += 1
            pair_rows.append(
                {
                    "genome1": result["genome1"],
                    "genome2": result["genome2"],
                    "naive_div": result["naive_div"],
                    "est_div": result["est_div"],
                    "genome_len": result["genome_len"],
                    "clonal_len": result["clonal_len"],
                }
            )
            transfer_dats.append(result["transfer_dat"])
            if completed % 50 == 0 or completed == len(close_pairs):
                print(
                    f"[{accession}] {completed}/{len(close_pairs)} pairs done "
                    f"at {time.ctime()}"
                )

    pair_dat = pd.DataFrame(
        pair_rows,
        columns=[
            "genome1",
            "genome2",
            "naive_div",
            "est_div",
            "genome_len",
            "clonal_len",
        ],
    )
    if transfer_dats:
        transfer_dat = pd.concat(transfer_dats).reset_index(drop=True)
    else:
        transfer_dat = pd.DataFrame(
            columns=["genome1", "genome2", "snp_vec_start", "snp_vec_end"]
        )
    return pair_dat, transfer_dat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CP-HMM inference for UHGG isolate accessions.",
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
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit the number of close pairs processed per accession.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(THIS_DIR / "results"),
    )
    parser.add_argument(
        "--prior-dir",
        default=str(HMM_PRIOR_PATH),
    )
    parser.add_argument(
        "--clade-cutoff-bin",
        type=int,
        default=40,
        help="CP-HMM clade-cutoff bin passed through to infer_pairs.",
    )
    parser.add_argument(
        "--ensure-prior",
        action="store_true",
        help=(
            "If a per-accession prior is missing, generate it with default "
            "settings before running inference."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log and continue when an accession fails, rather than aborting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Number of worker processes per accession. Set to 1 to run "
            "single-threaded (e.g. for debugging). Default: 4."
        ),
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help=(
            "Run iterative clonal-emission refinement (legacy "
            "_fit_and_count_transfers_iterative). Each outer iteration "
            "refits all contigs with a clonal-emission rate re-estimated "
            "from the previous pass's pooled clonal blocks. Output "
            "filenames gain an '__iterative' suffix so iterative and "
            "non-iterative results co-exist."
        ),
    )
    parser.add_argument(
        "--iterative-iters",
        type=int,
        default=3,
        help="Number of outer iterations when --iterative is set. Default: 3.",
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


def _ensure_prior(accession: str, prior_dir: Path) -> Path:
    """Return path to <accession>.csv, generating it if missing."""
    import generate_prior  # local import — keeps optional dependency local

    prior_path = prior_dir / f"{accession}.csv"
    if prior_path.exists():
        return prior_path
    print(
        f"[{accession}] prior {prior_path} missing; generating with defaults."
    )
    return generate_prior.generate_one(
        accession,
        prior_dir=prior_dir,
        num_samples=5000,
        block_size=1000,
        num_bins=__import__("cphmm.config", fromlist=["HMM_PRIOR_BINS"]).HMM_PRIOR_BINS,
        clade_cutoff=0.03,
        seed=0,
        save_samples=False,
        overwrite=False,
    )


def infer_one(
    accession: str,
    *,
    results_dir: Path,
    prior_dir: Path,
    clade_cutoff_bin: int,
    max_pairs: int | None,
    ensure_prior: bool,
    workers: int = 1,
    iterative: bool = False,
    n_iter: int = 3,
) -> tuple[Path, Path]:
    prior_path = prior_dir / f"{accession}.csv"
    if not prior_path.exists():
        if ensure_prior:
            _ensure_prior(accession, prior_dir)
        else:
            raise FileNotFoundError(
                f"Missing per-species prior {prior_path}. "
                "Run generate_prior.py first or pass --ensure-prior."
            )

    print(f"[{accession}] loading SNV helper at {time.ctime()}")
    datahelper = DataHelper_Isolate(accession, max_pairs=max_pairs)
    close_pairs = datahelper.get_close_pairs()
    print(
        f"[{accession}] {len(close_pairs)} close pairs to infer "
        f"(missing-sample pairs skipped: {len(datahelper.missing_pairs)})"
    )
    print(
        f"[{accession}] CP-HMM prior species: {datahelper.species}; "
        f"genome_len = {datahelper.genome_len} 4D core sites"
    )

    if len(close_pairs) == 0:
        print(f"[{accession}] no close pairs at cutoff; writing empty results.")

    start_time = time.time()
    use_parallel = workers > 1 and len(close_pairs) > 1
    mode_label = "iterative" if iterative else "single-pass"
    print(
        f"[{accession}] starting {mode_label} inference at {time.ctime()} "
        f"({'parallel, workers=' + str(workers) if use_parallel else 'serial'})"
    )
    if use_parallel:
        pair_dat, transfer_dat = _infer_pairs_parallel(
            accession,
            close_pairs,
            prior_dir=prior_dir,
            clade_cutoff_bin=clade_cutoff_bin,
            workers=workers,
            iterative=iterative,
            n_iter=n_iter,
        )
    else:
        pair_dat, transfer_dat = infer_pipelines.infer_pairs(
            datahelper,
            close_pairs,
            clade_cutoff_bin=clade_cutoff_bin,
            iterative=iterative,
            n_iter=n_iter,
        )
    elapsed = time.time() - start_time
    print(f"[{accession}] inference complete at {time.ctime()}, took {elapsed:.1f}s")

    base_suffix = (
        "all_pairs" if max_pairs is None else f"first_{max_pairs}_pairs"
    )
    suffix = f"{base_suffix}__iterative" if iterative else base_suffix
    results_dir.mkdir(parents=True, exist_ok=True)
    pair_path = results_dir / f"{accession}__{suffix}__inference_summary.csv"
    transfer_path = results_dir / f"{accession}__{suffix}__transfer_summary.csv"
    pair_dat.to_csv(pair_path, index=False)
    transfer_dat.to_csv(transfer_path, index=False)

    print(f"[{accession}] wrote inference summary: {pair_path}")
    print(f"[{accession}] wrote transfer summary:  {transfer_path}")
    return pair_path, transfer_path


def main() -> None:
    args = parse_args()
    accessions = _resolve_accessions(args)
    results_dir = Path(args.results_dir)
    prior_dir = Path(args.prior_dir)

    failures: list[tuple[str, str]] = []
    for accession in accessions:
        try:
            infer_one(
                accession,
                results_dir=results_dir,
                prior_dir=prior_dir,
                clade_cutoff_bin=args.clade_cutoff_bin,
                max_pairs=args.max_pairs,
                ensure_prior=args.ensure_prior,
                workers=args.workers,
                iterative=args.iterative,
                n_iter=args.iterative_iters,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[{accession}] FAILED: {exc!r}")
            failures.append((accession, repr(exc)))

    if failures:
        print("\nFailures:")
        for accession, msg in failures:
            print(f"  {accession}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
