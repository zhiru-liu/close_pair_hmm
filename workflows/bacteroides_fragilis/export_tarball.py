"""MAINTAINER step: (re)assemble the committed Bf SNV-catalog tarball.

Copies the minimal set of files the example needs from the full LiuGood2024 catalog
(only ``snv_catalog`` + ``coverage`` + ``biallelic_snvs`` -- ``alleles`` is not needed
because the reader runs with ``compute_bi_snvs=False`` -- plus the reference genome for
the 4D annotation), packs them into ``data/bf_data.tar.gz`` (the committed asset), and
prints the size + sha256.

    python workflows/bacteroides_fragilis/export_tarball.py \
        --snv-src /Volumes/Botein/GarudGood2019_snvs/snvs_feather \
        --ref-src /Volumes/Botein/GarudGood2019_snvs/midas_db_data/rep_genomes

Then commit the regenerated ``data/bf_data.tar.gz``.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import shutil
import tarfile
import tempfile

SPECIES = "Bacteroides_fragilis_54507"
THIS_DIR = Path(__file__).resolve().parent

DEFAULT_SNV_SRC = Path("/Volumes/Botein/GarudGood2019_snvs/snvs_feather")
DEFAULT_REF_SRC = Path("/Volumes/Botein/GarudGood2019_snvs/midas_db_data/rep_genomes")

SNV_FILES = ["snv_catalog.feather", "coverage.feather", "biallelic_snvs.feather"]
REF_FILES = ["genome.fna.gz", "genome.features.gz"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snv-src", type=Path, default=DEFAULT_SNV_SRC,
                   help="Dir containing <species>/<*.feather>.")
    p.add_argument("--ref-src", type=Path, default=DEFAULT_REF_SRC,
                   help="Dir containing <species>/{genome.fna.gz,genome.features.gz}.")
    p.add_argument("--out", type=Path, default=THIS_DIR / "data" / "bf_data.tar.gz")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    snv_species = args.snv_src / SPECIES
    ref_species = args.ref_src / SPECIES

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "bf_data"
        dst_snv = staging / "snvs" / SPECIES
        dst_ref = staging / "reference" / SPECIES
        dst_snv.mkdir(parents=True)
        dst_ref.mkdir(parents=True)

        for name in SNV_FILES:
            src = snv_species / name
            if not src.exists():
                raise FileNotFoundError(f"Missing {src}")
            shutil.copy2(src, dst_snv / name)
        for name in REF_FILES:
            src = ref_species / name
            if not src.exists():
                raise FileNotFoundError(f"Missing {src}")
            shutil.copy2(src, dst_ref / name)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {args.out} ...")
        with tarfile.open(args.out, "w:gz") as tar:
            tar.add(staging, arcname="bf_data")

    size_mb = args.out.stat().st_size / 1e6
    digest = _sha256(args.out)
    print(f"\nTarball:  {args.out}")
    print(f"Size:     {size_mb:.1f} MB")
    print(f"sha256:   {digest}")
    print("\nPaste into download_data.py: BF_DATA_SHA256 = "
          f'"{digest}"  and BF_DATA_URL = "<release asset url>"')


if __name__ == "__main__":
    main()
