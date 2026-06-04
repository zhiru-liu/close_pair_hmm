"""Provide the Bacteroides fragilis SNV catalog for this example.

The catalog is small (~15 MB compressed), so ``data/bf_data.tar.gz`` is committed to
the repo and simply extracted on first use -- the example is clone-and-run with no
network. A URL fallback remains for larger/other catalogs hosted as Release assets.

Resolution order for the data root (the extracted ``bf_data/`` directory):

1. ``$CPHMM_BF_DATA_DIR`` -- if set, use it directly. Point this at an already-extracted
   ``bf_data`` dir, e.g. for maintainers who built it locally.
2. A previously-extracted copy under ``data/extracted/bf_data/``.
3. The committed ``data/bf_data.tar.gz`` -- extracted into ``data/extracted/``.
4. Download ``$CPHMM_BF_DATA_URL`` (or the ``BF_DATA_URL`` constant below), verify its
   sha256, and extract it.

The tarball extracts to::

    bf_data/
      snvs/Bacteroides_fragilis_54507/{snv_catalog,coverage,biallelic_snvs}.feather
      reference/Bacteroides_fragilis_54507/{genome.fna.gz,genome.features.gz}
"""
from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

SPECIES = "Bacteroides_fragilis_54507"

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
EXTRACT_DIR = DATA_DIR / "extracted"
DATA_ROOT = EXTRACT_DIR / "bf_data"  # the directory the tarball unpacks to
TARBALL_NAME = "bf_data.tar.gz"
COMMITTED_TARBALL = DATA_DIR / TARBALL_NAME  # shipped in the repo

# --- Optional Release-asset fallback (only used if the committed tarball is absent) ---
# Set CPHMM_BF_DATA_URL to override without editing this file.
BF_DATA_URL = ""
# sha256 of the tarball; leave as None to skip integrity checking.
BF_DATA_SHA256: str | None = None


def _expected_files(data_root: Path) -> list[Path]:
    snv_dir = data_root / "snvs" / SPECIES
    ref_dir = data_root / "reference" / SPECIES
    return [
        snv_dir / "snv_catalog.feather",
        snv_dir / "coverage.feather",
        snv_dir / "biallelic_snvs.feather",
        ref_dir / "genome.fna.gz",
        ref_dir / "genome.features.gz",
    ]


def _is_complete(data_root: Path) -> bool:
    return data_root.is_dir() and all(p.exists() for p in _expected_files(data_root))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SNV catalog from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:6.1f} MB", end="")
    print()
    tmp.replace(dest)


def resolve_data_root(force_download: bool = False) -> Path:
    """Return the extracted ``bf_data`` directory, downloading/extracting if needed."""
    override = os.environ.get("CPHMM_BF_DATA_DIR")
    if override:
        root = Path(override).expanduser().resolve()
        if not _is_complete(root):
            missing = [str(p) for p in _expected_files(root) if not p.exists()]
            raise FileNotFoundError(
                f"CPHMM_BF_DATA_DIR={root} is missing expected files:\n  "
                + "\n  ".join(missing)
            )
        return root

    if _is_complete(DATA_ROOT) and not force_download:
        return DATA_ROOT

    # Prefer the committed tarball; fall back to downloading a hosted asset.
    tarball = COMMITTED_TARBALL
    if not tarball.exists() or force_download:
        url = os.environ.get("CPHMM_BF_DATA_URL", BF_DATA_URL)
        if not url:
            raise RuntimeError(
                f"No data available: committed tarball {COMMITTED_TARBALL} is missing and "
                "no CPHMM_BF_DATA_URL is set (also accepts $CPHMM_BF_DATA_DIR pointing at "
                "an already-extracted bf_data directory)."
            )
        _download(url, tarball)
        if BF_DATA_SHA256:
            digest = _sha256(tarball)
            if digest != BF_DATA_SHA256:
                raise RuntimeError(
                    f"sha256 mismatch for {tarball}: got {digest}, expected {BF_DATA_SHA256}"
                )

    print(f"Extracting {tarball} -> {EXTRACT_DIR}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)  # noqa: S202 - trusted self-published asset

    if not _is_complete(DATA_ROOT):
        missing = [str(p) for p in _expected_files(DATA_ROOT) if not p.exists()]
        raise FileNotFoundError(
            "Extraction did not produce the expected layout; missing:\n  "
            + "\n  ".join(missing)
        )
    return DATA_ROOT


def snv_and_reference_dirs(data_root: Path) -> tuple[Path, Path]:
    """Return ``(snvs_dir, reference_dir)`` to pass to the SNV reader."""
    return data_root / "snvs", data_root / "reference"


if __name__ == "__main__":
    root = resolve_data_root()
    print(f"Data ready at {root}")
