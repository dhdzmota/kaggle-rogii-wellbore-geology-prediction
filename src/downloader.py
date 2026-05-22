"""
Download competition data from Kaggle into data/raw/ and lock the directory read-only.

Authentication: set KAGGLE_API_TOKEN in a .env file at the project root (recommended),
or place ~/.kaggle/kaggle.json with {"username": "...", "key": "..."} for legacy credentials.

Usage:
    python src/downloader.py          # skip if data already present
    python src/downloader.py --force  # re-download even if data exists
"""

import argparse
import shutil
import stat
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

COMPETITION = "rogii-wellbore-geology-prediction"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _set_readonly(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _unlock(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_file():
            current = stat.S_IMODE(item.stat().st_mode)
            item.chmod(current | stat.S_IWUSR)


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = [p for p in RAW_DIR.iterdir() if p.is_file() and p.name != ".gitkeep"]
    if existing_files and not force:
        print(f"Raw data already present in {RAW_DIR} ({len(existing_files)} files). "
              "Pass --force to re-download.")
        return

    if force and existing_files:
        _unlock(RAW_DIR)

    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub package not found. Install it with: pip install kagglehub"
        )

    print(f"Downloading '{COMPETITION}' ...")
    try:
        cache_path = Path(kagglehub.competition_download(COMPETITION))
    except Exception as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    print(f"Copying files to {RAW_DIR} ...")
    for src in cache_path.rglob("*"):
        if src.is_file():
            dst = RAW_DIR / src.relative_to(cache_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    _set_readonly(RAW_DIR)
    files = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    print(f"Done. {len(files)} file(s) locked read-only in {RAW_DIR}:")
    for f in sorted(files):
        print(f"  {f.relative_to(RAW_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle competition data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if raw data is already present.",
    )
    args = parser.parse_args()
    download(force=args.force)
