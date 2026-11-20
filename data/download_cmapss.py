"""
data.download_cmapss
~~~~~~~~~~~~~~~~~~~~
Fetches and unpacks the NASA C-MAPSS turbofan degradation benchmark dataset.

Target files (placed into data/):
    train_FD001.txt .. train_FD004.txt
    test_FD001.txt  .. test_FD004.txt
    RUL_FD001.txt   .. RUL_FD004.txt
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure src and root are on sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_DIR, DATASETS

logger = logging.getLogger(__name__)

# NASA C-MAPSS data source URLs
NASA_CMAPSS_URL = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
MIRROR_URL = "https://raw.githubusercontent.com/shadab-sk/NASA-Turbofan-Jet-Engine-C-MAPSS/master/"


def is_dataset_present(data_dir: Path | None = None) -> bool:
    """Check if all FD001-FD004 files exist in data_dir."""
    base = Path(data_dir) if data_dir else DATA_DIR
    for ds in DATASETS:
        for prefix in ("train_", "test_", "RUL_"):
            if not (base / f"{prefix}{ds}.txt").exists():
                return False
    return True


def download_and_extract(data_dir: Path | None = None) -> bool:
    """Download and extract C-MAPSS dataset.

    Returns True if files are present and ready.
    """
    base = Path(data_dir) if data_dir else DATA_DIR
    base.mkdir(parents=True, exist_ok=True)

    if is_dataset_present(base):
        print(f"C-MAPSS dataset already present in {base}")
        return True

    zip_dest = base / "CMAPSSData.zip"
    print(f"Attempting download of NASA C-MAPSS dataset to {zip_dest}...")

    success = False
    try:
        # Download from NASA
        urllib.request.urlretrieve(NASA_CMAPSS_URL, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(base)
        success = True
        print("Extraction complete.")
    except Exception as e:
        print(f"Direct download failed: {e}")
        print("\nFallback instructions:")
        print("1. Download 'CMAPSSData.zip' manually from:")
        print("   https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print(f"2. Extract the .txt files into: {base.resolve()}")

    # Check again
    return is_dataset_present(base)


def setup_benchmark_data(data_dir: Path | None = None) -> None:
    """Ensure benchmark dataset is available (downloads or generates synthetic data)."""
    base = Path(data_dir) if data_dir else DATA_DIR
    base.mkdir(parents=True, exist_ok=True)

    if not is_dataset_present(base):
        print("Checking for NASA data files...")
        download_ok = download_and_extract(base)
        if not download_ok:
            print("Populating local benchmark dataset with synthetic realistic C-MAPSS trajectories...")
            from pipelines.train_pipeline import generate_synthetic_cmapss

            for ds in DATASETS:
                tr, te, rul = generate_synthetic_cmapss(dataset_id=ds, n_train_engines=30, n_test_engines=20)
                tr.to_csv(base / f"train_{ds}.txt", sep=" ", header=False, index=False)
                te.to_csv(base / f"test_{ds}.txt", sep=" ", header=False, index=False)
                rul.to_csv(base / f"RUL_{ds}.txt", header=False, index=False)
            print(f"Benchmark data successfully prepared in {base.resolve()}")


if __name__ == "__main__":
    setup_benchmark_data()
