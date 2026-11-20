"""
data_engineering.ingestion
~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads raw NASA C-MAPSS text files into structured pandas DataFrames.

Each C-MAPSS file (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) is a
space-separated file with no header. This module attaches correct column names
and returns tidy DataFrames ready for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from config import DATA_DIR, RAW_COLUMNS, DATASETS, PRIMARY_DATA_DIR, SECONDARY_DATA_DIR

logger = logging.getLogger(__name__)

RAW_DATASET_IDS = ["FD001", "FD002", "FD003", "FD004"]


def normalize_dataset_id(dataset_id: str) -> str:
    """Normalize any dataset label/alias to standard C-MAPSS ID (FD001-FD004)."""
    raw = str(dataset_id).strip()
    upper = raw.upper()
    if "FD001" in upper or "PRIMARY" in upper or "02_NASA" in upper:
        return "FD001"
    if "FD002" in upper:
        return "FD002"
    if "FD003" in upper:
        return "FD003"
    if "FD004" in upper:
        return "FD004"
    return raw


def _read_raw(filepath: Path) -> pd.DataFrame:
    """Read a single C-MAPSS space-delimited file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
        engine="python",
    )
    # Drop trailing NaN columns that sometimes appear due to trailing spaces
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def load_dataset(
    dataset_id: str = "FD001",
    data_dir: Path | None = None,
    split: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.Series], pd.DataFrame]:
    """Load train, test and RUL ground-truth for one C-MAPSS sub-dataset.

    Parameters
    ----------
    dataset_id : str
        One of ``"FD001"``, ``"FD002"``, ``"FD003"``, ``"FD004"``, or full alias.
    data_dir : Path, optional
        Directory containing the raw ``.txt`` files. Defaults to
        ``config.DATA_DIR`` (02_NASA_CMAPSS_Turbofan).
    split : str, optional
        If "train", returns only train_df.
        If "test", returns only test_df.
        If None (default), returns (train_df, test_df, rul_series).

    Returns
    -------
    train_df, test_df, rul_series (if split is None)
    or individual DataFrame for the requested split.
    """
    canon_id = normalize_dataset_id(dataset_id)
    if canon_id not in RAW_DATASET_IDS:
        raise ValueError(
            f"dataset_id must resolve to one of {RAW_DATASET_IDS}, got {dataset_id!r}"
        )

    # Search candidates: provided data_dir, DATA_DIR, PRIMARY_DATA_DIR, SECONDARY_DATA_DIR
    candidate_dirs = []
    if data_dir is not None:
        candidate_dirs.append(Path(data_dir))
    candidate_dirs.extend([DATA_DIR, PRIMARY_DATA_DIR, SECONDARY_DATA_DIR])

    base = None
    for cand in candidate_dirs:
        if (cand / f"train_{canon_id}.txt").exists():
            base = cand
            break

    if base is None:
        base = Path(data_dir) if data_dir is not None else DATA_DIR

    train_path = base / f"train_{canon_id}.txt"
    test_path = base / f"test_{canon_id}.txt"
    rul_path = base / f"RUL_{canon_id}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(
                f"C-MAPSS file not found: {p}\n"
                f"Download the NASA C-MAPSS dataset and place the .txt files in {base}/"
            )

    logger.info("Loading %s (canonical: %s) from %s …", dataset_id, canon_id, base)
    train_df = _read_raw(train_path)
    test_df = _read_raw(test_path)
    rul_series = pd.read_csv(rul_path, header=None, names=["rul"]).squeeze("columns")

    logger.info(
        "%s — train: %d rows / %d engines | test: %d rows / %d engines",
        canon_id,
        len(train_df),
        train_df["unit_id"].nunique(),
        len(test_df),
        test_df["unit_id"].nunique(),
    )
    if split == "train":
        return train_df
    elif split == "test":
        return test_df
    return train_df, test_df, rul_series


def load_all_datasets(
    data_dir: Path | None = None,
) -> dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
    """Convenience wrapper: loads all four C-MAPSS sub-datasets."""
    return {ds: load_dataset(ds, data_dir=data_dir) for ds in RAW_DATASET_IDS}
