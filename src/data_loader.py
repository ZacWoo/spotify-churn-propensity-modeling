"""
data_loader.py
--------------
Download the Spotify User Behavior dataset via Kaggle API and return a
clean pandas DataFrame.
"""

import kagglehub
import pandas as pd
from pathlib import Path


DATASET_SLUG = "sahilislam007/spotify-user-behavior-and-pattern"


def load_dataset() -> pd.DataFrame:
    """Download (or use cached) dataset and return as DataFrame."""
    path = kagglehub.dataset_download(DATASET_SLUG)
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    df = pd.read_csv(csv_files[0])
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {csv_files[0].name}")
    return df


def print_schema(df: pd.DataFrame) -> None:
    """Print a quick schema overview."""
    print("\n--- Dataset Schema ---")
    for col in df.columns:
        print(f"  {col:40s}  {str(df[col].dtype):10s}  "
              f"nunique={df[col].nunique():>6}  nulls={df[col].isnull().sum()}")
    print()
