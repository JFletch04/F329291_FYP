# data/splits.py
from pathlib import Path
from typing import List, Tuple


def list_parquets(folder: str) -> List[str]:
    p = Path(folder)
    files = sorted([str(f) for f in p.glob("*.parquet")])
    return files


def make_time_split(
    nov_dir: str,
    dec_dir: str,
    jan_dir: str,
    jan_val_days: int = 7,
) -> Tuple[List[str], List[str], List[str]]:
    nov = list_parquets(nov_dir)
    dec = list_parquets(dec_dir)
    jan = list_parquets(jan_dir)

    # nov + dec = train, first n jan days = val, remainder = test
    train_files = nov + dec
    val_files = jan[:jan_val_days]
    test_files = jan[jan_val_days:]
    return train_files, val_files, test_files


def make_time_split_from_root(
    data_root: str,
    jan_val_days: int = 7,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Expects:
      <data_root>/November/*.parquet
      <data_root>/December/*.parquet
      <data_root>/January/*.parquet
    """
    root = Path(data_root)
    nov_dir = str(root / "November")
    dec_dir = str(root / "December")
    jan_dir = str(root / "January")
    return make_time_split(nov_dir, dec_dir, jan_dir, jan_val_days=jan_val_days)