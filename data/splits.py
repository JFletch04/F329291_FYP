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
    jan_val_days: int = 7
) -> Tuple[List[str], List[str], List[str]]:
    nov = list_parquets(nov_dir)
    dec = list_parquets(dec_dir)
    jan = list_parquets(jan_dir)

    train_files = nov + dec
    val_files = jan[:jan_val_days]
    test_files = jan[jan_val_days:]

    return train_files, val_files, test_files
