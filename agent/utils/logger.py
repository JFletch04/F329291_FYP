import csv
import os
from typing import Dict, Any


class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._header_written = os.path.exists(path)

    def log(self, row: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
