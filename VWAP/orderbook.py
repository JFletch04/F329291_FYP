from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


@dataclass
class OrderBook:
    """
    Simple L2 order book that supports:
    - apply_snapshot(bids, asks)
    - apply_delta(bids, asks)
    - best_bid(), best_ask()
    
    bids/asks are stored as dict: price -> size
    Prices are floats here for simplicity. (Later we can switch to int ticks for safety/speed.)
    """
    bids: Dict[float, float]
    asks: Dict[float, float]
    initialized: bool = False

    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.initialized = False

    def apply_snapshot(self, bids: List[List[str]], asks: List[List[str]]) -> None:
        """
        bids/asks format: [[price_str, size_str], ...]
        """
        self.bids.clear()
        self.asks.clear()
        self._apply_levels(self.bids, bids)
        self._apply_levels(self.asks, asks)
        self.initialized = True

    def apply_delta(self, bids: List[List[str]], asks: List[List[str]]) -> None:
        """
        Delta updates:
        - if size == 0: remove the level
        - else: set/update size
        """
        if not self.initialized:
            # Can't safely apply deltas without an initial snapshot
            return
        self._apply_levels(self.bids, bids)
        self._apply_levels(self.asks, asks)

    @staticmethod
    def _apply_levels(side: Dict[float, float], levels: List[List[str]]) -> None:
        for p_str, q_str in levels:
            p = float(p_str)
            q = float(q_str)
            if q == 0.0:
                side.pop(p, None)
            else:
                side[p] = q

    def best_bid(self) -> Optional[Tuple[float, float]]:
        """
        Returns (price, size) of best bid, or None.
        """
        if not self.bids:
            return None
        p = max(self.bids)
        return p, self.bids[p]

    def best_ask(self) -> Optional[Tuple[float, float]]:
        """
        Returns (price, size) of best ask, or None.
        """
        if not self.asks:
            return None
        p = min(self.asks)
        return p, self.asks[p]

    def mid_price(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb[0] + ba[0])


def iter_orderbook_messages(jsonl_path: Path) -> Iterator[dict]:
    """
    Yields raw JSON messages from a JSONL file (one JSON object per line).
    """
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def replay_orderbook(jsonl_path: Path) -> Iterator[Tuple[int, OrderBook]]:
    """
    Streams through a JSONL day file and yields (ts, book) after each update.
    NOTE: 'book' is mutable; if you need a snapshot copy, copy bids/asks.
    """
    book = OrderBook()

    for msg in iter_orderbook_messages(jsonl_path):
        ts = int(msg["ts"])
        typ = msg.get("type")
        data = msg.get("data", {})

        if typ == "snapshot":
            book.apply_snapshot(data.get("b", []), data.get("a", []))
        elif typ == "delta":
            book.apply_delta(data.get("b", []), data.get("a", []))
        else:
            # ignore unknown types
            continue

        yield ts, book
