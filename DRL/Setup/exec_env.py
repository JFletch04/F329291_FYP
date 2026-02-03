import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

def walk_book_market(qty, prices, sizes):
    filled = 0.0
    notional = 0.0
    for p, s in zip(prices, sizes):
        if filled >= qty:
            break
        take = min(qty - filled, float(s))
        if take > 0:
            filled += take
            notional += take * float(p)
    if filled == 0:
        return 0.0, np.nan
    return filled, notional / filled


class ExecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        replay_parquet_path: str,
        horizon_steps: int = 180,
        side: str = "buy",
        target_qty: float = 0.5,      # base units for now
        max_child_qty: float = 0.05,  # cap per step
        pov_cap: float = 0.10,        # <= 10% of last 5s traded volume
        taker_fee_rate: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.df = pd.read_parquet(replay_parquet_path).reset_index(drop=True)

        self.horizon_steps = horizon_steps
        self.side = side.lower()
        self.target_qty = float(target_qty)
        self.max_child_qty = float(max_child_qty)
        self.pov_cap = float(pov_cap)
        self.taker_fee_rate = float(taker_fee_rate)

        self.rng = np.random.default_rng(seed)

        # Observation: [spread, trade_vol, signed_vol, imbalance, ret_1, remaining_frac, time_remaining_frac]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Action: fraction of remaining to trade now
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Episode state
        self.start_idx = None
        self.t = None
        self.remaining_qty = None
        self.arrival_mid = None
        self.filled_total = None
        self.notional_total = None
        self.is_cash_total = None

    def _get_row(self, idx):
        return self.df.iloc[idx]

    def _imbalance_top5(self, row):
        bid_sizes = np.array(row["bid_sizes"], dtype=float)
        ask_sizes = np.array(row["ask_sizes"], dtype=float)
        B = bid_sizes.sum()
        A = ask_sizes.sum()
        return float((B - A) / (B + A + 1e-12))

    def _ret_1(self, idx):
        if idx <= 0:
            return 0.0
        m0 = float(self.df.iloc[idx - 1]["mid"])
        m1 = float(self.df.iloc[idx]["mid"])
        return float((m1 - m0) / (m0 + 1e-12))

    def _obs(self):
        idx = self.start_idx + self.t
        row = self._get_row(idx)

        spread = float(row["spread"])
        trade_vol = float(row["trade_vol"])
        signed_vol = float(row["signed_vol"])
        imb = self._imbalance_top5(row)
        ret1 = self._ret_1(idx)

        remaining_frac = float(self.remaining_qty / self.target_qty)
        time_remaining_frac = float((self.horizon_steps - self.t) / self.horizon_steps)

        obs = np.array(
            [spread, trade_vol, signed_vol, imb, ret1, remaining_frac, time_remaining_frac],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.df) - self.horizon_steps - 1
        self.start_idx = int(self.rng.integers(1, max_start))  # start at >=1 for ret_1
        self.t = 0

        self.remaining_qty = self.target_qty
        self.filled_total = 0.0
        self.notional_total = 0.0
        self.is_cash_total = 0.0

        self.arrival_mid = float(self.df.iloc[self.start_idx]["mid"])

        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], 0.0, 1.0))
        idx = self.start_idx + self.t
        row = self._get_row(idx)

        mid_t = float(row["mid"])
        trade_vol = float(row["trade_vol"])

        # Convert action -> child qty
        child_qty = a * self.remaining_qty
        child_qty = min(child_qty, self.max_child_qty)

        # POV cap (avoid unrealistically large trades)
        if trade_vol > 0:
            child_qty = min(child_qty, self.pov_cap * trade_vol)

        # Simulate fill
        if self.side == "buy":
            prices = row["ask_prices"]
            sizes = row["ask_sizes"]
        else:
            prices = row["bid_prices"]
            sizes = row["bid_sizes"]

        filled, fill_price = walk_book_market(child_qty, prices, sizes)

        # Update totals
        if filled > 0:
            self.remaining_qty -= filled
            self.filled_total += filled
            self.notional_total += filled * fill_price

            fee_cash = self.taker_fee_rate * (filled * fill_price)

            # Per-step cost vs current mid (stable learning signal)
            if self.side == "buy":
                step_cost = filled * (fill_price - mid_t) + fee_cash
            else:
                step_cost = filled * (mid_t - fill_price) + fee_cash

            self.is_cash_total += step_cost
        else:
            step_cost = 0.0

        # Reward (negative cost, normalised)
        denom = self.target_qty * self.arrival_mid
        reward = - (step_cost / denom) if denom > 0 else 0.0

        # Advance time
        self.t += 1
        done = False

        # Terminal: force liquidation at last step
        if self.t >= self.horizon_steps or self.remaining_qty <= 1e-12:
            done = True
            # If leftover, force liquidate using current row book
            if self.remaining_qty > 1e-12:
                # use last available row
                last_row = self._get_row(self.start_idx + self.horizon_steps - 1)
                last_mid = float(last_row["mid"])
                if self.side == "buy":
                    p = last_row["ask_prices"]
                    s = last_row["ask_sizes"]
                else:
                    p = last_row["bid_prices"]
                    s = last_row["bid_sizes"]

                filled2, price2 = walk_book_market(self.remaining_qty, p, s)
                if filled2 > 0:
                    self.remaining_qty -= filled2
                    self.filled_total += filled2
                    self.notional_total += filled2 * price2
                    fee2 = self.taker_fee_rate * (filled2 * price2)

                    if self.side == "buy":
                        term_cost = filled2 * (price2 - last_mid) + fee2
                    else:
                        term_cost = filled2 * (last_mid - price2) + fee2

                    self.is_cash_total += term_cost
                    reward += - (term_cost / denom)

        exec_vwap = (self.notional_total / self.filled_total) if self.filled_total > 0 else np.nan

        info = {
            "filled_total": self.filled_total,
            "exec_vwap": exec_vwap,
            "arrival_mid": self.arrival_mid,
            "cost_cash_vs_mid": self.is_cash_total,
            "remaining_qty": self.remaining_qty,
        }

        obs = self._obs() if not done else self._obs()  # safe; final obs ok
        return obs, float(reward), done, False, info

env = ExecEnv("/Users/jackfletcher/Desktop/FYP_Data/2026-01-01_steps_5s.parquet")
obs, _ = env.reset()

done = False
total = 0
while not done:
    action = env.action_space.sample()
    obs, r, done, _, info = env.step(action)
    total += r

print("Return:", total)
print("Info:", info)

