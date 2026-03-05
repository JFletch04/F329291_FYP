import itertools
import json
import os
import subprocess
import csv
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RunResult:
    run_name: str
    config: Dict
    best_val_is_bps: float
    best_ckpt_path: str
    csv_path: str


# -----------------------------
# USER: set DOGE data + env knobs here
# -----------------------------
BTC_DATA_ROOT = "/Users/jackfletcher/Desktop/FYP_Data/replay_5s_BTC"

BTC_ENV_KWARGS = dict(
    horizon_steps=4320,
    side="buy",
    target_qty=50.0,
    max_child_qty=0.25,
    pov_cap=0.05,
    taker_fee_rate=0.0,
)

BTC_SYMBOL = "BTC"


def read_best_val_is_bps(csv_path: str) -> float:
    """
    Reads a train CSV log and returns min(best_val_is_bps) across all rows.
    Uses stdlib csv module to avoid pandas dependency issues.
    """
    best = float("inf")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "best_val_is_bps" not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} missing best_val_is_bps column. Found: {reader.fieldnames}")
        for row in reader:
            try:
                v = float(row["best_val_is_bps"])
                if v < best:
                    best = v
            except Exception:
                pass
    if best == float("inf"):
        raise ValueError(f"Could not parse any best_val_is_bps values in {csv_path}")
    return best


def run_training(config: Dict) -> RunResult:
    run_name = config["run_name"]

    # IMPORTANT: per-symbol folders (matches updated train_ppo_lstm.py)
    csv_path = os.path.join("logs", BTC_SYMBOL, f"{run_name}.csv")
    ckpt_dir = os.path.join("checkpoints", BTC_SYMBOL, run_name)
    best_ckpt_path = os.path.join(ckpt_dir, "best.weights.h5")

    cmd = [
        "python",
        "-m",
        "agent.experiments.train_ppo_lstm",

        # DATA / IDENTITY
        "--symbol", BTC_SYMBOL,
        "--data_root", BTC_DATA_ROOT,
        "--run_name", run_name,

        # PPO HYPERPARAMS
        "--lr", str(config["lr"]),
        "--clip_eps", str(config["clip_eps"]),
        "--ent_coef", str(config["ent_coef"]),
        "--ppo_epochs", str(config["ppo_epochs"]),
        "--batch_size_chunks", str(config["batch_size_chunks"]),
        "--chunk_len", str(config["chunk_len"]),
        "--gamma", str(config["gamma"]),
        "--lam", str(config["lam"]),

        # BUDGET / EARLY STOP
        "--rollout_episodes", str(config["rollout_episodes"]),
        "--patience", str(config["patience"]),

        # SEEDS
        "--seed", str(config["seed"]),
        "--val_seed", str(config["val_seed"]),

        # ENV KNOBS (DOGE)
        "--horizon_steps", str(config["horizon_steps"]),
        "--side", str(config["side"]),
        "--target_qty", str(config["target_qty"]),
        "--max_child_qty", str(config["max_child_qty"]),
        "--pov_cap", str(config["pov_cap"]),
        "--taker_fee_rate", str(config["taker_fee_rate"]),
    ]

    # Ensure base folders exist (train_ppo_lstm will also create them)
    os.makedirs(os.path.join("logs", BTC_SYMBOL), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", BTC_SYMBOL), exist_ok=True)

    print(f"\n=== Running: {run_name} ===")
    subprocess.run(cmd, check=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing CSV log at {csv_path}")

    best_val = read_best_val_is_bps(csv_path)

    return RunResult(
        run_name=run_name,
        config=config,
        best_val_is_bps=best_val,
        best_ckpt_path=best_ckpt_path,
        csv_path=csv_path,
    )


def make_configs() -> List[Dict]:
    """
    Stage 2: sweep ONLY gamma/lambda over the top-3 base configs from Stage 1.
    Keeps lr/clip/ent/epochs fixed to those top configs.
    """

    # ---- Top 3 base configs (from Stage 1) ----
    base_configs = [
        # #1
        dict(lr=0.001, clip_eps=0.2, ent_coef=0.01,  ppo_epochs=8, batch_size_chunks=16, chunk_len=32),
        # #2
        dict(lr=0.001, clip_eps=0.1, ent_coef=0.01,  ppo_epochs=8, batch_size_chunks=16, chunk_len=32),
        # #3
        dict(lr=0.001, clip_eps=0.2, ent_coef=0.005, ppo_epochs=8, batch_size_chunks=16, chunk_len=32),
    ]

    # ---- Stage 2 sweep grid ----
    gammas = [0.99, 0.995, 0.999]
    lams   = [0.90, 0.95]

    seeds = [1, 2, 3]
    val_seed = 999

    rollout_episodes = 16
    patience = 20

    # ---- Env knobs (unchanged) ----
    horizon_steps = BTC_ENV_KWARGS["horizon_steps"]
    side = BTC_ENV_KWARGS["side"]
    target_qty = BTC_ENV_KWARGS["target_qty"]
    max_child_qty = BTC_ENV_KWARGS["max_child_qty"]
    pov_cap = BTC_ENV_KWARGS["pov_cap"]
    taker_fee_rate = BTC_ENV_KWARGS["taker_fee_rate"]

    configs = []

    for base_i, base in enumerate(base_configs, start=1):
        for gamma, lam, seed in itertools.product(gammas, lams, seeds):
            run_name = (
                f"{BTC_SYMBOL}_S2B{base_i}_"
                f"lr{base['lr']:g}_clip{base['clip_eps']:g}_ent{base['ent_coef']:g}_ep{base['ppo_epochs']}"
                f"_bs{base['batch_size_chunks']}_cl{base['chunk_len']}"
                f"_gamma{gamma:g}_lam{lam:g}_seed{seed}"
            )

            configs.append({
                "run_name": run_name,

                # PPO knobs (fixed from base config)
                "lr": base["lr"],
                "clip_eps": base["clip_eps"],
                "ent_coef": base["ent_coef"],
                "ppo_epochs": base["ppo_epochs"],
                "batch_size_chunks": base["batch_size_chunks"],
                "chunk_len": base["chunk_len"],

                # Stage 2 sweep knobs
                "gamma": gamma,
                "lam": lam,

                # Seeds/budget
                "seed": seed,
                "val_seed": val_seed,
                "rollout_episodes": rollout_episodes,
                "patience": patience,

                # Env knobs
                "horizon_steps": horizon_steps,
                "side": side,
                "target_qty": target_qty,
                "max_child_qty": max_child_qty,
                "pov_cap": pov_cap,
                "taker_fee_rate": taker_fee_rate,
            })

    return configs


def write_summary_csv(path: str, results: List[RunResult]) -> None:
    """
    Write a flat summary csv with stdlib csv module.
    """
    if not results:
        return

    # union keys
    fieldnames = ["run_name", "best_val_is_bps", "ckpt", "csv"]
    # include config keys
    cfg_keys = sorted({k for r in results for k in r.config.keys() if k != "run_name"})
    fieldnames.extend(cfg_keys)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "run_name": r.run_name,
                "best_val_is_bps": r.best_val_is_bps,
                "ckpt": r.best_ckpt_path,
                "csv": r.csv_path,
            }
            for k in cfg_keys:
                row[k] = r.config.get(k)
            w.writerow(row)


def main():
    configs = make_configs()
    print(f"Total configs: {len(configs)}")

    results: List[RunResult] = []

    for cfg in configs:
        try:
            res = run_training(cfg)
            results.append(res)
            print(f"✅ {res.run_name}: best_val_is_bps={res.best_val_is_bps:.6f}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed: {cfg['run_name']} ({e})")
        except Exception as e:
            print(f"❌ Error in run {cfg['run_name']}: {e}")

    if not results:
        raise RuntimeError("No successful runs completed.")

    results.sort(key=lambda r: r.best_val_is_bps)
    top3 = results[:3]

    print("\n=== TOP 3 CONFIGS (by best_val_is_bps, lower is better) ===")
    for i, r in enumerate(top3, 1):
        print(f"\n#{i}: {r.run_name}")
        print(f"  best_val_is_bps: {r.best_val_is_bps:.6f}")
        print(f"  ckpt: {r.best_ckpt_path}")
        print(f"  log:  {r.csv_path}")
        print(f"  config: {json.dumps({k: v for k, v in r.config.items() if k != 'run_name'}, indent=2)}")

    # Save full summary in logs/DOGE/
    summary_path = os.path.join("logs", BTC_SYMBOL, "hparam_sweep2_summary.csv")
    write_summary_csv(summary_path, results)
    print(f"\nSaved sweep summary to: {summary_path}")


if __name__ == "__main__":
    main()