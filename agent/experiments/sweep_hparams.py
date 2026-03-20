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
# DOGE: data + env knobs
# -----------------------------
DOGE_SYMBOL = "DOGE"
DOGE_DATA_ROOT = "/Users/jackfletcher/Desktop/FYP_Data/replay_5s_DOGE"

DOGE_ENV_KWARGS = dict(
    horizon_steps=4320,
    side="buy",
    target_qty=3000000.0,
    max_child_qty=3500.0,
    pov_cap=0.10,
    taker_fee_rate=0.0,
)


def read_best_val_is_bps(csv_path: str) -> float:
    best = float("inf")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "best_val_is_bps" not in reader.fieldnames:
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


def get_run_paths(run_name: str):
    csv_path = os.path.join("logs", DOGE_SYMBOL, f"{run_name}.csv")
    ckpt_dir = os.path.join("checkpoints", DOGE_SYMBOL, run_name)
    best_ckpt_path = os.path.join(ckpt_dir, "best.weights.h5")
    return csv_path, ckpt_dir, best_ckpt_path


def is_run_complete(config: Dict) -> bool:
    """
    Treat a run as complete if its CSV exists and contains at least one parseable
    best_val_is_bps. Optionally require best checkpoint too.
    """
    run_name = config["run_name"]
    csv_path, _, best_ckpt_path = get_run_paths(run_name)

    if not os.path.exists(csv_path):
        return False

    try:
        _ = read_best_val_is_bps(csv_path)
    except Exception:
        return False

    # If you want to require checkpoint existence too, uncomment:
    # if not os.path.exists(best_ckpt_path):
    #     return False

    return True


def load_existing_result(config: Dict) -> RunResult:
    run_name = config["run_name"]
    csv_path, _, best_ckpt_path = get_run_paths(run_name)
    best_val = read_best_val_is_bps(csv_path)
    return RunResult(
        run_name=run_name,
        config=config,
        best_val_is_bps=best_val,
        best_ckpt_path=best_ckpt_path,
        csv_path=csv_path,
    )


def run_training(config: Dict) -> RunResult:
    run_name = config["run_name"]

    csv_path, _, best_ckpt_path = get_run_paths(run_name)

    cmd = [
        "python",
        "-m",
        "agent.experiments.train_ppo_lstm",

        # DATA / IDENTITY
        "--symbol", DOGE_SYMBOL,
        "--data_root", DOGE_DATA_ROOT,
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

        # ENV KNOBS
        "--horizon_steps", str(config["horizon_steps"]),
        "--side", str(config["side"]),
        "--target_qty", str(config["target_qty"]),
        "--max_child_qty", str(config["max_child_qty"]),
        "--pov_cap", str(config["pov_cap"]),
        "--taker_fee_rate", str(config["taker_fee_rate"]),
    ]

    os.makedirs(os.path.join("logs", DOGE_SYMBOL), exist_ok=True)
    os.makedirs(os.path.join("checkpoints", DOGE_SYMBOL), exist_ok=True)

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
    Stage 2: gamma/lambda sweep over top-3 Stage 1 base configs.
    Total runs = 3 bases * 3 gammas * 2 lams * 3 seeds = 54
    """

    base_configs = [
        dict(lr=3e-4, clip_eps=0.2, ent_coef=0.01,  ppo_epochs=4, batch_size_chunks=16, chunk_len=32),
        dict(lr=3e-4, clip_eps=0.2, ent_coef=0.005, ppo_epochs=4, batch_size_chunks=16, chunk_len=32),
        dict(lr=3e-4, clip_eps=0.2, ent_coef=0.01,  ppo_epochs=8, batch_size_chunks=16, chunk_len=32),
    ]

    gammas = [0.99, 0.995, 0.999]
    lams   = [0.90, 0.95]
    seeds = [1, 2, 3]
    val_seed = 999

    rollout_episodes = 16
    patience = 20

    horizon_steps = DOGE_ENV_KWARGS["horizon_steps"]
    side = DOGE_ENV_KWARGS["side"]
    target_qty = DOGE_ENV_KWARGS["target_qty"]
    max_child_qty = DOGE_ENV_KWARGS["max_child_qty"]
    pov_cap = DOGE_ENV_KWARGS["pov_cap"]
    taker_fee_rate = DOGE_ENV_KWARGS["taker_fee_rate"]

    configs = []

    for base_i, base in enumerate(base_configs, start=1):
        for gamma, lam, seed in itertools.product(gammas, lams, seeds):
            run_name = (
                f"{DOGE_SYMBOL}_S2B{base_i}_"
                f"lr{base['lr']:g}_clip{base['clip_eps']:g}_ent{base['ent_coef']:g}_ep{base['ppo_epochs']}"
                f"_bs{base['batch_size_chunks']}_cl{base['chunk_len']}"
                f"_gamma{gamma:g}_lam{lam:g}_seed{seed}"
                f"_H{horizon_steps}_Q{target_qty:g}_pov{pov_cap:g}_fee{taker_fee_rate:g}"
            )

            configs.append({
                "run_name": run_name,
                "lr": base["lr"],
                "clip_eps": base["clip_eps"],
                "ent_coef": base["ent_coef"],
                "ppo_epochs": base["ppo_epochs"],
                "batch_size_chunks": base["batch_size_chunks"],
                "chunk_len": base["chunk_len"],
                "gamma": gamma,
                "lam": lam,
                "seed": seed,
                "val_seed": val_seed,
                "rollout_episodes": rollout_episodes,
                "patience": patience,
                "horizon_steps": horizon_steps,
                "side": side,
                "target_qty": target_qty,
                "max_child_qty": max_child_qty,
                "pov_cap": pov_cap,
                "taker_fee_rate": taker_fee_rate,
            })

    return configs


def write_summary_csv(path: str, results: List[RunResult]) -> None:
    if not results:
        return

    fieldnames = ["run_name", "best_val_is_bps", "ckpt", "csv"]
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
    summary_path = os.path.join("logs", DOGE_SYMBOL, "hparam_sweep_stage2_gamma_lambda.csv")

    for i, cfg in enumerate(configs, start=1):
        run_name = cfg["run_name"]
        print(f"\n[{i}/{len(configs)}] {run_name}")

        try:
            if is_run_complete(cfg):
                res = load_existing_result(cfg)
                results.append(res)
                print(f"⏭️  Skipping completed run: {res.run_name} (best_val_is_bps={res.best_val_is_bps:.6f})")
            else:
                res = run_training(cfg)
                results.append(res)
                print(f"✅ {res.run_name}: best_val_is_bps={res.best_val_is_bps:.6f}")

            # Save progress after every successful/loaded run
            write_summary_csv(summary_path, results)

        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed: {run_name} ({e})")
        except Exception as e:
            print(f"❌ Error in run {run_name}: {e}")

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

    write_summary_csv(summary_path, results)
    print(f"\nSaved sweep summary to: {summary_path}")


if __name__ == "__main__":
    main()
