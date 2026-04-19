import argparse
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector_rl2 import collect_trials
from agent.rl.gae import compute_gae
from agent.rl.rollout_buffer import RolloutBuffer
from agent.rl.ppo_update import ppo_update_step
from agent.eval.evaluate_rl2 import evaluate_policy_rl2
from agent.utils.logger import CSVLogger

from data.splits import make_time_split, make_time_split_from_root
from env.multi_day_task_env import MultiDayTaskEnv


# presets — PPO-LSTM checkpoints used as RL² fine-tuning starting points
PRESETS: Dict[str, Dict[str, object]] = {
    "BTC_PPO_1": {
        "symbol": "BTC",
        "base_run_name": "BTC_S2B3_lr0.001_clip0.2_ent0.005_ep8_bs16_cl32_gamma0.99_lam0.9_seed2",
        "base_ckpt": "checkpoints/BTC/BTC_S2B3_lr0.001_clip0.2_ent0.005_ep8_bs16_cl32_gamma0.99_lam0.9_seed2/best.weights.h5",
        "lr": 0.001,
        "clip_eps": 0.2,
        "ent_coef": 0.005,
        "gamma": 0.99,
        "lam": 0.90,
        "ppo_epochs": 8,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 2,
        "target_qty": 50.0,
        "horizon_steps": 4320,
        "max_child_qty": 0.25,
        "pov_cap": 0.05,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
    "BTC_PPO_2": {
        "symbol": "BTC",
        "base_run_name": "BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.99_lam0.95_seed2",
        "base_ckpt": "checkpoints/BTC/BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.99_lam0.95_seed2/best.weights.h5",
        "lr": 0.001,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "ppo_epochs": 8,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 2,
        "target_qty": 50.0,
        "horizon_steps": 4320,
        "max_child_qty": 0.25,
        "pov_cap": 0.05,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
    "BTC_PPO_3": {
        "symbol": "BTC",
        "base_run_name": "BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.999_lam0.95_seed2",
        "base_ckpt": "checkpoints/BTC/BTC_S2B1_lr0.001_clip0.2_ent0.01_ep8_bs16_cl32_gamma0.999_lam0.95_seed2/best.weights.h5",
        "lr": 0.001,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "gamma": 0.999,
        "lam": 0.95,
        "ppo_epochs": 8,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 2,
        "target_qty": 50.0,
        "horizon_steps": 4320,
        "max_child_qty": 0.25,
        "pov_cap": 0.05,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
    "DOGE_PPO_1": {
        "symbol": "DOGE",
        "base_run_name": "DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0",
        "base_ckpt": "checkpoints/DOGE/DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
        "lr": 0.0003,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "gamma": 0.999,
        "lam": 0.95,
        "ppo_epochs": 4,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 1,
        "target_qty": 3000000.0,
        "horizon_steps": 4320,
        "max_child_qty": 3500.0,
        "pov_cap": 0.10,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
    "DOGE_PPO_2": {
        "symbol": "DOGE",
        "base_run_name": "DOGE_S2B2_lr0.0003_clip0.2_ent0.005_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0",
        "base_ckpt": "checkpoints/DOGE/DOGE_S2B2_lr0.0003_clip0.2_ent0.005_ep4_bs16_cl32_gamma0.999_lam0.95_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
        "lr": 0.0003,
        "clip_eps": 0.2,
        "ent_coef": 0.005,
        "gamma": 0.999,
        "lam": 0.95,
        "ppo_epochs": 4,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 1,
        "target_qty": 3000000.0,
        "horizon_steps": 4320,
        "max_child_qty": 3500.0,
        "pov_cap": 0.10,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
    "DOGE_PPO_3": {
        "symbol": "DOGE",
        "base_run_name": "DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.99_lam0.9_seed1_H4320_Q3e+06_pov0.1_fee0",
        "base_ckpt": "checkpoints/DOGE/DOGE_S2B1_lr0.0003_clip0.2_ent0.01_ep4_bs16_cl32_gamma0.99_lam0.9_seed1_H4320_Q3e+06_pov0.1_fee0/best.weights.h5",
        "lr": 0.0003,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.90,
        "ppo_epochs": 4,
        "batch_size_chunks": 16,
        "chunk_len": 32,
        "seed": 1,
        "target_qty": 3000000.0,
        "horizon_steps": 4320,
        "max_child_qty": 3500.0,
        "pov_cap": 0.10,
        "side": "buy",
        "taker_fee_rate": 0.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="RL² fine-tuning from existing best PPO-LSTM checkpoints.")
    p.add_argument("--preset", type=str, required=True, choices=sorted(PRESETS.keys()))

    p.add_argument("--data_root", type=str, default=None,
                   help="If set, expects subfolders: November/, December/, January/ containing parquet days.")
    p.add_argument("--nov_dir", type=str, default=None)
    p.add_argument("--dec_dir", type=str, default=None)
    p.add_argument("--jan_dir", type=str, default=None)
    p.add_argument("--jan_val_days", type=int, default=7)

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--base_ckpt", type=str, default=None,
                   help="Optional override. If omitted, uses checkpoint from preset.")

    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--rollout_trials", type=int, default=64)
    p.add_argument("--episodes_per_trial", type=int, default=4)
    p.add_argument("--val_trials", type=int, default=50)
    p.add_argument("--val_seed", type=int, default=999)
    p.add_argument("--seed", type=int, default=None,
                   help="Optional override. If omitted, uses seed from preset.")
    return p.parse_args()


def _resolve_split(args):
    if args.data_root is not None:
        return make_time_split_from_root(args.data_root, jan_val_days=args.jan_val_days)
    if args.nov_dir is None or args.dec_dir is None or args.jan_dir is None:
        raise ValueError("Provide either --data_root or all of --nov_dir --dec_dir --jan_dir.")
    return make_time_split(
        nov_dir=args.nov_dir,
        dec_dir=args.dec_dir,
        jan_dir=args.jan_dir,
        jan_val_days=args.jan_val_days,
    )


def _build_model_and_load(obs_dim: int, base_ckpt: str) -> RecurrentActorCritic:
    # warm up with dummy forward pass before loading weights
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    dummy_obs = tf.zeros((1, 1, obs_dim), dtype=tf.float32)
    _ = model(dummy_obs, initial_state=model.initial_state(batch_size=1), training=False)
    model.load_weights(base_ckpt)
    return model


def main():
    args = parse_args()
    cfg = dict(PRESETS[args.preset])

    # fix seeds for reproducibility
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    train_files, val_files, test_files = _resolve_split(args)
    print(f"Train days: {len(train_files)} | Val days: {len(val_files)} | Test days: {len(test_files)}")
    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("Train/Val file lists are empty. Check your parquet folders.")

    symbol = str(cfg["symbol"]).upper()
    base_ckpt = args.base_ckpt or str(cfg["base_ckpt"])

    if args.run_name is None:
        args.run_name = f"RL2_{symbol}_{cfg['base_run_name']}_trial{args.episodes_per_trial}_rt{args.rollout_trials}"

    log_root = os.path.join("logs", f"RL2_{symbol}")
    ckpt_root = os.path.join("checkpoints", f"RL2_{symbol}")
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    log_path = os.path.join(log_root, f"{args.run_name}.csv")
    ckpt_dir = os.path.join(ckpt_root, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    execenv_kwargs = dict(
        horizon_steps=int(cfg["horizon_steps"]),
        side=str(cfg["side"]),
        target_qty=float(cfg["target_qty"]),
        max_child_qty=float(cfg["max_child_qty"]),
        pov_cap=float(cfg["pov_cap"]),
        taker_fee_rate=float(cfg["taker_fee_rate"]),
    )

    env_train = MultiDayTaskEnv(train_files, seed=seed, **execenv_kwargs)
    env_val = MultiDayTaskEnv(val_files, seed=args.val_seed, **execenv_kwargs)
    obs_dim = env_train.observation_space.shape[0]

    model = _build_model_and_load(obs_dim=obs_dim, base_ckpt=base_ckpt)
    policy = LSTMPolicy(model)

    lr = float(cfg["lr"])
    clip_eps = float(cfg["clip_eps"])
    vf_coef = float(args.vf_coef)
    ent_coef = float(cfg["ent_coef"])
    max_grad_norm = float(args.max_grad_norm)
    gamma = float(cfg["gamma"])
    lam = float(cfg["lam"])
    chunk_len = int(cfg["chunk_len"])
    batch_size_chunks = int(cfg["batch_size_chunks"])
    ppo_epochs = int(cfg["ppo_epochs"])
    target_kl = float(args.target_kl)
    patience = int(args.patience)
    rollout_trials = int(args.rollout_trials)
    episodes_per_trial = int(args.episodes_per_trial)
    val_trials = int(args.val_trials)

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    logger = CSVLogger(log_path)
    buf = RolloutBuffer(chunk_len=chunk_len)

    best_val_is = float("inf")
    bad_iters = 0

    print("\n=== RL2 RUN CONFIG ===")
    print(f"preset={args.preset}")
    print(f"run_name={args.run_name}")
    print(f"symbol={symbol}")
    print(f"base_ckpt={base_ckpt}")
    print(f"rollout_trials={rollout_trials} episodes_per_trial={episodes_per_trial} val_trials={val_trials}")
    print(f"lr={lr} clip_eps={clip_eps} vf_coef={vf_coef} ent_coef={ent_coef} max_grad_norm={max_grad_norm}")
    print(f"gamma={gamma} lam={lam} chunk_len={chunk_len} batch_size_chunks={batch_size_chunks} ppo_epochs={ppo_epochs}")
    print(f"seed={seed} val_seed={args.val_seed}")
    print(f"execenv={execenv_kwargs}")
    print(f"log_path={log_path}")
    print(f"ckpt_dir={ckpt_dir}\n")

    # RL² training loop — hidden state persists across episodes within each trial
    for it in range(1, 10_000):
        t0 = time.time()

        rollout, ep_stats = collect_trials(
            env_train,
            policy,
            n_trials=rollout_trials,
            episodes_per_trial=episodes_per_trial,
            deterministic=False,
        )

        adv, ret = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], gamma=gamma, lam=lam
        )
        arrays = buf.make_training_arrays(rollout, adv, ret)

        losses, kls, entropies, clipfracs, gradnorms = [], [], [], [], []
        vlosses, pilosses = [], []

        for _epoch in range(ppo_epochs):
            for mb in buf.iter_minibatches(arrays, batch_size_chunks=batch_size_chunks, shuffle=True):
                total_loss, pi_loss, v_loss, entropy, approx_kl, clip_frac, grad_norm = ppo_update_step(
                    model,
                    opt,
                    mb.obs,
                    mb.raw_u,
                    mb.logp_old,
                    mb.adv,
                    mb.returns,
                    mb.mask,
                    mb.h0,
                    mb.c0,
                    clip_eps=clip_eps,
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    max_grad_norm=max_grad_norm,
                )
                losses.append(float(total_loss))
                pilosses.append(float(pi_loss))
                vlosses.append(float(v_loss))
                entropies.append(float(entropy))
                kls.append(float(approx_kl))
                clipfracs.append(float(clip_frac))
                gradnorms.append(float(grad_norm))

                if float(approx_kl) > 1.5 * target_kl:
                    break
            if len(kls) > 0 and float(kls[-1]) > 1.5 * target_kl:
                break

        val_metrics = evaluate_policy_rl2(
            env_val,
            policy,
            n_trials=val_trials,
            episodes_per_trial=episodes_per_trial,
            deterministic=True,
        )

        improved = val_metrics.mean_is_bps < best_val_is
        if improved:
            best_val_is = val_metrics.mean_is_bps
            bad_iters = 0
            model.save_weights(os.path.join(ckpt_dir, "best.weights.h5"))  # save on improvement
        else:
            bad_iters += 1

        row = {
            "iter": it,
            "run_name": args.run_name,
            "symbol": symbol,
            "preset": args.preset,
            "base_ckpt": base_ckpt,
            "base_run_name": cfg["base_run_name"],
            "steps_collected": len(rollout["rewards"]),
            "train_ep_mean_return": float(np.mean([s.episode_return for s in ep_stats])),
            "train_ep_mean_steps": float(np.mean([s.steps for s in ep_stats])),
            "loss_total": float(np.mean(losses)) if losses else np.nan,
            "loss_pi": float(np.mean(pilosses)) if pilosses else np.nan,
            "loss_v": float(np.mean(vlosses)) if vlosses else np.nan,
            "entropy": float(np.mean(entropies)) if entropies else np.nan,
            "approx_kl": float(np.mean(kls)) if kls else np.nan,
            "clip_frac": float(np.mean(clipfracs)) if clipfracs else np.nan,
            "grad_norm": float(np.mean(gradnorms)) if gradnorms else np.nan,
            "val_mean_is_bps": val_metrics.mean_is_bps,
            "val_mean_completion": val_metrics.mean_completion,
            "val_mean_return": val_metrics.mean_return,
            "val_n_trials": val_metrics.n_trials,
            "val_episodes_per_trial": val_metrics.episodes_per_trial,
            "time_sec": time.time() - t0,
            "best_val_is_bps": best_val_is,
            "bad_iters": bad_iters,
            "lr": lr,
            "clip_eps": clip_eps,
            "vf_coef": vf_coef,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "gamma": gamma,
            "lam": lam,
            "chunk_len": chunk_len,
            "batch_size_chunks": batch_size_chunks,
            "ppo_epochs": ppo_epochs,
            "target_kl": target_kl,
            "seed": seed,
            "val_seed": args.val_seed,
            "rollout_trials": rollout_trials,
            "episodes_per_trial": episodes_per_trial,
            **execenv_kwargs,
        }
        logger.log(row)

        print(
            f"run={args.run_name} it={it} steps={row['steps_collected']} "
            f"loss={row['loss_total']:.4g} kl={row['approx_kl']:.4g} ent={row['entropy']:.4g} "
            f"valIS={row['val_mean_is_bps']:.6g} best={best_val_is:.6g} bad={bad_iters} "
            f"time={row['time_sec']:.2f}s"
        )

        if bad_iters >= patience:
            print(f"Early stopping: no val improvement for {patience} iterations.")
            break


if __name__ == "__main__":
    main()