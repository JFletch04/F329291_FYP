# agent/experiments/train_ppo_lstm.py

import os
import time
import argparse
import random
import numpy as np
import tensorflow as tf

from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import collect_episodes
from agent.rl.gae import compute_gae
from agent.rl.rollout_buffer import RolloutBuffer
from agent.rl.ppo_update import ppo_update_step
from agent.eval.evaluate import evaluate_policy
from agent.utils.logger import CSVLogger

from data.splits import make_time_split, make_time_split_from_root
from env.multi_day_env import MultiDayExecEnv


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO-LSTM optimal execution agent.")

    # Identity / outputs
    p.add_argument("--symbol", type=str, default="BTC", help="Asset tag used in run_name/logs (e.g. BTC, DOGE).")
    p.add_argument("--run_name", type=str, default=None)

    # Data directories
    p.add_argument("--data_root", type=str, default=None,
                   help="If set, expects subfolders: November/, December/, January/ containing parquet days.")
    p.add_argument("--nov_dir", type=str, default=None)
    p.add_argument("--dec_dir", type=str, default=None)
    p.add_argument("--jan_dir", type=str, default=None)
    p.add_argument("--jan_val_days", type=int, default=7)

    # PPO hyperparameters
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)

    p.add_argument("--chunk_len", type=int, default=32)
    p.add_argument("--batch_size_chunks", type=int, default=16)
    p.add_argument("--ppo_epochs", type=int, default=8)

    p.add_argument("--target_kl", type=float, default=0.02)

    # Rollout / training budget knobs
    p.add_argument("--rollout_episodes", type=int, default=256)
    p.add_argument("--patience", type=int, default=20)

    # Seeding
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val_seed", type=int, default=999)

    # ExecEnv parameters (IMPORTANT for DOGE realism)
    p.add_argument("--horizon_steps", type=int, default=180)
    p.add_argument("--side", type=str, default="buy", choices=["buy", "sell"])
    p.add_argument("--target_qty", type=float, default=0.5)
    p.add_argument("--max_child_qty", type=float, default=0.05)
    p.add_argument("--pov_cap", type=float, default=0.10)
    p.add_argument("--taker_fee_rate", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    # =========================================================
    # 🔒 SEED EVERYTHING (reproducibility)
    # =========================================================
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    # =========================================================

    # Normalize symbol for folder names
    args.symbol = str(args.symbol).upper()

    # --------- Resolve train/val/test file lists ----------
    if args.data_root is not None:
        train_files, val_files, test_files = make_time_split_from_root(
            args.data_root, jan_val_days=args.jan_val_days
        )
    else:
        # Backward-compatible: explicit month dirs
        if args.nov_dir is None or args.dec_dir is None or args.jan_dir is None:
            raise ValueError(
                "Provide either --data_root or all of --nov_dir --dec_dir --jan_dir."
            )
        train_files, val_files, test_files = make_time_split(
            nov_dir=args.nov_dir,
            dec_dir=args.dec_dir,
            jan_dir=args.jan_dir,
            jan_val_days=args.jan_val_days,
        )

    print(f"Train days: {len(train_files)} | Val days: {len(val_files)} | Test days: {len(test_files)}")
    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("Train/Val file lists are empty. Check your parquet folders.")

    # --------- Outputs (PER SYMBOL) ----------
    log_root = os.path.join("logs", args.symbol)
    ckpt_root = os.path.join("checkpoints", args.symbol)
    os.makedirs(log_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    # Default run_name (include SYMBOL + key knobs so BTC/DOGE runs never collide)
    if args.run_name is None:
        args.run_name = (
            f"{args.symbol}_"
            f"lr{args.lr:g}_clip{args.clip_eps:g}_ent{args.ent_coef:g}_"
            f"ep{args.ppo_epochs}_bs{args.batch_size_chunks}_cl{args.chunk_len}_"
            f"gamma{args.gamma:g}_lam{args.lam:g}_seed{args.seed}"
        )

    log_path = os.path.join(log_root, f"{args.run_name}.csv")
    ckpt_dir = os.path.join(ckpt_root, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------- Env kwargs (passed into ExecEnv via MultiDayExecEnv) ----------
    execenv_kwargs = dict(
        horizon_steps=args.horizon_steps,
        side=args.side,
        target_qty=args.target_qty,
        max_child_qty=args.max_child_qty,
        pov_cap=args.pov_cap,
        taker_fee_rate=args.taker_fee_rate,
    )

    # --------- Multi-day environments ----------
    env_train = MultiDayExecEnv(train_files, seed=args.seed, **execenv_kwargs)
    env_val = MultiDayExecEnv(val_files, seed=args.val_seed, **execenv_kwargs)

    obs_dim = env_train.observation_space.shape[0]

    # --------- Model / policy ----------
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    # --------- Hyperparameters ----------
    chunk_len = args.chunk_len
    batch_size_chunks = args.batch_size_chunks
    ppo_epochs = args.ppo_epochs

    gamma = args.gamma
    lam = args.lam

    lr = args.lr
    clip_eps = args.clip_eps
    vf_coef = args.vf_coef
    ent_coef = args.ent_coef
    max_grad_norm = args.max_grad_norm

    target_kl = args.target_kl
    rollout_episodes = args.rollout_episodes

    patience = args.patience
    best_val_is = float("inf")
    bad_iters = 0

    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    logger = CSVLogger(log_path)
    buf = RolloutBuffer(chunk_len=chunk_len)

    print("\n=== RUN CONFIG ===")
    print(f"run_name={args.run_name}")
    print(f"symbol={args.symbol}")
    print(f"data_root={args.data_root}")
    print(f"lr={lr} clip_eps={clip_eps} vf_coef={vf_coef} ent_coef={ent_coef} max_grad_norm={max_grad_norm}")
    print(f"gamma={gamma} lam={lam} chunk_len={chunk_len} batch_size_chunks={batch_size_chunks} ppo_epochs={ppo_epochs}")
    print(f"rollout_episodes={rollout_episodes} target_kl={target_kl} patience={patience}")
    print(f"train_seed={args.seed} val_seed={args.val_seed}")
    print(f"execenv={execenv_kwargs}")
    print(f"log_path={log_path}")
    print(f"ckpt_dir={ckpt_dir}\n")

    # --------- Training loop ----------
    for it in range(1, 10_000):
        t0 = time.time()

        rollout, ep_stats = collect_episodes(
            env_train, policy, n_episodes=rollout_episodes, deterministic=False
        )

        adv, ret = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], gamma=gamma, lam=lam
        )
        arrays = buf.make_training_arrays(rollout, adv, ret)

        losses, kls, entropies, clipfracs, gradnorms = [], [], [], [], []
        vlosses, pilosses = [], []

        for epoch in range(ppo_epochs):
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

                if float(approx_kl) > target_kl:
                    break

            if kls and float(np.mean(kls[-min(10, len(kls)):])) > target_kl:
                break

        # Validation evaluation
        val_metrics = evaluate_policy(env_val, policy, n_episodes=200, deterministic=True)

        improved = val_metrics.mean_is_bps < best_val_is
        if improved:
            best_val_is = val_metrics.mean_is_bps
            bad_iters = 0
            model.save_weights(os.path.join(ckpt_dir, "best.weights.h5"))
        else:
            bad_iters += 1

        row = {
            "iter": it,
            "run_name": args.run_name,
            "symbol": args.symbol,

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

            "time_sec": time.time() - t0,
            "best_val_is_bps": best_val_is,
            "bad_iters": bad_iters,

            # key hyperparams logged for sweep analysis
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
            "rollout_episodes": rollout_episodes,
            "target_kl": target_kl,
            "seed": args.seed,
            "val_seed": args.val_seed,

            # env knobs
            "horizon_steps": args.horizon_steps,
            "side": args.side,
            "target_qty": args.target_qty,
            "max_child_qty": args.max_child_qty,
            "pov_cap": args.pov_cap,
            "taker_fee_rate": args.taker_fee_rate,
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