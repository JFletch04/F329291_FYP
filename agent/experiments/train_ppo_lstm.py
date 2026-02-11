import os
import time
import numpy as np
import tensorflow as tf

from env.exec_env import ExecEnv
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.runner.lstm_policy import LSTMPolicy
from agent.runner.collector import collect_episodes
from agent.rl.gae import compute_gae
from agent.rl.rollout_buffer import RolloutBuffer
from agent.rl.ppo_update import ppo_update_step
from agent.eval.evaluate import evaluate_policy
from agent.utils.logger import CSVLogger

from data.splits import make_time_split
from env.multi_day_env import MultiDayExecEnv


def main():
    # --------- Split paths (EDIT THESE FOLDERS) ----------
    train_files, val_files, test_files = make_time_split(
        nov_dir="/Users/jackfletcher/Desktop/FYP_Data/replay_5s/November",
        dec_dir="/Users/jackfletcher/Desktop/FYP_Data/replay_5s/December",
        jan_dir="/Users/jackfletcher/Desktop/FYP_Data/replay_5s/January",
        jan_val_days=7,
    )

    print(f"Train days: {len(train_files)} | Val days: {len(val_files)} | Test days: {len(test_files)}")
    if len(train_files) == 0 or len(val_files) == 0:
        raise RuntimeError("Train/Val file lists are empty. Check your folder paths and parquet filenames.")

    # --------- Outputs ----------
    log_path = "logs/train_ppo_lstm.csv"
    ckpt_dir = "checkpoints/ppo_lstm"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --------- Multi-day Environments ----------
    # You can pass ExecEnv kwargs through MultiDayExecEnv if you want (e.g. target_qty, side, pov_cap)
    env_train = MultiDayExecEnv(train_files, seed=1)
    env_val = MultiDayExecEnv(val_files, seed=999)

    obs_dim = env_train.observation_space.shape[0]

    # --------- Model ----------
    model = RecurrentActorCritic(obs_dim=obs_dim, hidden_units=128, lstm_units=128)
    policy = LSTMPolicy(model)

    # --------- PPO Hyperparams ----------
    chunk_len = 32
    batch_size_chunks = 16          # was 8
    ppo_epochs = 8                  # was 4

    gamma = 0.999
    lam = 0.95

    lr = 3e-4
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5

    target_kl = 0.02  # stop inside PPO epochs if KL explodes

    # Rollout collection per iteration (increase for longer training)
    rollout_episodes = 256          # was 32

    # Early stopping on validation IS (lower is better)
    # With a real split you can use early stopping; patience is now less aggressive.
    patience = 20                   # was 5
    best_val_is = float("inf")
    bad_iters = 0

    # Optimizer (M1/M2: legacy Adam is faster)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    logger = CSVLogger(log_path)
    buf = RolloutBuffer(chunk_len=chunk_len)

    # --------- Training Loop ----------
    for it in range(1, 10_000):  # stop by early stopping or manual
        t0 = time.time()

        # Collect rollout data (samples a random day each episode)
        rollout, ep_stats = collect_episodes(
            env_train, policy, n_episodes=rollout_episodes, deterministic=False
        )

        adv, ret = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], gamma=gamma, lam=lam
        )
        arrays = buf.make_training_arrays(rollout, adv, ret)

        # PPO updates (multiple epochs over same data)
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

                # Stop early if KL too high (policy moving too much)
                if float(approx_kl) > target_kl:
                    break

            # If KL drifting too high on average, stop remaining epochs this iteration
            if kls and float(np.mean(kls[-min(10, len(kls)):])) > target_kl:
                break

        # Validation evaluation (samples across validation days)
        val_metrics = evaluate_policy(env_val, policy, n_episodes=200, deterministic=True)

        # Track best and early stop
        improved = val_metrics.mean_is_bps < best_val_is
        if improved:
            best_val_is = val_metrics.mean_is_bps
            bad_iters = 0
            model.save_weights(os.path.join(ckpt_dir, "best.weights.h5"))
        else:
            bad_iters += 1

        # Log
        row = {
            "iter": it,
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
            "train_days": len(train_files),
            "val_days": len(val_files),
        }
        logger.log(row)

        print(
            f"it={it} steps={row['steps_collected']} "
            f"loss={row['loss_total']:.4g} kl={row['approx_kl']:.4g} ent={row['entropy']:.4g} "
            f"valIS={row['val_mean_is_bps']:.6g} best={best_val_is:.6g} bad={bad_iters} "
            f"time={row['time_sec']:.2f}s"
        )

        if bad_iters >= patience:
            print(f"Early stopping: no val improvement for {patience} iterations.")
            break


if __name__ == "__main__":
    main()
