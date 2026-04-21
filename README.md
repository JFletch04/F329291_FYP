# PPO-LSTM Optimal Trade Execution — F329291

This final year project is investigating whether a PPO-LSTM reinforcement learning agent can improve trade execution in cryptocurrency markets compared to TWAP and VWAP baselines. Its evaluated on BTC and DOGE across five different institutional order sizes using a custom Gymnasium environment thats been built from historical Bybit L2 order book and trade data. Also includes an RL² meta-learning extension fine-tuned from pretrained PPO-LSTM checkpoints with persistent hidden state across episodes.

---

## Setup

```bash
python3 -m venv drl_exec
source drl_exec/bin/activate
pip install -r requirements.txt
```

This project has been developed and tested on macOS with an Apple M-series CPU. No GPU required.

---

## Project Structure
FinalYearProject/
├── agent/
│   ├── models/actor_critic_lstm.py       # recurrent actor-critic network
│   ├── runner/                            # policy wrapper, episode/trial collectors
│   ├── rl/                                # GAE, PPO update, rollout buffer
│   ├── eval/                              # evaluate.py and evaluate_rl2.py
│   ├── utils/logger.py                    # CSV training logger
│   └── experiments/
│       ├── train_ppo_lstm.py             # PPO-LSTM training script
│       ├── train_rl2_lstm.py             # RL² fine-tuning script
│       └── sweep_hparams.py              # two-stage hyperparameter sweep
├── env/
│   ├── exec_env.py                        # custom Gymnasium execution environment
│   ├── multi_day_env.py                   # multi-day wrapper for PPO training
│   └── multi_day_task_env.py             # multi-day wrapper for RL² training
├── data/
│   ├── splits.py                          # train/val/test date-based splitting
│   ├── build_replay_day.py               # builds single-day parquet from raw LOB + trades
│   └── build_replay_month.py             # builds full month replay dataset
├── baselines/
│   ├── TWAP/TWAP_tester.py               # TWAP baseline
│   └── VWAP/                              # VWAP baseline and curve utilities
├── eval_final.py                          # main evaluation script (PPO-LSTM + TWAP)
├── final_eval_rl2_btc.py                 # RL² evaluation script for BTC
├── final_eval_rl2_doge.py                # RL² evaluation script for DOGE
├── requirements.txt
└── README.md

---

## Data

The data hasnt been included due to the file sizes. The data is 92 days of Bybit L2 order book snapshots and trade records for BTC/USDT and DOGE/USDT (November 2025 — January 2026), which is preprocessed into fixed 5-second interval parquet files.

Expected structure:
data_root/
├── November/
│   ├── 2025-11-01_steps_5s.parquet
│   └── ...
├── December/
└── January/

To build from raw the Bybit feeds, update the file path variables at the top of `data/build_replay_day.py` or `data/build_replay_month.py` and run directly.

---

## Training

**PPO-LSTM:**
```bash
python agent/experiments/train_ppo_lstm.py \
    --symbol BTC \
    --data_root /path/to/data \
    --lr 0.001 \
    --clip_eps 0.2 \
    --ent_coef 0.005 \
    --ppo_epochs 8 \
    --gamma 0.99 \
    --lam 0.9 \
    --seed 2
```

**Hyperparameter sweep (two-stage grid search):**
```bash
python agent/experiments/sweep_hparams.py --data_root /path/to/data
```

**RL² fine-tuning from pretrained PPO-LSTM checkpoint:**
```bash
python agent/experiments/train_rl2_lstm.py \
    --preset BTC_PPO_1 \
    --data_root /path/to/data \
    --rollout_trials 64 \
    --episodes_per_trial 4
```

---

## Evaluation

**PPO-LSTM and TWAP across all order sizes:**
```bash
python eval_final.py \
    --asset BTC \
    --data_root /path/to/data \
    --output_root results/ \
    --n_episodes 500 \
    --eval_seed 123
```

**RL² evaluation:**
```bash
python final_eval_rl2_btc.py \
    --asset BTC \
    --data_root /path/to/data \
    --output_root results_rl2/ \
    --n_episodes 500 \
    --eval_seed 123
```

Outputs are written to `output_root/ASSET/` and include:
- `summary_all.csv` — aggregated metrics per strategy and order size
- `per_episode_all.csv` — raw per-episode IS, completion and steps for every strategy
- `episodes_all.csv` — fixed episode seeds used across all strategies
- `manifest.json` — full evaluation configuration

The primary metric is `true_is_bps` — implementation shortfall in basis points relative to arrival mid-price. More negative values indicate more favourable execution.

---

## Key Design Decisions

- **TensorFlow over Stable-Baselines3** — required for low-level control over recurrent hidden state management, sequence chunking for truncated BPTT and custom GAE
- **Custom Gymnasium environment** — built on historical L2 replay with book-walking execution, participation caps and forced terminal liquidation; ABIDES was ruled out due to L3 data requirements
- **Two-stage hyperparameter search** — Stage 1 searches core PPO parameters, Stage 2 refines discount factor and GAE lambda over the top Stage 1 configurations
- **RL² via fine-tuning** — initialised from pretrained PPO-LSTM checkpoint with hidden state persisting across episodes within each trial of 4 episodes

---

## Notes

- Hardcoded paths in `tests/` and `agent/experiments/` point to the original development machine and will need updating locally
- `eval_final.py` and both RL² eval scripts use command-line arguments throughout — no path changes needed for evaluation
- Transaction fees are set to zero throughout all experiments
- Training logs, model checkpoints and experiment results have not been included in this submission. These were excluded as the submission requirements specified source code only.