import numpy as np
from typing import Dict, Optional
from env.exec_env import ExecEnv


class MultiDayTaskEnv:
    """
    New env wrapper for RL² training.
    - Samples a parquet day at each reset, exactly like the existing multi-day env.
    - Keeps env kwargs in a mutable task dict so a future RL² variant can switch
      tasks between trials without touching any existing code.
    - For your current request, task is fixed at 50 BTC or 3,000,000 DOGE.
    """

    def __init__(self, parquet_paths, seed: int = 0, **execenv_kwargs):
        self.parquet_paths = list(parquet_paths)
        if len(self.parquet_paths) == 0:
            raise ValueError("No parquet files provided.")
        self.rng = np.random.default_rng(seed)
        self.task_kwargs: Dict[str, float] = dict(execenv_kwargs)

        # create one env upfront so observation and action spaces are available immediately
        self._env = ExecEnv(self.parquet_paths[0], seed=int(seed), **self.task_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def target_qty(self):
        return self._env.target_qty

    def set_task(self, **task_kwargs):
        self.task_kwargs.update(task_kwargs)

    def get_task(self) -> Dict[str, float]:
        return dict(self.task_kwargs)

    def reset(self, seed: Optional[int] = None, options=None):
        # sample a random day and create a fresh env for this episode
        path = self.parquet_paths[int(self.rng.integers(0, len(self.parquet_paths)))]
        env_seed = int(self.rng.integers(0, 1_000_000)) if seed is None else int(seed)
        self._env = ExecEnv(path, seed=env_seed, **self.task_kwargs)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)