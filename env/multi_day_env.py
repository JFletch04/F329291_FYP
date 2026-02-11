import numpy as np
from env.exec_env import ExecEnv


class MultiDayExecEnv:
    """
    Wraps ExecEnv but randomly picks a parquet file each episode (reset).
    """
    def __init__(self, parquet_paths, seed=0, **execenv_kwargs):
        self.parquet_paths = list(parquet_paths)
        assert len(self.parquet_paths) > 0, "No parquet files provided."
        self.rng = np.random.default_rng(seed)
        self.execenv_kwargs = execenv_kwargs

        # Create one env initially so we have action/obs spaces available
        self._env = ExecEnv(self.parquet_paths[0], seed=int(seed), **execenv_kwargs)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def target_qty(self):
        return self._env.target_qty

    def reset(self, seed=None, options=None):
        path = self.parquet_paths[int(self.rng.integers(0, len(self.parquet_paths)))]
        # recreate internal env for that day
        self._env = ExecEnv(path, seed=int(self.rng.integers(0, 1_000_000)), **self.execenv_kwargs)
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)
