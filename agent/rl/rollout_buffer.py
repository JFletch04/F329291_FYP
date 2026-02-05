from dataclasses import dataclass
from typing import Dict, Tuple, Iterator
import numpy as np


@dataclass
class SequenceBatch:
    """
    One minibatch of sequence chunks for PPO-LSTM training.
    Shapes:
      obs:      [B, T, obs_dim]
      raw_u:    [B, T, 1]
      actions:  [B, T, 1]
      logp_old: [B, T]
      values:   [B, T]
      adv:      [B, T]
      returns:  [B, T]
      mask:     [B, T]  (1.0 for valid timesteps, 0.0 for padded)
      h0, c0:   [B, lstm_units]  initial LSTM state for each chunk
    """
    obs: np.ndarray
    raw_u: np.ndarray
    actions: np.ndarray
    logp_old: np.ndarray
    values: np.ndarray
    adv: np.ndarray
    returns: np.ndarray
    mask: np.ndarray
    h0: np.ndarray
    c0: np.ndarray


class RolloutBuffer:
    """
    Stores a flat rollout (possibly multiple episodes concatenated),
    then creates fixed-length sequence chunks suitable for LSTM PPO.
    """

    def __init__(self, chunk_len: int = 32):
        self.chunk_len = int(chunk_len)

    def _split_into_chunks(self, rollout: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          chunk_starts: indices in the flat arrays where each chunk begins
          chunk_lens:   actual length for each chunk (<= chunk_len, last chunk of an episode may be shorter)
        Rule:
          - Chunks never cross episode boundaries (dones=True ends an episode).
        """
        dones = rollout["dones"]
        N = len(dones)

        starts = []
        lens = []

        i = 0
        while i < N:
            # determine episode segment [i, j] inclusive
            j = i
            while j < N and not dones[j]:
                j += 1
            # j is either N or index where done=True
            ep_end = min(j, N - 1)  # last index in episode
            if j < N and dones[j]:
                ep_end = j

            # now chunk this episode segment
            ep_i = i
            while ep_i <= ep_end:
                L = min(self.chunk_len, ep_end - ep_i + 1)
                starts.append(ep_i)
                lens.append(L)
                ep_i += L

            i = ep_end + 1

        return np.asarray(starts, dtype=np.int32), np.asarray(lens, dtype=np.int32)

    def make_training_arrays(
        self,
        rollout: Dict[str, np.ndarray],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Converts flat rollout -> padded chunk arrays.
        Output arrays are chunked with shape [C, T, ...], where:
          C = number of chunks, T = chunk_len
        """
        T = self.chunk_len

        obs = rollout["obs"]                  # [N, obs_dim]
        actions = rollout["actions"]          # [N, 1]
        raw_u = rollout["raw_u"]              # [N]
        logp = rollout["logp"]                # [N]
        values = rollout["values"]            # [N]
        h = rollout["h"]                      # [N, lstm_units] (state before action)
        c = rollout["c"]                      # [N, lstm_units]
        dones = rollout["dones"]              # [N]

        N, obs_dim = obs.shape
        lstm_units = h.shape[1]

        chunk_starts, chunk_lens = self._split_into_chunks(rollout)
        Cn = len(chunk_starts)

        # Allocate padded arrays
        obs_c = np.zeros((Cn, T, obs_dim), dtype=np.float32)
        actions_c = np.zeros((Cn, T, 1), dtype=np.float32)
        raw_u_c = np.zeros((Cn, T, 1), dtype=np.float32)
        logp_c = np.zeros((Cn, T), dtype=np.float32)
        values_c = np.zeros((Cn, T), dtype=np.float32)
        adv_c = np.zeros((Cn, T), dtype=np.float32)
        ret_c = np.zeros((Cn, T), dtype=np.float32)
        mask_c = np.zeros((Cn, T), dtype=np.float32)

        # Initial LSTM state per chunk (taken from stored state BEFORE first step in chunk)
        h0 = np.zeros((Cn, lstm_units), dtype=np.float32)
        c0 = np.zeros((Cn, lstm_units), dtype=np.float32)

        for ci, (st, L) in enumerate(zip(chunk_starts, chunk_lens)):
            sl = slice(st, st + L)

            obs_c[ci, :L] = obs[sl]
            actions_c[ci, :L] = actions[sl]
            raw_u_c[ci, :L, 0] = raw_u[sl]
            logp_c[ci, :L] = logp[sl]
            values_c[ci, :L] = values[sl]
            adv_c[ci, :L] = advantages[sl]
            ret_c[ci, :L] = returns[sl]
            mask_c[ci, :L] = 1.0

            h0[ci] = h[st]
            c0[ci] = c[st]

            # Sanity: if this chunk doesn't start at episode start,
            # it still uses the correct carried state because we stored (h,c) per step.

        return {
            "obs": obs_c,
            "actions": actions_c,
            "raw_u": raw_u_c,
            "logp_old": logp_c,
            "values": values_c,
            "adv": adv_c,
            "returns": ret_c,
            "mask": mask_c,
            "h0": h0,
            "c0": c0,
        }

    def iter_minibatches(
        self,
        arrays: Dict[str, np.ndarray],
        batch_size_chunks: int = 8,
        shuffle: bool = True,
    ) -> Iterator[SequenceBatch]:
        """
        Yields SequenceBatch minibatches, sampling by chunk (sequence) not by timestep.
        """
        Cn = arrays["obs"].shape[0]
        idx = np.arange(Cn)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, Cn, batch_size_chunks):
            mb = idx[start:start + batch_size_chunks]

            yield SequenceBatch(
                obs=arrays["obs"][mb],
                raw_u=arrays["raw_u"][mb],
                actions=arrays["actions"][mb],
                logp_old=arrays["logp_old"][mb],
                values=arrays["values"][mb],
                adv=arrays["adv"][mb],
                returns=arrays["returns"][mb],
                mask=arrays["mask"][mb],
                h0=arrays["h0"][mb],
                c0=arrays["c0"][mb],
            )
