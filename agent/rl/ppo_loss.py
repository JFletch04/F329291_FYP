import numpy as np
import tensorflow as tf
from agent.models.actor_critic_lstm import log_prob_squashed_gaussian


@tf.function
def masked_mean(x: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Mean over [B,T] with mask in {0,1}."""
    denom = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(x * mask) / denom


@tf.function
def ppo_losses(
    mu: tf.Tensor,                 # [B,T,1]
    log_std: tf.Tensor,            # [1]
    v_pred: tf.Tensor,             # [B,T,1]
    raw_u: tf.Tensor,              # [B,T,1]  pre-sigmoid actions taken
    logp_old: tf.Tensor,           # [B,T]
    adv: tf.Tensor,                # [B,T]
    returns: tf.Tensor,            # [B,T]
    mask: tf.Tensor,               # [B,T]  1 valid, 0 padded
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
):
    """
    Returns:
      total_loss, policy_loss, value_loss, entropy, approx_kl, clip_frac
    """
    # new log-prob under current policy
    logp_new = log_prob_squashed_gaussian(raw_u, mu, log_std)  # [B,T]
    ratio = tf.exp(logp_new - logp_old)                        # [B,T]

    # normalise advantages (masked)
    adv_mean = masked_mean(adv, mask)
    adv_var = masked_mean(tf.square(adv - adv_mean), mask)
    adv_norm = (adv - adv_mean) / (tf.sqrt(adv_var) + 1e-8)

    # PPO clipped objective
    unclipped = ratio * adv_norm
    clipped = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    policy_loss = -masked_mean(tf.minimum(unclipped, clipped), mask)

    # value loss (MSE)
    v = tf.squeeze(v_pred, axis=-1)   # [B,T]
    value_loss = masked_mean(tf.square(returns - v), mask)

    # entropy of gaussian in raw space — H = 0.5 * (1 + log(2*pi)) + log_std
    entropy_per_t = 0.5 * (1.0 + np.log(2.0 * np.pi)) + log_std  # [1]
    entropy = tf.reduce_mean(entropy_per_t)

    # approx KL (PPO diagnostic)
    approx_kl = masked_mean(logp_old - logp_new, mask)

    # clip fraction — how often ratio was outside the clip range
    clipped_mask = tf.cast(tf.abs(ratio - 1.0) > clip_eps, tf.float32)
    clip_frac = masked_mean(clipped_mask, mask)

    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return total_loss, policy_loss, value_loss, entropy, approx_kl, clip_frac