import tensorflow as tf
from agent.models.actor_critic_lstm import RecurrentActorCritic
from agent.rl.ppo_loss import ppo_losses


@tf.function
def ppo_update_step(
    model: RecurrentActorCritic,
    optimizer: tf.keras.optimizers.Optimizer,
    obs,            # [B,T,obs_dim]
    raw_u,          # [B,T,1]
    logp_old,       # [B,T]
    adv,            # [B,T]
    returns,        # [B,T]
    mask,           # [B,T]
    h0,             # [B,lstm]
    c0,             # [B,lstm]
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
):
    obs = tf.convert_to_tensor(obs, dtype=tf.float32)
    raw_u = tf.convert_to_tensor(raw_u, dtype=tf.float32)
    logp_old = tf.convert_to_tensor(logp_old, dtype=tf.float32)
    adv = tf.convert_to_tensor(adv, dtype=tf.float32)
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    h0 = tf.convert_to_tensor(h0, dtype=tf.float32)
    c0 = tf.convert_to_tensor(c0, dtype=tf.float32)

    with tf.GradientTape() as tape:
        mu, log_std, v_pred, _ = model(obs, initial_state=(h0, c0), training=True)
        total_loss, pi_loss, v_loss, entropy, approx_kl, clip_frac = ppo_losses(
            mu=mu,
            log_std=log_std,
            v_pred=v_pred,
            raw_u=raw_u,
            logp_old=logp_old,
            adv=adv,
            returns=returns,
            mask=mask,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm(grads)

    if max_grad_norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, pi_loss, v_loss, entropy, approx_kl, clip_frac, grad_norm
