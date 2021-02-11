import jax.numpy as jnp
from jax import jit
from jax.nn import sigmoid, log_sigmoid
from sdm_ml.checklist_level.utils import evaluate_on_chunks


def predict_env_from_samples(env_covs, env_slope_samples, env_intercept_samples):
    @jit
    def predict(X):

        logits = jnp.einsum(
            "nc,dcs->dns", X, env_slope_samples
        ) + env_intercept_samples.reshape(env_slope_samples.shape[0], 1, -1)

        prob = jnp.mean(sigmoid(logits), axis=0)

        return prob

    prob = evaluate_on_chunks(predict, 2, env_covs, is_df=False)

    return prob


def predict_obs_from_samples(
    env_covs, env_slope_samples, env_intercept_samples, obs_covs, obs_slope_samples
):
    @jit
    def predict(X_env, X_obs):

        env_logits = jnp.einsum(
            "nc,dcs->dns", X_env, env_slope_samples
        ) + env_intercept_samples.reshape(env_slope_samples.shape[0], 1, -1)

        obs_logits = jnp.einsum("nc,dcs->dns", X_obs, obs_slope_samples)

        prob = jnp.exp(log_sigmoid(env_logits) + log_sigmoid(obs_logits)).mean(axis=0)

        return prob

    result = evaluate_on_chunks(predict, 2, env_covs, obs_covs, is_df=False)

    return result
