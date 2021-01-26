import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import norm
from jax_advi.constraints import constrain_positive
from sdm_ml.checklist_level.likelihoods import compute_checklist_likelihood
from ml_tools.jax import half_normal_logpdf

theta_constraints = {
    "obs_coef_prior_sds": constrain_positive,
}


def calculate_likelihood_single(
    X_checklist,
    X_env,
    cur_obs_coefs,
    cur_env_slopes,
    cur_env_intercept,
    cur_y,
    cell_ids,
):

    cell_ids = jnp.array(cell_ids)

    obs_logits = X_checklist @ cur_obs_coefs
    env_logits = X_env @ cur_env_slopes + cur_env_intercept

    cur_lik = compute_checklist_likelihood(
        env_logits, obs_logits, 1 - cur_y, cell_ids, env_logits.shape[0]
    )

    return jnp.sum(cur_lik)


def calculate_likelihood(theta, X_env, X_checklist, y_checklist, cell_ids):

    curried_lik = lambda cur_obs_coefs, cur_env_slopes, cur_env_intercept, cur_y: calculate_likelihood_single(
        X_checklist,
        X_env,
        cur_obs_coefs,
        cur_env_slopes,
        cur_env_intercept,
        cur_y,
        cell_ids,
    )

    lik = vmap(curried_lik)(
        theta["obs_coefs"].T,
        theta["env_slopes"].T,
        theta["env_intercepts"],
        y_checklist.T,
    )

    return jnp.sum(lik)


def calculate_prior_non_centered(theta):

    prior = jnp.sum(norm.logpdf(theta["obs_coefs_raw"], 0.0, 1.0))

    prior = prior + jnp.sum(norm.logpdf(theta["env_slopes"], 0.0, 1.0))
    prior = prior + jnp.sum(norm.logpdf(theta["obs_coef_prior_means"]))
    prior = prior + jnp.sum(half_normal_logpdf(theta["obs_coef_prior_sds"], 1.0))
    prior = prior + jnp.sum(norm.logpdf(theta["env_intercepts"], 0.0, 10.0))

    return prior


def transform_non_centred(theta):

    theta["obs_coefs"] = (
        theta["obs_coefs_raw"] * theta["obs_coef_prior_sds"]
        + theta["obs_coef_prior_means"]
    )

    return theta


def initialise_shapes_non_centred(n_env_covs, n_s, n_check_covs):

    theta_shapes = {
        "env_slopes": (n_env_covs, n_s),
        "env_intercepts": (n_s,),
        "obs_coefs_raw": (n_check_covs, n_s),
        "obs_coef_prior_means": (n_check_covs, 1),
        "obs_coef_prior_sds": (n_check_covs, 1),
    }

    return theta_shapes
