from jax import jit, vmap
from jax.scipy.stats import norm
from jax_advi.constraints import constrain_positive
import jax.numpy as jnp
from sdm_ml.checklist_level.likelihoods import compute_checklist_likelihood
from jax_advi.advi import optimize_advi_mean_field
from jax_advi.advi import get_posterior_draws
from jax.nn import sigmoid
from functools import partial

theta_constraints = {
    "obs_coef_prior_sds": constrain_positive,
    "env_slope_prior_sds": constrain_positive,
}


@partial(jit, static_argnums=6)
def calculate_likelihood_single(
    X_checklist,
    X_env,
    cur_obs_coefs,
    cur_env_slopes,
    cur_env_intercept,
    cur_y,
    cell_ids,
):

    obs_logits = X_checklist @ cur_obs_coefs
    env_logits = X_env @ cur_env_slopes + cur_env_intercept

    cur_lik = compute_checklist_likelihood(
        env_logits, obs_logits, 1 - cur_y, cell_ids, max(cell_ids) + 1
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


def calculate_prior(theta):

    prior = jnp.sum(
        norm.logpdf(
            theta["obs_coefs"],
            theta["obs_coef_prior_means"],
            theta["obs_coef_prior_sds"],
        )
    )

    prior = prior + jnp.sum(
        norm.logpdf(
            theta["env_slopes"], 0.0, theta["env_slope_prior_sds"].reshape(-1, 1)
        )
    )

    prior = prior + jnp.sum(norm.logpdf(theta["env_slope_prior_sds"]))

    # TODO: Maybe put hyperpriors on the obs coef stuff too

    return prior


def fit(
    X_env,
    X_checklist,
    y_checklist,
    cell_ids,
    M=20,
    seed=3,
    verbose=True,
    opt_method="trust-ncg",
):

    n_env_covs = X_env.shape[1]
    n_s = y_checklist.shape[1]
    n_check_covs = X_checklist.shape[1]

    # Define shapes
    theta_shapes = {
        "env_slopes": (n_env_covs, n_s),
        "env_intercepts": (n_s),
        "obs_coefs": (n_check_covs, n_s),
        "obs_coef_prior_means": (n_check_covs, 1),
        "obs_coef_prior_sds": (n_check_covs, 1),
        "env_slope_prior_sds": (n_env_covs),
    }

    # Curry and jit the likelihood
    lik_fun = jit(
        partial(
            calculate_likelihood,
            X_env=X_env,
            X_checklist=X_checklist,
            y_checklist=y_checklist,
            cell_ids=cell_ids,
        )
    )

    result = optimize_advi_mean_field(
        theta_shapes,
        calculate_prior,
        lik_fun,
        n_draws=None,
        verbose=verbose,
        M=M,
        constrain_fun_dict=theta_constraints,
        seed=seed,
        opt_method=opt_method,
    )

    return result


def predict_direct(fit_result, X_env, n_draws=100):
    def pred_fun(theta, X_env):

        return sigmoid(X_env @ theta["env_slopes"] + theta["env_intercepts"])

    prob_obs_direct = get_posterior_draws(
        fit_result["free_means"],
        fit_result["free_sds"],
        theta_constraints,
        fun_to_apply=partial(pred_fun, X_env=X_env),
        n_draws=n_draws,
    ).mean(axis=0)

    return prob_obs_direct


# TODO: Add a function to predict observed prob
