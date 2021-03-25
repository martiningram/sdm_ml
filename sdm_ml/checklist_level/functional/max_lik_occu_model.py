import numpy as np
import pandas as pd
import jax.numpy as jnp
from ml_tools.max_lik import find_map_estimate
from patsy import dmatrix
from functools import partial
from jax import jit
from sdm_ml.checklist_level.likelihoods import compute_checklist_likelihood
from sklearn.preprocessing import StandardScaler

# TODO: Remove this dependency
from jax_advi.diagnostics import compute_ev
from jax_advi.utils.autodiff import hvp


def likelihood_fun(theta, m, X_env, X_checklist, checklist_cell_ids, n_cells):

    env_logit = X_env @ theta["env_coefs"]
    obs_logit = X_checklist @ theta["obs_coefs"]

    likelihood = compute_checklist_likelihood(
        env_logit, obs_logit, m, checklist_cell_ids, n_cells
    )

    return jnp.sum(likelihood)


def fit(
    X_env: pd.DataFrame,
    X_checklist: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    env_formula: str,
    checklist_formula: str,
    scale_env_data=False,
    gtol=1e-3,
):

    env_design_mat = dmatrix(env_formula, X_env)
    checklist_design_mat = dmatrix(checklist_formula, X_checklist)

    env_covs = np.asarray(env_design_mat)
    checklist_covs = np.asarray(checklist_design_mat)

    if scale_env_data:
        scaler = StandardScaler()
        env_covs = scaler.fit_transform(env_covs)
    else:
        scaler = None

    theta = {
        "env_coefs": np.zeros(env_covs.shape[1]),
        "obs_coefs": np.zeros(checklist_covs.shape[1]),
    }

    lik_curried = jit(
        partial(
            likelihood_fun,
            m=1 - y,
            X_env=env_covs,
            X_checklist=checklist_covs,
            checklist_cell_ids=cell_ids,
            # TODO: Is that right?
            n_cells=X_env.shape[0],
        )
    )

    fit_result, opt_result = find_map_estimate(
        theta, lik_curried, opt_method="trust-ncg", gtol=gtol
    )

    # assert opt_result.success, "Optimisation failed!"

    env_coef_results = pd.Series(
        fit_result["env_coefs"], index=env_design_mat.design_info.column_names
    )

    obs_coef_results = pd.Series(
        fit_result["obs_coefs"], index=checklist_design_mat.design_info.column_names
    )

    return {
        "env_coefs": env_coef_results,
        "obs_coefs": obs_coef_results,
        "env_scaler": scaler,
        "env_formula": env_formula,
        "checklist_formula": checklist_formula,
        "optimisation_successful": opt_result.success,
        "opt_result": opt_result,
    }


def predict_env_logit(X_env, env_formula, env_coefs, scaler=None):

    env_design_mat = np.asarray(dmatrix(env_formula, X_env))

    if scaler is not None:
        env_design_mat = scaler.transform(env_design_mat)

    env_logit = env_design_mat @ env_coefs

    return env_logit


def predict_obs_logit(X_obs, obs_formula, obs_coefs):

    obs_design_mat = np.asarray(dmatrix(obs_formula, X_obs))

    obs_logit = obs_design_mat @ obs_coefs

    return obs_logit


def predict_env_prob(X_env, env_formula, env_coefs):

    env_logit = predict_env_logit(X_env, env_formula, env_coefs)

    return jnp.nn.sigmoid(env_logit)
