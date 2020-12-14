import numpy as np
import pandas as pd
import jax.numpy as jnp
from ml_tools.max_lik import find_map_estimate
from patsy import dmatrix
from functools import partial
from jax import jit
from sdm_ml.checklist_level.likelihoods import compute_checklist_likelihood


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
):

    env_design_mat = dmatrix(env_formula, X_env)
    checklist_design_mat = dmatrix(checklist_formula, X_checklist)

    env_covs = np.asarray(env_design_mat)
    checklist_covs = np.asarray(checklist_design_mat)

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

    fit_result, opt_result = find_map_estimate(theta, lik_curried)

    assert opt_result.success, "Optimisation failed!"

    env_coef_results = pd.Series(
        fit_result["env_coefs"], index=env_design_mat.design_info.column_names
    )

    obs_coef_results = pd.Series(
        fit_result["obs_coefs"], index=checklist_design_mat.design_info.column_names
    )

    return env_coef_results, obs_coef_results
