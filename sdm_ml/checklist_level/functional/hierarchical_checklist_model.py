import arviz as az
import numpy as np
from jax import jit, vmap
from patsy import dmatrix, build_design_matrices
import jax.numpy as jnp
from jax_advi.advi import optimize_advi_mean_field
from jax_advi.advi import get_posterior_draws
from jax.nn import sigmoid
from functools import partial
from .model import (
    calculate_prior_non_centered,
    transform_non_centred,
    initialise_shapes_non_centred,
    theta_constraints,
    calculate_likelihood,
    calculate_likelihood_for_loop,
)
from .hierarchical_checklist_model_mcmc import predict_obs, predict_env
from sklearn.preprocessing import StandardScaler
from ml_tools.patsy import remove_intercept_column


def fit(
    X_env,
    X_checklist,
    y_checklist,
    checklist_cell_ids,
    env_formula,
    checklist_formula,
    scale_env=True,
    draws=1000,
    M=20,
    seed=3,
    verbose=True,
    opt_method="trust-ncg",
    # opt_method="L-BFGS-B",
):

    # TODO: Currently this is the same as the MCMC version. If it stays that
    # way, should probably abstract away some stuff.
    env_design_mat = dmatrix(env_formula, X_env)
    checklist_design_mat = dmatrix(checklist_formula, X_checklist)

    env_covs = np.asarray(env_design_mat)
    checklist_covs = np.asarray(checklist_design_mat)

    env_covs = remove_intercept_column(env_covs, env_design_mat.design_info)

    if scale_env:
        scaler = StandardScaler()
        env_covs = scaler.fit_transform(env_covs)

    n_env_covs = env_covs.shape[1]
    n_s = y_checklist.shape[1]
    n_check_covs = checklist_covs.shape[1]

    shapes = initialise_shapes_non_centred(n_env_covs, n_s, n_check_covs)

    lik_fun = jit(
        lambda x: partial(
            calculate_likelihood_for_loop,
            X_env=env_covs,
            X_checklist=checklist_covs,
            y_checklist=y_checklist.values,
            cell_ids=checklist_cell_ids,
        )(transform_non_centred(x))
    )

    design_info = {
        "env": env_design_mat.design_info,
        "obs": checklist_design_mat.design_info,
        "species_names": y_checklist.columns,
    }

    result = optimize_advi_mean_field(
        shapes,
        jit(calculate_prior_non_centered),
        lik_fun,
        n_draws=draws,
        verbose=verbose,
        M=M,
        constrain_fun_dict=theta_constraints,
        seed=seed,
        opt_method=opt_method,
    )

    draws = result["draws"]

    # Add a dimension "chain":
    draws = {x: jnp.expand_dims(y, axis=0) for x, y in draws.items()}
    az_trace = az.from_dict(posterior=draws)

    if scale_env:
        design_info["env_scaler"] = scaler

    return az_trace, {x: y for x, y in result.items() if x != "draws"}, design_info
