from ml_tools.numpyro_mcmc import sample_nuts
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix, build_design_matrices
import numpy as np
from .model import (
    calculate_likelihood,
    theta_constraints,
    calculate_prior_non_centered,
    transform_non_centred,
    initialise_shapes_non_centred,
)
from functools import partial
from jax import jit
from sdm_ml.checklist_level.utils import evaluate_on_chunks
from jax.nn import log_sigmoid, sigmoid
import pandas as pd
import jax.numpy as jnp
from jax.scipy.stats import norm
from .utils import predict_env_from_samples, predict_obs_from_samples
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
    tune=1000,
    thinning=1,
    chain_method="vectorized",
):

    env_design_mat = dmatrix(env_formula, X_env)
    checklist_design_mat = dmatrix(checklist_formula, X_checklist)

    env_covs = np.asarray(env_design_mat)
    checklist_covs = np.asarray(checklist_design_mat)

    # Workaround to remove intercept
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
            calculate_likelihood,
            X_env=env_covs,
            X_checklist=checklist_covs,
            y_checklist=y_checklist.values,
            cell_ids=checklist_cell_ids,
        )(transform_non_centred(x))
    )

    samples = sample_nuts(
        shapes,
        jit(calculate_prior_non_centered),
        lik_fun,
        constrain_fun_dict=theta_constraints,
        draws=draws,
        tune=tune,
        thinning=thinning,
        chain_method=chain_method,
        use_tfp=False,
    )

    design_info = {
        "env": env_design_mat.design_info,
        "obs": checklist_design_mat.design_info,
        "species_names": y_checklist.columns,
    }

    if scale_env:
        design_info["env_scaler"] = scaler

    return samples, design_info


def fetch_env_samples(samples):

    env_slope_samples = samples.posterior["env_slopes"].values
    env_intercept_samples = samples.posterior["env_intercepts"].values

    # Concatenate across chains
    # TODO: Maybe arviz has a nicer method for this?
    env_slope_samples = flatten_chains(env_slope_samples)
    env_intercept_samples = flatten_chains(env_intercept_samples)

    return env_slope_samples, env_intercept_samples


def flatten_chains(draws):

    return np.reshape(draws, (-1, *draws.shape[2:]))


def predict_env(X_env, samples, design_info):

    design_mat = build_design_matrices([design_info["env"]], X_env)[0]
    env_covs = np.asarray(design_mat)
    env_covs = remove_intercept_column(env_covs, design_mat.design_info)

    if "env_scaler" in design_info:
        env_covs = design_info["env_scaler"].transform(env_covs)

    env_slope_samples, env_intercept_samples = fetch_env_samples(samples)

    prob = predict_env_from_samples(env_covs, env_slope_samples, env_intercept_samples)

    return pd.DataFrame(prob, index=X_env.index, columns=design_info["species_names"])


def predict_obs(X_env, X_obs, samples, design_info):

    env_design_mat = build_design_matrices([design_info["env"]], X_env)[0]
    env_covs = np.asarray(env_design_mat)

    env_covs = remove_intercept_column(env_covs, env_design_mat.design_info)

    if "env_scaler" in design_info:
        env_covs = design_info["env_scaler"].transform(env_covs)

    env_slope_samples, env_intercept_samples = fetch_env_samples(samples)

    obs_design_mat = build_design_matrices([design_info["obs"]], X_obs)[0]
    obs_covs = np.asarray(obs_design_mat)

    transformed_samples = transform_non_centred(
        {x: y.values for x, y in samples.posterior.items()}
    )

    obs_slope_samples = flatten_chains(transformed_samples["obs_coefs"])

    result = predict_obs_from_samples(
        env_covs, env_slope_samples, env_intercept_samples, obs_covs, obs_slope_samples
    )

    return pd.DataFrame(result, columns=design_info["species_names"])
