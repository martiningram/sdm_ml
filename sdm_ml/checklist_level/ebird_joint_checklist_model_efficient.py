from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd
from tqdm import tqdm
from patsy import dmatrix
from jax_advi.advi import get_pickleable_subset, optimize_advi_mean_field
from ml_tools.utils import save_pickle_safely
from os import makedirs
from os.path import join
from sklearn.preprocessing import StandardScaler, LabelEncoder
from jax_advi.constraints import constrain_positive
from jax.scipy.stats import norm
from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
from sdm_ml.checklist_level.likelihoods import compute_checklist_likelihood
from .functional.hierarchical_checklist_model import predict_direct


def create_shapes_and_constraints(n_protocols, n_s, n_env_covs, n_daytimes):

    theta_shapes = {
        "env_slopes": (n_env_covs, n_s),
        "env_intercepts": (n_s),
        "duration_slopes": (n_s),
        "obs_intercepts": (n_s),
        "duration_slope_mean": (),
        "duration_slope_sd": (),
        "obs_intercept_mean": (),
        "obs_intercept_sd": (),
        "protocol_slope_additions": (n_protocols - 1, n_s),
        "protocol_intercept_additions": (n_protocols - 1, n_s),
        "protocol_slope_additions_mean": (n_protocols - 1),
        "protocol_slope_additions_sd": (n_protocols - 1),
        "protocol_intercept_additions_mean": (n_protocols - 1),
        "protocol_intercept_additions_sd": (n_protocols - 1),
        # "env_slope_prior_sd": (n_env_covs),
        "daytime_intercept_additions": (n_daytimes - 1, n_s),
        "daytime_intercept_additions_sd": (n_daytimes - 1),
        "daytime_intercept_additions_mean": (n_daytimes - 1),
    }

    theta_constraints = {
        "duration_slope_sd": constrain_positive,
        "obs_intercept_sd": constrain_positive,
        "protocol_slope_additions_sd": constrain_positive,
        "protocol_intercept_additions_sd": constrain_positive,
        # "env_slope_prior_sd": constrain_positive,
        "daytime_intercept_additions_sd": constrain_positive,
    }

    return theta_shapes, theta_constraints


def calculate_likelihood(theta, n_s, covs, obs_covs, cell_ids, y_pa):

    protocol_slope_additions = jnp.concatenate(
        [jnp.zeros((1, n_s)), theta["protocol_slope_additions"]]
    )

    protocol_intercept_additions = jnp.concatenate(
        [jnp.zeros((1, n_s)), theta["protocol_intercept_additions"]]
    )

    daytime_intercept_additions = jnp.concatenate(
        [jnp.zeros((1, n_s)), theta["daytime_intercept_additions"]]
    )

    env_logits = covs @ theta["env_slopes"] + theta["env_intercepts"].reshape(1, -1)

    obs_logits = obs_covs["log_duration"].reshape(-1, 1) * (
        theta["duration_slopes"].reshape(1, -1)
        + protocol_slope_additions[obs_covs["protocol_ids"]]
    ) + (
        theta["obs_intercepts"].reshape(1, -1)
        + protocol_intercept_additions[obs_covs["protocol_ids"]]
        + daytime_intercept_additions[obs_covs["daytime_ids"]]
    )

    curried_lik = lambda cur_env_logit, cur_obs_logit, cur_y: compute_checklist_likelihood(
        cur_env_logit, cur_obs_logit, 1 - cur_y, cell_ids, max(cell_ids) + 1
    )

    lik = vmap(curried_lik)(env_logits.T, obs_logits.T, y_pa.T)

    return jnp.sum(lik)


@jit
def calculate_prior(theta):

    prior = jnp.sum(
        norm.logpdf(
            theta["duration_slopes"],
            theta["duration_slope_mean"],
            theta["duration_slope_sd"],
        )
    )

    prior = prior + jnp.sum(
        norm.logpdf(
            theta["obs_intercepts"],
            theta["obs_intercept_mean"],
            theta["obs_intercept_sd"],
        )
    )

    prior = prior + jnp.sum(
        norm.logpdf(
            theta["protocol_slope_additions"],
            theta["protocol_slope_additions_mean"].reshape(-1, 1),
            theta["protocol_slope_additions_sd"].reshape(-1, 1),
        )
    )

    prior = prior + jnp.sum(
        norm.logpdf(
            theta["protocol_intercept_additions"],
            theta["protocol_intercept_additions_mean"].reshape(-1, 1),
            theta["protocol_intercept_additions_sd"].reshape(-1, 1),
        )
    )

    # prior = prior + jnp.sum(
    #     norm.logpdf(
    #         theta["env_slopes"], 0.0, theta["env_slope_prior_sd"].reshape(-1, 1)
    #     )
    # )

    # N(0, 1) prior
    prior = prior + jnp.sum(norm.logpdf(theta["env_slopes"], 0.0, 1.0))

    prior = prior + jnp.sum(
        norm.logpdf(
            theta["daytime_intercept_additions"],
            theta["daytime_intercept_additions_mean"].reshape(-1, 1),
            theta["daytime_intercept_additions_sd"].reshape(-1, 1),
        )
    )

    # Hyperpriors
    # Env slopes: SD only
    # prior = prior + norm.logpdf(theta["env_slope_prior_sd"])

    # Others: Mean and sd get N(0, 1)
    for cur_var in [
        "protocol_slope_additions",
        "protocol_intercept_additions",
        "daytime_intercept_additions",
        "duration_slope",
        "obs_intercept",
    ]:
        prior = prior + jnp.sum(norm.logpdf(theta[cur_var + "_mean"]))
        prior = prior + jnp.sum(norm.logpdf(theta[cur_var + "_sd"]))

    return prior


class EBirdJointChecklistModel(ChecklistModel):
    def __init__(
        self,
        M=20,
        n_pred_draws=1000,
        verbose_fit=True,
        opt_method="trust-ncg",
        env_interactions=True,
    ):

        self.scaler = None
        self.M = M
        self.n_pred_draws = n_pred_draws
        self.verbose_fit = verbose_fit
        self.opt_method = opt_method
        self.env_interactions = env_interactions

    @staticmethod
    def assemble_dataset(
        X_env_cell, X_checklist_fun, y_checklist, checklist_cell_ids, species_names
    ):

        # Assemble the dataset
        y_full = list()
        cell_ids = None
        X_checklist = None

        print("Preparing data...")
        for cur_species in tqdm(species_names):

            cur_y = y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            y_full.append(cur_y)

            if cell_ids is None:
                cell_ids = cur_cell_ids
            else:
                assert np.all(cur_cell_ids == cell_ids)

            cur_X_checklist = X_checklist_fun(cur_species)

            if X_checklist is None:
                X_checklist = cur_X_checklist
            else:
                assert np.all(
                    X_checklist["sampling_event_identifier"].values
                    == cur_X_checklist["sampling_event_identifier"].values
                )

        y_full = np.stack(y_full, axis=1)

        # TODO: Is that right?
        X_env = X_env_cell
        print("Done.")

        return X_checklist, y_full, X_env, cell_ids

    @staticmethod
    def derive_obs_covariates(X_checklist):

        protocol_types = (
            X_checklist["protocol_type"]
            .str.replace("Traveling - Property Specific", "Traveling")
            .values
        )

        hour_of_day = (
            X_checklist["time_observations_started"]
            .str.split(":")
            .str.get(0)
            .astype(int)
        )

        time_of_day = np.select(
            [
                (hour_of_day >= 5) & (hour_of_day < 12),
                (hour_of_day > 12) & (hour_of_day < 21),
            ],
            ["morning", "afternoon/evening"],
            default="night",
        )

        log_duration = np.log(X_checklist["duration_minutes"].values)

        daytime_encoder = LabelEncoder()
        daytime_ids = daytime_encoder.fit_transform(time_of_day)

        protocol_encoder = LabelEncoder()
        protocol_ids = protocol_encoder.fit_transform(protocol_types)

        obs_covariates = {
            "daytime_ids": daytime_ids,
            "protocol_ids": protocol_ids,
            "log_duration": log_duration,
        }

        encoders = {
            "daytime_encoder": daytime_encoder,
            "protocol_encoder": protocol_encoder,
        }

        return obs_covariates, encoders

    @staticmethod
    def create_design_matrix_env(X_env, add_interactions=True):

        cov_names = X_env.columns

        main_effects = "+".join(cov_names)
        quad_terms = "+".join([f"I({x}**2)" for x in cov_names])

        model_str = main_effects + "+" + quad_terms

        if add_interactions:
            inter_terms = "(" + main_effects + ")"
            inter_terms = inter_terms + ":" + inter_terms
            model_str = model_str + "+" + inter_terms

        env_design_mat = dmatrix(model_str + "- 1", X_env)

        return env_design_mat

    def fit(
        self,
        X_env_cell: pd.DataFrame,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ):

        self.species_names = species_names

        X_checklist, y_full, X_env, cell_ids = self.assemble_dataset(
            X_env_cell, X_checklist, y_checklist, checklist_cell_ids, species_names
        )

        n_s = len(species_names)

        # Next, extract the checklist data we need
        obs_covs, encoders = self.derive_obs_covariates(X_checklist)

        # Create the design matrix for the environmental variables
        env_design_mat = self.create_design_matrix_env(X_env, self.env_interactions)

        self.env_cov_names = env_design_mat.design_info.column_names

        self.scaler = StandardScaler()

        X_env = self.scaler.fit_transform(np.asarray(env_design_mat))

        # Create functions for ADVI
        lik_fun = jit(
            partial(
                calculate_likelihood,
                n_s=n_s,
                covs=X_env,
                obs_covs=obs_covs,
                cell_ids=cell_ids,
                y_pa=y_full,
            )
        )

        shapes, constraints = create_shapes_and_constraints(
            len(encoders["protocol_encoder"].classes_),
            n_s,
            X_env.shape[1],
            len(encoders["daytime_encoder"].classes_),
        )

        # Fit ADVI
        self.fit_result = optimize_advi_mean_field(
            shapes,
            calculate_prior,
            lik_fun,
            M=self.M,
            constrain_fun_dict=constraints,
            verbose=self.verbose_fit,
            opt_method=self.opt_method,
            n_draws=None,
        )

    def predict_log_marginal_probabilities(self, X: pd.DataFrame) -> np.ndarray:

        X_design = self.create_design_matrix_env(X)
        X = self.scaler.transform(X_design)

        preds = predict_direct(self.fit_result, X, n_draws=self.n_pred_draws)

        log_preds = np.log(preds)
        log_not_preds = np.log(1 - preds)

        combined = np.stack([log_not_preds, log_preds], axis=-1)

        return combined

    def calculate_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder: str) -> None:

        makedirs(target_folder, exist_ok=True)

        to_pickle = {
            "fit_result": get_pickleable_subset(self.fit_result),
            "env_cov_names": self.env_cov_names,
            "species_names": self.species_names,
        }

        save_pickle_safely(to_pickle, join(target_folder, "fit_results.pkl"))
        save_pickle_safely(self.scaler, join(target_folder, "scaler.pkl"))
