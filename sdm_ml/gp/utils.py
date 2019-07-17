import os
import gpflow
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans, KMeans


def calculate_log_joint_bernoulli_likelihood(latent_prob_samples: np.ndarray,
                                             outcomes: np.ndarray) -> float:
    # latent_prob_samples is n_samples x n_outcomes array of probabilities on
    # the probit scale
    # outcomes is (n_outcomes,) array of binary outcomes (1 and 0)
    assert(latent_prob_samples.shape[1] == outcomes.shape[0])

    # Make sure broadcasting is unambiguous
    assert(latent_prob_samples.shape[0] != outcomes.shape[0])

    n_samples = latent_prob_samples.shape[0]

    # Get log likelihood for each draw
    individual_liks = np.sum(
        outcomes * norm.logcdf(latent_prob_samples)
        + (1 - outcomes) * norm.logcdf(-latent_prob_samples),
        axis=1)

    # Compute the Monte Carlo expectation
    return logsumexp(individual_liks - np.log(n_samples))


def log_probability_via_sampling(means: np.ndarray, stdevs: np.ndarray,
                                 n_draws: int) -> np.ndarray:

    # TODO: Currently expects means and stdevs to be 1D. Maybe could do n-d.
    # TODO: This could really do with the odd unit test.

    draws = np.random.normal(means, stdevs, size=(
        n_draws, means.shape[0]))

    # OK, now to do the logsumexp trick.
    pre_factor = -np.log(n_draws)
    presence_log_probs = norm.logcdf(draws)
    absence_log_probs = norm.logcdf(-draws)

    presence_results = logsumexp(pre_factor + presence_log_probs, axis=0)
    absence_results = logsumexp(pre_factor + absence_log_probs, axis=0)

    return np.stack([absence_results, presence_results], axis=1)


def find_starting_z(X, num_inducing, use_minibatching=False):
    # Find starting locations for inducing points

    if use_minibatching:
        k_means = MiniBatchKMeans(n_clusters=num_inducing)
    else:
        k_means = KMeans(n_clusters=num_inducing)

    k_means.fit(X)
    Z = k_means.cluster_centers_

    return Z


def save_gpflow_model(model_object, target_file, exist_ok=True):

    if exist_ok and os.path.isfile(target_file):
        os.remove(target_file)

    saver = gpflow.saver.Saver()
    saver.save(target_file, model_object)
