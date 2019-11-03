import os
import gpflow
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp, expit
from sklearn.cluster import MiniBatchKMeans, KMeans
from gpflow.multioutput.conditionals import conditional
from gpflow.multioutput.features import SeparateIndependentMof
from gpflow.multioutput.kernels import SeparateIndependentMok
from gpflow.multioutput.conditionals import _mix_latent_gp
import tensorflow as tf


def calculate_log_joint_bernoulli_likelihood(
        latent_prob_samples: np.ndarray, outcomes: np.ndarray,
        link: str = 'probit') -> float:
    # latent_prob_samples is n_samples x n_outcomes array of probabilities on
    # the probit scale
    # outcomes is (n_outcomes,) array of binary outcomes (1 and 0)
    assert(latent_prob_samples.shape[1] == outcomes.shape[0])

    # Make sure broadcasting is unambiguous
    assert(latent_prob_samples.shape[0] != outcomes.shape[0])

    n_samples = latent_prob_samples.shape[0]

    # Get log likelihood for each draw

    assert link in ['logit', 'probit'], \
        'Only logit and probit links supported!'

    if link == 'probit':
        individual_liks = np.sum(
            outcomes * norm.logcdf(latent_prob_samples)
            + (1 - outcomes) * norm.logcdf(-latent_prob_samples),
            axis=1)
    else:
        individual_liks = np.sum(
            outcomes * np.log(expit(latent_prob_samples))
            + (1 - outcomes) * np.log(1 - expit(latent_prob_samples)),
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


def load_saved_gpflow_model(gpflow_model_path: str):

    gpflow.reset_default_graph_and_session()
    m = gpflow.saver.Saver().load(gpflow_model_path)
    return m


def compute_latent_predictions(gpflow_model, test_point, mix_latents=False,
                               full_output_cov_if_mix=True):
    """
    Given a GPFlow model object, returns the mean and variance at the test
    point for each of the L latent predictions.

    Args:
        gpflow_model: The model object returned by GPFlow.
        test_point: The point to calculate the latents at. This should be a
            single point.

    Returns:
        A Tuple containing the means and variances of the L latent processes
        at the test site.
    """

    m = gpflow_model

    feat = m.feature
    kern = m.kern
    f = tf.constant(m.q_mu.value)
    q_sqrt = tf.constant(m.q_sqrt.value)
    white = m.whiten

    independent_cond = conditional.dispatch(object, SeparateIndependentMof,
                                            SeparateIndependentMok, object)
    g_mu, g_var = independent_cond(test_point, feat, kern, f, white=white,
                                   q_sqrt=q_sqrt, full_output_cov=False,
                                   full_cov=False)

    if not mix_latents:
        return m.enquire_session().run((g_mu, g_var))
    else:
        # See:
        # https://github.com/GPflow/GPflow/blob/develop/gpflow/multioutput/conditionals.py
        return m.enquire_session().run(_mix_latent_gp(
            m.kern.W.value, g_mu, g_var, False, full_output_cov_if_mix))
