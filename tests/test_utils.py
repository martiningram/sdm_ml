import numpy as np
from scipy.stats import norm
from sdm_ml.gp.utils import log_probability_via_sampling


def test_log_probability_via_sampling():

    np.random.seed(1)

    n_draws = 100000
    n_obs = 100

    # Compare against simpler non-log scale sampling.
    f_mean = np.random.randn(n_obs)
    f_std = np.random.randn(n_obs)**2

    sampling_prob = log_probability_via_sampling(f_mean, f_std, n_draws)

    # Sample non-log instead
    draws = np.random.normal(f_mean, f_std, size=(n_draws, n_obs))

    presence_probs = norm.cdf(draws)
    absence_probs = 1 - norm.cdf(draws)

    mean_presence_probs = np.mean(presence_probs, axis=0)
    mean_absence_probs = np.mean(absence_probs, axis=0)

    combined = np.stack([mean_absence_probs, mean_presence_probs], axis=1)

    log_version = np.log(combined)

    assert np.allclose(log_version, sampling_prob, atol=1e-1)
