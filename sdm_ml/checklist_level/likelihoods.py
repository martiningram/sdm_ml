import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.nn import log_sigmoid


def compute_checklist_likelihood(pres_abs_logit, obs_logit, m, cell_nums, n_cells):
    # This computes the log probability of observing the data m (the missingness
    # indicator, 1 if no observation, 0 if observation) given the probability of
    # presence in the cell on the logit scale and the probability of observing the
    # species if present on the logit scale.
    # The pres_abs_logit is on the grid cell scale. Since there can be
    # multiple observations per grid cell, the "cell_nums" array should match
    # them up. Specifically, entry cell_nums[i] should contain the index j so
    # that it belongs to the grid cell whose presence probability is in
    # pres_abs_logit[j]. n_cells is the total number of cells.

    log_prob_pres = log_sigmoid(pres_abs_logit)
    log_prob_abs = log_sigmoid(-pres_abs_logit)

    log_prob_miss_if_pres = log_sigmoid(-obs_logit)
    log_prob_obs_if_pres = log_sigmoid(obs_logit)

    rel_log_probs = jnp.where(m == 0, log_prob_obs_if_pres, log_prob_miss_if_pres)
    summed_liks = jnp.bincount(cell_nums, weights=rel_log_probs, length=n_cells)

    lik_term_1 = log_prob_abs
    lik_term_2 = log_prob_pres + summed_liks

    lik_if_at_least_one_obs = lik_term_2
    lik_if_all_missing = logsumexp(jnp.stack([lik_term_1, lik_term_2], axis=1), axis=1)

    not_miss = 1 - m

    obs_per_cell = jnp.bincount(cell_nums, weights=not_miss, length=n_cells)
    log_lik = jnp.where(obs_per_cell == 0, lik_if_all_missing, lik_if_at_least_one_obs)

    return log_lik
