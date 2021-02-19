data {
  int K; // The number of checklists
  int N; // The number of cells
  int cell_ids[K]; // The corresponding cell indices for each checklist

  int n_obs_covs; // The number of checklist-level covariates
  int n_env_covs; // The number of environment-level covariates

  matrix[N, n_env_covs] env_covs; // Design matrix for environment variables
  matrix[K, n_obs_covs] obs_covs; // Design matrix for observation variables

  int n_species; // Number of species

  int y[K, n_species]; // Presence or absence for each species on each checklist
} 
transformed data {
  int observed_at_least_once[N, n_species] = rep_array(0, N, n_species);

  for (i in 1:K) {
    for (j in 1:n_species) {
      if (observed_at_least_once[cell_ids[i], j] != 0) {
	continue;
      }
      // Otherwise check whether there's an observation
      observed_at_least_once[cell_ids[i], j] += y[i, j];
    }
  }
}
parameters {

  row_vector[n_species] env_intercepts;

  // matrix[n_env_covs, n_species] env_slopes_raw;
  matrix[n_env_covs, n_species] env_slopes;

  // This includes an intercept
  matrix[n_obs_covs, n_species] obs_coefs_raw;

  // The standard deviations
  // vector<lower=0>[n_env_covs] env_slope_sds;
  vector<lower=0>[n_obs_covs] obs_slope_sds;

  vector[n_obs_covs] obs_slope_means;

}
transformed parameters {

  // Not sure this is most efficient but hey...
  // matrix[n_env_covs, n_species] env_slopes =
  //   rep_matrix(env_slope_sds, n_species) .* env_slopes_raw;
  
  matrix[n_obs_covs, n_species] obs_coefs =
    rep_matrix(obs_slope_sds, n_species) .* obs_coefs_raw +
    rep_matrix(obs_slope_means, n_species);

}
model {
  matrix[N, n_species] env_logits = env_covs * env_slopes +
    rep_matrix(env_intercepts, N);
  matrix[K, n_species] obs_logits = obs_covs * obs_coefs;
  matrix[N, n_species] log_obs_prob_given_present = rep_matrix(0., N, n_species);

  // Some priors
  // to_vector(env_slopes_raw) ~ normal(0, 1);
  to_vector(env_slopes) ~ normal(0, 1);
  to_vector(obs_coefs_raw) ~ normal(0, 1);
  // Half-normal prior on slope sds
  // env_slope_sds ~ normal(0, 1);
  obs_slope_means ~ normal(0, 1);
  obs_slope_sds ~ normal(0, 1);
  env_intercepts ~ normal(0, 10);

  // Summarise the log observation likelihoods per cell
  for (k in 1:K) {
    for (j in 1:n_species) {
      // TODO: Is there a way of combining log and inv_logit to prevent underflow?
      real log_prob_obs = log(inv_logit(obs_logits[k, j]));
      real log_prob_not_obs = log(inv_logit(-obs_logits[k, j]));
      log_obs_prob_given_present[cell_ids[k], j] +=
	y[k, j] * log_prob_obs + (1 - y[k, j]) * log_prob_not_obs;
    }
  }

  // Compute the log likelihood
  for (i in 1:N) {
    for (j in 1:n_species) {
      real cur_log_prob_pres = log(inv_logit(env_logits[i, j]));

      if (observed_at_least_once[i, j] == 0) {
	real cur_log_prob_abs = log(inv_logit(-env_logits[i, j]));
	target +=
	  log_sum_exp(cur_log_prob_abs * (1 - observed_at_least_once[i, j]),
		      cur_log_prob_pres + log_obs_prob_given_present[i, j]);
      } else {
	target += cur_log_prob_pres + log_obs_prob_given_present[i, j];
      }
    }
  }
}
