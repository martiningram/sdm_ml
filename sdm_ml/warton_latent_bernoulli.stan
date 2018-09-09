data {
  int n; // number of sites
  int k; // number of species
  int p; // number of predictors
  int l; // number of latents

  int y[n, k]; // whether or not each species was observed at each site
  matrix[n, p] X; // design matrix of predictors
}
transformed data {
  int flattened_y[n*k] = to_array_1d(y);
}
parameters {
  matrix[p, k] beta_1; // species response to predictors
  cholesky_factor_cov[k, l] beta_z; // species response to latents
  matrix[n, l] latents; // latents at each site
}
model {
  matrix[n, k] theta;

  // Theta is a linear function of the covariates
  theta = X * beta_1 + latents * beta_z';

  to_vector(beta_1) ~ normal(0, 1);
  to_vector(beta_z) ~ normal(0, 1);

  to_vector(latents) ~ normal(0, 1);
  flattened_y ~ bernoulli_logit(to_vector(theta'));
}
