data {
    int N; // number of data points
    int n_species;
    int n_cov; // Number of covariates
        
    int y[N, n_species]; // Presence / absence
    matrix[N, n_cov] X; // Design matrix
}
parameters {
    matrix[n_cov, n_species] coefficients_raw;
    
    // Do these uncorrelated for starters
    vector<lower=0>[n_cov] coeff_sd;
    vector[n_cov] coeff_mean;
}
transformed parameters {
    // Priors on coefficients
    matrix[n_cov, n_species] coefficients;
    for (i in 1:n_cov) {
        coefficients[i, :] = coefficients_raw[i, :] * coeff_sd[i] +
	  coeff_mean[i];
    }
}
model {
    matrix[N, n_species] y_pred = X * coefficients;

    to_vector(coefficients_raw) ~ normal(0, 1);
    coeff_sd ~ normal(0, 1);
    coeff_mean ~ normal(0, 1);
    
    // Note the transpose due to Stan's weirdness
    to_array_1d(y) ~ bernoulli_logit(to_vector(y_pred'));
    
}
