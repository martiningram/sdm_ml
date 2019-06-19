library(greta)
library(boot)


fit <- function(X, y, n_latents = 8, n_samples = 2000, warmup = 2000, 
                chains = 4) {

  x_greta <- as_data(X)
  y_greta <- as_data(y)

  n_pred <- ncol(x_greta)
  n_species <- ncol(y_greta)

  beta_1_greta <- normal(0, 1., dim = c(n_pred, n_species))

  ZZ_est <- zeros(n_species, n_latents)
  ## diagonal (must be positive)
  idx_d <- row(ZZ_est) == col(ZZ_est)
  ZZ_est_raw_d = normal(0, 1, dim = sum(idx_d), truncation = c(0, Inf))
  ZZ_est[idx_d] <- ZZ_est_raw_d
  ## sub-diagonal
  idx_s <- lower.tri(ZZ_est)
  ZZ_est_raw_s = normal(0, 1, dim = sum(idx_s))
  ZZ_est[idx_s] <- ZZ_est_raw_s

  latents <- normal(0, 1, dim=c(nrow(x_greta), n_latents))

  rates <- x_greta %*% beta_1_greta + latents %*% t(ZZ_est)
  distribution(y_greta) = bernoulli(ilogit(rates))

  m <- model(beta_1_greta, ZZ_est, precision = 'double')

  draws <- greta::mcmc(m, n_samples = n_samples, warmup = warmup, 
                       chains = chains)

  list(draws = draws, model = m)

}

predict_site <- function(site_vars, samples) {

  site_vars <- t(as.matrix(site_vars))

  mult <- apply(samples, 1, function(x) site_vars %*% x)
  # Apply non-linearity
  mult <- inv.logit(mult)
  average <- apply(mult, 1, mean)

  average

}

get_parameter <- function(model_draws, greta_model, parameter_name) {

  variable_names <- coda::varnames(model_draws)
  relevant_vars <- grepl(parameter_name, variable_names)

  stopifnot(sum(relevant_vars) > 0)

  relevant_draws <- model_draws[, relevant_vars]
  relevant_draws <- as.matrix(relevant_draws)

  # Reshape these to their original shape
  target_dims <- dim(greta_model$visible_greta_arrays[[parameter_name]])
  target_dims <- c(nrow(relevant_draws), target_dims)
  relevant_draws <- array(relevant_draws, dim = target_dims)

  relevant_draws

}

predict_data <- function(X, model_draws, greta_model) {

  # Extract the parameter of interest
  param_values <- get_parameter(model_draws, greta_model, 'beta_1_greta')

  # FIXME: I should probably be doing something with the variance of the latents
  # here.

  # Predict
  predictions <- apply(X, 1, function(cur_site) 
                       predict_site(cur_site, param_values))

  t(predictions)

}

log_lik_single <- function(y_pred, y_true) {

  y_p_for_truth <- base::rep(0, times = length(y_pred))
  y_p_for_truth[y_true == 1] <- y_pred[y_true == 1]
  y_p_for_truth[y_true == 0] <- 1 - y_pred[y_true == 0]
  log_lik <- log(y_p_for_truth)
  log_lik

}

log_likelihood <- function(y_pred, y_true) {

  species_log_lik <- matrix(NA, nrow = nrow(y_pred), ncol = ncol(y_pred))
  for (i in 1:ncol(y_pred)) {
    species_log_lik[, i] <- log_lik_single(y_pred[, i], y_true[, i])
  }

  species_log_lik
}

source('./get_data.R')

bird_path <- "/Users/ingramm/Projects/uni_melb/multi_species/bbs/dataset/csv_bird_data/"
target_path <- './experiments/warton_greta/'
dir.create(file.path(target_path), showWarnings = FALSE)

# TODO: Test whether this refactoring actually works!
data <- load_bbs_data(bird_path)

train_x <- data[['train_x']]
train_y <- data[['train_y']]
test_y <- data[['test_y']]
holdout_x <- data[['test_x']]

# Scale it
scaled_bio <- scale(train_x)

scaled_holdout <- scale(holdout_x, center = attr(scaled_bio, 'scaled:center'),
                        scale = attr(scaled_bio, 'scaled:scale'))

# Add an intercept
scaled_bio <- cbind(scaled_bio, intercept = rep(1, nrow(scaled_bio)))
scaled_holdout <- cbind(scaled_holdout, 
                        intercept = rep(1, nrow(scaled_holdout)))

species <- colnames(pres_abs)

fit_result <- fit(scaled_bio, train_y, n_latents = 8, n_samples = 1000)

# Can I predict with this?
# Just predict all the data for now.
# Start simple: one site at a time.
y_pred <- predict_data(scaled_holdout, fit_result$draws, fit_result$model)

log_loss <- -apply(log_likelihood(y_pred, test_y), 2, mean)
names(log_loss) <- species

# Store results
write.csv(log_loss, file.path(target_path, 'log_loss.csv'))
saveRDS(fit_result, file.path(target_path, 'fit.Rds'))
