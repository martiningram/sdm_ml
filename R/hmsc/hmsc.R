library(Hmsc)

get_data <- function(dataset_name, base_dir = './datasets/') {

  dataset_path <- paste0(base_dir, '/', dataset_name, '/')

  X <- read.csv(paste0(dataset_path, 'X.csv'), row.names = 1)
  X_test <- read.csv(paste0(dataset_path, 'X_test.csv'), row.names = 1)

  Y <- read.csv(paste0(dataset_path, 'Y.csv'), row.names = 1)
  Y_test <- read.csv(paste0(dataset_path, 'Y_test.csv'), row.names = 1)

  list(
    X = X,
    X_test = X_test,
    train_outcomes = Y,
    test_outcomes = Y_test
  )

}

set.seed(2)

args <- commandArgs(trailingOnly = TRUE)
dataset_name <- args[1]
dataset <- get_data(dataset_name)
target_base_dir <- args[2]

print(paste0('Fitting dataset with name ', dataset_name))

n_transient <- as.numeric(args[3])
n_sample <- as.numeric(args[4])
n_thin <- as.numeric(args[5])
n_chains <- as.numeric(args[6])

add_poly_terms <- args[7] == 'true'
add_interaction_terms <- args[8] == 'true'

target_dir <- paste0(target_base_dir, '/', dataset_name, '/')
dir.create(target_dir, showWarnings = FALSE, recursive = TRUE)

Y <- as.matrix(dataset$train_outcomes) * 1
Y_test <- as.matrix(dataset$test_outcomes) * 1

X <- data.frame(dataset$X)
X_test <- data.frame(dataset$X_test)

chosen <- seq(nrow(X))
chosen_birds <- seq(ncol(Y))

# Remove species with 5 or fewer presences
species_presences <- colSums(Y)
species_names <- colnames(Y)
meet_criterion <- species_presences > 5
chosen_species <- species_names[meet_criterion]

print(paste0('Originally, had ', length(species_names), ' species.'))
print(paste0('After removing rare ones, now have ', length(chosen_species),
             ' species.'))

# Optionally run only a subset:
# chosen <- sample.int(nrow(X), size=100)
# chosen_species <- sample.int(ncol(Y), size=64)

X <- X[chosen, ]
Y <- Y[chosen, chosen_species]

# We also need to subset the test species!
Y_test <- Y_test[, chosen_species]

# Try HMSC
studyDesign = data.frame(sample =sprintf('sample_%.3d',1:nrow(X)))

rL = HmscRandomLevel(units = studyDesign$sample)
rL$nfMax = 15

if (add_poly_terms) {
  formula_raw <- paste('~ 1 +',paste('poly(',colnames(X),',2)',collapse = ' + '))
} else {
  formula_raw <- '~ .'
}

if (add_interaction_terms) {
  formula_raw <- paste0(formula_raw, '+ .^2')
}

fit_formula <- as.formula(formula_raw)

test <- Hmsc(Y, XData=as.data.frame(X), XFormula=fit_formula, distr = 'probit',
             ranLevels =list(sample = rL), studyDesign = studyDesign)

print(colnames(X))
print(test)

start_time <- Sys.time()
samples <- sampleMcmc(test, n_sample, nChains = n_chains, transient =
                      n_transient, thin = n_thin, nParallel = 1)
end_time <- Sys.time()

saveRDS(samples, paste0(target_dir, '/samples.Rds'))
runtime <- end_time - start_time
write.csv(as.numeric(runtime, units='secs'), 
          paste0(target_dir, '/runtime.csv'))

print(runtime)

# Predict & evaluate
test_studyDesign = data.frame(sample = as.factor(1:nrow(X_test)))

test_rL = HmscRandomLevel(units = test_studyDesign$sample)

predictions <- predict(samples, XData = as.data.frame(X_test), studyDesign =
                       test_studyDesign, ranLevels = list(sample = test_rL), 
                       expected=TRUE)

# Compute the mean predictions
preds <- predictions[[1]]

for (i in 2:length(predictions)) {
  preds <- preds + predictions[[i]] 
}

probs <- preds / length(predictions)

write.csv(probs, paste0(target_dir, '/marginal_species_predictions.csv'))

# Compute the joint likelihoods
clip_between <- function(values, lower_bound=10^(-15), upper_bound=1 - 10^(-15))
{
   matrix(pmax(lower_bound, pmin(values, upper_bound)), nrow=nrow(values),
          ncol=ncol(values))
}

library(matrixStats)
# Try another way; maybe this was wrong.
# First, get all the predictions of a single site.
compute_site_likelihood <- function(cur_site_preds, site_truth) {
  
  clipped_preds <- clip_between(cur_site_preds)

  log_site_preds <- log(clipped_preds)
  log_neg_site_preds <- log(1 - clipped_preds)
  
  true_contrib <- sweep(log_site_preds, 2, site_truth, `*`)
  false_contrib <- sweep(log_neg_site_preds, 2, 1 - site_truth, `*`)
  
  lik_draws <- true_contrib + false_contrib
  lik_draws <- rowSums(lik_draws)
  lik_draws <- lik_draws - log(length(lik_draws))
  
  logSumExp(lik_draws)
  
}

mean_site_liks <- rep(0, length=nrow(X_test))

for (cur_site in 1:nrow(X_test)) {
  
  print(cur_site)

  cur_site_preds <- do.call(rbind, lapply(predictions, function (x) x[cur_site, ]))
  
  cur_lik <- compute_site_likelihood(cur_site_preds, Y_test[cur_site, ])
  
  print(cur_lik)
  
  mean_site_liks[cur_site] <- cur_lik
  
}

write.csv(mean_site_liks, paste0(target_dir, '/test_site_likelihoods.csv'))
