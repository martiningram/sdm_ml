library(coda)
library(Hmsc)

compute_rhats_one_by_one <- function(coda_draws) {

    coda_rhats <- list()

    variable_names <- colnames(coda_draws[[1]])
    num_variables <- length(variable_names)

    for (cur_index in 1:num_variables) {

        cur_var_name <- variable_names[cur_index]
        cur_rhat <- coda::gelman.diag(coda_draws[, cur_index],
                                      multivariate = FALSE)
        coda_rhats[cur_var_name] <- cur_rhat[[1]][, 1]

    }

    coda_rhats <- do.call(rbind, coda_rhats)
    coda_rhats

}

args <- commandArgs(trailingOnly = TRUE)

sample_filename <- args[1]
target_dir <- args[2]

dir.create(target_dir, showWarnings = FALSE, recursive = TRUE)

# Load the samples
samples <- readRDS(sample_filename)
mpost <- convertToCodaObject(samples)
ns <- length(samples$spNames)

psrf.beta = gelman.diag(mpost$Beta, multivariate=FALSE)$psrf[, 1]
psrf.gamma = gelman.diag(mpost$Gamma, multivariate=FALSE)$psrf[, 1]
psrf.omega = compute_rhats_one_by_one(mpost$Omega[[1]])

# Compute some summaries
compute_summaries <- function(rhats, quantiles=c(0.01, 0.5, 0.99)) {
  results <- quantile(rhats, quantiles)
  results[['max']] <- max(rhats)
  results[['min']] <- min(rhats)
  results[['frac_under_1.1']] <- mean(rhats < 1.1)
  
  results
}


individual_summaries <- lapply(list(beta = psrf.beta, gamma = psrf.gamma, 
                                    omega = psrf.omega, 
                                    all = c(psrf.beta, psrf.gamma, psrf.omega)),
                               compute_summaries)

combined <- do.call(rbind, individual_summaries)

# Store summaries
get_path <- function(target_dir, name) {
  paste0(target_dir, '/', name, '.csv')
}

write.csv(combined, get_path(target_dir, 'rhat_summaries'))

write.csv(psrf.beta, get_path(target_dir, 'beta_rhats'))
write.csv(psrf.gamma, get_path(target_dir, 'gamma_rhats'))
write.csv(psrf.omega, get_path(target_dir, 'omega_rhats'))
