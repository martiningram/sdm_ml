library(unmarked)

args <- commandArgs(trailingOnly = TRUE)

data_dir <- args[1]
target_dir <- args[2]

cov_df <- data.frame(scale(read.csv(paste0(data_dir, '/env_cov_unmarked.csv'), row.names = 1)))
presence_df <- read.csv(paste0(data_dir, '/sample_observations.csv'), row.names = 1)
duration_df <- read.csv(paste0(data_dir, '/log_duration.csv'), row.names=1)
protocol_df <- read.csv(paste0(data_dir, '/protocol_type.csv'), row.names=1)
daytime_df <- read.csv(paste0(data_dir, '/daytimes_alt.csv'), row.names=1)

print(presence_df)

obs_dfs <- list(
    duration=duration_df,
    protocol=protocol_df,
    daytime=daytime_df
)

umf <- unmarkedFrameOccu(y = presence_df, siteCovs = cov_df, obsCovs = obs_dfs)

start_time <- Sys.time()

i <- 1

fm <- occu(formula = ~ duration + protocol + daytime
                    ~ bio1+bio2+bio3+bio4+bio5+bio6+bio7+bio8+bio9+bio10+bio12+bio13+bio14+bio15+bio18+bio19+I(bio1^2)+I(bio2^2)+I(bio3^2)+I(bio4^2)+I(bio5^2)+I(bio6^2)+I(bio7^2)+I(bio8^2)+I(bio9^2)+I(bio10^2)+I(bio12^2)+I(bio13^2)+I(bio14^2)+I(bio15^2)+I(bio18^2)+I(bio19^2),
        data = umf, method='BFGS', se = FALSE)

end_time <- Sys.time()

time_diff <- as.double(end_time - start_time, units='secs')

print(time_diff)

write.csv(time_diff, paste0(target_dir, 'time_taken_unmarked_', i, '.csv'))
saveRDS(fm, paste0(target_dir, 'occu_fit_', i, '.Rds'))

write.csv(coef(fm, 'det'), paste0(target_dir, 'det_coefs_unmarked_', i, '.csv'))
write.csv(coef(fm, 'state'), paste0(target_dir, 'env_coefs_unmarked_', i, '.csv'))
