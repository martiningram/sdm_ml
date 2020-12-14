library(unmarked)

cov_df <- data.frame(scale(read.csv('unmarked_data/env_cov_unmarked.csv', row.names = 1)))
presence_df <- read.csv('unmarked_data/sample_observations.csv', row.names = 1)
duration_df <- read.csv('unmarked_data/log_duration.csv', row.names=1)
protocol_df <- read.csv('unmarked_data/protocol_type.csv', row.names=1)
daytime_df <- read.csv('unmarked_data/time_of_day.csv', row.names=1)

obs_dfs <- list(
    duration=duration_df,
    protocol=protocol_df,
    daytime=daytime_df
)

umf <- unmarkedFrameOccu(y = presence_df, siteCovs = cov_df, obsCovs = obs_dfs)

start_time <- Sys.time()

fm <- occu(formula = ~ duration + protocol + daytime + (duration * protocol) 
                     ~ bio1+bio2+bio3+bio4+bio5+bio6+bio7+bio8+bio9+bio10+bio12+bio13+bio14+bio15+bio18+bio19+I(bio1^2)+I(bio2^2)+I(bio3^2)+I(bio4^2)+I(bio5^2)+I(bio6^2)+I(bio7^2)+I(bio8^2)+I(bio9^2)+I(bio10^2)+I(bio12^2)+I(bio13^2)+I(bio14^2)+I(bio15^2)+I(bio18^2)+I(bio19^2),
           data = umf, method='BFGS')

end_time <- Sys.time()

time_diff <- end_time - start_time

write.csv(time_diff, 'time_taken_unmarked.csv')
saveRDS(fm, 'occu_fit.Rds')
