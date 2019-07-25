library(lme4)
library(argparse)
library(reshape2)


parser <- ArgumentParser()
parser$add_argument('--train-x-csv', required=TRUE, type="character")
parser$add_argument('--train-y-csv', required=TRUE, type="character")
parser$add_argument('--test-x-csv', required=TRUE, type="character")
parser$add_argument('--target-dir', required=TRUE, type="character")
parser$add_argument('--test-run', action='store_true')
arguments <- parser$parse_args()

X <- read.csv(arguments$train_x_csv, row.names=1)
y <- read.csv(arguments$train_y_csv, row.names=1, check.names=FALSE)
test_x <- read.csv(arguments$test_x_csv, row.names=1)

# Scale X
X <- scale(X)
test_x <- scale(test_x, center=attr(X, 'scaled:center'), 
                scale=attr(X, 'scaled:scale'))

X <- data.frame(X)
test_x <- data.frame(test_x)

# Rename to be consistent
colnames(X) <- paste0('X', 0:(ncol(X) - 1))
colnames(test_x) <- paste0('X', 0:(ncol(X) - 1))

if (arguments$test_run) {
  site_subset <- sample.int(n=nrow(X), size=50, replace = FALSE)
  species_subset <- sample.int(n=ncol(y), size=3, replace = FALSE)

  X <- X[site_subset, ]
  y <- y[site_subset, species_subset]
}

# I need to make this one long dataframe.
combined_x_y <- cbind(X, y)
melted <- melt(combined_x_y, id.vars=colnames(X), variable.name='species',
               value.name='is_present')

if (ncol(X) == 8) {
  # We're fitting BBS
  fit <- glmer(is_present ~ (X0 | species) + (X1 | species) + (X2 | species) +
               (X3 | species) + (X4 | species) + (X5 | species) + (X6 | species)
             + (X7 | species) + (1 | species), melted, family = 'binomial')
} else if (ncol(X) == 5) {
  # We're fitting one of the Norberg sets
  fit <- glmer(is_present ~ (X0 | species) + (X1 | species) + (X2 | species) +
               (X3 | species) + (X4 | species) + (1 | species), melted, 
               family = 'binomial')
} else {
  fit <- glmer(is_present ~ (X0 | species) + (X1 | species) + (X2 | species) +
               (X3 | species) + (1 | species), melted, family = 'binomial')
}


saveRDS(fit, file=paste(arguments$target_dir, 'mixed_fit.Rds', sep='/'))

test_preds <- matrix(0, ncol=ncol(y), nrow=nrow(test_x))

# Now predict:
for (i in 1:ncol(y)) {

  cur_to_predict <- test_x
  cur_to_predict$species <- colnames(y)[i]
  predictions <- predict(fit, cur_to_predict, type='response')
  test_preds[, i] <- predictions

}

test_preds <- data.frame(test_preds)
colnames(test_preds) <- colnames(y)

target_file <- paste(arguments$target_dir, 'mixed_predictions.csv', sep='/')
write.csv(test_preds, target_file)
