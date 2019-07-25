source('brt_fit.R')
library(argparse)


parser <- ArgumentParser()
parser$add_argument('--train-x-csv', required=TRUE, type="character")
parser$add_argument('--train-y-csv', required=TRUE, type="character")
parser$add_argument('--test-x-csv', required=TRUE, type="character")
parser$add_argument('--target-dir', required=TRUE, type="character")
parser$add_argument('--test-run', action='store_true')
arguments <- parser$parse_args()

train_x <- read.csv(arguments$train_x_csv, row.names=1)
train_y <- read.csv(arguments$train_y_csv, row.names=1)
test_x <- read.csv(arguments$test_x_csv, row.names=1)

if (arguments$test_run) {
  # Subset for testing
  train_y <- train_y[1:100, 1:5]
  train_x <- train_x[1:100, ]
}

test_preds <- matrix(0, ncol=ncol(train_y), nrow=nrow(test_x))

for (i in 1:ncol(train_y)) {

  print(paste0('On species number ', i))

  cur_y <- train_y[, i]
  fit <- brtFit(cur_y, train_x)
  preds <- brtPredict(fit$m, test_x, fit$ntree)
  test_preds[, i] <- preds

}

test_preds <- data.frame(test_preds, row.names=rownames(test_x))
colnames(test_preds) <- colnames(train_y)

print(head(test_preds))

target_file <- paste(arguments$target_dir, 'brt_predictions.csv', sep='/')
write.csv(test_preds, target_file)
