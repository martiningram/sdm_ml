library(dismo)
library(gbm)

# From Nick's code
# predict from a BRT model
brtFitAlt <- function(y, x.train) {

  # We're passing in a matrix because that's easier from python
  x.train <- data.frame(x.train)
  
  # fit a BRT with 10,000 trees and Jane Elith's defaults otherwise
  m <- gbm(y ~ .,
           distribution = 'bernoulli',
           data = x.train,
           interaction.depth = 5,
           shrinkage = 0.001,
           n.trees = 10000,
           cv.folds = 5,
           bag.fraction = 0.5,
           verbose = FALSE,
           n.cores = 1)
  
  ntree <- gbm.perf(m,
                    plot.it = FALSE,
                    method = 'cv')

  return (list(ntree=ntree, m=m))
  
}

brtFit <- function(y, x.train) {

  x.train <- data.frame(x.train)
  training <- cbind(y, x.train)

  m <- gbm.step(data = training, 
                gbm.x = 2:ncol(training), # column indices for covariates 
                gbm.y = 1, # column index for response 
                family = "bernoulli", 
                tree.complexity = ifelse(sum(y) < 50, 1, 5), # On recommendation
                learning.rate = 0.001, 
                bag.fraction = 0.75, 
                max.trees = 10000, 
                n.trees = 50, 
                plot.main = FALSE,
                n.folds = 4, # 4-fold cross-validation 
                silent = TRUE) # avoid printing the cv resutls 

  ntree <- m$n.trees

  return (list(ntree=ntree, m=m))

}

brtPredict <- function(model, x.test, ntree) {

  x.test <- data.frame(x.test)

  p <- predict(model, x.test, n.trees = ntree, type = 'response')

  return (p)

}
