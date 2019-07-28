# library(dismo)
library(gbm)

# From Nick's code
# predict from a BRT model
brtFit <- function(y, x.train) {

  # We're passing in a matrix because that's easier from python
  x.train <- data.frame(x.train)
  
  # fit a BRT with 10,000 trees and Jane Elith's defaults otherwise
  m <- gbm(y ~ .,
           distribution = 'bernoulli',
           data = x.train,
           interaction.depth = 5, # For rare species: 1
           shrinkage = 0.001,
           n.trees = 10000,
           cv.folds = 5,
           bag.fraction = 0.5,
           verbose = FALSE,
           n.cores = 1)

  # GBM.step?
  
  ntree <- gbm.perf(m,
                    plot.it = FALSE,
                    method = 'cv')

  return (list(ntree=ntree, m=m))
  
}


brtPredict <- function(model, x.test, ntree) {

  x.test <- data.frame(x.test)

  p <- predict(model, x.test, n.tree = ntree, type = 'response')

  return (p)

}
