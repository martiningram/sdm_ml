# Species distribution modelling using machine learning (SDM_ML)

This repository contains code to fit and evaluate (multi-) species distribution
models in python.

## Requirements & setup

* Python 3 (preferably using anaconda / miniconda)
* The requirements listed in `requirements.txt` (preferably installed using
  `conda install --file requirements.txt`)
* The requirements listed in `requirements_pip.txt` (to be installed used pip,
  i.e.`pip install -r requirements_pip.txt`)
* If you would like to use the BRT model, you will also need to make sure that
  your R installation has the packages `dismo` and `gbm` installed.
* The library `ml_tools`. Please clone the repository from here:
  https://github.com/martiningram/ml_tools . Then, install by cd ing into the
  cloned directory and running `pip install -e .`
* With all these requirements in place, install the `sdm_ml` package by running
  `pip install -e .` in the base directory.
  
A Dockerfile is also available in `docker/Dockerfile`.

## Getting started

All models implemented in this framework share the same API, which consists of
the functions `fit`, `predict_log_marginal_probabilities`,
`predict_marginal_probabilities`, `save_model`, and `calculate_log_likelihood`.
We start by showing how to fit the simplest model -- a logistic regression.

```python
from sdm_ml.scikit_model import ScikitModel
from sklearn.linear_model import LogisticRegression
from functools import partial
import numpy as np

model = ScikitModel(model_fun=partial(LogisticRegression, penalty='none',
                                      solver='newton-cg'))

# Make up some fake data
X = np.random.randn(5, 4)
y = np.random.randint(0, 2, size=(5, 2))

# We can now fit the model like so:
model.fit(X, y)

# We can predict the training set like so:
y_pred = model.predict_marginal_probabilities(X)
print('The predicted probabilities are:')
print(y_pred)

# And the log likelihood:
train_lik = model.calculate_log_likelihood(X, y)
print(f'The log likelihood at each site is: {train_lik}')

# Finally, we save the model results in case we want to look at them later
model.save_model('./saved_logistic_regression')
```
