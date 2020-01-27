# Species distribution modelling using machine learning (SDM_ML)

This repository contains code to fit and evaluate (multi-) species distribution
models in python. It accompanies the manuscript "Multi-Output Gaussian Processes
for Species Distribution Modelling", which is currently under review.

## Requirements & setup

* Python 3 (preferably using anaconda / miniconda)
* The requirements listed in `requirements.txt` (preferably installed using
  `conda install --file requirements.txt`)
* If you would like to use the BRT model, you will also need to make sure that
  your R installation has the packages `dismo` and `gbm` installed.
* The library `ml_tools`. Please clone the repository from here:
  https://github.com/martiningram/ml_tools . Then, install by cd ing into the
  cloned directory and running `pip install -e .`
* The library `svgp`. Please clone from here:
  https://github.com/martiningram/svgp . The install just like `ml_tools` by
  cd-ing into the cloned directory and running `pip install -e .`.
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

### Fitting a multi-output GP (MOGP)

Fitting the MOGP is very similar, see below.

```python
from sdm_ml.gp.hierarchical_mogp import HierarchicalMOGP
import numpy as np

# Configuration for the MOGP
n_latent = 2
n_inducing = 10
kernel = 'matern_3/2'

# Build the model
model = HierarchicalMOGP(n_inducing=n_inducing, n_latent=n_latent,
                         kernel=kernel)

# The rest is the same as the logistic regression example:
n_data = 20
n_cov = 4
n_outputs = 5

np.random.seed(2)

# Make up some fake data
X = np.random.randn(n_data, n_cov)
y = np.random.randint(0, 2, size=(n_data, n_outputs))

# We can now fit the model like so (please allow a few minutes for this,
# especially on CPU):
model.fit(X, y)

# We can predict the training set like so:
y_pred = model.predict_marginal_probabilities(X)
print('The predicted probabilities are:')
print(y_pred)

# And the log likelihood:
train_lik = model.calculate_log_likelihood(X, y)
print(f'The log likelihood at each site is: {train_lik}')

# Finally, we save the model results in case we want to look at them later
model.save_model('./saved_mogp')
```

### Reproducing the paper experiments

Code to run the experiments in the paper can be found in the script:

`scripts/evaluate_model.py`

To run this code, you will have to obtain the datasets used in the paper:

* The breeding bird survey dataset can be fetched using the helpers in this
  repository: [repo](https://github.com/martiningram/bbs_bird_mistnet_split).
  Once downloaded, please set the environment variable `BBS_PATH` to point to
  the folder `csv_bird_data` generated.
* The datasets used in Anna Norberg's review paper are available here: [zenodo
  link](https://zenodo.org/record/2637812#.XeSjab9S-u4). Once downloaded, please
  set the environment variable `NORBERG_PATH` to point to the `DATA` folder in
  that dataset containing the individual `.csv` files.

Please also set the environment variable `SDM_ML_EVAL_PATH` to the directory you
would like to save results to.

In the script, you will find a dictionary called `models` which defines which
models are run. You can comment and uncomment them to select which to run.
Please note that trying to run all of them at once may consume too much RAM on
your machine. If you run into trouble, try running them one (or a few) at a
time.

Please note also that the SOGP and particularly the MOGP models benefit strongly
from GPU acceleration and will run much more quickly if one is available.
