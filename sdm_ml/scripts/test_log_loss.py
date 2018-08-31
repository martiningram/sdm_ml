import numpy as np
from sdm_ml.evaluator import Evaluator
from sklearn.metrics import log_loss


truth = np.random.choice([0, 1], size=100)
prediction = np.random.uniform(0, 1, size=100)

sk = log_loss(truth, prediction)
custom = Evaluator.log_loss(truth, prediction)

print(sk - custom)
