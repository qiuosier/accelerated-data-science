import numpy as np
from sklearn.metrics import make_scorer


def customerize_score(y_true, y_pred, sample_weight=None):
    score = y_true == y_pred
    return np.average(score, weights=sample_weight)


scoring = make_scorer(customerize_score)
