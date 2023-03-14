import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.random import check_random_state
from sklearn.multiclass import check_classification_targets
from typing import Optional
from .loss import make_loss
from .ensemble import HRBMEnsemble


class BaseHRBMBoosting(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 1000,
                 loss: str = 'mse',
                 learning_rate: float = 0.1,
                 random_state: Optional[int] = None,
                 classification: bool = False,
                 margin: float = 0.0):
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.classification = classification
        self.margin = margin

    def _sample_boundaries(self, X, y, sample_weight=None, rng=None):
        raise NotImplementedError()

    def _fill(self, X, y, sample_weight=None, rng=None, loss_fn=None):
        n_points = X.shape[0]
        cum_pred = np.zeros_like(y)
        target = y
        mask_in = self.hrbms_.mask_in(X)
        gamma = self.learning_rate
        for i in range(self.n_estimators):
            # fill the current APR
            mask_in_i = mask_in[:, i]
            n_in = mask_in_i.sum()
            if n_in > 0:
                v_in = target[mask_in_i].mean(axis=0)
            else:
                v_in = 0.0
            if n_in < n_points:
                v_out = target[~mask_in_i].mean(axis=0)
            else:
                v_out = 0.0
            self.hrbms_.out_bias += v_out * gamma
            self.hrbms_.in_values[i] = (v_in - v_out) * gamma
            # update cumulative prediction
            cum_pred += v_out * gamma
            cum_pred[mask_in_i] += (v_in - v_out) * gamma
            # recalculate target
            target = (y - cum_pred)
        return

    def _classifier_init(self, X, y):
        check_classification_targets(y)

        y_encoded = np.zeros(y.shape, dtype=int)
        self.classes_, y_encoded[:] = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y = y_encoded
        y_hot = np.zeros((y.shape[0], self.n_classes_), dtype=np.float64)
        y_hot[np.arange(y.shape[0]), y] = 1
        return y, y_hot

    def fit(self, X, y, sample_weight=None):
        if self.classification:
            # one-hot encode y
            y_labels, y = self._classifier_init(X, y)
        rng = check_random_state(self.random_state)
        loss_fn = make_loss(self.loss)
        out_shape = 1 if y.ndim == 1 else y.shape[1]
        self.hrbms_ = HRBMEnsemble(
            self.n_estimators,
            X.shape[1],
            out_shape
        )
        self._sample_boundaries(X, y, sample_weight=sample_weight, rng=rng)
        self.mean_ = loss_fn.find_opt_constant(y)
        init_value = np.zeros_like(y) + self.mean_
        self._fill(X, y, init_value, sample_weight=sample_weight, rng=rng, loss_fn=loss_fn)
        return self

    def _predict(self, X):
        preds = (self.hrbms_(X) + self.mean_)
        if preds.shape[1] == 1:
            return preds.squeeze(axis=1)
        return preds

    def predict_proba(self, X):
        if not self.classification:
            raise RuntimeError(
                f'Cannot call `predict_proba` on estimator with `classification=False`.'
            )
        loss = make_loss(self.loss)
        return loss.link(self._predict(X))

    def predict(self, X):
        if self.classification:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        # else regression
        loss = make_loss(self.loss)
        return loss.link(self._predict(X))
