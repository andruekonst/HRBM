import numpy as np


class HRBMEnsemble:
    """HRBM Ensemble implemented with parallel arrays.
    """
    def __init__(self, n_estimators: int, n_features: int, n_outputs: int = 1):
        self._prepare(n_estimators, n_features, n_outputs)

    def _prepare(self, n_estimators: int, n_features: int, n_outputs: int):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lefts = np.full((n_estimators, n_features), -np.inf, dtype=np.float64)
        self.rights = np.full((n_estimators, n_features), np.inf, dtype=np.float64)
        self.out_bias = np.zeros((n_outputs,), dtype=np.float64)
        self.in_values = np.zeros((n_estimators, n_outputs), dtype=np.float64)

    def __call__(self, X):
        predicates = self.mask_in(X)  # shape (n_samples, n_estimators)
        result = (predicates[:, :, np.newaxis] * self.in_values[np.newaxis]).sum(axis=1)
        result += self.out_bias[np.newaxis]
        return result

    def mask_in(self, X):
        left_predicates = np.all(X[:, np.newaxis] >= self.lefts[np.newaxis], axis=2)
        right_predicates = np.all(X[:, np.newaxis] <= self.rights[np.newaxis], axis=2)
        predicates = left_predicates & right_predicates  # shape (n_samples, n_estimators)
        return predicates

    def mask_in_apr(self, X, apr_id: int):
        predicates = np.all(X >= self.lefts[np.newaxis, apr_id], axis=1) & \
                     np.all(X <= self.rights[np.newaxis, apr_id], axis=1)
        return predicates

    def to_tree(self):
        raise NotImplementedError()
