import numpy as np
import numba
from ..boosting.base import BaseHRBMBoosting


@numba.njit
def _fast_apr_shap(tests: np.ndarray, n_in: np.ndarray, n_features: int,
                   minus_v_diff: float) -> np.ndarray:
    n_samples = tests.shape[0]
    result = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        if n_in[i] < n_features:
            for j in range(n_features):
                if not tests[i, j]:
                    result[i, j] = minus_v_diff / (n_features - n_in[i])
    return result


class ModelBasedEnsembleSHAP:
    def __init__(self, ens: BaseHRBMBoosting):
        self.ens = ens

    def __call__(self, X):
        """Calculate SHAP values.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Tuple (shap_values, bias_value).

        """
        n_features = X.shape[1]
        bias_value = self.ens.mean_ + self.ens.hrbms_.out_bias
        shap_values = 0.0
        gamma = self.ens.learning_rate
        for i in range(self.ens.n_estimators):
            # current bias value
            b = self.ens.hrbms_.in_values[i][0]  # self.apr.value_in
            tests = (self.ens.hrbms_.lefts[i][np.newaxis] <= X) &\
                    (X <= self.ens.hrbms_.rights[i][np.newaxis])
            # tests shape: (n_samples, n_features)
            n_in = tests.sum(axis=1)  # shape: (n_samples)
            minus_v_diff = (-(self.ens.hrbms_.in_values[i] - 0))[0]
            # current shap values
            s = _fast_apr_shap(
                tests,
                n_in,
                n_features,
                minus_v_diff
            )
            # update sums
            bias_value += b * gamma
            shap_values += s * gamma

        return (shap_values, bias_value)
