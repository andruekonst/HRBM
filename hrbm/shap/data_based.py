import numpy as np
import numba
from .util import _pack_to_uint32, _count_ones


@numba.njit
def _fast_expectation_apr_shap(shap_values: np.ndarray,
                               tests: np.ndarray,
                               n_in: np.ndarray,
                               phi_out_minus_phi_empty: float) -> np.ndarray:
    """Compute the final SHAP values.

    Args:
        shap_values: Input-output SHAP values.
                     Only values for `inside` features should be precomputed.
        tests: Tests results for each sample and feature.
               `tests[i, j] == True` iff i-th point is inside by feature j.
        n_in: Number of features inside (sum of `tests` over axis 1).
        phi_out_minus_phi_empty: Difference between a model prediction and a bias.

    Returns:
        Updated `shap_values`.

    """
    n_samples = tests.shape[0]
    n_features = tests.shape[1]
    # result = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        if n_in[i] < n_features:  # if point is outside for the i-th point
            sum_inner = 0.0
            for j in range(n_features):  # calculate sum over `inside` features
                if tests[i, j]:
                    sum_inner += shap_values[i, j]
            for j in range(n_features):  # calculate values for `outside` features
                if not tests[i, j]:
                    shap_values[i, j] = (phi_out_minus_phi_empty - sum_inner) / (n_features - n_in[i])
    return shap_values


@numba.njit
def _make_inv_binomial_coefficients(n_features: int):
    """Generate inverted binomial coefficients for SHAP general equation.

    Args:
        n_features: Number of features.
    """
    coef = np.zeros((n_features + 1,), dtype=np.float64)
    k = 1 / n_features
    for s in range(n_features):
        if s > 0:
            k *= s / (n_features - s)
        coef[s] = k
    return coef


@numba.njit
def _fast_precompute_means(background_tests_bitset: np.ndarray,
                           coef: np.ndarray,
                           n_features: int) -> np.ndarray:
    """Precompute average values for every feature subset.

    Args:
        background_tests_bitset: Tests for background points of shape (n_samples,).
        coef: Inverted binomial coefficients times v_diff divided by n_samples.

    Returns:
        Array of means indexed by subsets of shape (n_features, 2 ** n_features).

    """
    # background_tests shape: (n_background_samples, n_features)
    # result[i, S] is a number of points with feature i outside the rectangle,
    # ... and features out of S inside.
    n_points = background_tests_bitset.shape[0]
    n_subsets = 2 ** n_features
    general_set = (1 << n_features) - 1

    result = np.zeros((n_features, n_subsets), dtype=np.float64)
    for i in range(n_features):
        two_power_i = int(1 << i)

        for t in range(n_points):
            if background_tests_bitset[t] & two_power_i:
                continue
            # iterate over subsets
            tbt = background_tests_bitset[t]
            inv_s = tbt
            while True:
                # assert ((background_tests_bitset[t] & inv_s) == inv_s)
                s = np.invert(inv_s)
                result[i, (two_power_i ^ s) & general_set] += 1
                if inv_s == 0:
                    break
                inv_s = (inv_s - 1) & tbt
    for i in range(n_features):
        # two_power_i = int(1 << i)
        for index_s in range(n_subsets):
            index_s_size = _count_ones(index_s)
            result[i, index_s] = result[i, index_s] * coef[index_s_size]

    return result


@numba.njit
def _fast_compute_feature_shap(tests_bitset: np.ndarray,
                               precomputed_means: np.ndarray,
                               n_features: int) -> np.ndarray:
    """Precompute SHAP values for features that are `inside`.
    """
    n_samples = tests_bitset.shape[0]
    shap_values = np.zeros((n_samples, n_features), dtype=np.float64)

    for t in range(n_samples):
        # iterate over subsets of tests_bitset[t]
        tbt = tests_bitset[t]
        s = tbt
        while True:
            # assert ((tbt & s) == s)
            # update shap values for feature subset s
            for i in range(n_features):
                if not (tbt & (1 << i)):
                    continue
                shap_values[t, i] += precomputed_means[i, s]
            # go to the next subset
            if s == 0:  # avoid infinite loop over subsets
                break
            s = ((s - 1) & tbt)

    return shap_values


class DataBasedEnsembleSHAP:
    """Data- or Expectation-Based Ensemble SHAP.
    """
    def __init__(self, ens, background=None):
        self.ens = ens
        self.background = background
        self._precompute_means()

    def _precompute_means(self):
        self.background_in_ratio_ = []
        self.precomputed_means_ = []
        n_samples, n_features = self.background.shape
        inv_coef = _make_inv_binomial_coefficients(n_features) / n_samples
        for i in range(self.ens.n_estimators):
            background_tests = (self.ens.hrbms_.lefts[i][np.newaxis] <= self.background) & \
                               (self.background <= self.ens.hrbms_.rights[i][np.newaxis])
            n_in = np.all(background_tests, axis=1).sum(axis=0)
            self.background_in_ratio_.append(n_in / self.background.shape[0])
            background_tests_bitset = _pack_to_uint32(background_tests)
            v_diff = self.ens.hrbms_.in_values[i][0]
            self.precomputed_means_.append(
                _fast_precompute_means(
                    background_tests_bitset,
                    inv_coef * v_diff,
                    background_tests.shape[1]
                )
            )
        self.background_in_ratio_ = np.array(self.background_in_ratio_)
        self.precomputed_means_ = np.stack(self.precomputed_means_, axis=0)

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
            tests = (self.ens.hrbms_.lefts[i][np.newaxis] <= X) &\
                    (X <= self.ens.hrbms_.rights[i][np.newaxis])
            # tests shape: (n_samples, n_features)
            n_in = tests.sum(axis=1)  # shape: (n_samples)
            v_diff = (self.ens.hrbms_.in_values[i][0] - 0)
            b = v_diff * self.background_in_ratio_[i]
            # s = _fast_compute_feature_shap(tests, self.precomputed_means_[i])
            # tests_bitset = np.packbits(tests, axis=1, bitorder='little').squeeze(1)
            tests_bitset = _pack_to_uint32(tests)
            s = _fast_compute_feature_shap(
                tests_bitset,
                self.precomputed_means_[i],
                n_features
            )
            phi_out_minus_phi_empty = -v_diff * self.background_in_ratio_[i]
            _fast_expectation_apr_shap(
                s,
                tests,
                n_in,
                phi_out_minus_phi_empty
            )
            # update sums
            bias_value += b * gamma
            shap_values += s * gamma

        return (shap_values, bias_value)
