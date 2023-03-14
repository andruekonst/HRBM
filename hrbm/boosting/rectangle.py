import numpy as np
from .regularized import RegularizedHRBMBoosting


def find_nearest(arr, values):
    """
    Args:
        arr: Array to search in.
        values: Values to search for of shape (n_values).
    """
    ids = np.searchsorted(arr, values, side='left')
    result = np.full_like(values, arr[0])
    ids[ids == 0] = 1
    ids[ids == len(arr)] -= 1
    predicate = (np.abs(arr[ids] - values) < np.abs(arr[ids - 1] - values))
    result[predicate] = arr[ids][predicate]
    result[~predicate] = arr[ids - 1][~predicate]
    return result


class NearestFarestSearcher:
    """Find the nearest and the farest feature-wise vectors.
    """
    def __init__(self, X):
        self.sorted_X_ = np.sort(X, axis=0)
        self.min_X_ = self.sorted_X_[0]
        self.max_X_ = self.sorted_X_[-1]

    def nearest(self, centers):
        """
        Args:
            centers: Centers of shape (n_centers, n_features).
        """
        n_features = centers.shape[1]
        return np.stack([
            find_nearest(self.sorted_X_[:, j], centers[:, j])
            for j in range(n_features)
        ], axis=1)

    def farest_dist(self, centers):
        return np.maximum(
            np.abs(centers - self.min_X_[np.newaxis]),
            np.abs(centers - self.max_X_[np.newaxis])
        )



class UniformRectMixinForReg:
    # def __init__(self, n_estimators: int = 1000,
    #              learning_rate: float = 0.1,
    #              random_state: Optional[int] = None,
    #              need_rsm: bool = False,
    #              rsm_size: int = -1):
    #     super().__init__(
    #         n_estimators=n_estimators,
    #         learning_rate=learning_rate,
    #         random_state=random_state
    #     )
    #     self.need_rsm = need_rsm
    #     self.rsm_size = rsm_size

    def _sample_boundaries(self, X, y, sample_weight=None, rng=None):
        """Uniform corner sampling.
        """
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)
        self.searcher_ = NearestFarestSearcher(X)

        centers = rng.uniform(self.X_min_, self.X_max_, size=(self.n_estimators, X.shape[1]))
        # sizes = rng.uniform(0, self.X_max_ - self.X_min_, size=(self.n_estimators, X.shape[1]))
        sizes = rng.uniform(
            2 * np.abs(centers - self.searcher_.nearest(centers)),
            2 * (self.X_max_ - self.X_min_),
            size=(self.n_estimators, X.shape[1])
        )
        self.hrbms_.rights = centers + sizes / 2
        self.hrbms_.lefts = centers - sizes / 2

        # RSM
        if not self.need_rsm:
            return
        n_features = X.shape[1]
        n_used_features = self.rsm_size if self.rsm_size > 0 else n_features
        n_unused_features = max(n_features - n_used_features, 0)

        unused_features_ids = rng.choice(n_features, size=(self.n_estimators, n_features))
        for i in range(self.n_estimators):
            unused_ids = unused_features_ids[i, :n_unused_features]
            if len(unused_ids) > 0:
                self.hrbms_.lefts[i, unused_ids] = -np.inf
                self.hrbms_.rights[i, unused_ids] = np.inf

    def _rebuild_apr(self, i: int, X, rng=None):
        n_features = X.shape[1]
        centers = rng.uniform(self.X_min_, self.X_max_, size=(n_features,))
        # sizes = rng.uniform(0, self.X_max_ - self.X_min_, size=(n_features,))
        sizes = rng.uniform(
            np.abs(centers[np.newaxis] - self.searcher_.nearest(centers[np.newaxis]))[0],
            self.X_max_ - self.X_min_,
            size=(n_features,)
        )
        self.hrbms_.rights[i] = centers + sizes / 2
        self.hrbms_.lefts[i] = centers - sizes / 2
        # RSM
        if not self.need_rsm:
            return
        n_used_features = self.rsm_size if self.rsm_size > 0 else n_features
        n_unused_features = max(n_features - n_used_features, 0)

        unused_features_ids = rng.choice(n_features, size=(n_features,))
        unused_ids = unused_features_ids[:n_unused_features]
        if len(unused_ids) > 0:
            self.hrbms_.lefts[i, unused_ids] = -np.inf
            self.hrbms_.rights[i, unused_ids] = np.inf


class RectangleBoosting(UniformRectMixinForReg, RegularizedHRBMBoosting):
    ...
