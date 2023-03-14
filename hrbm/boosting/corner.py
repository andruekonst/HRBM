import numpy as np
from typing import Optional
from .regularized import RegularizedHRBMBoosting


class UniformCornerBoundariesMixin:
    def _sample_boundaries(self, X, y, sample_weight=None, rng=None):
        """Uniform corner sampling.
        """
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)
        if self.margin > 0.0:
            x_span = self.X_max_ - self.X_min_
            self.X_min_ -= x_span * self.margin
            self.X_max_ += x_span * self.margin

        centers = rng.uniform(self.X_min_, self.X_max_, size=(self.n_estimators, X.shape[1]))
        directions = rng.choice(2, size=centers.shape)
        directions_left = (directions == 0)
        directions_right = ~directions_left
        self.hrbms_.rights[directions_left] = centers[directions_left]
        self.hrbms_.lefts[directions_right] = centers[directions_right]

        if not self.need_rsm:
            return
        # RSM
        n_features = X.shape[1]
        n_used_features = n_features if self.rsm_size <= 0 else self.rsm_size
        n_unused_features = max(n_features - n_used_features, 0)

        for i in range(self.n_estimators):
            cur_n_unused_features = rng.randint(n_unused_features, n_features)
            unused_ids = rng.choice(n_features, size=cur_n_unused_features, replace=False)
            self.hrbms_.lefts[i, unused_ids] = -np.inf
            self.hrbms_.rights[i, unused_ids] = np.inf

    def _rebuild_apr(self, i: int, X, rng=None):
        n_features = X.shape[1]
        centers = rng.uniform(self.X_min_, self.X_max_, size=(n_features,))
        directions = rng.choice(2, size=centers.shape)
        directions_left = (directions == 0)
        directions_right = ~directions_left
        self.hrbms_.rights[i, directions_left] = np.inf
        self.hrbms_.lefts[i, directions_left] = -np.inf
        self.hrbms_.rights[i, directions_left] = centers[directions_left]
        self.hrbms_.lefts[i, directions_right] = centers[directions_right]

        if not self.need_rsm:
            return
        # RSM
        n_used_features = n_features if self.rsm_size <= 0 else self.rsm_size
        n_unused_features = max(n_features - n_used_features, 0)

        unused_features_ids = rng.choice(n_features, size=(n_features,))
        unused_ids = unused_features_ids[:n_unused_features]
        self.hrbms_.lefts[i, unused_ids] = -np.inf
        self.hrbms_.rights[i, unused_ids] = np.inf


class TotallyRandomCornerBoundariesMixin:
    def _sample_boundaries(self, X, y, sample_weight=None, rng=None):
        """Uniform corner sampling.
        """
        self.X_min_ = np.min(X, axis=0)
        self.X_max_ = np.max(X, axis=0)
        if self.margin > 0.0:
            x_span = self.X_max_ - self.X_min_
            self.X_min_ -= x_span * self.margin
            self.X_max_ += x_span * self.margin

        n_features = X.shape[1]
        centers = rng.uniform(self.X_min_, self.X_max_, size=(self.n_estimators, n_features))
        directions = rng.choice(2, size=centers.shape)
        directions_left = (directions == 0)
        directions_right = ~directions_left
        self.hrbms_.rights[directions_left] = centers[directions_left]
        self.hrbms_.lefts[directions_right] = centers[directions_right]
        # RSM
        n_used_features = rng.randint(1, n_features + 1, size=(self.n_estimators,))
        n_unused_features = n_features - n_used_features

        unused_features_ids = rng.choice(n_features, size=(self.n_estimators, n_features))
        for i in range(self.n_estimators):
            unused_ids = unused_features_ids[i, :n_unused_features[i]]
            self.hrbms_.lefts[i, unused_ids] = -np.inf
            self.hrbms_.rights[i, unused_ids] = np.inf

    def _rebuild_apr(self, i: int, X, rng=None):
        n_features = X.shape[1]
        centers = rng.uniform(self.X_min_, self.X_max_, size=(n_features,))
        directions = rng.choice(2, size=centers.shape)
        directions_left = (directions == 0)
        directions_right = ~directions_left
        self.hrbms_.rights[i, directions_left] = np.inf
        self.hrbms_.lefts[i, directions_left] = -np.inf
        self.hrbms_.rights[i, directions_left] = centers[directions_left]
        self.hrbms_.lefts[i, directions_right] = centers[directions_right]
        # RSM
        n_used_features = rng.randint(1, n_features + 1)
        n_unused_features = max(n_features - n_used_features, 0)

        unused_features_ids = rng.choice(n_features, size=(n_features,))
        unused_ids = unused_features_ids[:n_unused_features]
        self.hrbms_.lefts[i, unused_ids] = -np.inf
        self.hrbms_.rights[i, unused_ids] = np.inf




class CornerBoosting(UniformCornerBoundariesMixin, RegularizedHRBMBoosting):
    ...


class TotallyRandomCornerBoosting(TotallyRandomCornerBoundariesMixin, RegularizedHRBMBoosting):
    ...
