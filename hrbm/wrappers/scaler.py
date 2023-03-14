from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler


class ScalerWrapper(BaseEstimator):
    """Target scaler wrapper.
    """
    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        self.scaler_ = RobustScaler()
        scaled_y = self.scaler_.fit_transform(y.reshape((-1, 1))).squeeze(1)
        self.model.fit(X, scaled_y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return self.scaler_.inverse_transform(preds.reshape((-1, 1))).squeeze(1)
