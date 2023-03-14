import numpy as np
from typing import Optional, Union
from abc import ABC, abstractmethod
from scipy.special import softmax, xlogy


class LossFn(ABC):
    CONSTANT_HESSIAN = False

    @abstractmethod
    def __call__(self, y_true, y_pred, no_return: bool = False):
        ...

    @abstractmethod
    def link(self, y):
        ...

    @property
    @abstractmethod
    def gradient(self) -> Optional[np.ndarray]:
        ...

    @property
    @abstractmethod
    def hessian(self) -> Optional[Union[np.ndarray, float]]:
        ...

    @abstractmethod
    def clear(self):
        """Clear the state."""
        ...

    @abstractmethod
    def find_opt_constant(self, y):
        ...

    def get_hessian_constant(self, y_true):
        if not self.CONSTANT_HESSIAN:
            raise Exception(
                'Cannot get Hessian constant for the loss with '
                'non-constant Hessian'
            )
        else:
            raise NotImplementedError(
                'Hessian constant getter is not implemented'
            )


class MSELossFn(LossFn):
    CONSTANT_HESSIAN = True

    def __call__(self, y_true, y_pred, no_return: bool = False):
        self._residuals = y_pred - y_true
        if no_return:
            return
        return (self._residuals ** 2).sum()

    def link(self, y):
        return y

    @property
    def gradient(self) -> np.ndarray:
        return 2 * self._residuals

    @property
    def hessian(self) -> float:
        return 2

    def clear(self):
        if hasattr(self, '_residuals'):
            del self._residuals

    def find_opt_constant(self, y):
        return y.mean(axis=0)

    def get_hessian_constant(self, y_true):
        return np.full_like(y_true, 2)


class CrossEntropyLossFn(LossFn):
    def __call__(self, y_true, y_pred, no_return: bool = False):
        """
        Args:
            y_true: One-hot class vector.
            y_pred: Logits for each class.
        """
        probas = self.link(y_pred)
        self._residuals = probas - y_true
        self._hessians = probas * (1 - probas)
        eps = np.finfo(y_pred.dtype).eps
        self._hessians[np.isclose(self._hessians, 0, atol=eps)] = 1.0
        if no_return:
            return
        probas = np.clip(probas, eps, 1 - eps)
        probas_sum = probas.sum(axis=1)
        probas = probas / probas_sum[:, np.newaxis]
        return -xlogy(y_true, probas).sum()

    def link(self, logits):
        return softmax(logits, axis=1)

    @property
    def gradient(self) -> np.ndarray:
        return self._residuals

    @property
    def hessian(self) -> float:
        return self._hessians

    def clear(self):
        if hasattr(self, '_residuals'):
            del self._residuals
            del self._hessians

    def find_opt_constant(self, y_hot):
        return y_hot.mean(axis=0)


LOSSES = {
    'mse': MSELossFn,
    'ce': CrossEntropyLossFn,
}


def make_loss(loss: Union[str, LossFn]):
    if isinstance(loss, LossFn):
        return loss
    if loss not in LOSSES:
        raise ValueError(
            f'Cannot find loss with name {loss!r}. Available losses: {list(LOSSES.keys())!r}'
        )
    return LOSSES[loss]()
