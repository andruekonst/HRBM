# Interpretable Ensembles of Hyper-Rectangles as Base Models (HRBM)

## Prerequisites

Please, make sure:

- Python version is smaller than 3.11 (*some dependencies like numba currently are not implemented for >= 3.11*);
- Your CPU is little-endian (*only for this format bitset packing is currently implemented and tested. It is required for Data-based SHAP explanation*).

## How to Use

Usage examples are located in [notebooks/](notebooks),
but a brief introduction is provided below.

### Model training

Currently two base models are available:

- [`hrbm.boosting.corner.CornerBoosting`](hrbm/boosting/corner.py);
- [`hrbm.boosting.rectangle.RectangleBoosting`](hrbm/boosting/rectangle.py).

**Note** that by default these models assume that target variable is scaled to small range (like `[-1, 1]`).
[`hrbm.wrappers.scaler.ScalerWrapper`](hrbm/wrappers/scaler) can be used to wrap a model and automatically transform target values, and rescale them back after prediction.

Example:

```{python}
from hrbm.wrappers.scaler import ScalerWrapper
from hrbm.boosting.corner import CornerBoosting

model = ScalerWrapper(
    CornerBoosting(
        learning_rate=1.0,
        need_rsm=True,
        rsm_size=5,
        val_size=0.5,
        hold_validation_set=False,
        enable_validation=False,
        reg_type='value',
        l1_reg=0.0,
        l2_reg='auto',
        l1_reg_min=0.0,
        l2_reg_min=0.01,
        n_estimators=10000,
        reg_beta=0.01,
        random_state=12345,
    )
)

model.fit(X_train, y_train)
```

### Explanation

Two explainers are available:

- [`hrbm.shap.model_based.ModelBasedEnsembleSHAP`](hrbm/shap/model_based.py) – Linear-time model-based data-agnostic;
- [`hrbm.shap.data_based.DataBasedEnsembleSHAP`](hrbm/shap/data_based.py) – Fast, but, strictly speaking, exponential (number of features) implementation that is equivalent to model-agnostic SHAP and much more effective.


```{python}
from hrbm.shap.model_based import ModelBasedEnsembleSHAP
from hrbm.shap.data_based import DataBasedEnsembleSHAP

model_explainer = ModelBasedEnsembleSHAP(model.model)
model_values, model_bias = model_explainer(explain_points)

data_explainer = DataBasedEnsembleSHAP(model.model, X_train)
data_values, data_bias = data_explainer(explain_points)
```
