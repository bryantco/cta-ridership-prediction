"""Utility functions for model training and hyperparameter optimisation."""

import numpy as np
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_validate

import xgboost as xgb
import lightgbm as lgb


def r2_scorer_inverse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² after inverse log-transforming both arrays.

    Parameters
    ----------
    y_true:
        Ground-truth values (log-transformed).
    y_pred:
        Predicted values (log-transformed).

    Returns
    -------
    float
        R² score computed on the original (exponentiated) scale.
    """
    return r2_score(np.expm1(y_true), np.expm1(y_pred))


def xgb_objective(trial, x: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for tuning an XGBoost regressor.

    Parameters
    ----------
    trial:
        An :class:`optuna.Trial` instance.
    x:
        Feature matrix.
    y:
        Log-transformed target vector.

    Returns
    -------
    float
        Mean cross-validated R² on the original scale.
    """
    params = {
        "tree_method": "hist",
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }

    model = xgb.XGBRegressor(**params)
    cv_results = cross_validate(
        model, x, y, cv=3,
        scoring={"r2_inverse": make_scorer(r2_scorer_inverse)},
    )
    return cv_results["test_r2_inverse"].mean()


def lgb_objective(trial, x: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for tuning a LightGBM regressor.

    Parameters
    ----------
    trial:
        An :class:`optuna.Trial` instance.
    x:
        Feature matrix.
    y:
        Log-transformed target vector.

    Returns
    -------
    float
        Mean cross-validated R² on the original scale.
    """
    params = {
        "random_state": 42,
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
    }

    model = lgb.LGBMRegressor(**params)
    cv_results = cross_validate(
        model, x, y, cv=3,
        scoring={"r2_inverse": make_scorer(r2_scorer_inverse)},
    )
    return cv_results["test_r2_inverse"].mean()
