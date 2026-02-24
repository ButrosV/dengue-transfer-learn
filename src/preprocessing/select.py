import pandas as pd
import numpy as np

from typing import Iterable

import lightgbm as lgb

from sklearn.metrics import r2_score, mean_absolute_error

from src.utils.utils import _check_feature_presence
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()

 
def remove_features(X: pd.DataFrame, feats_to_drop: Iterable[str]=None)->pd.DataFrame:
    """
    Remove specified feature columns from DataFrame using config defaults or explicit list.
    :param X: Input DataFrame.
    :param feats_to_drop: Features to drop. If None, uses config multicollinearity removal list.
                          Default None.
    :return: New DataFrame copy with specified features removed. Raises ValueError if features missing.
    """
    if feats_to_drop is None:
        feats_to_drop = cnfg.preprocess.multicolinear["removal_list"]
    X_slim = X.copy()
    if isinstance(feats_to_drop, str):
        feats_to_drop = [feats_to_drop]
    _check_feature_presence(target_list=feats_to_drop, source_list=X_slim.columns)
    return X_slim.drop(columns=feats_to_drop)
    
    
def get_wfcv(X: pd.DataFrame,
             y: pd.Series | pd.DataFrame,
             target_feture:str | None = None,
             log_target: bool = True,
             test_size:int = 52,
             min_train_size:int = 104,
             stride_size:int = 52    
):
    """
    Perform walk-forward cross-validation with LightGBM for time series forecasting 
    and aggregate feature importances across folds.
    
    Use growing-window splits. Optionally train on log(1+y) targets 
    and back-transform predictions for evaluation.
    
    :param X: Features DataFrame (time-ordered rows).
    :param y: Target Series or DataFrame. If DataFrame, extracts target_feature column.
    :param target_feture: Target column name if y is DataFrame. Defaults to config target.
    :param log_target: If True, train on log1p(y_train), predict and expm1 for metrics.
    :param test_size: Size of test window in rows (default: 52 weeks).
    :param min_train_size: Minimum training rows before first fold (default: 104 weeks).
    :param stride_size: Step between test window starts (default: 52 weeks).
    
    :return: Dict with 'aggregated' (feature-level stats + mean metrics) 
                        and 'raw' (per-fold metrics):
             - aggregated: mean_r2, mean_mae, feature[list], mean_importances[array], 
                          stability_cv[array]
             - raw: raw_r2[list], raw_mae[list] (one value per fold)
             - fold_test_starts: fold test start indexes
    """
    target_feture = target_feture or cnfg.preprocess.feature_groups["target"]
    y = y if isinstance(y, pd.Series) else y.copy()[target_feture]

    assert all(X.index == y.index), "Indices for 'X' and 'y' must be aligned." 
    
    results = {"r2s": [], "mae": [], "importance_gains": [], "fold_test_starts" : []}
    
    for test_start in range(min_train_size, len(X) - test_size + 1, stride_size):
        test_start = min(len(X) - test_size, test_start)
        X_train, X_test = X.iloc[:test_start], X.iloc[test_start:test_start+test_size]
        y_train = np.log1p(y.iloc[:test_start]) if log_target else y.iloc[:test_start]
        y_test = y.iloc[test_start:test_start+test_size]
    
        model = lgb.LGBMRegressor(verbose=-1)
        model.fit(X=X_train, y=y_train)
        y_pred = np.expm1(model.predict(X=X_test)) if log_target else model.predict(X=X_test)

        results["mae"].append(mean_absolute_error(y_pred=y_pred, y_true=y_test))
        results["r2s"].append(r2_score(y_pred=y_pred, y_true=y_test))
        results["importance_gains"].append(model.booster_.feature_importance(importance_type='gain'))
        results["fold_test_starts"].append(test_start)
    
    std_importances = np.std(results["importance_gains"], axis=0)
    mean_importances = np.mean(results["importance_gains"], axis=0)
    
    aggregated_results = {"mean_r2": np.mean(results["r2s"]),
                          "mean_mae": np.mean(results["mae"]),
                          "feature": X.columns.to_list(),
                          "mean_importances": mean_importances,
                          "stability_cv": std_importances / (mean_importances + 1e-8)}
                          
    raw_stats = {"raw_r2": results["r2s"],
                 "raw_mae": results["mae"]}
                                 
    return {"aggregated": aggregated_results, "raw": raw_stats,
    		"fold_test_starts": results["fold_test_starts"]}
    
