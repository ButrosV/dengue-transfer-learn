import pandas as pd
import numpy as np

from typing import Iterable, Tuple

import lightgbm as lgb

from sklearn.metrics import r2_score, mean_absolute_error

from src.utils.utils import _check_feature_presence
from src.preprocessing.preprocess import encode_categorical
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
             target_feature:str | None = None,
             log_target: bool | None = None,
             test_size:int | None = None,
             min_train_size:int | None = None,
             stride_size:int | None = None    
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
    :param test_size: Size of test window in rows. Defaults to config value.
    :param min_train_size: Minimum training rows before first fold. Defaults to config value.
    :param stride_size: Step between test window starts. Defaults to config value.
    
    :return: Dict with 'aggregated' (feature-level stats + mean metrics) 
                        and 'raw' (per-fold metrics):
             - aggregated: mean_r2, mean_mae, feature[list], mean_importances[array], 
                          stability_cv[array]
             - raw: raw_r2[list], raw_mae[list] (one value per fold)
             - fold_test_starts: fold test start indexes
    """
    target_feature = target_feature or cnfg.preprocess.feature_groups["target"]
    log_target = log_target or cnfg.preprocess.target["logtransform"]
    window_splits = cnfg.preprocess.wfcv["window_splits"]
    test_size = test_size or window_splits["test_size"]
    min_train_size = min_train_size or window_splits["min_train_size"]
    stride_size = stride_size or window_splits["stride_size"]
    
    y = y if isinstance(y, pd.Series) else y.copy()[target_feature]

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


def wfcv_feature_autoselect(X: pd.DataFrame, y: pd.DataFrame | pd.Series,
                           mean_imp_thresh: float | None = None,
                           lowest_mean_thresh:float | None = None,
                           stability_cv_thresh:float | None = None,
                           do_not_drop:Tuple[str] or None = None):
    """
    Perform walk-forward CV feature selection, dropping low-importance and unstable features.
    
    Run WFCV via get_wfcv(), then apply two-stage feature selection policy:
    1. Drop features below mean_importance threshold
    2. Among remainder, drop bottom N% of mean_importance IF stability_cv > threshold
    
    Protect features matching do_not_drop prefix (e.g., 'city' → protects 'city_sj').
    
    :param X: Input features DataFrame (auto-encodes categoricals if needed).
    :param y: Target Series or DataFrame containing target column.
    :param mean_imp_thresh: Minimum mean importance to keep. Uses config default.
    :param lowest_mean_thresh: Fraction of lowest mean_importance features to consider 
                              dropping (e.g. 0.2 = bottom 20%). Uses config default.
    :param stability_cv_thresh: Maximum stability_cv for drop consideration 
                               (high CV = unstable). Uses config default.
    :param do_not_drop: Prefixes/patterns of features to always keep (e.g. "city").
                        Defaults to config "city" feature.
    
    :return: Dict with 'data' (reduced DataFrame) and "removed_features" (list of dropped names).
    """
    feat_sel_policy = cnfg.preprocess.wfcv["feature_selection_policy"]
    if mean_imp_thresh is None:  # 0 can be falsy, thus "or" may fail
        mean_imp_thresh = feat_sel_policy["mean_imp_thresh"]
    if lowest_mean_thresh is None:
        lowest_mean_thresh = feat_sel_policy["lowest_mean_thresh"]
    if stability_cv_thresh is None:
        stability_cv_thresh = feat_sel_policy["stability_cv_thresh"]
    do_not_drop = do_not_drop or tuple([cnfg.preprocess.feature_groups["city"]])

    if any(X.dtypes == "object"):
        X=encode_categorical(X)["data"]
        
    vfcv_dict = get_wfcv(X, y=y)
    vfcv_df = pd.DataFrame(data=vfcv_dict["aggregated"])
    
    mask_high_means = vfcv_df["mean_importances"] > mean_imp_thresh
    vfcv_df_red = vfcv_df[mask_high_means].copy()
    
    low_importances = int(len(vfcv_df_red) * lowest_mean_thresh)
    low_importances = vfcv_df_red["mean_importances"].nsmallest(low_importances)
        
    mask_low_imprtn_cv = (
        vfcv_df_red["mean_importances"].isin(low_importances) & 
        (vfcv_df_red["stability_cv"] > stability_cv_thresh))
    vfcv_df_red = vfcv_df_red[~mask_low_imprtn_cv]

    drop_feats = set(vfcv_df["feature"].values) - set(vfcv_df_red["feature"].values)
    drop_feats = [feat for feat in drop_feats if not feat.startswith(do_not_drop)]

    X = X.drop(columns = drop_feats)

    return {"data": X, "removed_features": drop_feats}
        
