import pandas as pd
from typing import List, Dict, Any
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def cap_outliers(data: pd.DataFrame, features: List[str]=None,
                 group_keys: List[str]=None,
                 lower_cap:float=None, upper_cap:float=None,
                output_stats:bool=True) -> Dict[str, Any]:
    """
    Perform groupwise Winsorization (percentile clipping) on specified features to handle outliers.
    Automatically filter environmental features from config prefixes if not specified.
    :param data: Input pandas DataFrame.
    :param features: List of column names to clip. Default None auto-selects env features 
           from prefixes defined in config.yaml.
    :param group_keys: List of columns to group by for quantile calculation. Default None uses 
           config.yaml 'city' grouping.
    :param lower_cap: Lower percentile for clipping (0-1). Default None uses config.yaml 
           'outlier_perc.lower' (originally 0.01).
    :param upper_cap: Upper percentile for clipping (0-1). Default None uses config.yaml 
           'outlier_perc.upper' (originally 0.99).
    :param output_stats: If True, returns % rows changed per feature. Default True.
    :return: Dict containing:
           - 'data': Clipped DataFrame copy (original unchanged)
           - 'capped_row_prc': Series of % rows clipped per feature (if output_stats=True)
    """
    if features is None:
        features = [f for f in data.columns if f.startswith(
            tuple(cnfg.preprocess.feature_groups["env_prefixes"]))]
    if group_keys is None:
        group_keys = cnfg.preprocess.feature_groups["city"]
    if lower_cap is None:
        lower_cap = cnfg.preprocess.outlier_perc["lower"]
    if upper_cap is None:
        upper_cap = cnfg.preprocess.outlier_perc["upper"]

    data_no_outliers = data.copy()
    data_no_outliers[features] = data_no_outliers.groupby(by=group_keys)[features].transform(
        lambda group: group.clip(
            lower=group.quantile(lower_cap), upper=group.quantile(upper_cap)))

    if output_stats:
        capped_row_percent = round(
            ((data[features] != data_no_outliers[features]).sum() / len(data) * 100), 2)
        return {"data": data_no_outliers,
                "capped_row_prc": capped_row_percent}
    return {"data": data_no_outliers}
    

def drop_nan_rows(X: pd.DataFrame, y: pd.Series | None=None,
                  threshold_percent: float=None) -> pd.DataFrame:
    """
    Drop rows with NaN values exceeding threshold_percent of columns.
    
    :param X: pandas DataFrame of features.
    :param y: Optional target array/series. Default None.
    :param threshold_percent: Optional min non-null fraction required [0,1].
          Default None pulls 'nan_threshold' from config.yaml.
    :return: Filtered X (and y if provided), both with reset_index().
    """
    if threshold_percent is None:
        threshold_percent = cnfg.preprocess.nan_threshold
    row_drop_threshold = int(len(X.columns) * threshold_percent)
    X_no_nan = X.dropna(thresh=row_drop_threshold)
    if y is not None:
        return X_no_nan.reset_index(), y.iloc[X_no_nan.index].reset_index()
    return X_no_nan.reset_index()


def median_groupwise_impute(X: pd.DataFrame,
                            group_keys: List[str]=None) -> pd.DataFrame:
    """
    Impute NaN values in numeric columns using median within specified group keys.
    
    :param X: pandas DataFrame containing grouping columns and features to impute.
    :param group_keys: Optional list of column names for grouping.
                 Default None pills 'city' and 'week' feature names from config.yaml.
    :return: Copy of input DataFrame with NaNs filled by group-wise medians.
    """
    if group_keys is None:
        group_keys = [cnfg.preprocess.feature_groups["city"],
                      cnfg.preprocess.feature_groups["week"]]
    missing_keys = set(group_keys) - set(X.columns)
    if missing_keys:
        raise ValueError(f"Missing group keys {missing_keys}")

    X_no_nan = X.copy()
    cols_with_nan = X_no_nan.select_dtypes(include="number")\
        .columns[X_no_nan.select_dtypes(include="number").isna().sum() > 0].to_list()

    if len(cols_with_nan) > 0:
        X_no_nan[cols_with_nan] = X_no_nan[cols_with_nan + group_keys]\
            .groupby(by=group_keys)[cols_with_nan]\
            .transform(lambda group: group.fillna(group.median()))
    return X_no_nan
    
