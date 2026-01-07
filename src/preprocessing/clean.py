import pandas as pd
from typing import List


def drop_nan_rows(X: pd.DataFrame, y: pd.Series | None = None,
                  threshold_percent: float = 0.5):
    """
    Drop rows with NaN values exceeding threshold_percent of columns.
    
    :param X: pandas DataFrame of features.
    :param y: Optional target array/series. Default None.
    :param threshold_percent: Min non-null fraction required [0,1]. Default 0.5.
    :return: Filtered X (and y if provided), both with reset_index().
    """
    row_drop_threshold = int(len(X.columns) * threshold_percent)
    X_no_nan = X.dropna(thresh=row_drop_threshold)
    if y is not None:
        return X_no_nan.reset_index(), y.iloc[X_no_nan.index].reset_index()
    return X_no_nan.reset_index()


def median_groupwise_impute(X: pd.DataFrame,
                            group_keys: List[str] = ['city', 'weekofyear']):
    """
    Impute NaN values in numeric columns using median within specified group keys.
    
    :param X: pandas DataFrame containing grouping columns and features to impute.
    :param group_keys: List of column names for grouping. Default ['city', 'weekofyear'].
    :return: Copy of input DataFrame with NaNs filled by group-wise medians.
    """
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
    