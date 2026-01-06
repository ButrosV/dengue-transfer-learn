import pandas as pd

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
    result = X.dropna(thresh=row_drop_threshold)
    if y is not None:
        return result.reset_index(), y.iloc[result.index].reset_index()
    return result.reset_index()
    