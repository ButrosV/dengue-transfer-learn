import pandas as pd
import numpy as np

from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def split_by_feature(X: np.ndarray,
                    y: np.ndarray,
                    group_source: pd.DataFrame,
                    feature:str) -> tuple[dict, dict]:
    """
    A helper to split NumPy feature and target arrays into groups based on a DataFrame column.

    :param X: Input feature array (e.g., X_train or X_valid).
    :param y: Input target array aligned with X.
    :param group_source: DataFrame containing grouping feature aligned with X/y.
    :param feature: Column name used to split data (e.g., 'city').
    :return: Tuple of two dictionaries:
             - X_grouped: {group_value: X_subset}
             - y_grouped: {group_value: y_subset}
    """
    X_grouped = dict()
    y_grouped = dict()
    for city in group_source[feature].unique():
        mask = (group_source[feature] == city).values
        X_grouped[city] = X[mask]
        y_grouped[city] = y[mask]
    return X_grouped, y_grouped
    
