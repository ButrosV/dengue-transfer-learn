import pandas as pd
import numpy as np
from typing import List

from src.config import ProjectConfig  # project config file parser
cnfg = ProjectConfig.load_configuration()

def encode_categorical(X: pd.DataFrame, cetegoricals: List[str] | None=None):
    """
    One-hot encode specified categorical columns with drop_first=True.
    :param X: Input DataFrame containing categorical columns to encode.
    :param cetegoricals: List or string of categorical column name(s) to encode. 
                           Defaults to None, useing config city feature group.
    :return: Dict with 'data' (encoded DataFrame copy)
             and 'features' dict containing 'new_names' (dummy column names)
             and 'source_categories' (mapping original column to its 
             unique categories before encoding).
    """
    X_dummies = X.copy()
    cetegoricals = cetegoricals or cnfg.preprocess.feature_groups["city"]
    if isinstance (cetegoricals, str):
        cetegoricals = [cetegoricals]
    source_categories = {feat: list(X_dummies[feat].unique())
                         for feat in cetegoricals}
    X_dummies = pd.get_dummies(X_dummies, columns=cetegoricals,
                               drop_first=True, dtype=int, prefix_sep="_")
    new_features = [f for f in X_dummies if f.startswith(tuple(cetegoricals))]
    
    return {"data": X_dummies,
            "features": {"new_names": new_features,
                         "source_categories": source_categories
                        }}
                        

def time_aware_group_split(X: pd.DataFrame,
                           y: pd.DataFrame | pd.Series,
                           group_feat:str | None=None,
                           test_size:int | None=None,
                           group_aware_frame: pd.DataFrame | None=None
                          ) -> tuple:
    """
    Generate time-aware train-test split preserving group structure (e.g., city, region).
    
    Split data by taking the last `test_size` rows from each group defined by `group_feat`.
    When no grouping is specified, takes last `test_size` rows from entire dataset.
    Handles different dataframes for splitting logic vs final output (e.g., original vs dummy-encoded).
    
    :param X: Input pandas DataFrame containing features for train/test split.
    :param y: Input pandas DataFrame or Series containing target values for train/test split.
    :param group_feat: Name of grouping feature (e.g., 'city') for temporal split. 
                       If None, uses default from configuration.
    :param test_size: Number of rows to take from end of each group for test set. 
                      If None, uses default from configuration (e.g., test_weeks).
    :param group_aware_frame: DataFrame used for grouping logic. Defaults to `X`. 
                              Use original pre-dummy data when `X` contains dummy variables.
    :return: Tuple of (X_train, X_test, y_train, y_test) DataFrames/Series.
    """
    group_feat = group_feat or cnfg.preprocess.feature_groups.get("city")
    test_size = test_size or cnfg.preprocess.train_test_split["test_weeks"]
    
    X = X.copy()
    y = y.copy()
    
    if group_aware_frame is None:
        group_aware_frame = X
        
    if group_feat is None:
        group_feat = pd.Series(
            data=np.ones(len(group_aware_frame), dtype=bool),
            index=group_aware_frame.index)
        
    test_nan_mask = group_aware_frame.index.isin(
        group_aware_frame.groupby(by=group_feat).tail(test_size).index)
    
    X_train = X[~test_nan_mask]
    X_test = X[test_nan_mask]
    y_train = y[~test_nan_mask]
    y_test = y[test_nan_mask]
    
    return X_train, X_test, y_train, y_test
    
