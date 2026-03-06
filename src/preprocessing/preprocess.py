from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime 
import joblib
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler

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
    

def robust_scale_data_train(X_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            X_valid: pd.DataFrame,
                            y_valid: pd.DataFrame,
                            target_feat:str | None=None,
                            X_scaler_path: Path | None = None,
                            y_scaler_path: Path | None = None,
                            overwrite_files: bool = False) -> Dict[str, Any]:
    """
    Fit and apply RobustScaler to training and validation data.
    
    Optionally save fitted scalers to disk with directory creation and handling 
    file overwrites with timestamped filenames. Scalers are saved if 
    BOTH X_scaler_path and y_scaler_path are provided (via params or config).
    
    :param X_train: Training features DataFrame.
    :param y_train: Training targets DataFrame.
    :param X_valid: Validation features DataFrame.
    :param y_valid: Validation targets DataFrame.
    :param target_feat: Target column name. Falls back to config default if None.
    :param X_scaler_path: Path to save X scaler. Falls back to config if None.
    :param y_scaler_path: Path to save y scaler. Falls back to config if None.
    :param overwrite_files: If True, overwrite existing scaler files. If False (default),
                           create timestamped versions like `scaler_X_20260306_1755.joblib`.
    :return: Dictionary containing:
             - ``scaled_data``: Scaled X_train_sc, X_valid_sc, y_train_sc, y_valid_sc arrays
             - ``scalers``: Fitted scaler_X and scaler_y objects
             - ``paths``: Actual paths where scalers were saved (timestamped if created)
    :raises KeyError: If target_feat not found in y_train/y_valid or config paths missing.
    """
    directory = cnfg.data.dirs.get("model")
    X_file = cnfg.data.files.get("X_scaler")
    y_file = cnfg.data.files.get("y_scaler")
    target_feat = target_feat or cnfg.preprocess.feature_groups["target"]
    
    X_scaler_path = X_scaler_path or (directory / X_file if X_file else None)
    y_scaler_path = y_scaler_path or (directory / y_file if y_file else None)

    rob_scaler_X = RobustScaler()
    
    rob_scaler_X.fit(X_train)
    X_train_sc = rob_scaler_X.transform(X_train)
    X_valid_sc = rob_scaler_X.transform(X_valid)
    
    rob_scaler_y = RobustScaler()

    y_train = y_train[target_feat].values.reshape(-1,1)
    rob_scaler_y.fit(y_train)
    y_train_sc = rob_scaler_y.transform(y_train).ravel()
    y_valid = y_valid[target_feat].values.reshape(-1,1)
    y_valid_sc = rob_scaler_y.transform(y_valid).ravel()

    path_names = [None, None]
    if (X_scaler_path is not None) and (y_scaler_path is not None):
        path_names = []
        for path, scaler in (
            (X_scaler_path, rob_scaler_X), (y_scaler_path, rob_scaler_y)):
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                if overwrite_files:
                    logging.info("Path file present, overwriting.")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    path = path.with_stem(f"{path.stem}_{timestamp}")
                    logging.info(f"Path file present, creating new one: {path.name}.")
            joblib.dump(scaler, path)
            path_names.append(path)
        
    paths = {"X_scaler_path": path_names[0], "y_scaler_path": path_names[1]}
    scalers = {"scaler_X": rob_scaler_X, "scaler_y": rob_scaler_y}
    scaled_data = {"X_train_sc": X_train_sc, "X_valid_sc": X_valid_sc,
                  "y_train_sc": y_train_sc, "y_valid_sc": y_valid_sc}

    return {"scaled_data": scaled_data,
            "scalers": scalers, "paths": paths}
    
