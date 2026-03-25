from pathlib import Path
import pandas as pd
from datetime import datetime
import joblib

from sklearn.preprocessing import RobustScaler

import logging
logging.basicConfig(level=logging.INFO)

from typing import Iterable
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def load_file(path: str | Path, datetime_col: str=None) -> pd.DataFrame:
    """
    Load data from CSV or Parquet file into a pandas DataFrame.  
    :param path: File path as string or Path object.
    :param datetime_col: Name of column to parse as datetime with CSV files. 
                        Optional; pass None to skip.
    :return: Loaded DataFrame with data from the file.
    :raises FileNotFoundError: If file does not exist.
    :raises ValueError: If file format is unsupported (.csv or .parquet only).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, parse_dates=[datetime_col] if datetime_col else None)
    elif path.suffix.lower() in [".parquet", ".pqt"]:
        data = pd.read_parquet(path, engine='fastparquet')
    else:
        raise ValueError(f"Unsuported file format: {path.suffix}")

    return data
    

def save_file(df: pd.DataFrame, path: str | Path, overwrite: bool = False):
    """
    Save pandas DataFrame to CSV or Parquet file with automatic directory creation 
    and timestamped to avoid overwrites.
    :param df: DataFrame to save.
    :param path: File path as string or Path object (.csv, .parquet, or .pqt).
    :param overwrite: If True, overwrite existing file. If False (default), 
                      create timestamped version like `file_20260201_1947.csv`.
    :return: Final Path object where file was saved.
    :raises ValueError: If DataFrame is empty or file format unsupported.
    """
    path=Path(path)
    if df.empty:
        raise ValueError("Attemting to save empty DataFrame")
    if not path.parent.is_dir():
        logging.info("No directory for provided path. Creating one.")
        path.parent.mkdir(parents=True)
    if path.is_file():
        if overwrite:
            logging.info("Path file present, overwriting.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = path.with_stem(f"{path.stem}_{timestamp}")
            logging.info(f"Path file present, creating new one: {path.name}.")
            
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in [".parquet", ".pqt"]:
        df.to_parquet(path, index=False, engine='fastparquet')
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    logging.info(f"Saved {df.shape} shaped data as {path.name}")  # remove .name for non-public notebooks
    return path
    
     
def _check_feature_presence(target_list: Iterable[str], source_list: Iterable[str]) -> None:
    """
    Validate that all required features exist in source column list.
    :param target_list: Iterable of required feature names to check for.
    :param source_list: Iterable of available column names from input DataFrame.
    :return: None. Raises ValueError if any required features are missing.
    """
    if isinstance(target_list, str):
        target_list = [target_list]
    if isinstance(source_list, str):
        source_list = [source_list]
    missing_features = set(target_list) - set(source_list)
    if missing_features:
        raise ValueError(f"No {missing_features} features in input dataframe columns: {source_list}") 
        
        
def load_scaler(scaler_type: str | None = None,
                 scaler_path: Path | None = None,
                ) -> RobustScaler | None:
    """
    Load fitted RobustScaler from explicit path or config default.
    Resolve X/y scalers from config if no path provided.
    
    :param scaler_type: 'X' for features or 'y' for targets. Required if no `scaler_path`.
    :param scaler_path: Explicit path to scaler file. Overrides config lookup.
    
    :return: Fitted RobustScaler instance or None if path invalid/missing.
    """
    if scaler_path is None:
        directory = cnfg.data.dirs.get("model")
        if scaler_type == "X":
            scaler_file = cnfg.data.files.get("X_scaler")
        elif scaler_type == "y":
            scaler_file = cnfg.data.files.get("y_scaler")
        else:
            raise ValueError("if no scaler_path provided, scaler_type must be 'X' or 'y'.")

        scaler_path = directory / scaler_file if scaler_file else None

    scaler = joblib.load(filename=scaler_path) if scaler_path else None

    return scaler


def invert_scaled_X(X,
                    X_scaler_path: Path | None = None,
                    scaler_X: RobustScaler | None = None
                   ) -> pd.DataFrame:
    """
    Invert RobustScaler transformation on feature predictions.
    Auto-loads X scaler from config or disk if not provided.
    
    :param X: Scaled features of shape (n_samples, n_features) or (n_features,) for single row.
    :param X_scaler_path: Path to X scaler file. Auto-fallback to config.
    :param scaler_X: Live fitted RobustScaler instance (training use).
    :return: Unscaled features as DataFrame with original column names.
    
    Raises:
        ValueError: No valid scaler or scaler path provided.
    """
    scaler_X = scaler_X or load_scaler(scaler_type="X", scaler_path=X_scaler_path)
    if scaler_X is None:
        raise ValueError("No scaler or valid scaler path provided")
        
    if X.ndim == 1:
        X = X.reshape(1, -1)  # assumes one row provided
        
    inverted_X = pd.DataFrame(
        data = scaler_X.inverse_transform(X),
        columns = getattr(scaler_X, "feature_names_in_", None)
    )

    return inverted_X


def invert_scaled_y(y,
                    y_scaler_path: Path | None = None,
                    scaler_y: RobustScaler | None = None
                   ) -> pd.DataFrame:
    """
    Invert RobustScaler transformation on target predictions.
    
    Auto-load y scaler from config or disk. Handles both single predictions 
    and prediction arrays.
    
    :param y: Scaled targets of shape (n_samples,) or (n_samples, 1).
    :param y_scaler_path: Path to y scaler file. Auto-fallback to config.
    :param scaler_y: Live fitted RobustScaler instance (training use).
    :return: Unscaled dengue case counts as DataFrame.
    
    Raises:
        ValueError: No valid scaler or scaler path provided.
    """
    scaler_y = scaler_y or load_scaler(scaler_type="y", scaler_path=y_scaler_path)
    if scaler_y is None:
        raise ValueError("No scaler or valid scaler path provided")

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    inverted_y = pd.DataFrame(
        data = scaler_y.inverse_transform(y),
        columns = getattr(scaler_y, "feature_names_in_", None)
    )

    return inverted_y
    
        
