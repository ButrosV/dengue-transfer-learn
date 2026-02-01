from pathlib import Path
import pandas as pd
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO)

from typing import Iterable, Union
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def load_file(path: Union[str, Path], datetime_col: str=None) -> pd.DataFrame:
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
    

def save_file(df: pd.DataFrame, path: Union[str, Path], overwrite=False):
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
    logging.info(f"Saved {df.shape} shaped data to {path}")
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
        
