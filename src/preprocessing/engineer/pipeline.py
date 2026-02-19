from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import ProjectConfig
from src.utils.utils import load_file, save_file
from src.preprocessing.engineer.base import (reduce_features,
                                            add_missingness_features,
                                            low_value_targets)

from src.preprocessing.engineer.temporal import (circular_time_features,
                                                dynamic_temporal_features)
                                                
cnfg = ProjectConfig.load_configuration()


def pipe_engineer(X: pd.DataFrame | None=None,
                  y: pd.DataFrame | None=None,
                  nan_mask: pd.DataFrame | None=None,
                  manual_dirs: Dict[str, Path] | None=None,
                  manual_files: Dict[str, Path] | None=None,
                  datetime_col: str=None,
                  overwrite_files: bool=False) -> Dict[str, Any]:
    """
    Feature engineering pipeline for dengue case prediction.
    Combines missingness indicators, feature reduction, outlier handling, 
    cyclical encoding, and dynamic temporal lags/rolling statistics.
    
    :param X: Input features DataFrame. Default None uses config.yaml data.files.
    :param y: Input targets DataFrame. Default None uses config.yaml data.files.
    :param nan_mask: NaN mask DataFrame tracking imputation locations. 
        Default None uses config.yaml data.dirs.
    :param manual_dirs: Dict of directory paths to override config.yaml dirs. 
        Default None uses config.yaml data.dirs.
    :param manual_files: Dict of filenames to override config.yaml files. 
       Default None uses config.yaml data.files.
    :param datetime_col: Name of datetime column for parsing. Default None uses
        config.yaml preprocess.feature_groups["datetime"].
    :param overwrite_files: If True, overwrites existing engineered feature files.
        Default False.
    
    :return: Dict containing:
        - 'X_eng_save_path': Path where engineered features saved
        - 'X_eng_data': Engineered features DataFrame
                        (missingness, temporal lags/rolls, cyclical)
        - 'y_eng_save_path': Path where engineered targets saved  
        - 'y_eng_data': Engineered targets DataFrame (Remowed first rows due to
                        NaNs created by lag/rolling stat feature engineering)
    """
    dirs = manual_dirs or cnfg.data.dirs
    filenames = manual_files or cnfg.data.files
    datetime_col = datetime_col or cnfg.preprocess.feature_groups["datetime"]

    file_loaders = {
        'X': lambda: load_file(path=dirs["intermediate"] / 
                               filenames["features_clean"],
                               datetime_col=datetime_col),
        'y': lambda: load_file(path=dirs["intermediate"] /
                               filenames["labels_clean"],
                               datetime_col=datetime_col),
        "nan_mask": lambda: load_file(path=dirs["intermediate"] /
                                      filenames["nan_mask"],
                                      datetime_col=datetime_col)
    }

    X_eng = X.copy() if X is not None else file_loaders['X']()
    y_eng = y.copy() if y is not None else file_loaders['y']()
    nan_mask = (nan_mask.copy() if nan_mask is not None 
                else file_loaders["nan_mask"]())

    X_eng = add_missingness_features(X=X_eng, nan_mask=nan_mask)
    X_eng = reduce_features(X=X_eng)
    X_eng = low_value_targets(X=X_eng, y=y_eng)
    X_eng = circular_time_features(X=X_eng)
    X_eng, y_eng = dynamic_temporal_features(X=X_eng, y=y_eng)
      
    X_save_path = save_file(df=X_eng, path=dirs["intermediate"] / 
                            filenames["features_eng"],
                            overwrite=overwrite_files)
    y_save_path = save_file(df=y_eng, path=dirs["intermediate"] / 
                            filenames["labels_eng"],
                            overwrite=overwrite_files)
    
    return {"X_eng_save_path": X_save_path,
            "X_eng_data": X_eng,
            "y_eng_save_path": y_save_path,
            "y_eng_data": y_eng,         
           }
