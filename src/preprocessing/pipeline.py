from pathlib import Path
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from typing import Dict, Any

from src.config import ProjectConfig

from src.preprocessing.clean import pipe_clean
from src.preprocessing.engineer.pipeline import pipe_engineer
from src.preprocessing.select import pipe_select
from src.preprocessing.preprocess import time_aware_group_split, robust_scale_data
from src.utils.utils import save_file
                                
cnfg = ProjectConfig.load_configuration()
DIRS = cnfg.data.dirs
FILES = cnfg.data.files





def full_preprocess_pipe(
    manual_dirs: Dict[str, Path] | None=None,
    manual_files: Dict[str, Path] | None=None,
    grouping_feature: str | None = None,
    target: str | None = None,
    logtransform_target: bool | None = None,
    overwrite_files: bool=False) -> Dict[str, Any]:
    """
    Complete dengue preprocessing pipeline: clean → engineer → select → split → scale → save.
    
    Orchestrates dataset preparation with intermediate file persistence
    and time-aware group splits for city-wise transfer learning.
    
    :param manual_dirs: Override config data directories {'raw', 'intermediate', 'processed', 'model'}.
    :param manual_files: Override config filenames for all pipeline stages.
    :param grouping_feature: City grouping column ('city'). Defaults to config.
    :param target: Target column name. Defaults to config 'total_cases'.
    :param logtransform_target: Apply log1p() to targets before scaling. Defaults to config.
    :param overwrite_files: Overwrite existing artifacts. Default False (timestamped backups).
    
    :return: Complete processed dataset with paths:
        - `scaled_data`: {'X_train', 'X_valid', 'y_train', 'y_valid'} NumPy arrays (scaled)
        - `data_paths`: Save paths for all scaled arrays (.npy/.parquet)
        - `scalers`: Fitted RobustScaler objects for X/y inversion
        - `scaler_paths`: Paths where scalers saved (.joblib)
        - `grouping_feature`: Preserved city Series aligned with train/valid splits
    
    :raises AssertionError: Index misalignment, empty splits, NaN values detected.

    Note:
        - WFCV feature selection reduces to top ~25 stable features (config-driven)
        - Log1p + RobustScaler handles skewed input data
    """
    cfg_pre = cnfg.preprocess
    
    dirs = manual_dirs or DIRS
    filenames = manual_files or FILES
    target = target or cfg_pre.feature_groups["target"]
    city = grouping_feature or cfg_pre.feature_groups["city"]
    logtransform_target = (logtransform_target if logtransform_target is not None 
                           else cfg_pre.target.get("logtransform"))
    
    cleaned_data = pipe_clean(overwrite_files=overwrite_files)
    
    enginered_data = pipe_engineer(X=cleaned_data["X_clean_data"],
                                   y=cleaned_data["y_clean_data"],
                                   overwrite_files=overwrite_files)
    groups = enginered_data["X_eng_data"][[city]]
    
    selected_data = pipe_select(X=enginered_data["X_eng_data"],
                                   y=enginered_data["y_eng_data"],
                                   overwrite_files=overwrite_files)
     # make sure selection did not introduce index issues for downstream grouping (ulikely, thou)
    groups = groups.loc[selected_data['X_select_data'].index]
    
    logging.info(
        f"Selected {selected_data['X_select_data'].shape[1]} features "
        f"from original + engineered feature set of "
        f"{enginered_data['X_eng_data'].shape[1]} features.")
    
    y_selected = selected_data["y_select_data"].copy()
    assert all(selected_data["X_select_data"].index == y_selected.index)
    
    if logtransform_target:
        y_selected[target] = np.log1p(y_selected[target])
        
    X_train, X_valid, y_train, y_valid = time_aware_group_split(
        X=selected_data["X_select_data"],
        y=y_selected,
        group_aware_frame=groups)

    assert len(X_train) > 0 and len(X_valid) > 0
    assert not X_train.isna().any().any()
    assert not y_train.isna().any().any()
    
    scaler_output = robust_scale_data(X_train=X_train, y_train=y_train,
                                      X_valid=X_valid, y_valid=y_valid,
                                      overwrite_files=overwrite_files)
    data_paths = dict()
    for name, data in scaler_output["scaled_data"].items():
        if name in filenames:
            save_path = dirs["processed"] / filenames[name]
            save_path = save_file(data=data, path=save_path,
                        overwrite=overwrite_files)
            data_paths[name + "_save_path"] = save_path
    
    return {"scaled_data": scaler_output["scaled_data"], "data_paths": data_paths,
            "scalers": scaler_output["scalers"], "scaler_paths": scaler_output["paths"],
           "grouping_feature": groups}
           
