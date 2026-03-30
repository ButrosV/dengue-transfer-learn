from dataclasses import dataclass
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class PreprocessOutput:
    """
    Container for outputs of full preprocessing pipeline.

    Attributes:
        scaled_data: {'X_train', 'X_valid', 'y_train', 'y_valid'} arrays
        data_paths: Paths where scaled datasets are saved
        scalers: Fitted scalers for features/target
        scaler_paths: Paths where scalers are persisted
        grouping_masks: (train_groups, valid_groups) DataFrames aligned with splits
    """
    scaled_data: Dict[str, np.ndarray]
    data_paths: Dict[str, Path]
    scalers: Dict[str, Any]
    scaler_paths: Dict[str, Path]
    grouping_masks: Tuple[pd.DataFrame, pd.DataFrame]
    
