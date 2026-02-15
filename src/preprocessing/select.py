import pandas as pd
from typing import Iterable

from src.utils.utils import _check_feature_presence
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()

 
def remove_features(X: pd.DataFrame, feats_to_drop: Iterable[str]=None)->pd.DataFrame:
    """
    Remove specified feature columns from DataFrame using config defaults or explicit list.
    :param X: Input DataFrame.
    :param feats_to_drop: Features to drop. If None, uses config multicollinearity removal list.
                          Default None.
    :return: New DataFrame copy with specified features removed. Raises ValueError if features missing.
    """
    if feats_to_drop is None:
        feats_to_drop = cnfg.preprocess.multicolinear["removal_list"]
    X_slim = X.copy()
    if isinstance(feats_to_drop, str):
        feats_to_drop = [feats_to_drop]
    _check_feature_presence(target_list=feats_to_drop, source_list=X_slim.columns)
    return X_slim.drop(columns=feats_to_drop)
    
