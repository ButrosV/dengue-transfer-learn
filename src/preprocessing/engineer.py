import pandas as pd
import itertools
from typing import List, Iterable

from src.utils.utils import _check_feature_presence
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()
    
def reduce_features(X: pd.DataFrame, 
                    input_feat_groups: List[List[str]]=None,
                    output_feat_names: List[str]=None,
                   function: str=None):
    """
    Aggregate multiple feature groups into single reduced features using specified function.
    Combine input features and drop originals.
    
    :param X: Input pandas DataFrame.
    :param input_feat_groups: List of feature group lists to aggregate. Default None uses 
           config.yaml settings (e.g., [['ndvi_ne', 'ndvi_nw']]).
    :param output_feat_names: Output column names for aggregated features. Default None uses 
           config.yaml settings (e.g., ['ndvi_north']).
    :param function: Aggregation function string ('mean', 'sum', 'median'). Default None uses 
           config.yaml settings (e.g 'mean').
    :return: DataFrame with reduced features. Original input columns dropped.
    """
    X_reduced = X.copy()
    if input_feat_groups is None:
        input_feat_groups = cnfg.preprocess.combine_features["input_groups"]
    if output_feat_names is None:
        output_feat_names = cnfg.preprocess.combine_features["output_names"]
    if function is None:
        function = cnfg.preprocess.combine_features["aggregation"]
      
    if not len(input_feat_groups) == len(output_feat_names):
        raise ValueError(f"Input feature groups {input_feat_groups} mismatch target keys {output_feat_names}")
    
    _check_feature_presence(target_list=itertools.chain(*input_feat_groups), source_list=X.columns)
    
    for name, group in zip(output_feat_names, input_feat_groups):
        X_reduced[name] = X_reduced[group].agg(function, axis=1)
        X_reduced.drop(columns=group, inplace=True)
        
    return X_reduced
    
    
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
    
