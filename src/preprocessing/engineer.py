import pandas as pd
import itertools
import logging
logging.basicConfig(level=logging.INFO)

from typing import List, Iterable, Optional

from src.utils.utils import _check_feature_presence
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def add_missingness_features(X: pd.DataFrame,
                             nan_mask: pd.DataFrame,
                             aggregated_feat_name: Optional[str]=None,
                             input_env_feat_prefixes: Optional[List[str]]=None,
                             input_group_prefix: Optional[list]=None,
                             output_feature_prefix: Optional[str]=None
                            ) -> pd.DataFrame:
    """
    Add missingness indicator features to DataFrame using column prefix patterns.
    1. (optional) Aggregated ratio of missing values across environment prefix columns.
    2. Max missingness indicator (0/1) per feature group defined by prefixes
    Fall back to config values if parameters unspecified.
    :param X: Input DataFrame to add missingness features to.
    :param nan_mask: Boolean DataFrame same shape as X where True indicates missing.
    :param aggregated_feat_name: Name for aggregated missingness ratio feature. 
                                Uses config default if None.
    :param input_env_feat_prefixes: List of prefixes to match environment columns 
                                   for aggregated ratio. Uses config if None.
    :param input_group_prefix: List of prefix lists or single strings defining 
                              feature groups for max missingness indicators. 
                              Mixed format supported: `[["station", "precip"], "ndvi_s"]`.
                              Uses config if None.
    :param output_feature_prefix: Prefix for new group missingness columns 
                                 (e.g., "missing_station_max"). Uses config if None.
    :return: X with additional missingness feature column(s).
    :raises ValueError: If no columns match environment prefixes.
    """
    X_missing = X.copy()
    config = cnfg.preprocess.missingness_features
    aggregated_feat_name = aggregated_feat_name or config.get("aggregated_feat_n")
    input_env_feat_prefixes = input_env_feat_prefixes or cnfg.preprocess.feature_groups["env_prefixes"]
    input_group_prefix = input_group_prefix or config["group_prefixes"]
    output_feature_prefix = output_feature_prefix or config["new_feature_prefix"]

    if aggregated_feat_name and input_env_feat_prefixes:
        env_cols = nan_mask.columns[nan_mask.columns.str.startswith(tuple(input_env_feat_prefixes))]
        denominator = len(env_cols)
        if denominator == 0:
            raise ValueError("No columns match environment prefix/es '{input_env_feat_prefixes}'.")
        X_missing[f"{output_feature_prefix}{aggregated_feat_name}"
            ] = nan_mask[env_cols].sum(axis=1) / denominator

    for prefix in input_group_prefix:
        if isinstance(prefix, str):
            prefix = [prefix]
        features = nan_mask.columns[
            nan_mask.columns.str.startswith(tuple(prefix))]
        if len(features) == 0:
            logging.warning(f"No columns match prefix/es '{prefix}' - skipping.")
            continue
        feature_name = f"{output_feature_prefix}{prefix[0]}"
        X_missing[feature_name] = nan_mask[features].agg("max", axis=1)
    
    return X_missing
    
    
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
    
