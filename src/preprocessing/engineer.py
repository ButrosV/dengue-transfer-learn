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
        X_missing[feature_name] = nan_mask[features].agg("max", axis=1).astype(int)
    
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
    
    
def _get_cumulative_streaks(group: pd.Series,
                            low_value_range:int,
                           filter_initial_streaks: bool=False):
    """
    Compute low-value streak lengths within a 1D series, optionally filtering to initial streak only.

    :param group: pandas Series representing a single grouped sequence (e.g., one city).
    :param low_value_range: Upper bound (exclusive) for values considered "low".
    :param filter_initial_streaks: If True, keep only initial low-value streak when it starts at first
                                   position (all others 0). If False, return complete streak lengths.
                                   Default is False.
    :return: pandas Series (same index/shape as `group`) with cumulative low-value streak lengths,
             filtered according to `filter_initial_streaks`.
    """
    
    mask_lows = group.isin(range(low_value_range))
    streak_groups = (mask_lows != mask_lows.shift(fill_value=False)).cumsum()
    low_count_streaks = mask_lows.groupby(by=streak_groups).cumsum()

    if not filter_initial_streaks:
        return low_count_streaks

    else:
        if low_count_streaks.iloc[0] == 1:
            initial_low_streaks = low_count_streaks.where(streak_groups == 1, 0)
        else:
            initial_low_streaks = pd.Series(0, index=low_count_streaks.index)
        return initial_low_streaks


def low_value_targets(X: pd.DataFrame, y: pd.DataFrame,
                      target_feature: str | None=None,
                      group_feature:str | None=None,
                      new_feat_name:str | None=None,
                      initial_streaks_only:bool | None=None,
                      min_initial_streak_len:int | None=None,
                      low_value_range:int | None=None
                     ) -> pd.DataFrame:
    """
    Generate low-value streak features for ML pipelines. Creates continuous streak length feature
    (`low_case_streak`) and optional boolean initial-streak indicator with minimum length filtering.
    
    Continuous streak preserves magnitude info for model gradients. Boolean initial streak supports
    minimum length thresholding.

    :param X: Input features DataFrame.
    :param y: Target DataFrame (same index as X).
    :param target_feature: Target column name. Uses config default if None.
    :param group_feature: Grouping column (e.g., 'city'). None processes entire series.
                            Defaults to config settings.
    :param new_feat_name: Output base name (e.g., 'low_case_streak'). None returns X unchanged.
                            Defaults to config settings.
    :param initial_streaks_only: If True, adds thresholded boolean `initial_{new_feat_name}`.
                            Defaults to config settings.
    :param min_initial_streak_len: Min length threshold. Applies **only** to boolean `initial_*` feature.
                            Defaults to config settings.
    :param low_value_range: Values < this are "low" (exclusive upper bound).
                            Defaults to config settings.
    :return: X with `low_case_streak` (continuous) ± `initial_low_case_streak` (boolean).
    """
    
    assert all(X.index == y.index), "Indices for 'X' and 'y' must be aligned." 
    
    config_values = cnfg.preprocess
    
    target_feature = target_feature or config_values.feature_groups["target"]
    group_feature = group_feature or config_values.feature_groups["city"]
    min_initial_streak_len = min_initial_streak_len or (config_values.
        low_value_streak_features["target_streak_len_threshold"])
    new_feat_name = new_feat_name or (config_values.
        low_value_streak_features.get("target_streak_feat_n"))
    low_value_range = low_value_range or (config_values.
        low_value_streak_features["low_value_range"])

    if initial_streaks_only is None:
        initial_streaks_only = config_values.low_value_streak_features.get("initial_streaks"
                                                                          ) or False
    if new_feat_name is None:
        return X
        
    X_low_streaks = X.copy()

    if group_feature is not None:
        low_count_streaks = y.groupby(by=group_feature)[target_feature].transform(
            lambda x: _get_cumulative_streaks(x, low_value_range))
    else:
        low_count_streaks = _get_cumulative_streaks(y[target_feature], low_value_range)
        
    temp_output_dict = {new_feat_name: low_count_streaks}

    if initial_streaks_only:
        if group_feature is not None:
            initial_streaks = y.groupby(by=group_feature)[target_feature].transform(
                lambda x: _get_cumulative_streaks(x, low_value_range, initial_streaks_only))
        else:
            initial_streaks = _get_cumulative_streaks(y[target_feature], low_value_range,
                                                      initial_streaks_only)
            
        if min_initial_streak_len:
            long_streak_mask = initial_streaks >= min_initial_streak_len
            long_strek_threshpoints = long_streak_mask[
                initial_streaks == min_initial_streak_len].index
            
            for point in long_strek_threshpoints:
                start = max(0, point - min_initial_streak_len + 1)
                long_streak_mask[start:point] = True
            initial_streaks = initial_streaks.where(long_streak_mask, 0)
            
        temp_output_dict[f"initial_{new_feat_name}"] = initial_streaks
            
            
    for feat_name, streak in temp_output_dict.items():
        if feat_name.startswith("initial_"):
            streak = streak.astype(bool).astype(int)
        X_low_streaks[feat_name] = streak
        
    return X_low_streaks
    
