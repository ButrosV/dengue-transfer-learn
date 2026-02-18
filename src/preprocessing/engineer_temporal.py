import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from src.utils.utils import _check_feature_presence
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def circular_time_features(X: pd.DataFrame,
                           source_feature: str | None = None,
                           period: int | None = None,
                           drop_source_feature: bool = True):
    """
    Generate cyclical sin/cos features for periodic time variables. Transforms integer week/month/day
    features into continuous circular encodings that preserve temporal continuity across period boundaries.

    :param X: Input features DataFrame.
    :param source_feature: Periodic column name (e.g., 'weekofyear'). Uses config default if None.
    :param period: Fixed cycle length (e.g., 52 for weeks, 12 for months). Uses column max if None.
    :param drop_source_feature: If True, drops original source column after encoding.
    :return: X with `sin_{source_feature}` and `cos_{source_feature}` columns added.
    """
    source_feature = (source_feature 
                      or cnfg.preprocess.feature_groups["week"])
    
    _check_feature_presence(source_feature, X.columns)
    
    X_circular = X.copy()
    divisor = period or X_circular[source_feature].max()
    
    X_circular[f"sin_{source_feature}"] = np.sin(
        2 * np.pi * X_circular[source_feature] / divisor)
    X_circular[f"cos_{source_feature}"] = np.cos(
        2 * np.pi * X_circular[source_feature] / divisor)

    if drop_source_feature:
        X_circular.drop(columns=[source_feature], inplace=True)

    return X_circular
    
    
def _rolling_aggregations(groups, task, step):
    """Handle rolling aggregation (mean, sum, etc.) for the given step."""
    return groups.transform(lambda group: group.rolling(
        window=step + 1, min_periods=1, closed="left").agg(task))

def _lags(groups, step):
    """Handle lag features for the given step."""
    return groups.transform(lambda group: group.shift(step))
    
    
def dynamic_temporal_features(X: pd.DataFrame,
                             y: pd.Series | None = None,
                             target_feature: str | None=None,
                              group_feature: str | None=None,
                             source_feature: str | None=None,
                              new_feature: str | None=None,
                             task: str | None=None,
                             backward_steps: int | None=None):

    """
    Generate dynamic temporal features (e.g., rolling aggregates (mean, sum, etc.),
    lag features).
    Remove NaNs rows created in the proces. 
    WARNING: If input X contains NaN rows, those will be removed.
    
    :param X: Input pandas DataFrame containing features.
    :param y: Optional pandas Series of target values.
    :param target_feature: The name of the target feature in `y` to include for lag creation. 
                            If None, uses default from configuration.
    :param group_feature: The name of the grouping feature (e.g., city, region) for temporal 
                            aggregation. Defaults to 'city' from config.
    :param source_feature: The name of the feature to transform (e.g., 'total_cases',
                            'precipitation_amt_mm'). If None, uses default from configuration.
    :param new_feature: The name of the new feature column to be created. 
                        If None, uses default from configuration.
    :param task: The type of transformation to apply. Options are 'lag', 'sum', or 'mean'. 
                    If None, uses default from configuration.
    :param backward_steps: List or a single integer of steps to look back for lag or rolling 
                            aggregation. Example: [1, 4] means lags/rolling aggregation for 1 
                            and 4 steps. If None, uses default from configuration. 
    :return: DataFrame with dynamic temporal features added. If `y` is provided, returns 
                2 dataframes X with new fetures and y both filtered to remove 
                any rows with NaNs.
    """
    
    target_feature = target_feature or cnfg.preprocess.feature_groups.get("target")
    group_feature = group_feature or cnfg.preprocess.feature_groups.get("ciy")

    X_temporal = X.copy()
    
    feature_params = dict()
    if all((param is None for param in (
        source_feature, new_feature, task, backward_steps))):
        feature_params = cnfg.preprocess.dynamic_temporal_features
    else:
        feature_params = {source_feature:
                          {"source_feature": source_feature,
                           "task": task,
                           "backward_steps": backward_steps}}

    if group_feature is None:
        group_feature = pd.Series(data=np.ones(len(X_temporal), dtype=bool),
                                  index=X_temporal.index)
        
    switch=False
    
    if (any(target_feature == v["source_feature"] for k,v in feature_params.items()) 
        and y is not None):
        X_temporal[target_feature] = y[target_feature]
        switch = True
     
    for param, rules in feature_params.items():
        if not isinstance(rules["backward_steps"], list):
            rules["backward_steps"] = [rules["backward_steps"]]

        if rules["task"] not in ["lag", "sum", "mean", "max", "min"]:
            logging.warning(f"Wrong task type provided: {rules['task']}.")
            
        _check_feature_presence(rules["source_feature"], X_temporal.columns)
        groups = X_temporal.groupby(by=group_feature)[rules["source_feature"]]     
        for step in rules["backward_steps"]:
            if rules["task"] != "lag":
                X_temporal[
                    f'{param}_{rules["task"]}_{step}'
                    ] = _rolling_aggregations(groups, rules["task"], step)
            else:
                X_temporal[
                    f'{param}_rolling_{rules["task"]}_{step}'
                    ] = _lags(groups, step)
    
    if switch:
        X_temporal.drop(columns=[target_feature], inplace=True)
        
    mask = X_temporal.notna().all(axis=1)
    X_temporal = X_temporal[mask]

    if y is not None:
        return X_temporal, y.copy()[mask]
        
    return X_temporal
    
