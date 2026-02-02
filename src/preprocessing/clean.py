from pathlib import Path
import pandas as pd
from typing import List, Dict, Any

from src.utils.utils import _check_feature_presence, load_file, save_file

from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def cap_outliers(data: pd.DataFrame, features: List[str]=None,
                 group_keys: List[str]=None,
                 lower_cap:float=None, upper_cap:float=None,
                output_stats:bool=True) -> Dict[str, Any]:
    """
    Perform groupwise Winsorization (percentile clipping) on specified features to handle outliers.
    Automatically filter environmental features from config prefixes if not specified.
    :param data: Input pandas DataFrame.
    :param features: List of column names to clip. Default None auto-selects env features 
           from prefixes defined in config.yaml.
    :param group_keys: List of columns to group by for quantile calculation. Default None uses 
           config.yaml 'city' grouping.
    :param lower_cap: Lower percentile for clipping (0-1). Default None uses config.yaml 
           'outlier_perc.lower' (originally 0.01).
    :param upper_cap: Upper percentile for clipping (0-1). Default None uses config.yaml 
           'outlier_perc.upper' (originally 0.99).
    :param output_stats: If True, returns % rows changed per feature. Default True.
    :return: Dict containing:
           - 'data': Clipped DataFrame copy (original unchanged)
           - 'capped_row_prc': Series of % rows clipped per feature (if output_stats=True)
    """
    if features is None:
        features = [f for f in data.columns if f.startswith(
            tuple(cnfg.preprocess.feature_groups["env_prefixes"]))]
    if group_keys is None:
        group_keys = cnfg.preprocess.feature_groups["city"]
    if lower_cap is None:
        lower_cap = cnfg.preprocess.outlier_perc["lower"]
    if upper_cap is None:
        upper_cap = cnfg.preprocess.outlier_perc["upper"]
        
    _check_feature_presence(target_list=group_keys, source_list=data.columns)

    data_no_outliers = data.copy()
    data_no_outliers[features] = data_no_outliers.groupby(by=group_keys)[features].transform(
        lambda group: group.clip(
            lower=group.quantile(lower_cap), upper=group.quantile(upper_cap)))

    if output_stats:
        capped_row_percent = round(
            ((data[features] != data_no_outliers[features]).sum() / len(data) * 100), 2)
        return {"data": data_no_outliers,
                "capped_row_prc": capped_row_percent}
    return {"data": data_no_outliers}
    

def drop_nan_rows(X: pd.DataFrame, y: pd.Series | None=None,
                  threshold_percent: float=None) -> pd.DataFrame:
    """
    Drop rows with NaN values exceeding threshold_percent of columns.
    
    :param X: pandas DataFrame of features.
    :param y: Optional target array/series. Default None.
    :param threshold_percent: Optional min non-null fraction required [0,1].
          Default None pulls 'nan_threshold' from config.yaml.
    :return: Filtered X (and y if provided), both with reset_index().
    """
    if threshold_percent is None:
        threshold_percent = cnfg.preprocess.nan_threshold
    row_drop_threshold = int(len(X.columns) * threshold_percent)
    X_no_nan = X.dropna(thresh=row_drop_threshold)
    if y is not None:
        return X_no_nan.reset_index(drop=True), y.iloc[X_no_nan.index].reset_index(drop=True)
    return X_no_nan.reset_index(drop=True)
    
    
def median_groupwise_impute(X: pd.DataFrame,
                            group_keys: List[str]=None) -> pd.DataFrame:
    """
    Impute NaN values in numeric columns using median within specified group keys.
    
    :param X: pandas DataFrame containing grouping columns and features to impute.
    :param group_keys: Optional list of column names for grouping.
                 Default None pills 'city' and 'week' feature names from config.yaml.
    :return: Copy of input DataFrame with NaNs filled by group-wise medians,
    		DataFrame with NaN mask for those values thate were imputed.
    """
    if group_keys is None:
        group_keys = [cnfg.preprocess.feature_groups["city"],
                      cnfg.preprocess.feature_groups["week"]]
                      
    _check_feature_presence(target_list=group_keys, source_list=X.columns)

    X_no_nan = X.copy()
    cols_with_nan = X_no_nan.select_dtypes(include="number")\
        .columns[X_no_nan.select_dtypes(include="number").isna().sum() > 0].to_list()
        
    nan_mask = X_no_nan[cols_with_nan].isna()

    if len(cols_with_nan) > 0:
        X_no_nan[cols_with_nan] = X_no_nan[cols_with_nan + group_keys]\
            .groupby(by=group_keys)[cols_with_nan]\
            .transform(lambda group: group.fillna(group.median()))
    return X_no_nan, nan_mask
    

def pipe_clean(manual_dirs: Dict[str, Path] | None=None,
               manual_files: Dict[str, Path] | None=None,
               datetime_col: str=None,
               overwrite_files: bool=False) -> Dict[str, Any]:
    """
    Data cleaning pipeline for features and targets.
    Handle outlier capping, NaN row dropping, and groupwise median imputation in sequence.
    :param manual_dirs: Dict of directory paths to override config.yaml dirs. 
        Keys: 'raw', 'intermediate'. Default None uses config.yaml data.dirs.
    :param manual_files: Dict of filenames to override config.yaml files. 
        Keys: 'features_train', 'labels_train', 'features_clean', 'labels_clean', 'nan_mask'. 
        Default None uses config.yaml data.files.
    :param datetime_col: Name of datetime column for parsing. Default None uses
                            config.yaml preprocess.feature_groups["datetime"].
    :param overwrite_files: If True, overwrites existing saved files. Default False.
    
    :return: Dict containing:
        - 'X_clean_save_path': Path where cleaned features saved
        - 'X_clean_data': Cleaned features DataFrame
        - 'X_capped_rows_prc': Float % of rows affected by outlier capping
        - 'y_clean_save_path': Path where cleaned targets saved  
        - 'y_clean_data': Cleaned targets DataFrame
        - 'nan_mask_save_path': Path where NaN mask saved
        - 'nan_mask_data': DataFrame tracking imputed locations (for feature engineering)
    """
    dirs = cnfg.data.dirs
    filenames = cnfg.data.files
    if manual_dirs is not None:
        dirs = manual_dirs
    if manual_files is not None:
        filenames = manual_files

    X_clean = load_file(path=dirs["raw"] / filenames["features_train"],
                        datetime_col=datetime_col)
    y_clean = load_file(path=dirs["raw"] / filenames["labels_train"],
                        datetime_col=datetime_col)
    
    caping_output = cap_outliers(data=X_clean)
    X_clean = caping_output["data"]
    X_clean, y_clean = drop_nan_rows(X=X_clean, y=y_clean)
    X_clean, df_nan_mask = median_groupwise_impute(X=X_clean)

    X_save_path = save_file(df=X_clean, path=dirs["intermediate"] / filenames["features_clean"],
                            overwrite=overwrite_files)
    y_save_path = save_file(df=y_clean, path=dirs["intermediate"] / filenames["labels_clean"],
                            overwrite=overwrite_files)
    nan_mask_save_path = save_file(df=df_nan_mask, path=dirs["intermediate"] / filenames["nan_mask"],
                                   overwrite=overwrite_files)
    
    return {"X_clean_save_path": X_save_path,
            "X_clean_data": X_clean,
            "X_capped_rows_prc": caping_output["capped_row_prc"],
            "y_clean_save_path": y_save_path,
            "y_clean_data": y_clean,
            "nan_mask_save_path": nan_mask_save_path,
            "nan_mask_data": df_nan_mask,            
           }
    
