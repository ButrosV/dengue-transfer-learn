from typing import Iterable
from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


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
        
