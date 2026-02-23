import pandas as pd
from typing import List
from src.config import ProjectConfig  # project config file parser
cnfg = ProjectConfig.load_configuration()

def encode_categorical(X: pd.DataFrame, cetegoricals: List[str] | None=None):
    """
    One-hot encode specified categorical columns with drop_first=True.
    :param X: Input DataFrame containing categorical columns to encode.
    :param cetegoricals: List or string of categorical column name(s) to encode. 
                           Defaults to None, useing config city feature group.
    :return: Dict with 'data' (encoded DataFrame copy)
             and 'features' dict containing 'new_names' (dummy column names)
             and 'source_categories' (mapping original column to its 
             unique categories before encoding).
    """
    X_dummies = X.copy()
    cetegoricals = cetegoricals or cnfg.preprocess.feature_groups["city"]
    if isinstance (cetegoricals, str):
        cetegoricals = [cetegoricals]
    source_categories = {feat: list(X_dummies[feat].unique())
                         for feat in cetegoricals}
    X_dummies = pd.get_dummies(X_dummies, columns=cetegoricals,
                               drop_first=True, dtype=int, prefix_sep="_")
    new_features = [f for f in X_dummies if f.startswith(tuple(cetegoricals))]
    
    return {"data": X_dummies,
            "features": {"new_names": new_features,
                         "source_categories": source_categories
                        }}
