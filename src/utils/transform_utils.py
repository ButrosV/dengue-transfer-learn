from pathlib import Path
import pandas as pd

from sklearn.preprocessing import RobustScaler

from src.config import ProjectConfig


def invert_scaled_X(X,
                    X_scaler_path: Path | None = None,
                    scaler_X: RobustScaler | None = None
                   ) -> pd.DataFrame:
    """
    Invert RobustScaler transformation on feature predictions.
    Auto-loads X scaler from config or disk if not provided.
    
    :param X: Scaled features of shape (n_samples, n_features) or (n_features,) for single row.
    :param X_scaler_path: Path to X scaler file. Auto-fallback to config.
    :param scaler_X: Live fitted RobustScaler instance (training use).
    :return: Unscaled features as DataFrame with original column names.
    
    Raises:
        ValueError: No valid scaler or scaler path provided.
    """
    scaler_X = scaler_X or load_scaler(scaler_type="X", scaler_path=X_scaler_path)
    if scaler_X is None:
        raise ValueError("No scaler or valid scaler path provided")
        
    if X.ndim == 1:
        X = X.reshape(1, -1)  # assumes one row provided
        
    inverted_X = pd.DataFrame(
        data = scaler_X.inverse_transform(X),
        columns = getattr(scaler_X, "feature_names_in_", None)
    )

    return inverted_X


def invert_scaled_y(y,
                    y_scaler_path: Path | None = None,
                    scaler_y: RobustScaler | None = None
                   ) -> pd.DataFrame:
    """
    Invert RobustScaler transformation on target predictions.
    
    Auto-load y scaler from config or disk. Handles both single predictions 
    and prediction arrays.
    
    :param y: Scaled targets of shape (n_samples,) or (n_samples, 1).
    :param y_scaler_path: Path to y scaler file. Auto-fallback to config.
    :param scaler_y: Live fitted RobustScaler instance (training use).
    :return: Unscaled dengue case counts as DataFrame.
    
    Raises:
        ValueError: No valid scaler or scaler path provided.
    """
    scaler_y = scaler_y or load_scaler(scaler_type="y", scaler_path=y_scaler_path)
    if scaler_y is None:
        raise ValueError("No scaler or valid scaler path provided")

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    inverted_y = pd.DataFrame(
        data = scaler_y.inverse_transform(y),
        columns = getattr(scaler_y, "feature_names_in_", None)
    )

    return inverted_y
    
