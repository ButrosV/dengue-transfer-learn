import numpy as np

from src.config import ProjectConfig  # project config file parser
cnfg = ProjectConfig.load_configuration()


def make_windows(X: np.ndarray,
                 y: np.ndarray,
                 window_size: int | None = None,
                stride: int | None = None,
                horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Create zero-copy sliding windows for time series forecasting using stride tricks.
    
    Past `window_size` timesteps predict the next week's `horizon` timesteps ahead.
    
    :param X: Input features array of shape `(n_timesteps, n_features)`.
    :param y: Target array of shape `(n_timesteps,)` or `(n_timesteps, n_targets)`.
    :param window_size: Number of past timesteps for input window. Falls back to 
                        config file settings (default: 52).
    :param stride: Step size between consecutive windows. Falls back to 
                   config file settings (default: 1).
    :param horizon: Number of future timesteps to predict. Falls back to 
                    config file settings  (default: 1).
    
    :return: Tuple of zero-copy window arrays:
             - ``X_windows``: Shape `(n_windows, window_size, n_features)`
             - ``y_windows``: Shape `(n_windows, horizon)`
    """
    settings = cnfg.preprocess.windowing
    window_size = window_size or settings["input_weeks"]
    stride = stride or settings["stride"]
    horizon = horizon or settings["output_weeks"]
    
    strider = np.lib.stride_tricks.sliding_window_view
    X_windows = strider(X, window_shape=window_size, axis=0)[::stride]
    
    y_start = window_size 
    y_windows = strider(y[y_start:], window_shape=horizon, axis=0)[::stride]

    cuttof = min(X_windows.shape[0], y_windows.shape[0])

    return X_windows[:cuttof], y_windows[:cuttof]
