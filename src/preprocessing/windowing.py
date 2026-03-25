import numpy as np
import tensorflow as tf

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
             
    Note:
        The number of X and y windows may differ due to horizon offset. Both are
        truncated to the same length (`cutoff`) to ensure proper alignment.
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
    
    
def make_tf_windows(X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int | None = None,
                 shuffle: bool = True
                ) -> tf.data.Dataset:
    """
    Create batched tf.data.Dataset from pre-windowed time series arrays for LSTM training.
    Convert NumPy window arrays (ie from `make_windows()`) to optimized TensorFlow Dataset.
    
    :param X: Pre-windowed input features array of shape `(n_windows, window_size, n_features)`.
    :param y: Pre-windowed target array of shape `(n_windows, horizon)`.
    :param batch_size: Number of windows per batch. Falls back to config (default: 32).
    :param shuffle: Whether to shuffle windows between epochs. Defaults to True.
    
    :return: Batched tf.data.Dataset yielding tuples `(batch_X, batch_y)` where:
             - `batch_X`: Shape `(batch_size, window_size, n_features)`
             - `batch_y`: Shape `(batch_size, horizon)`
             
    Note:
        - Expects `X`, `y` from `make_windows()` - already properly aligned and truncated.
        - No `drop_remainder=True` (enable if LSTM shape mismatch occurs).
        - Includes prefetch for CPU/GPU performance optimization.
    """
    batch_size = batch_size or cnfg.preprocess.windowing["batch_size"]

    tf_dataset = tf.data.Dataset.from_tensor_slices(tensors=(X, y))

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(X))
            
    tf_dataset = tf_dataset.batch(batch_size=batch_size)

    return tf_dataset.prefetch(tf.data.AUTOTUNE)
    
