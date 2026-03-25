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
    
    
def make_tf_dataset(X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int | None = None,
                 shuffle: bool = True
                ) -> tf.data.Dataset:
    """
    Convert NumPy arrays into a batched `tf.data.Dataset` suitable for LSTM training.
    
    Handles both windowed sequences (from `make_windows`) and pointwise (1→1) data.
    Adds a dummy timestep dimension for 2D inputs if needed.
    
    :param X: Input features array. Shape can be:
              - `(n_windows, window_size, n_features)` for windowed sequences
              - `(n_samples, n_features)` for tabular 2D data
    :param y: Target array. Shape can be:
              - `(n_windows, horizon)` for windowed sequences
              - `(n_samples,)` or `(n_samples, 1)` for tabular 2D data
    :param batch_size: Number of samples per batch. Defaults to config value (e.g., 32).
    :param shuffle: Whether to shuffle samples/windows between epochs. Defaults to True.

    :return: A `tf.data.Dataset` yielding tuples `(batch_X, batch_y)`:
             - `batch_X`: `(batch_size, window_size, n_features)` (or `(batch_size, 1, n_features)` for 1→1)
             - `batch_y`: `(batch_size, horizon)` (or `(batch_size, 1)` for 1→1)

    Notes:
        - Does **not** use `drop_remainder=True` by default; enable if exact batch sizes are required for LSTM.
        - Prepares data efficiently for GPU/CPU via `.prefetch(tf.data.AUTOTUNE)`.
        - Compatible with both windowed sequence datasets and non-windowed 1→1 datasets.
    """
    batch_size = batch_size or cnfg.preprocess.windowing["batch_size"]

    if X.ndim == 2:
        X = X.copy()[:, np.newaxis, :]
    if y.ndim == 1:
        y = y.copy()[:, np.newaxis]

    tf_dataset = tf.data.Dataset.from_tensor_slices(tensors=(X, y))

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(X))
            
    tf_dataset = tf_dataset.batch(batch_size=batch_size)

    return tf_dataset.prefetch(tf.data.AUTOTUNE)
    
