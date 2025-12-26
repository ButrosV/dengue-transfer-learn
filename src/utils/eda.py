import pandas as pd
import numpy as np
from typing import Any


def top_correlations(data: pd.DataFrame, corr_threshold:float=0.95) -> pd.Series:
    """
    Identify and return highly correlated feature pairs above a specified threshold.
    :param data: pandas DataFrame containing the input dataset. Defaults to df_clean.
    :param corr_threshold: Correlation threshold to filter feature pairs. Default is 0.95.
    :return: A pandas Series with multi-index (feature pairs) and correlation values.
    """
    correlations = data.select_dtypes(include="number").corr()
    correlations = correlations.unstack().sort_values(ascending=False)
    correlations = correlations[(correlations.abs() > corr_threshold) &
        (correlations.index.get_level_values(0) != correlations.index.get_level_values(1))]
    correlations = correlations.drop_duplicates()
    return correlations



def value_streaks(data: pd.DataFrame, column: str, value: Any, run_threshold: int = 0
                    ) -> pd.DataFrame:
    """
    Identify and return streaks of specified values (including NaN) in a specified 
    column exceeding a minimum length threshold.
    
    **Usage Example:**
    ```
    # Check for streaks of 0
    result = value_streaks(df, "total_cases", 0, run_threshold=5)
    result = value_streaks(df, "ndvi_ne", run_threshold=5)
    df["ndvi_ne"].iloc[result["first_pos"]:result["last_pos"] + 1]  # Extract 2nd NaN streak
    ```
    
    :param data: pandas DataFrame containing the input dataset.
    :param column: Name of the column to analyze for NaN streaks.
    :param value: Value to detect streaks of (use `np.nan` for NaN streaks).
    :param run_threshold: Minimum streak length to include. Default is 0 (includes all streaks).
    :return: A pandas DataFrame with columns 'first_pos', 'last_pos', and 'streak_len' 
                for qualifying NaN streaks, sorted by length descending.
    """
    if pd.isna(value):
        mask_sequences = data[column].isna()
    else:
        mask_sequences = data[column] == value
    
    run_boundaries = mask_sequences != mask_sequences.shift()  # shift by one  + `!=` indicates where runs of equal values start/end
    run_boundaries = run_boundaries.cumsum()  # asign numeric value to each run sequence (previous sequence+=1)
    sequence_run_boundaries = run_boundaries[mask_sequences]  # filter mask == True value run labels
    run_groups = sequence_run_boundaries.groupby(by=sequence_run_boundaries)  # group these value sequences

    sequence_streak_df = run_groups.agg(
                        first_pos = lambda x: x.index[0],
                        last_pos = lambda x: x.index[-1],
                        streak_len = "size"  # counts rows per group, similar to len, but mre stable if NaNs in data
                        )
    sequence_streak_df = sequence_streak_df.query(f"streak_len > {run_threshold}")  #filter aggregated DatFrame
    sequence_streak_df = sequence_streak_df.sort_values(by="streak_len", ascending=False)
    sequence_streak_df = sequence_streak_df.reset_index(drop=True)  # drop now meaningless `run_boundaries` index

    return sequence_streak_df
    
