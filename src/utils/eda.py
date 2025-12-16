import pandas as pd


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



def nan_streaks(data: pd.DataFrame, column: str, run_threshold: int = 0
                    ) -> pd.DataFrame:
    """
    Identify and return streaks of NaN values in a specified column exceeding a minimum length threshold.
    
    **Usage Example:**
    ```
    result = nan_streaks(df, "ndvi_ne", run_threshold=5)
    df["ndvi_ne"].iloc[result["first_pos"]:result["last_pos"] + 1]  # Extract 2nd NaN streak[1]
    ```
    
    :param data: pandas DataFrame containing the input dataset.
    :param column: Name of the column to analyze for NaN streaks.
    :param run_threshold: Minimum streak length to include. Default is 0 (includes all streaks).
    :return: A pandas DataFrame with columns 'first_pos', 'last_pos', and 'streak_len' for qualifying NaN streaks, sorted by length descending.
    """
    null_sequences = data[column].isnull()
    run_boundaries = null_sequences != null_sequences.shift()  # shift by one  + `!=` indicates where runs of equal values start/end
    run_boundaries = run_boundaries.cumsum()  # asign numeric value to each run sequence (previous sequence+=1)
    nan_run_boundaries = run_boundaries[null_sequences]  # filter nan == True run labels
    run_groups = nan_run_boundaries.groupby(by=nan_run_boundaries)  # group these NaN sequences

    nan_streak_df = run_groups.agg(
                        first_pos = lambda x: x.index[0],
                        last_pos = lambda x: x.index[-1],
                        streak_len = "size"  # counts rows per group, similar to len, but mre stable with NaNs
                        )
    nan_streak_df = nan_streak_df.query(f"streak_len > {run_threshold}")  #filter aggregated DatFrame
    nan_streak_df = nan_streak_df.sort_values(by="streak_len", ascending=False)
    nan_streak_df = nan_streak_df.reset_index(drop=True)  # drop now meaningless `run_boundaries` index

    return nan_streak_df
    
