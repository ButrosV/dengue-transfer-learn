import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

def random_color():
    """
    get random matplot-lib colour - just for fun
    """
    color_names = list(mcolors.get_named_colors_mapping().keys())
    color_count = len(color_names)
    random_num = random.randint(0, color_count - 1)
    rand_col = mcolors.get_named_colors_mapping()[color_names[random_num]]
    return rand_col


def random_colormap(seaborn: bool = False, n_colors: int = 6):
    """
    get random matplot-lib colourmap - just for fun
    :param seaborn: set to True if if use with seaborn
    :param n_colors: also for seaborn palette - how many colors to use
                    Default 6 is also seaborn default. 
                    For hues use category count.
    """
    c_maps = plt.colormaps()
    random_cmap_name = random.choice(c_maps)
    if not seaborn:
        return random_cmap_name
    else:
        return sns.color_palette(random_cmap_name, n_colors=n_colors)


def compute_correlations_matrix(data: pd.DataFrame,
                               annot: bool = False,
                               figsize: tuple = (17, 5),
                                cmap: str = "PuBuGn"):
    """
    Compute and display a heatmap of the correlation matrix for numerical features.    
    :param data: pandas DataFrame containing the input data.
    :return: pandas DataFrame of the correlation matrix.
    """
    plt.figure(figsize=figsize)
    correlation_matrix = data.select_dtypes(include="number").corr()
    triangular_matrix = np.triu(correlation_matrix, k=1)

    sns.heatmap(data=correlation_matrix,
                center=0,
                cmap=cmap,
                cbar=False,
                # cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                annot=annot,
                linewidths=0.5,
                mask=triangular_matrix,
               )
    plt.xticks(rotation=30, ha="right", rotation_mode="anchor")
    plt.title("Correlation matrix for feature and target columns.",
                fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout();



def display_distributions(data: pd.DataFrame, features: List[str],
                          fig_width: int = 17,
                          hue_palette: Tuple[str | None, Any | None] = (None, None),
                          x_range: Tuple[int,int] | None = None,
                          title_prefix: str | None =None) -> None:
    """Display distribution graphs for specified numerical features.
    All subplots share the same x-axis scale.
    :param data: DataFrames with numerical categories for visualization.
    :param features: list of column names/features  from 'data' Dataframe.
    :param hue_palette: Coloring by category. Use (None, None) for no hue 
                coloring (default).
    :param x_range: Optional tuple (min, max) to set uniform x-axis limits 
                across all subplots (default: None, auto-scaled).
    :param title_prefix: Optional, prefix for visualization title.
    """
    n_subplots = len(features) * 2
    fig, axs = plt.subplots(nrows=n_subplots, figsize = (fig_width, n_subplots * 2), 
                            sharex=True
                           )                   
    index = 0
    for feature in features:
        sns.boxenplot(data=data, x=data[feature],
                      hue=hue_palette[0],
                      palette=hue_palette[1], color=random_color(), 
                      ax=axs[index])
        sns.kdeplot(data=data, x=data[feature], fill=True,
                      hue=hue_palette[0], palette=hue_palette[1],
                    color=random_color(), 
                    ax=axs[index + 1])
        index += 2
        
    if title_prefix is not None:
        fig.suptitle(f"{title_prefix.capitalize()} feature distribution analysis.", fontsize=18,
                    fontweight='bold', y=0.98)
    else:
        fig.suptitle("Feature distribution analysis.", fontsize=18,
                    fontweight='bold', y=0.98)

    if x_range is not None:
        axs[0].set_xlim(x_range)
        
    fig.tight_layout();


def display_timeseries(data: pd.DataFrame, x: str, y: str,
                        hue: str | None = None, grid: bool = True,
                        month_ticks: Tuple[int, ...] = (1,4,7,10),
                        shift: int | None = None,
                        title_prefix: str | None = None) -> None:
    """
    Display timeseries line plots for specified features.
    All subplots share the same x-axis scale.
    
    :param data: DataFrame with numerical categories for visualization.
    :param x: Column name for x-axis (datetime).
    :param y: Column name for y-axis.
    :param hue: Optional column name for hue coloring (default: None).
    :param grid: Whether to show grid lines (default: True).
    :param shift: Optional number of weekly periods to shift the time axis 
        for an overlaid comparison line.
    :param month_ticks: Tuple of month numbers for minor x-axis ticks (default: (1,4,7,10) quarterly).
    :param title_prefix: Optional, prefix for visualization title.
    """
    
    fig, ax = plt.subplots(figsize=(19, 5))
    sns.lineplot(data=data, x=x, y=y, hue=hue)
    if shift is not None:
        shifted_data = data[[x, y, hue]]
        shifted_data[x] = shifted_data[x] + pd.Timedelta(52, "W")
        sns.lineplot(data=shifted_data, x=x, y=y, hue=hue, linestyle=":")
    # # Configure x-axis ticks (quarterly minors, yearly majors)
    ax.xaxis.set(major_locator=YearLocator(),
                 minor_locator=MonthLocator(bymonth=month_ticks),
                 minor_formatter=(DateFormatter(fmt="%b")))
        
    # Style ticks
    ax.tick_params(axis='x', which="major", pad=20, length=20,
                   colors=random_color()  # remove for cleanup
                  )
    ax.tick_params(axis='x', which="minor", labelrotation=60,
                  colors=random_color()  # remove for cleanup
                  )
    
    # Grid styling
    if grid:
        plt.grid(alpha=0.7,
            linestyle="dashed",
            color=random_color()  # remove for cleanup
        )
        plt.grid(alpha=0.5, axis='x', which="minor", linestyle="dotted",
                color=random_color()  # remove for cleanup
                )

    if title_prefix is not None:
        plt.title(f"{title_prefix.capitalize()} distribution over time.",
                    fontsize=13, fontweight="bold");
    else:
        plt.title("Target distribution over time.",
                 fontsize=13, fontweight="bold");
    
