import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Any, Dict
from pathlib import Path
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

import seaborn as sns

from src.config import ProjectConfig

cnfg = ProjectConfig.load_configuration()


def save_picture(path: str | Path, show: bool = True):
    """Save current figure with folder resolution and auto dir creation.
    Optionally display (default for notebooks)"""
    path = Path(path)
    if path.parent.name == '':
        folder = cnfg.data.dirs.get("pics")
        if folder is not None:
            path = folder / path.name


    path.parent.mkdir(parents=True, exist_ok=True)
        
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    if not show:
        plt.close() # Prevents memory buildup
    
    return path

        
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
                                cmap: str = "PuBuGn",
                                savefile_name: str | None = None):
    """
    Compute and display a heatmap of the correlation matrix for numerical features.    
    :param data: pandas DataFrame containing the input data.
    :savefile_name: Optional path for saving visualization.
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
                
    if savefile_name is not None:
        saved_path = save_picture(path=savefile_name)
        
    plt.tight_layout()
    plt.show()
   

def display_distributions(data: pd.DataFrame, features: List[str],
                          fig_width: int = 17,
                          hue_palette: Tuple[str | None, Any | None] = (None, None),
                          x_range: Tuple[int,int] | None = None,
                          title_prefix: str | None =None,
                          savefile_name: str | None = None) -> None:
    """Display distribution graphs for specified numerical features.
    All subplots share the same x-axis scale.
    :param data: DataFrames with numerical categories for visualization.
    :param features: list of column names/features  from 'data' Dataframe.
    :param hue_palette: Coloring by category. Use (None, None) for no hue 
                coloring (default).
    :param x_range: Optional tuple (min, max) to set uniform x-axis limits 
                across all subplots (default: None, auto-scaled).
    :param title_prefix: Optional, prefix for visualization title.
    :savefile_name: Optional path for saving visualization.
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
        
    if savefile_name is not None:
        saved_path = save_picture(path=savefile_name)  
              
    fig.tight_layout()
    plt.show()
    

def display_timeseries(data: pd.DataFrame, x: str, y: str,
                        hue: str | None = None, grid: bool = True,
                        month_ticks: Tuple[int, ...] = (1,4,7,10),
                        shift: int | None = None,
                        title_prefix: str | None = None,
                        savefile_name: str | None = None) -> None:
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
    :savefile_name: Optional path for saving visualization.
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
                 fontsize=13, fontweight="bold")
                 
    if savefile_name is not None:
        saved_path = save_picture(path=savefile_name)            
             
    plt.show()
             
                
def display_wfcv_folds(data:Dict,
                       X:pd.DataFrame | None = None,
                       group_feat:str | None = None,
                       savefile_name: str | None = None):
    """
    Visualize walk-forward cross-validation (WFCV) fold evaluation metrics 
    across multiple statistics with feature group boundaries.
    
    Create subplots (one per statistic) showing metrics over folds. 
    Optionally overlays feature group boundaries as vertical lines.
    
    :param data: Dict containing WFCV results with keys:
                 - "raw": Dict of {statistic: metrics_list} for each fold
                 - "fold_test_starts": Array of fold test start positions
    :param X: Optional features DataFrame for group boundaries. 
              Index should match data fold positions or be sequential.
    :param group_feat: Optional column name in X to group by for boundaries.
    :savefile_name: Optional path for saving visualization.
    
    :return: Tuple of (figure, axs) for further customization.
    
    Example:
        display_wfcv_folds(wfcv_results, X=features_df, group_feat='city_id')
    """
    figure, axs = plt.subplots(ncols=1, nrows=len(data["raw"]),
                               figsize=(13, len(data["raw"]) * 2), sharex=True)
    
    if (X is not None) and (group_feat is not None):
        if all(X.index == range(len(X))):
            feat_boundaries=X.groupby(by=group_feat).nth(0).index
        else:
            temp_series = X[group_feat].reset_index(drop=True)
            feat_boundaries=temp_series.groupby(level=0).nth(0).index
        feat_boundaries = [boundary for boundary in feat_boundaries if boundary != 0]
        fold_adjusted_boundaries = np.searchsorted(data["fold_test_starts"], feat_boundaries)

        for index, scaled in zip(feat_boundaries, fold_adjusted_boundaries):
            figure.text(x=scaled, y=1.07, fontweight="bold",
                        s=f"{X[group_feat].iloc[index - 1]} <-> {X[group_feat].iloc[index]}",
                        ha="center",
                        transform=axs[-1].get_xaxis_transform()
                       )
    else:
        fold_adjusted_boundaries = []
    
    index = 0
    for stat, metrics in data["raw"].items():
        ax = axs[index]
        ax.plot(metrics, marker='o', markersize=3,
                mfc=random_color(), color=random_color())
        
        tick_col = random_color()    
        ax.grid(alpha=0.7, linestyle="dashed", color=tick_col)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="major", length=7, color=tick_col)
    
        ax.vlines(x=fold_adjusted_boundaries,
                  ymin=0, ymax=1, transform=ax.get_xaxis_transform(), 
                  linestyles="dashdot", colors=random_color())

        ax.set_ylabel(stat, fontweight="bold")
        index += 1
    
    figure.suptitle("WFCV fold evaluation metrics.", fontsize=13, fontweight="bold")
    
    if savefile_name is not None:
        saved_path = save_picture(path=savefile_name)
        
    plt.show()
    
