import pandas as pd
import numpy as np
import random
from typing import List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def random_color():
    """
    get random matplot-lib colour - just for fun
    """
    color_names = list(mcolors.get_named_colors_mapping().keys())
    color_count = len(color_names)
    random_num = random.randint(0, color_count - 1)
    rand_col = mcolors.get_named_colors_mapping()[color_names[random_num]]
    return rand_col


def compute_correlations_matrix(data: pd.DataFrame,
                               annot: bool = False,
                               figsize: tuple = (17, 5),
                                cmap:str = "PuBuGn"):
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
                          title_prefix: str=None) -> None:
    """Display distribution graphs for specified categorical columns.
    Graphs are displayed as a vertical stack of box plots.
    :param data: DataFrames with numerical categories for visualization.
    :param features: list of column names/features  from 'data' Dataframe.
    :param title_prefix: Optional, prefix for visualization title.
    """
    n_subplots = len(features) * 2
    fig, axs = plt.subplots(nrows=n_subplots, figsize = (fig_width, n_subplots * 2))
    index = 0
    for feature in features:
        sns.boxenplot(data=data, x=data[feature],
                      color=random_color(), ax=axs[index])
        sns.kdeplot(data=data, x=data[feature], fill=True,
                    color=random_color(), ax=axs[index + 1])
        index += 2

    if title_prefix is not None:
        fig.suptitle(f"{title_prefix} feature distribution analysis.", fontsize=18,
                    fontweight='bold', y=0.98)
    else:
        fig.suptitle("Feature distribution analysis.", fontsize=18,
                    fontweight='bold', y=0.98)
    fig.tight_layout();

