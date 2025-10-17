import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    count_data: np.array, 
    output_path: str, 
    title: str = "", 
    xlabel: str = "DSB Count (log scale)", 
    ylabel: str = "Frequency", 
    bins: int = 80,
    figsize: tuple = (10, 6), 
    fontsize: int = 12,
    alpha: float = 0.75
    ) -> None:
    # Log-transform the count data; log(x+1)
    count_data = np.log1p(count_data)
    
    plt.figure(figsize=figsize)
    plt.hist(count_data, bins=bins, alpha=alpha, color='gray', edgecolor='black')
    
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    
    # Quantiles and median lines
    q1 = np.percentile(count_data, 25)
    median = np.median(count_data)
    q3 = np.percentile(count_data, 75)
    print(f"Q1: {q1}, Median: {median}, Q3: {q3}")
    plt.axvline(q1, color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(median, color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(q3, color='green', linestyle='dashed', linewidth=1)

    # plt.show()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()