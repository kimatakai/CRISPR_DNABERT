import matplotlib.pyplot as plt
import numpy as np





def plot_linegraph_cpg(
    config: dict,
    data_list: list,
    y_labels: str,
    labels: list,
    colors: list,
    title: str = "",
    fig_path: str = None,
    ):
    # Load parameters from config
    both_window_size = config["parameters"]["eda_window_size"]*2
    cpg_sliding_window_size = config["parameters"]["cpg_sliding_window_size"]
    cpg_sliding_window_step = config["parameters"]["cpg_sliding_window_step"]

    # X-coordinates range
    side_range = (both_window_size // 2) - (cpg_sliding_window_size // 2)
    x_coords = np.arange(-side_range, side_range + 1, cpg_sliding_window_step)
    
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, data_list[0], label=labels[0], color=colors[0], marker='o', markersize=1)
    plt.plot(x_coords, data_list[1], label=labels[1], color=colors[1], marker='o', markersize=1)
    
    # Set labels and title
    plt.xlim(-x_coords[-1], x_coords[-1])
    plt.xlabel("Distance from DSB (bp)", fontsize=12)
    plt.ylabel(y_labels, fontsize=12)
    plt.title(title)
    plt.grid(False)
    plt.legend(loc="lower right", fontsize=12)
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)

