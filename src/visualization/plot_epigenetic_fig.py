
import matplotlib.pyplot as plt
import numpy as np



def plot_epigenetic_line_graph_by_cm(config: dict, epigenetic_data: dict, type: str, save_path: str=None) -> None:
    # Plotting line graphs for epigenetic features by confusion matrix categories
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    # Define colors for each category
    colors = {
        "TN": config["colors"]["TN"],
        "FP": config["colors"]["FP"],
        "FN": config["colors"]["FN"],
        "TP": config["colors"]["TP"]
    }
    legends = {
        "TN": "True Negative",
        "FP": "False Positive",
        "FN": "False Negative",
        "TP": "True Positive"
    }
    
    # X-axis range
    x_axis = np.arange(-config["parameters"]["window_size"][type], config["parameters"]["window_size"][type], config["parameters"]["bin_size"][type])
    
    # Y-axis limits
    y_max, y_min = 0, 0

    for label, signal_array in epigenetic_data.items():
        data = np.nanmean(signal_array, axis=0)
        # Update y_max for setting y-axis limits
        y_max = max(y_max, np.nanmax(data)*1.3)
        y_min = min(y_min, np.nanmin(data))
        ax.plot(x_axis, data, color=colors[label], label=legends[label] + f" (n={signal_array.shape[0]})", linewidth=1.5)
    
    # ax.yaxis.set_visible(False)
    y_min = y_min - abs(y_max)*0.1 if y_min < 0 else -y_min - abs(y_max)*0.1
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([-config["parameters"]["window_size"][type], 0, config["parameters"]["window_size"][type]])
    ax.set_xticklabels([str(-config["parameters"]["window_size"][type]) + "bp", "center", str(config["parameters"]["window_size"][type]) + "bp"])
    ax.set_xlabel("Genomic Position relative to Center", fontsize=16)
    ax.legend(loc='upper right', fontsize=9) # frameon=True, 
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Epigenetic line graph saved to {save_path}")
    else:
        plt.show()


def plot_confusion_mismatch_heatmap(config: dict, confusion_mismatch_epigenetic: dict, window_size: int, bin_size: int, save_path: int) -> None:
    # ===== Create data for heat map =====
    n_mismatch = [1, 2, 3, 4, 5, 6]
    confusion_entries = ["tn", "fp", "fn", "tp"]
    confusion_entries_labels = {"tn": "TN", "fp": "FP", "fn": "FN", "tp": "TP"}
    blocks = []
    row_boundaries = []
    row_labels = []
    current_row = 0
    
    for entry in confusion_entries:
        for mm in n_mismatch:
            block = confusion_mismatch_epigenetic[entry][mm] 
            if block.size > 0:
                blocks.append(block)
                current_row += 1
                row_labels.append(f"{confusion_entries_labels[entry]} : MM{mm}")
        row_boundaries.append(current_row)
    heatmap_data = np.vstack(blocks)
    
    # ===== Plot heat map =====
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    
    for boundary in row_boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="black", linewidth=3)
    
    x_axis = np.arange(-window_size, window_size, bin_size)
    ax.set_xticks([0, len(x_axis)//2, len(x_axis)-1])
    ax.set_xticklabels([str(-window_size) + "bp", "center", str(window_size) + "bp"], fontsize=14)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=13)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(length=0) 
    cbar.ax.set_yticklabels([])
    
    print(f"Epigenetic heatmap saved to {save_path}")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


def plot_epi_mark_shap_importance(aggregate_importance: dict, using_epi_type: list, epi_name_map: dict, save_path: str=None) -> None:
    # Calculate mean and standard deviation of absolute SHAP values for each epigenetic mark
    mean_importance = []
    std_importance = []
    for type_of_data in using_epi_type:
        values = aggregate_importance[type_of_data]
        mean_importance.append(np.mean(np.abs(values)))
        std_importance.append(np.std(np.abs(values)))
    marks = [epi_name_map[type_of_data] for type_of_data in using_epi_type]
    # Plot
    plt.figure(figsize=(8, 6))
    plt.bar(marks, mean_importance, yerr=std_importance, capsize=5, color='skyblue', edgecolor='black')
    plt.ylabel('Mean Absolute SHAP Value', fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_epi_mark_shap_position_importance(position_importance: dict, positions: np.ndarray, using_epi_type: list, epi_name_map: dict, color_map: dict, save_path: str=None) -> None:
    plt.figure(figsize=(15, 7))
    for type_of_data in using_epi_type:
        plt.plot(positions[type_of_data], position_importance[type_of_data]["mean"], label=epi_name_map[type_of_data], color=color_map[type_of_data])
        plt.fill_between(
            positions[type_of_data], 
            position_importance[type_of_data]["mean"] - position_importance[type_of_data]["std"], 
            position_importance[type_of_data]["mean"] + position_importance[type_of_data]["std"], 
            color=color_map[type_of_data], alpha=0.3
            )
    plt.xlabel("Genomic position relative to cleavage site (±bp)", fontsize=16)
    plt.ylabel("Mean Absolute SHAP value", fontsize=16)
    plt.legend(fontsize=16, loc='upper left')
    plt.axvline(x=0, color="black", linestyle="--")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    