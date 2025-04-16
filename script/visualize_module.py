


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import config

from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve, precision_recall_curve




def histgram_visualize(data : np.array, bins: int=100, edgecolor="black", title: str="", xlabel: str="", ylabel: str="", save_path=None, dpi: int=150) -> None:
    plt.hist(data, bins=bins, edgecolor=edgecolor)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def boxplot_visualize(data_list : list, data_names : list=None, title : str="", xlabel: str="", ylabel: str="", save_path=None, dpi: int=150) -> None:
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data_list, vert=True, patch_artist=True, widths=0.6, showmeans=False)
    for i, data in enumerate(data_list):
        mean = np.mean(data)
        plt.scatter(i+1, mean, color='red', zorder=3)
        if data_names:
            plt.text(i+1, -1, data_names[i], horizontalalignment='center', verticalalignment='bottom', fontsize=12)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def line_graph_chromatin_state(data_array_dict: list, title: str="", ylabel: str="", scope_range: int=5000, bin_size: int=50, divide_value: int=100, save_path: str=None, dpi: int=300):
    # X axis
    x_axis = np.linspace(-scope_range, scope_range, scope_range*2//bin_size)
    
    # Plot figure
    plt.figure(figsize=(8, 6))
    
    # Plot line graph
    for idx, data_name_ in enumerate(data_array_dict.keys()):
        data_array = np.nan_to_num(data_array_dict[data_name_], nan=0)
        data_mean_array = np.mean(data_array, axis=0)
        color = config.type_colors[data_name_]
        plt.plot(x_axis/divide_value, data_mean_array, color=color)
    
    # Axis label and title
    plt.ylabel(ylabel)
    plt.title(title, fontsize=20)
    if divide_value == 1000:
        plt.xticks([-scope_range/divide_value, 0, scope_range/divide_value], [f"{-scope_range/divide_value} kb", "Center", f"{scope_range/divide_value} kb"], fontsize=16)
    else:
        plt.xticks([-scope_range/divide_value, 0, scope_range/divide_value], [f"{-scope_range/divide_value} bp", "Center", f"{scope_range/divide_value} bp"], fontsize=16)
    # plt.legend(loc="upper right")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()


def line_graph_legend(data_name_dict: dict, save_path: str=None, dpi: int=300):
    # Set font size ad line width
    plt.rcParams.update({
        'font.size': 14,        # Increase font size
        'legend.fontsize': 18,  # Increase legend font size
        'lines.linewidth': 2    # Increase line width
    })
    # Create a dummy plot to generate the legend
    fig_legend = plt.figure(figsize=(10, 1))
    # Add dummy lines to the plot for each entry in data_names_dict
    for data_name, display_name in data_name_dict.items():
        color = config.type_colors[data_name]
        plt.plot([], [], color=color, label=display_name, linewidth=8)
    
    # Generate and save the legend
    plt.legend(loc='center', frameon=False, ncol=len(data_name_dict))
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    plt.show()



def plot_reductioned_data(result_data: np.array, type_list, title: str="", xlabel: str="", ylabel: str="", save_path: str=None, dpi: int=150) -> None:
    # regend
    colors = [config.type_colors[sample_type] for sample_type in type_list]
    type_colors_dict = {key: config.type_colors[key] for key in config.type_colors if key in set(type_list)}
    
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=sample_type, markersize=10, markerfacecolor=color) for sample_type, color in type_colors_dict.items()]
    # plot
    plt.figure(figsize=(10, 6))
    plt.scatter(result_data[:, 0], result_data[:, 1], c=colors, alpha=1, s=5)
    plt.legend(title='Type', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'{title}')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}", dpi=dpi)
    plt.show()

def annotate_significance(ax, x1, x2, y, text):
    ax.plot([x1+0.005, x1+0.005, x2-0.005, x2-0.005], [y, y+0.01, y+0.01, y], lw=1.5, color='k')
    ax.text((x1 + x2) * .5, y + 0.008, text, ha='center', va='bottom', color='k', fontsize=6)

def save_boxplot(data: dict, model_name_mapping: dict, title: str="", ylabel: str="", save_path: str=None, dpi: int=300, if_dnabert_epi: bool=False) -> None:
    # dataはdata = {"model_name": [val1, val2, val3, ...], ...}
    # model_name_mappingはmodel_name_mapping = {"model_name": "Model Name", ...}
    # 各モデルの色はconfig.pyのtype_colorsに記載されている
    plt.figure(figsize=(6, 6))
    
    # Prepare data for plotting
    data_list = [data[model_name] for model_name in data.keys()]
    data_names = [model_name_mapping[model_name] for model_name in data.keys()]
    
    # Create boxplot
    box = plt.boxplot(data_list, vert=True, patch_artist=True, widths=0.6, showmeans=True,
                      meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":2},
                      flierprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":2},
                      medianprops={"color":"black", "linewidth":1.5})
    
    # Apply colors to each box
    for patch, color in zip(box['boxes'], [config.type_colors[model_name] for model_name in data.keys()]):
        patch.set_facecolor(color)
    
    # Annotate medians
    # for i, data in enumerate(data_list):
    #     median = np.median(data)
    #     plt.text(i+1, median, f'{median:.4f}', horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    
    # Annotate means
    for i, data in enumerate(data_list):
        mean = np.mean(data)
        if if_dnabert_epi:
            add_range_x = 1.525
        else:
            add_range_x = 1.5
        plt.text(i+add_range_x, mean, f'{mean:.4f}', horizontalalignment='right', verticalalignment='center', fontsize=8, rotation=90)
    
    # Set labels and title
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(ticks=np.arange(1, len(data_names) + 1), labels=data_names, rotation=15, ha='center')
    
    # Annotate significance
    y_max = 1
    for i in range(len(data_list) - 1):
        for j in range(i + 1, len(data_list)):
            _, p_value1 = wilcoxon(data_list[i], data_list[j], alternative='less')
            _, p_value2 = wilcoxon(data_list[i], data_list[j], alternative='greater')
            p_value = min(p_value1, p_value2)
            if p_value < 1e-4:
                text = '****'
            elif p_value >= 1e-4:
                text = 'ns' if p_value > 0.05 else '*' * int(-np.log10(p_value))
            else:
                text = ""
            if if_dnabert_epi:
                if data_names[j] == "DNABERT":
                    annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i), text)
                if data_names[j] == "DNABERT-Epi":
                    annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(data_list)-3) + 0.05 * (j - i), text)
                if data_names[j] == "Ensemble":
                    annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * ((len(data_list)-2)+(len(data_list)-3)) + 0.05 * (j - i), text)
            else:
                if data_names[j] == "DNABERT":
                    annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i), text)
                if data_names[j] == "Ensemble":
                    annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(data_list)-2) + 0.05 * (j - i), text)
    
    # Set y-axis range
    if not if_dnabert_epi:
        plt.ylim(0, y_max + 0.6)
    else:
        plt.ylim(0, y_max + 1.0)
    plt.yticks(np.arange(0, 1.1, 0.2))
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    # plt.show()
    

def plot_roc_curve(ground_truth_array: np.array, probability_dict: dict, rocauc_value_dict: dict, model_name_mapping: dict, title: str="", save_path: str=None, dpi: int=300) -> None:
    # Color mapping
    color = [config.type_colors[model_name] for model_name in probability_dict.keys()]
    # Plot figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    for (model_name, probability), color in zip(probability_dict.items(), color):
        fpr, tpr, _ = roc_curve(ground_truth_array, probability[:, 1])
        plt.plot(fpr, tpr, color=color, label=f'{model_name_mapping[model_name]} (AUC={rocauc_value_dict[model_name]["ROC-AUC"]})')
    
    # Axis label and title, legend
    plt.title(title)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)

def plot_prroc_curve(ground_truth_array: np.array, probability_dict: dict, prauc_value_dict: dict, model_name_mapping: dict, title: str="", save_path: str=None, dpi: int=300) -> None:
    # Color mapping
    color = [config.type_colors[model_name] for model_name in probability_dict.keys()]
    # Plot figure
    plt.figure(figsize=(8, 6))
    
    # Plot PR curve
    for (model_name, probability), color in zip(probability_dict.items(), color):
        precision, recall, _ = precision_recall_curve(ground_truth_array, probability[:, 1])
        plt.plot(recall, precision, color=color, label=f'{model_name_mapping[model_name]} (AUC={prauc_value_dict[model_name]["PR-AUC"]})')
    
    # Axis label and title, legend
    plt.title(title)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
def attention_weight_visualize(attention_weight_array: np.array, seq_len: int=24, kmer: int=3, title: str="", save_path: str=None, dpi: int=300) -> None:
    # Attention weight array shape: (num layers, seq_len, seq_len) = (12, 47, 47)
    print(attention_weight_array.shape)
    
    # y-axis labels
    y_labels = ["[CLS]"] + [f"DNA {i+1}" for i in range(seq_len-kmer+1)] + ["[SEP]"] + [f"sgRNA {i+1}" for i in range(seq_len-kmer+1)] + ["[SEP]"]
    
    # Visualize DNABERT Attention weights
    fig, ax1 = plt.subplots(figsize=(24, 15))
    ax2 = ax1.twinx()
    
    # Set axis infortmation
    ax1.set_yticks(np.arange(len(y_labels)))
    ax2.set_yticks(np.arange(len(y_labels)))
    ax1.set_ylim(-1, len(y_labels))
    ax2.set_ylim(-1, len(y_labels))
    ax1.set_yticklabels(y_labels[::-1], fontsize=14)
    ax2.set_yticklabels(y_labels[::-1], fontsize=14)
    ax1.set_xticks(np.arange(13))
    ax1.set_xlim(-0.1, 12.1)
    
    # Draw attention weights
    for n_layer in range(12):
        n_attention = attention_weight_array[n_layer]
        # threshold = np.percentile(n_attention, 90)
        # n_attention[n_attention < threshold] = 0
        for i in range(len(y_labels)):
            for j in range(len(y_labels)):
                if n_attention[i, j] > 0:
                    ax1.plot([n_layer, n_layer+0.9], [i, j], color='red', alpha=n_attention[i, j].item())
                else:
                    ax2.plot([n_layer, n_layer+0.9], [i, j], color='blue', alpha=-n_attention[i, j].item())
    red_patch = mpatches.Patch(color='red', label='Active OTS Attention')
    blue_patch = mpatches.Patch(color='blue', label='Inactive OTS Attention')
    plt.legend(handles=[red_patch, blue_patch], loc='upper left', fontsize=14)
    plt.title(title, fontsize=24)
    plt.grid(False)
    
    # Save plot if save_path is provided 
    if save_path:
        plt.savefig(save_path, dpi=dpi)

