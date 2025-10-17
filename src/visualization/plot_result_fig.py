
import matplotlib.pyplot as plt

import numpy as np



def annotate_significance(ax, x1, x2, y, y_range, text):
    ax.plot([x1+0.005, x1+0.005, x2-0.005, x2-0.005], [y, y+0.01*y_range, y+0.01*y_range, y], lw=1.5, color='k')
    ax.text((x1 + x2) * .5, y + 0.008 * y_range, text, ha='center', va='bottom', color='k', fontsize=6)

def _plot_boxplot(config: dict, aggregated_results: dict, aggregated_p_values: dict, model_names_list: list, metrics: list, save_dir: str=None, dpi: int=300) -> None:
    # If model_names_list contains "Ensemble", "DNABERT", or "DNABERT-Epi", ensure the order is Other models, "DNABERT", "DNABERT-Epi", "Ensemble"
    if "Ensemble" in model_names_list or "DNABERT" in model_names_list or "DNABERT-Epi" in model_names_list:
        other_models = [m for m in model_names_list if m not in ["Ensemble", "DNABERT", "DNABERT-Epi"]]
        ordered_models = other_models
        if "DNABERT" in model_names_list:
            ordered_models.append("DNABERT")
        if "DNABERT-Epi" in model_names_list:
            ordered_models.append("DNABERT-Epi")
        if "Ensemble" in model_names_list:
            ordered_models.append("Ensemble")
        model_names_list = ordered_models
    
    for metric in [m for m in metrics if m!= "confusion_matrix"]:
    # for metric in ["pr_auc"]:
        plt.figure(figsize=(6, 6))
        # Prepare data for plotting
        data_list = []
        for model_name in model_names_list:
            data_list.append(aggregated_results[model_name][metric])
        box = plt.boxplot(data_list, vert=True, patch_artist=True, widths=0.6, showmeans=True,
                            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":2},
                            flierprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":2},
                            medianprops={"color":"black", "linewidth":1.5})
        # Apply colors to each box
        for patch, color in zip(box['boxes'], [config["colors"][model_name] for model_name in model_names_list]):
            patch.set_facecolor(color)
        # Annotate means
        range_x_map = {3: 1, 5: 1.45, 6: 1.475, 7: 1.5, 8: 1.525}
        for i, data in enumerate(data_list):
            mean = np.mean(data)
            add_range_x = range_x_map[len(model_names_list)]
            plt.text(i+add_range_x, mean, f'{mean:.4f}', horizontalalignment='right', verticalalignment='center', fontsize=8, rotation=90)
        # Set xlabels
        plt.xticks(ticks=np.arange(1, len(model_names_list) + 1), labels=model_names_list, rotation=15, ha='center')
        # Annotate significance
        y_max = 1
        for i, model_name_1 in enumerate(model_names_list):
            for j, model_name_2 in enumerate(model_names_list):
                p_value_key = (model_name_1, model_name_2)
                if model_name_2 == "Ensemble":
                    p_value_key = (model_name_2, model_name_1)
                if p_value_key in aggregated_p_values:
                    p_value = aggregated_p_values[p_value_key][metric]
                    if p_value < 1e-4:
                        text = '****'
                    elif p_value >= 1e-4:
                        text = 'ns' if p_value > 0.05 else '*' * int(-np.log10(p_value))
                    else:
                        text = ""
                    if "DNABERT-Epi" in model_names_list:
                        if model_name_2 == "DNABERT" and model_name_1 not in ["DNABERT-Epi", "Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i), text)
                        elif model_name_2 == "DNABERT-Epi" and model_name_1 not in ["Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(model_names_list)-3) + 0.05 * (j - i), text)
                        elif model_name_2 == "Ensemble":
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * ((len(model_names_list)-2)+(len(model_names_list)-3)) + 0.05 * (j - i), text)
                    else:
                        if model_name_2 == "DNABERT" and model_name_1 not in ["Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i), text)
                        elif model_name_2 == "Ensemble":
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(model_names_list)-2) + 0.05 * (j - i), text)
        # Set y-axis range
        if "DNABERT-Epi" not in model_names_list:
            plt.ylim(0, y_max + 0.6)
        else:
            plt.ylim(0, y_max + 1.0)
        yticks = plt.yticks()[0] 
        yticks_under_1 = [y for y in yticks if y <= 1]
        if 1 not in yticks_under_1:
            yticks_under_1.append(1)
        yticks_under_1 = sorted(set(yticks_under_1))
        plt.yticks(yticks_under_1)
    
        # Save plot if save_dir is provided
        if save_dir:
            plt.savefig(save_dir + f"/{metric}.png", dpi=dpi, bbox_inches='tight')                
        # plt.show()


def plot_boxplot(config: dict, aggregated_results: dict, aggregated_p_values: dict, model_names_list: list, metrics: list, save_dir: str=None, dpi: int=300) -> None:
    # If model_names_list contains "Ensemble", "DNABERT", or "DNABERT-Epi", ensure the order is Other models, "DNABERT", "DNABERT-Epi", "Ensemble"
    if "Ensemble" in model_names_list or "DNABERT" in model_names_list or "DNABERT-Epi" in model_names_list:
        other_models = [m for m in model_names_list if m not in ["Ensemble", "DNABERT", "DNABERT-Epi"]]
        ordered_models = other_models
        if "DNABERT" in model_names_list:
            ordered_models.append("DNABERT")
        if "DNABERT-Epi" in model_names_list:
            ordered_models.append("DNABERT-Epi")
        if "Ensemble" in model_names_list:
            ordered_models.append("Ensemble")
        model_names_list = ordered_models
    
    for metric in [m for m in metrics if m!= "confusion_matrix"]:
        plt.figure(figsize=(6, 6))
        # Prepare data for plotting
        data_list = []
        for model_name in model_names_list:
            data_list.append(aggregated_results[model_name][metric])
        box = plt.boxplot(data_list, vert=True, patch_artist=True, widths=0.6, showmeans=True,
                            meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":2},
                            flierprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":2},
                            medianprops={"color":"black", "linewidth":1.5})
        # Apply colors to each box
        for patch, color in zip(box['boxes'], [config["colors"][model_name] for model_name in model_names_list]):
            patch.set_facecolor(color)
        # Annotate means
        range_x_map = {3: 1, 5: 1.45, 6: 1.475, 7: 1.5, 8: 1.525}
        for i, data in enumerate(data_list):
            mean = np.mean(data)
            add_range_x = range_x_map[len(model_names_list)]
            plt.text(i+add_range_x, mean, f'{mean:.4f}', horizontalalignment='right', verticalalignment='center', fontsize=8, rotation=90)
        # Set xlabels
        # plt.xticks(ticks=np.arange(1, len(model_names_list) + 1), labels=model_names_list, rotation=15, ha='center')
        plt.xticks(ticks=np.arange(1, len(model_names_list) + 1), labels=model_names_list, rotation=90, ha='center')
        # Annotate significance
        y_max, y_min = 0, 1
        for data in data_list:
            if max(data) > y_max:
                y_max = max(data)
            if min(data) < y_min:
                y_min = min(data)
        y_range = y_max - y_min
        for i, model_name_1 in enumerate(model_names_list):
            for j, model_name_2 in enumerate(model_names_list):
                p_value_key = (model_name_1, model_name_2)
                if model_name_2 == "Ensemble":
                    p_value_key = (model_name_2, model_name_1)
                if p_value_key in aggregated_p_values:
                    p_value = aggregated_p_values[p_value_key][metric]
                    if p_value < 1e-4:
                        text = '****'
                    elif p_value >= 1e-4:
                        text = 'ns' if p_value > 0.05 else '*' * int(-np.log10(p_value))
                    else:
                        text = ""
                    if "DNABERT-Epi" in model_names_list:
                        if model_name_2 == "DNABERT" and model_name_1 not in ["DNABERT-Epi", "Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i) * y_range, y_range, text)
                        elif model_name_2 == "DNABERT-Epi" and model_name_1 not in ["Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(model_names_list)-3)* y_range + 0.05 * (j - i) * y_range, y_range, text)
                        elif model_name_2 == "Ensemble":
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * ((len(model_names_list)-2)+(len(model_names_list)-3))* y_range + 0.05 * (j - i) * y_range, y_range, text)
                    else:
                        if model_name_2 == "DNABERT" and model_name_1 not in ["Ensemble"]:
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (j - i) * y_range, y_range, text)
                        elif model_name_2 == "Ensemble":
                            annotate_significance(plt.gca(), i + 1, j + 1, y_max + 0.05 * (len(model_names_list)-2)* y_range + 0.05 * (j - i) * y_range, y_range, text)
        # Set y-axis range
        if "DNABERT-Epi" not in model_names_list:
            plt.ylim(y_min, y_max + y_range* 0.6)
        else:
            plt.ylim(y_min, y_max + y_range* 1.0)
        yticks = plt.yticks()[0] 
        yticks_under_1 = [y for y in yticks if y <= y_max]
        yticks_under_1 = sorted(set(yticks_under_1))
        plt.yticks(yticks_under_1)
        plt.subplots_adjust(bottom=0.25) 
    
        # Save plot if save_dir is provided
        if save_dir:
            plt.savefig(save_dir + f"/{metric}.png", dpi=dpi)


def plot_ensemble_importance_barplot(config: dict, importance_scores: dict, save_dir: str=None, dpi: int=300) -> None:
    # Importance score
    base_score = importance_scores["all"]
    model_names = [name for name in importance_scores.keys() if name != "all"]
    contributions = {}
    contributions = {name: base_score - importance_scores[name] for name in model_names}
    sorted_items = sorted(contributions.items(), key=lambda x: x[1])
    model_names = [k for k, _ in sorted_items]
    deltas = [v for _, v in sorted_items]
    # Plot barplot
    plt.figure(figsize=(6, 4))
    bars = plt.barh(model_names, deltas, color="skyblue")
    plt.yticks(fontsize=14)
    plt.xticks([])
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + "/ensemble_importance.png", dpi=dpi, bbox_inches='tight')


