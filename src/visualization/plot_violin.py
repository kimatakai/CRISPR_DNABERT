import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats



def plot_violin(
    config: dict,
    data_list: list,
    y_labels: str,
    labels: list,
    colors: list,
    p_values: list,
    title: str="",
    fig_path: str=None,
    ):
    # Data -> Pandas DataFrame
    df_data = pd.DataFrame()
    for i, data in enumerate(data_list):
        # Remove outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = [x for x in data if lower_bound <= x <= upper_bound]
        temp_df = pd.DataFrame({"Data": data, "Label": [labels[i]] * len(data)})
        df_data = pd.concat([df_data, temp_df], ignore_index=True)
    
    # Plot
    plt.figure(figsize=(max(6, 2 * len(data_list)), 6)) 
    sns.set(style="whitegrid")

    # Create a violin plot
    for i in range(len(labels)):
        sns.violinplot(x="Label", y="Data", data=df_data[df_data["Label"] == labels[i]], inner="quartile", color=colors[i], linewidth=1.25, label=labels[i], alpha=0.7)
    
    # Set labels and title
    plt.xlabel("")
    plt.xticks([])
    plt.ylabel(y_labels)
    violin_ylim = plt.gca().get_ylim()
    y_max = violin_ylim[1]
    plt.ylim(violin_ylim[0], y_max * 1.4)
    plt.legend(title="Groups", loc='upper left')
    plt.title(title)
    
    # Plot significance annotation
    p_value = p_values[0]
    if p_value > 0.05:
        text = "ns"
    elif 0.01 < p_value and p_value <= 0.05:
        text = "*"
    elif 0.001 < p_value and p_value <= 0.01:
        text = "**"
    elif 0.0001 < p_value and p_value <= 0.001:
        text = "***"
    elif p_value <= 0.0001:
        text = "****"
    x1_pos = 0
    x2_pos = 1
    y_range = y_max * 0.05
    bar_top_y = y_max * 1.05
    bar_bottom_y = bar_top_y - y_range
    y_text_pos = bar_top_y + y_range * 0.05
    plt.plot([x1_pos, x2_pos], [bar_top_y, bar_top_y], color='black', lw=1.5)
    plt.plot([x1_pos, x1_pos], [bar_bottom_y, bar_top_y], color='black', lw=1.5)
    plt.plot([x2_pos, x2_pos], [bar_bottom_y, bar_top_y], color='black', lw=1.5)
    plt.text((x1_pos + x2_pos) / 2, y_text_pos, text, ha='center', va='center', fontsize=12, color='black')
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)