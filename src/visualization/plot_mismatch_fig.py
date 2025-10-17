

import matplotlib.pyplot as plt
import numpy as np
import itertools



def plot_mismatch_bar_graph(mismatch_data: dict, titles: list, save_path: str=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    cm_keys = ["TN", "FP", "FN", "TP"]
    
    # y-axis scaling
    max_ = 0
    max_val = 0
    for cm_key in cm_keys:
        for count in mismatch_data[cm_key].values():
            log_count = np.log10(count + 1)
            if log_count > max_val:
                max_val = log_count
            if count > max_:
                max_ = count
    
    # Plot each confusion matrix category
    for i, cm_key in enumerate(cm_keys):
        ax = axes[i]
        mismatches = np.arange(7)
        
        # No bulge (bulge=0)
        counts_no_bulge = [mismatch_data[cm_key].get((mismatch, 0), 0) for mismatch in mismatches]
        log_counts_no_bulge = np.log10(np.array(counts_no_bulge) + 1)
        
        # With bulge (bulge=1)
        counts_with_bulge = [mismatch_data[cm_key].get((mismatch, 1), 0) for mismatch in mismatches]
        log_counts_with_bulge = np.log10(np.array(counts_with_bulge) + 1)
        
        # Bar width
        bar_width = 0.4
        
        # Upper (No Bulge)
        bars_upper = ax.bar(mismatches, log_counts_no_bulge, bar_width, color='skyblue', label='Bulge: No')
        # Lower (With Bulge)
        bars_lower = ax.bar(mismatches, [-val if val != -np.inf else 0 for val in log_counts_with_bulge], bar_width, color='lightcoral', label='Bulge: Yes')
        
        ax.set_title(titles[i])
        ax.set_xticks(mismatches)
        ax.set_xticklabels([str(m) for m in mismatches])
        ax.set_xlabel('Mismatch Count')
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
        # Set y-axis limits
        ax.set_ylim(-max_val - 1, max_val + 1)
        
        # Center x-line
        ax.axhline(0, color='black', linewidth=0.8)
        
    plt.tight_layout()
    # plt.legend(loc='upper right')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def plot_mismatch_frequency(mismatch_data: dict, config: dict, titles: list, save_path: str=None) -> None:
    max_pairseq_len = config["parameters"]["max_pairseq_len"]
    
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    axes = axes.flatten()
    cm_keys = ["TN", "FP", "FN", "TP"]
    
    bases = ["A", "T", "C", "G"]
    mismatch_types = []
    for base1 in bases:
        for base2 in bases:
            if base1 != base2:
                mismatch_types.append(f"r{base1}>d{base2}")
    num_rows = len(mismatch_types)
    
    for idx, cm_key in enumerate(cm_keys):
        heatmap_data = np.zeros((num_rows, max_pairseq_len))
        row_label_to_idx = {label: i for i, label in enumerate(mismatch_types)}
        
        for rna_seq, dna_seq in zip(mismatch_data[cm_key]["rna_seq"], mismatch_data[cm_key]["dna_seq"]):
            for i in range(max_pairseq_len):
                nt1 = rna_seq[i]
                nt2 = dna_seq[i]
                if nt1 == "-" or nt2 == "-":
                    continue
                elif nt1 != nt2:
                    mismatch_type = f"r{nt1}>d{nt2}"
                    if mismatch_type in row_label_to_idx:
                        heatmap_data[row_label_to_idx[mismatch_type], i] += 1
        
        ax = axes[idx]
        cax = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(np.arange(max_pairseq_len))
        ax.set_xticklabels(np.arange(max_pairseq_len))
        ax.set_xlabel('Position (nt)')
        
        ax.set_yticks(np.arange(num_rows))
        ax.set_yticklabels(mismatch_types)
        ax.set_ylabel('Mismatch Type')
        
        ax.set_title(f'{cm_key} Mismatch Frequency')
    
    # fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:    
        plt.show()
    return None