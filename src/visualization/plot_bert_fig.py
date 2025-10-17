
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import umap
import seaborn as sns


def plot_attention_weights(attention_array: np.ndarray, pair_seq_len: int, kmer: int, layer: int, mode: str="diff", title: str=None, save_path: str=None) -> None:
    # Attention array shape: (k_layer, seq_len, seq_len) = (6, 47, 47)
    # Y-axis: From 3' to 5' (top to bottom)
    y_labels = ["[CLS]"] + [f"RNA {i+1}" for i in range(pair_seq_len-kmer+1)] + ["[SEP]"] + [f"DNA {i+1}" for i in range(pair_seq_len-kmer+1)] + ["[SEP]"]
    print(len(y_labels), attention_array.shape[1])
    print(y_labels)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 9))
    ax2 = ax1.twinx()
    
    # Set axis information
    ax1.set_yticks(np.arange(len(y_labels)))
    ax2.set_yticks(np.arange(len(y_labels)))
    ax1.set_ylim(-1, len(y_labels))
    ax2.set_ylim(-1, len(y_labels))
    ax1.set_yticklabels(y_labels[::-1], fontsize=12)
    ax2.set_yticklabels(y_labels[::-1], fontsize=12)
    ax1.set_xticks(np.arange(12-layer, 12+1))
    ax1.set_xlim(12-layer -0.1, 12 + 0.1)
    
    # Value scale
    if mode == "diff":
        max_value = np.max(np.abs(attention_array))
        attention_array = attention_array / max_value  # Normalize to -1 to 1
        print(attention_array.shape)
    
    # Draw Attention weights
    for n_layer in range(layer):
        n_attention = attention_array[n_layer, :, :]  # Shape: (seq_len, seq_len)
        for i in range(len(y_labels)):
            for j in range(len(y_labels)):
                if mode == "diff":
                    if n_attention[i, j] > 0:
                        ax1.plot([12-layer+n_layer, 12-layer+n_layer + 0.9], [i, j], color="red", alpha=n_attention[i, j])
                    elif n_attention[i, j] < 0:
                        ax1.plot([12-layer+n_layer, 12-layer+n_layer + 0.9], [i, j], color="blue", alpha=-n_attention[i, j])
                elif mode == "pvalue":
                    if n_attention[i, j] < 0.001:
                        ax1.plot([12-layer+n_layer, 12-layer+n_layer + 0.9], [i, j], color="red", alpha=0.5, lw=2)
                    elif n_attention[i, j] < 0.01:
                        ax1.plot([12-layer+n_layer, 12-layer+n_layer + 0.9], [i, j], color="orange", alpha=0.5, lw=1)
                    elif n_attention[i, j] < 0.05:
                        ax1.plot([12-layer+n_layer, 12-layer+n_layer + 0.9], [i, j], color="green", alpha=0.5, lw=0.5)
    if mode == "diff":
        red_patch = mpatches.Patch(color='red', label='Active OTS Attention')
        blue_patch = mpatches.Patch(color='blue', label='Inactive OTS Attention')
        plt.legend(handles=[red_patch, blue_patch], loc='upper left', fontsize=16)
    elif mode == "pvalue":
        red_patch = mpatches.Patch(color='red', label='p < 0.001')
        orange_patch = mpatches.Patch(color='orange', label='0.001 < p < 0.01')
        yellow_patch = mpatches.Patch(color='green', label='0.01 < p < 0.05')
        plt.legend(handles=[red_patch, orange_patch, yellow_patch], loc='upper left', fontsize=16)
    
    plt.title(title, fontsize=16)
    plt.grid(False)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

         
def plot_token_importance(all_token_importance: dict, seq_list: str, pair_seq_len: int, kmer: int, hotspot_sgrna: list, title: str=None, save_path: str=None) -> None:
    # Token importance shape: (seq_len,) = (47,)
    n_cols = pair_seq_len - kmer + 1
    fig, axs = plt.subplots(len(seq_list), 1, figsize=(15, 2.5 * len(seq_list)),
                            constrained_layout=False)
    plt.subplots_adjust(hspace=1.0)
    if len(seq_list) == 1:
        axs = [axs]

    ims = []
    for i, _sgrna in enumerate(seq_list):
        if _sgrna in hotspot_sgrna:
            text = r"$\mathbf{" + _sgrna + "}$" # bold
        else:
            text = _sgrna
        
        ax_hm = axs[i]

        token_importance = all_token_importance[_sgrna][pair_seq_len - kmer + 1 + 2: -1].reshape(1, -1)
        row = token_importance / np.max(np.abs(token_importance))
        vmin, vmax = np.min(row), np.max(row)

        im = ax_hm.imshow(row, aspect='auto', cmap='Reds', vmin=vmin, vmax=vmax)
        ims.append(im)

        ax_hm.text(0.5, 1.05, text, transform=ax_hm.transAxes,
                   ha='center', va='bottom', fontsize=18)

        ax_hm.set_xticks(np.arange(n_cols))
        ax_hm.set_xticklabels([f"DNA{j+1}" for j in range(n_cols-1)]+["PAM"], fontsize=10)
        ax_hm.set_yticks([])


    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_ig_importance_umap(importance_array: np.ndarray, attention_sgrna: list, title: str=None, save_path: str=None) -> None:
    # importance_array shape: (n_sgrna, seq_len) = (n_sgrna, 47)
    # attention_sgrna shape: (n_sgrna,) = (n_sgrna,) = [0 or 1]
    # UMAP reduction (attention1 -> red, attention0 -> gray)
    reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.1, metric='correlation')
    embedding = reducer.fit_transform(importance_array)  # shape: (n_sgrna, 2)
    print(embedding.shape)
    plt.figure(figsize=(6, 6))
    attention_sgrna = np.array(attention_sgrna)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=attention_sgrna, palette={0: 'gray', 1: 'red', 2: 'blue'}, s=100, alpha=0.7, legend=False)
    # plt.legend([],[], frameon=False) 
    plt.title(title, fontsize=16)
    plt.xlabel("UMAP 1", fontsize=14)
    plt.xticks([])
    plt.ylabel("UMAP 2", fontsize=14)
    plt.yticks([])
    # plt.legend(title="", labels=["Hotspot Pattern Observed", "Hotspot Pattern Not Observed"], fontsize=10, title_fontsize=14, loc='lower left')
    handles = [
        mpatches.Patch(color='gray', label="No Hotspot Pattern"),
        mpatches.Patch(color='red',  label="Hotspot (PAM-distal)"),
        mpatches.Patch(color='blue', label="Hotspot (PAM-proximal)")
    ]
    plt.legend(handles=handles, fontsize=10, title_fontsize=14, loc='lower left', bbox_to_anchor=(0, 0.4))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
def _plot_ig_importance_umap(importance_array: np.ndarray, attention_sgrna: list, title: str=None, save_path: str=None) -> None:
    # importance_array shape: (n_sgrna, seq_len) = (n_sgrna, 47)
    # UMAP reduction
    reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.1, metric='correlation')
    embedding = reducer.fit_transform(importance_array)  # shape: (n_sgrna, 2)
    
    # Color calculation
    red_strength = np.sum(importance_array[:, [3,4,5]], axis=1)  # PAM-distal
    blue_strength = np.sum(importance_array[:, [13,14,15]], axis=1)  # PAM-proximal
    # Normalize to 0-1
    if np.max(red_strength) > 0:
        red_strength = red_strength / np.max(red_strength)
    if np.max(blue_strength) > 0:
        blue_strength = blue_strength / np.max(blue_strength)
    # RGB color
    colors = np.zeros((importance_array.shape[0], 3))
    colors[:, 0] = red_strength  # Red channel
    colors[:, 2] = blue_strength  # Blue channel
    colors = np.clip(colors,0,1)
    print(colors)
    
    print(embedding.shape)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x=embedding[:, 0], y=embedding[:, 1], c=colors, s=100, alpha=0.7)
    plt.xlabel("UMAP 1", fontsize=14)
    plt.xticks([])
    plt.ylabel("UMAP 2", fontsize=14)
    plt.yticks([])
    # cbar = plt.colorbar(sc)
    # cbar.set_ticks([])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_mismatch_pattern(mismatch_array: np.ndarray, sgrna_list: list, max_pairseq_len: int, title: str=None, save_path: str=None) -> None:
    # mismatch_array shape: (mismatch types, max_pairseq_len) = (12, 24)
    n_cols = max_pairseq_len
    fig, axs = plt.subplots(len(sgrna_list), 1, figsize=(15, 3.0 * len(sgrna_list)),
                            constrained_layout=False)
    plt.subplots_adjust(hspace=1.0)
    
    if len(sgrna_list) == 1:
        axs = [axs]
    
    ims = []
    for i, _sgrna in enumerate(sgrna_list):
        text = _sgrna
        ax = axs[i]
        data = mismatch_array[i] # shape: (12, 24)
        
        vmax = np.max(data) if np.max(data) > 0 else 1
        im = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
        
        ims.append(im)
        ax.text(0.5, 1.05, text, transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=20)
        ax.set_xticks(np.arange(0, max_pairseq_len, 1))
        ax.set_xticklabels([f"{j+1}" for j in range(0, max_pairseq_len, 1)], fontsize=12, rotation=0)
        ax.set_yticks(np.arange(12))
        ax.set_yticklabels([
            "rA:dC","rA:dG","rA:dT",
            "rC:dA","rC:dG","rC:dT",
            "rG:dA","rG:dC","rG:dT",
            "rT:dA","rT:dC","rT:dG"
        ], fontsize=12)
        
    # fig.colorbar(ims[0], ax=axs, orientation="vertical", fraction=0.02, pad=0.01, label="Mismatch frequency")
    
    if title:
        ax.set_title(f"{title} - {_sgrna}", fontsize=16)
        
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
        