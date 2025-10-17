import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles



def plot_venn_diagram_CS_GS(
    venn_counts: dict,
    output_path: str,
    title: str="",
    figsize: tuple = (8, 8),
    fontsize: int = 12,
    alpha: float = 0.5,
    set_labels: tuple = ("CHANGE-seq", "GUIDE-seq"),
    set_colors: tuple = ("#1f77b4", "#ff7f0e")
    ) -> None:
    
    subset_sizes = (
        venn_counts["change_seq_only_count"],
        venn_counts["guide_seq_only_count"],
        venn_counts["intersection_count"]
    )
    subset_sizes_ = (
        venn_counts["change_seq_only_count"] // 20,
        venn_counts["guide_seq_only_count"],
        venn_counts["intersection_count"]
    )
    
    # Plot Venn diagram
    plt.figure(figsize=figsize)
    venn = venn2(subsets=subset_sizes_, set_labels=set_labels, set_colors=set_colors, alpha=alpha)
    if venn.set_labels[0]: # CHANGE-seq
        venn.set_labels[0].set_fontsize(fontsize)
    if venn.set_labels[1]: # GUIDE-seq
        venn.set_labels[1].set_fontsize(fontsize)
    
    if venn.get_label_by_id('10'):
        venn.get_label_by_id('10').set_text(str(subset_sizes[0]))
        venn.get_label_by_id('10').set_fontsize(fontsize)
    if venn.get_label_by_id('01'):
        venn.get_label_by_id('01').set_text(str(subset_sizes[1]))
        venn.get_label_by_id('01').set_fontsize(fontsize)
    if venn.get_label_by_id('11'):
        venn.get_label_by_id('11').set_text(str(subset_sizes[2]))
        venn.get_label_by_id('11').set_fontsize(fontsize)
    
    plt.title(title, fontsize=fontsize)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    # plt.show()