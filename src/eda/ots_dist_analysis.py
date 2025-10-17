
import os
import pysam
import numpy as np
import tqdm
import subprocess
from collections import defaultdict
from scipy.stats import mannwhitneyu
from Bio import SeqIO

import visualization.plot_venn_diagram as plot_venn_diagram
import visualization.plot_dsb_count as plot_dsb_count
import visualization.plot_violin as plot_violin
import visualization.plot_linecurve as plot_linecurve
import tools.meme as meme
import utils.file_handlers as file_handlers
import utils.fasta_handlers as fasta_handlers

"""
Dataset columns
sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,reads
"""


def return_common_dataset_CS_GS(config: dict) -> dict:
    Lazzarotto_2020_GUIDE_seq_sgrna_list_old_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_sgrna_list_old"]
    Lazzarotto_2020_GUIDE_seq_sgrna_list = file_handlers.load_csv_list(Lazzarotto_2020_GUIDE_seq_sgrna_list_old_path)

    # Load CHANGE-seq and GUIDE-seq dataset
    Lazzarotto_2020_CHANGE_seq_dataset_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_dataset"]
    Lazzarotto_2020_GUIDE_seq_dataset_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_dataset"]
    Lazzarotto_2020_CHANGE_seq_dataset_df = file_handlers.load_csv_dataset(Lazzarotto_2020_CHANGE_seq_dataset_path) # shape: (4936278, 11)
    Lazzarotto_2020_GUIDE_seq_dataset_df = file_handlers.load_csv_dataset(Lazzarotto_2020_GUIDE_seq_dataset_path)   # shape: (3271049, 11)
    
    # Filter CHANGE-seq dataset to include only GUIDE-seq sgRNAs
    Lazzarotto_2020_CHANGE_seq_dataset_df = Lazzarotto_2020_CHANGE_seq_dataset_df[
        Lazzarotto_2020_CHANGE_seq_dataset_df["sgRNA"].isin(Lazzarotto_2020_GUIDE_seq_sgrna_list)]  # shape: (2094659, 11)
    Lazzarotto_2020_GUIDE_seq_dataset_df  = Lazzarotto_2020_GUIDE_seq_dataset_df[
        Lazzarotto_2020_GUIDE_seq_dataset_df["sgRNA"].isin(Lazzarotto_2020_GUIDE_seq_sgrna_list)]   # shape: (2120340, 11)
    
    return {
        "Lazzarotto_2020_CHANGE_seq_dataset_df": Lazzarotto_2020_CHANGE_seq_dataset_df,
        "Lazzarotto_2020_GUIDE_seq_dataset_df": Lazzarotto_2020_GUIDE_seq_dataset_df
        }


def venn_analysis_between_CS_GS(config: dict) -> dict:
    """
    Perform Venn diagram analysis between Lazzarotto_2020_CHANGE-seq and Lazzarotto_2020_GUIDE-seq datasets.

    :param config: Configuration dictionary containing paths and parameters.
    :return: Dictionary with results of the Venn diagram analysis.
    """
    # Load CHANGE-seq and GUIDE-seq dataset
    Lazzarotto_2020_CHANGE_seq_dataset_df, Lazzarotto_2020_GUIDE_seq_dataset_df = return_common_dataset_CS_GS(config).values()
    
    # Filter "reads" col > 0 (active OTS)
    Lazzarotto_2020_CHANGE_seq_dataset_df = Lazzarotto_2020_CHANGE_seq_dataset_df[
        Lazzarotto_2020_CHANGE_seq_dataset_df["reads"] > 0]  # shape: (62798, 11)
    Lazzarotto_2020_GUIDE_seq_dataset_df = Lazzarotto_2020_GUIDE_seq_dataset_df[
        Lazzarotto_2020_GUIDE_seq_dataset_df["reads"] > 0]  # shape: (1520, 11)
    
    # Prepare data for Venn diagram analysis
    # set (sgRNA, chhrom, strand, Align.chromStart)
    Lazzarotto_2020_CHANGE_seq_set = set(
        zip(Lazzarotto_2020_CHANGE_seq_dataset_df["sgRNA"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["chrom"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["Align.strand"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["Align.chromStart"])
    )
    Lazzarotto_2020_GUIDE_seq_set = set(
        zip(Lazzarotto_2020_GUIDE_seq_dataset_df["sgRNA"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["chrom"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["Align.strand"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["Align.chromStart"])
    )
    
    # Calculate intersection and differences
    intersection = Lazzarotto_2020_CHANGE_seq_set.intersection(Lazzarotto_2020_GUIDE_seq_set)
    change_seq_only = Lazzarotto_2020_CHANGE_seq_set - intersection
    guide_seq_only = Lazzarotto_2020_GUIDE_seq_set - intersection
    
    # Prepare counts for Venn diagram
    venn_counts = {
        "change_seq_only_count": len(change_seq_only),
        "guide_seq_only_count": len(guide_seq_only),
        "intersection_count": len(intersection)
    }
    
    # Plot Venn diagram
    output_dir_path = config["paths"]["figure"]["eda"]
    os.makedirs(output_dir_path, exist_ok=True)
    output_path = output_dir_path + "/venn_diagram_CS_GS.png"
    plot_venn_diagram.plot_venn_diagram_CS_GS(
        venn_counts=venn_counts,
        output_path=output_path,
        title="",
        figsize=(8, 8),
        fontsize=16,
        alpha=0.3
    )


def save_fasta_common_between_CS_GS(config: dict) -> None:
    """
    Perform sequence analysis between Lazzarotto_2020_CHANGE-seq and Lazzarotto_2020_GUIDE-seq datasets.

    :param config: Configuration dictionary containing paths and parameters.
    """
    # Load CHANGE-seq and GUIDE-seq dataset
    Lazzarotto_2020_CHANGE_seq_dataset_df, Lazzarotto_2020_GUIDE_seq_dataset_df = return_common_dataset_CS_GS(config).values()
    
    # Filter "reads" col > 0 (active OTS)
    Lazzarotto_2020_CHANGE_seq_dataset_df = Lazzarotto_2020_CHANGE_seq_dataset_df[
        Lazzarotto_2020_CHANGE_seq_dataset_df["reads"] > 0]  # shape: (62798, 11)
    Lazzarotto_2020_GUIDE_seq_dataset_df = Lazzarotto_2020_GUIDE_seq_dataset_df[
        Lazzarotto_2020_GUIDE_seq_dataset_df["reads"] > 0]  # shape: (1520, 11)
    
    # Lazzarotto_2020_GUIDE_seq_dataset_dfについて、sgRNAのカウントを集計
    Lazzarotto_2020_CHANGE_seq_sgrna_count = Lazzarotto_2020_CHANGE_seq_dataset_df["sgRNA"].value_counts()
    Lazzarotto_2020_GUIDE_seq_sgrna_count = Lazzarotto_2020_GUIDE_seq_dataset_df["sgRNA"].value_counts()
    # カウント数上位と下位を表示
    print(Lazzarotto_2020_CHANGE_seq_sgrna_count.head(30))
    print(Lazzarotto_2020_GUIDE_seq_sgrna_count.head(30))
    print(Lazzarotto_2020_CHANGE_seq_sgrna_count.tail(30))
    print(Lazzarotto_2020_GUIDE_seq_sgrna_count.tail(30))
    
    # 各sgRNAでCHANGE-seqの数/GUIDE-seqの数を集計
    Lazzarotto_2020_GUIDE_seq_sgrna_count = Lazzarotto_2020_GUIDE_seq_sgrna_count.to_dict()
    Lazzarotto_2020_CHANGE_seq_sgrna_count = Lazzarotto_2020_CHANGE_seq_sgrna_count.to_dict()
    Lazzarotto_2020_GUIDE_seq_sgrna_ratio = {}
    for sgrna in Lazzarotto_2020_GUIDE_seq_sgrna_count.keys():
        if sgrna in Lazzarotto_2020_CHANGE_seq_sgrna_count:
            Lazzarotto_2020_GUIDE_seq_sgrna_ratio[sgrna] = Lazzarotto_2020_CHANGE_seq_sgrna_count[sgrna] / Lazzarotto_2020_GUIDE_seq_sgrna_count[sgrna]
        else:
            Lazzarotto_2020_GUIDE_seq_sgrna_ratio[sgrna] = 0.0
    print("Lazzarotto_2020_GUIDE_seq_sgrna_ratio:")
    print(Lazzarotto_2020_GUIDE_seq_sgrna_ratio)
    # 一番割合が小さいのは
    print("Minimum ratio:", min(Lazzarotto_2020_GUIDE_seq_sgrna_ratio.values()))
    
    # Prepare data for sequence analysis
    # set (sgRNA, chhrom, strand, Align.chromStart)
    Lazzarotto_2020_CHANGE_seq_set = set(
        zip(Lazzarotto_2020_CHANGE_seq_dataset_df["sgRNA"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["chrom"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["Align.strand"],
            Lazzarotto_2020_CHANGE_seq_dataset_df["Align.chromStart"])
    )
    Lazzarotto_2020_GUIDE_seq_set = set(
        zip(Lazzarotto_2020_GUIDE_seq_dataset_df["sgRNA"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["chrom"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["Align.strand"],
            Lazzarotto_2020_GUIDE_seq_dataset_df["Align.chromStart"])
    )
    Lazzarotto_2020_only_CHANGE_seq_set = Lazzarotto_2020_CHANGE_seq_set - Lazzarotto_2020_GUIDE_seq_set
    Lazzarotto_2020_common_set = Lazzarotto_2020_CHANGE_seq_set.intersection(Lazzarotto_2020_GUIDE_seq_set)
    
    # Fasta object
    fasta_path = config["paths"]["reference_genome"]["hg38"]
    reference_fa = pysam.FastaFile(fasta_path)
    chrom_size = fasta_handlers.return_chrom_size(reference_fa)
    
    # Save flanking sequences
    if not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1"]) or not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r2"]):
        fasta_handlers.save_flanking_sequence_fasta(
            reference_fa=reference_fa,
            off_target_data=Lazzarotto_2020_only_CHANGE_seq_set,
            chrom_size=chrom_size,
            window_size=config["parameters"]["eda_window_size"],
            output_fasta_path=[config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1"], config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r2"]]
        )
    if not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1"]) or not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r2"]):
        fasta_handlers.save_flanking_sequence_fasta(
            reference_fa=reference_fa,
            off_target_data=Lazzarotto_2020_common_set,
            chrom_size=chrom_size,
            window_size=config["parameters"]["eda_window_size"],
            output_fasta_path=[config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1"], config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r2"]]
        )
    
    # Concat the two FASTA files into one
    if not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_all"]):
        with open(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_all"], "w") as outfile:
            # Forward strand sequences
            for record in SeqIO.parse(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1"], "fasta"):
                record.id = record.id + "_forward"
                record.description = record.id
                SeqIO.write(record, outfile, "fasta")
            # Reverse strand sequences
            for record in SeqIO.parse(config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r2"], "fasta"):
                SeqIO.write(record, outfile, "fasta")

    if not os.path.exists(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_all"]):
        with open(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_all"], "w") as outfile:
            # Forward strand sequences
            for record in SeqIO.parse(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1"], "fasta"):
                SeqIO.write(record, outfile, "fasta")
            # Reverse strand sequences
            for record in SeqIO.parse(config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r2"], "fasta"):
                SeqIO.write(record, outfile, "fasta")
    
    # Close the FASTA file
    reference_fa.close()


def calculate_gc_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) * 100


def calculate_cpg_oe_ratio(sequence: str) -> float:
    if len(sequence) < 2:
        return 0.0
    # Count CpG dinucleotides
    observed_cpg = sequence.count('CG')
    # Count C and G separately
    C_count = sequence.count('C')
    G_count = sequence.count('G')
    total_bases = len(sequence)
    
    # Calculate expected CpG count
    if C_count == 0 or G_count == 0 or total_bases < 2:
        return 0.0
    
    expected_cpg = (C_count / total_bases) * (G_count / total_bases) * (total_bases - 1)
    if expected_cpg == 0:
        return 0.0
    return observed_cpg / expected_cpg if expected_cpg > 0 else 0.0


def return_cpg_matrix(config: dict, sequence_list: list) -> np.ndarray:
    cpg_oe_matrix = np.zeros((len(sequence_list), (config["parameters"]["eda_window_size"]*2 - config["parameters"]["cpg_sliding_window_size"]) // config["parameters"]["cpg_sliding_window_step"] + 1))
    # Iterate through each sequence and calculate CpG O/E ratio
    for i, sequence in enumerate(sequence_list):
        # Calculate CpG O/E ratio for each sliding window
        for j in range(0, len(sequence) - config["parameters"]["cpg_sliding_window_size"] + 1, config["parameters"]["cpg_sliding_window_step"]):
            window_sequence = sequence[j:j + config["parameters"]["cpg_sliding_window_size"]]
            cpg_oe_ratio = calculate_cpg_oe_ratio(window_sequence)
            cpg_oe_matrix[i, j // config["parameters"]["cpg_sliding_window_step"]] = cpg_oe_ratio
    return cpg_oe_matrix

def remove_similar_sequences_90_percent(sequence_list: list) -> list:
    """
    Remove sequences that are 90% similar to each other.
    Calculates similarity based on the shorter sequence length for matching characters.
    """
    unique_sequences = []
    for seq in tqdm.tqdm(sequence_list, total=len(sequence_list), desc="Removing similar sequences"):
        is_similar = False
        for unique_seq in unique_sequences:
            # Calculate similarity as the fraction of matching characters in the common prefix
            # This is still not robust for insertions/deletions
            min_len = min(len(seq), len(unique_seq))
            if min_len == 0: # Handle empty sequences
                similarity = 1.0 if len(seq) == len(unique_seq) else 0.0
            else:
                similarity = sum(1 for a, b in zip(seq[:min_len], unique_seq[:min_len]) if a == b) / min_len

            # Or, if you want to consider overall length (more like Jaccard index for set of positions)
            # This is also not ideal for sequence similarity
            # intersection = sum(1 for a, b in zip(seq, unique_seq) if a == b)
            # union = max(len(seq), len(unique_seq)) # or len(seq) + len(unique_seq) - intersection for true union
            # similarity = intersection / union if union > 0 else 0.0

            if similarity >= 0.9:
                is_similar = True
                break
        if not is_similar:
            unique_sequences.append(seq)
    return unique_sequences


def eda_cpg_site_analysis(config: dict, fasta_filepaths: list[str], output_path_prefix: str, labels: list, colors: list) -> None:
    # Check the number of filepaths
    if len(fasta_filepaths) != 2:
        raise ValueError("Two filepaths are required for the analysis.")
    
    # Load the sequences from the FASTA files
    flanking_sequences_dict_1 = fasta_handlers.load_fasta_file(fasta_filepaths[0])
    flanking_sequences_dict_2 = fasta_handlers.load_fasta_file(fasta_filepaths[1])
    print(f"Loaded {len(flanking_sequences_dict_1['seq'])} sequences from {fasta_filepaths[0]}")
    print(f"Loaded {len(flanking_sequences_dict_2['seq'])} sequences from {fasta_filepaths[1]}")
    # Remove duplicate sequences
    # flanking_sequences_dict_1["seq"] = remove_similiar_sequences_90_percent(flanking_sequences_dict_1["seq"])
    flanking_sequences_dict_2["seq"] = remove_similar_sequences_90_percent(flanking_sequences_dict_2["seq"])
    print(f"After removing duplicates, {len(flanking_sequences_dict_1['seq'])} sequences remain in {fasta_filepaths[0]}")
    print(f"After removing duplicates, {len(flanking_sequences_dict_2['seq'])} sequences remain in {fasta_filepaths[1]}")
    
    # Extract CpG O/E ratios for group
    cpg_oe_matrix_1 = return_cpg_matrix(config, flanking_sequences_dict_1["seq"])
    cpg_oe_matrix_2 = return_cpg_matrix(config, flanking_sequences_dict_2["seq"])
    
    # ランダムに100サンプルずつ取り出し
    cpg_oe_matrix_1 = cpg_oe_matrix_1[np.random.choice(cpg_oe_matrix_1.shape[0], 60000, replace=False), :]
    cpg_oe_matrix_2 = cpg_oe_matrix_2[np.random.choice(cpg_oe_matrix_2.shape[0], 300, replace=False), :]
    
    # Plot mean of CpG O/E ratio violin plot
    u_statistic, p_value = mannwhitneyu(np.mean(cpg_oe_matrix_2, axis=1).flatten(), np.mean(cpg_oe_matrix_1, axis=1).flatten(), alternative='greater')
    print(f"Mann-Whitney U test: U-statistic = {u_statistic}, p-value = {p_value}")
    plot_violin.plot_violin(
        config=config,
        data_list=[np.mean(cpg_oe_matrix_1, axis=1), np.mean(cpg_oe_matrix_2, axis=1)],
        y_labels="CpG O/E ratio",
        labels=labels,
        colors=colors,
        p_values=[p_value],
        fig_path=output_path_prefix + "_viloin_plot.png"
    )
    
    # Plot median of CpG O/E ratio line plot
    plot_linecurve.plot_linegraph_cpg(
        config=config,
        data_list=[np.mean(cpg_oe_matrix_1, axis=0), np.mean(cpg_oe_matrix_2, axis=0)],
        y_labels="CpG O/E ratio",
        labels=labels,
        colors=colors,
        fig_path=output_path_prefix + "_position_plot.png"
    )

    
def eda_seauence_common_between_CS_GS(config: dict) -> None:
    # Save flanking sequences for CHANGE-seq and GUIDE-seq datasets
    save_fasta_common_between_CS_GS(config)
    
    # Load Only CHANGE-seq and Common Fasta files
    Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1"]
    Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r2_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r2"]
    Lazzarotto_2020_CHANGE_seq_CSGS_flanking_all_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_CHANGE_seq_CSGS_flanking_all"]
    Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1"]
    Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r2_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r2"]
    Lazzarotto_2020_GUIDE_seq_CSGS_flanking_all_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_CSGS_flanking_all"]
    
    eda_cpg_site_analysis(
        config = config, 
        fasta_filepaths = [Lazzarotto_2020_CHANGE_seq_CSGS_flanking_r1_path, Lazzarotto_2020_GUIDE_seq_CSGS_flanking_r1_path],
        output_path_prefix = config["paths"]["figure"]["eda"] + "/CS_GS_common",
        labels=["in vitro-specific OTS", "Common OTS"],
        colors=[config["colors"]["only_CS"], config["colors"]["common_CS_GS"]]
        )
    
    # 
    # meme.run_meme(
    #     config=config,
    #     input_fasta=Lazzarotto_2020_CHANGE_seq_CSGS_flanking_all_path,
    #     output_dir=config["paths"]["motif"]["Lazzarotto_2020_CHANGE_seq_CSGS_dir"]
    # )
    # meme.run_meme(
    #     config=config,
    #     input_fasta=Lazzarotto_2020_GUIDE_seq_CSGS_flanking_all_path,
    #     output_dir=config["paths"]["motif"]["Lazzarotto_2020_GUIDE_seq_CSGS_dir"]
    # )


def dsb_count_analysis(config: dict) -> None:
    """
    Perform DSB count analysis for all datasets.

    :param config: Configuration dictionary containing paths and parameters.
    """
    # Load GUIDE-seq dataset
    Lazzarotto_2020_GUIDE_seq_sgrna_list_old_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_sgrna_list_old"]
    Lazzarotto_2020_GUIDE_seq_sgrna_list = file_handlers.load_csv_list(Lazzarotto_2020_GUIDE_seq_sgrna_list_old_path)
    Lazzarotto_2020_GUIDE_seq_dataset_path = config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq_dataset"]
    Lazzarotto_2020_GUIDE_seq_dataset_df = file_handlers.load_csv_dataset(Lazzarotto_2020_GUIDE_seq_dataset_path)
    
    # Load TTISS dataset
    SchmidBurgk_2020_TTISS_sgrna_list_path = config["paths"]["off_target_dataset"]["SchmidBurgk_2020_TTISS_sgrna_list"]
    SchmidBurgk_2020_TTISS_dataset_path = config["paths"]["off_target_dataset"]["SchmidBurgk_2020_TTISS_dataset"]
    SchmidBurgk_2020_TTISS_dataset_df = file_handlers.load_csv_dataset(SchmidBurgk_2020_TTISS_dataset_path)

    # Filter DSB count
    Lazzarotto_2020_GUIDE_seq_dataset_df = Lazzarotto_2020_GUIDE_seq_dataset_df[
        Lazzarotto_2020_GUIDE_seq_dataset_df["reads"] > 0]  # shape: (2166, 11)
    Lazzarotto_2020_GUIDE_seq_dsb_count = Lazzarotto_2020_GUIDE_seq_dataset_df["reads"]
    SchmidBurgk_2020_TTISS_dataset_df = SchmidBurgk_2020_TTISS_dataset_df[
        SchmidBurgk_2020_TTISS_dataset_df["reads"] > 0]  # shape: (1381, 11)
    SchmidBurgk_2020_TTISS_dsb_count = SchmidBurgk_2020_TTISS_dataset_df["reads"]
    # Convert to numpy array
    Lazzarotto_2020_GUIDE_seq_dsb_count = Lazzarotto_2020_GUIDE_seq_dsb_count.to_numpy()
    SchmidBurgk_2020_TTISS_dsb_count = SchmidBurgk_2020_TTISS_dsb_count.to_numpy()
    
    # Plot DSB histogram
    output_dir_path = config["paths"]["figure"]["eda"]
    os.makedirs(output_dir_path, exist_ok=True)
    output_path = output_dir_path + "/dsb_histogram_Lazzarotto_2020_GUIDE_seq.png"
    plot_dsb_count.plot_histogram(
        count_data=Lazzarotto_2020_GUIDE_seq_dsb_count,
        output_path=output_path,
    )
    output_path = output_dir_path + "/dsb_histogram_SchmidBurgk_2020_TTISS.png"
    plot_dsb_count.plot_histogram(
        count_data=SchmidBurgk_2020_TTISS_dsb_count,
        output_path=output_path,
        bins=200
    )


