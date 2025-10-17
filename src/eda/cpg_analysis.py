
import pandas as pd
import tqdm
import pysam
import numpy as np
import random
from scipy.stats import mannwhitneyu

import utils.file_handlers as file_handlers
import utils.fasta_handlers as fasta_handlers



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


class CpGanalysisClass:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_1_str = config["dataset_name"]["dataset_1"]
        self.dataset_2_str = config["dataset_name"]["dataset_2"]
        if self.dataset_1_str == "Lazzarotto_2020_CHANGE_seq" and self.dataset_2_str == "Lazzarotto_2020_GUIDE_seq":
            self.sgrna_list_path = config["paths"]["off_target_dataset"][self.dataset_2_str]["sgrna_list_old"]
        else:
            self.sgrna_list_path = config["paths"]["off_target_dataset"][self.dataset_1_str]["sgrna_list"]
        self.sgrna_list = file_handlers.load_csv_list(self.sgrna_list_path)

    def load_data_CSGS(self, chrom_size: dict) -> None:
        self.dataset_1_df = file_handlers.load_csv_dataset(self.config["paths"]["off_target_dataset"][self.dataset_1_str]["dataset"])
        self.dataset_2_df = file_handlers.load_csv_dataset(self.config["paths"]["off_target_dataset"][self.dataset_2_str]["dataset"])
        
        # Filter datasets by sgRNA list
        self.dataset_1_df = file_handlers.filter_datsaet_by_sgrna(self.dataset_1_df, self.sgrna_list)
        self.dataset_2_df = file_handlers.filter_datsaet_by_sgrna(self.dataset_2_df, self.sgrna_list)
        
        # Filter OTS
        self.dataset_1_df = file_handlers.filter_OTS_dataset(self.dataset_1_df)
        self.dataset_2_df = file_handlers.filter_OTS_dataset(self.dataset_2_df)
        
        # Filter chromosomes
        self.dataset_1_df = file_handlers.filter_chromosome(self.dataset_1_df, chrom_size)
        self.dataset_2_df = file_handlers.filter_chromosome(self.dataset_2_df, chrom_size)
        
        return {"CS_df": self.dataset_1_df, "GS_df": self.dataset_2_df}

    def return_row_idx_by_sgrna(self, dataset_df: pd.DataFrame, sgrna_list: list) -> dict: # {sgrna: [row_idx, ...]}
        row_idx_dict = {}
        for sgrna in tqdm.tqdm(sgrna_list):
            row_idx = dataset_df[dataset_df["sgRNA"] == sgrna].index.tolist()
            if row_idx:
                row_idx_dict[sgrna] = row_idx
        return row_idx_dict
    
    def return_reads_ratio_by_sgrna(self, row_num_dict_1, row_num_dict_2, sgrna_list: list) -> dict:
        reads_ratio_dict = {}
        for sgrna in tqdm.tqdm(sgrna_list):
            if sgrna in row_num_dict_1 and sgrna in row_num_dict_2:
                reads_ratio = row_num_dict_1[sgrna] / row_num_dict_2[sgrna]
                reads_ratio_dict[sgrna] = reads_ratio
            else:
                reads_ratio_dict[sgrna] = 0
        return reads_ratio_dict
    
    def return_idx_sampling_ratio_times(self, row_idx_dict_1: dict, row_idx_dict_2: dict, min_ratio: int, sgrna_list: list, random_seed: int=42) -> list:
        sampled_row_idx = []
        random.seed(random_seed)
        for sgrna in sgrna_list:
            if sgrna in row_idx_dict_1 and sgrna in row_idx_dict_2:
                # sampling_num = len(row_idx_dict_2[sgrna]) * min_ratio
                sampling_num = len(row_idx_dict_2[sgrna]) * 1
                random_row_idx = random.sample(row_idx_dict_1[sgrna], sampling_num)
                sampled_row_idx.extend(random_row_idx)
        return sampled_row_idx
    
    def calculate_which_cpg_levels_by_sgrna(self, CS_cpg_matrix: np.ndarray, GS_cpg_matrix: np.ndarray, CS_row_idx_dict: dict, GS_row_idx_dict: dict, min_ratio: int, random_seed: int=42) -> dict:
        cpg_levels_array = []
        for sgrna in self.sgrna_list:
            if sgrna in CS_row_idx_dict and sgrna in GS_row_idx_dict and len(GS_row_idx_dict[sgrna]) > 5:
                sampled_CS_row_idx = random.sample(CS_row_idx_dict[sgrna], min(len(CS_row_idx_dict[sgrna]), len(GS_row_idx_dict[sgrna]) * min_ratio))
                filtered_CS_cpg_matrix = CS_cpg_matrix[sampled_CS_row_idx, :]
                CS_cpg_matrix_mean = np.mean(filtered_CS_cpg_matrix, axis=0)
                filtered_GS_cpg_matrix = GS_cpg_matrix[GS_row_idx_dict[sgrna], :]
                GS_cpg_matrix_mean = np.mean(filtered_GS_cpg_matrix, axis=0)
                print(f"sgRNA: {sgrna}, CS_cpg_matrix_mean: {np.mean(CS_cpg_matrix_mean)}, GS_cpg_matrix_mean: {np.mean(GS_cpg_matrix_mean)}, num GS:, {len(GS_row_idx_dict[sgrna])}")
                # 
                window_cpg_levels_diff = np.zeros((len(CS_cpg_matrix_mean),))
                for i in range(CS_cpg_matrix_mean.shape[0]):
                    if CS_cpg_matrix_mean[i] > GS_cpg_matrix_mean[i]:
                        window_cpg_levels_diff[i] = 1
                cpg_levels_array.append(window_cpg_levels_diff)
        cpg_levels_array = np.array(cpg_levels_array)
        return cpg_levels_array


def run_cpg_analysis_CSGS(config: dict) -> None:
    # Fasta object
    fasta_path = config["paths"]["reference_genome"]["hg38"]
    reference_fa = pysam.FastaFile(fasta_path)
    chrom_size = fasta_handlers.return_chrom_size(reference_fa)
    
    cpg_analysis = CpGanalysisClass(config)
    
    dataset_dict = cpg_analysis.load_data_CSGS(chrom_size=chrom_size)
    
    # Remove duplicate row from CHANGE-seq dataset to GUIDE-seq dataset
    merged = dataset_dict["CS_df"].merge(dataset_dict["GS_df"], on=["sgRNA", "chrom", "Align.strand", "Align.chromStart"], how="left", indicator=True)
    CS_df_copy = dataset_dict["CS_df"].copy()
    dataset_dict["CS_df"] = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    # Extract only duplicate rows from GUIDE-seq dataset
    dataset_dict["GS_df"] = dataset_dict["GS_df"].merge(CS_df_copy, on=["sgRNA", "chrom", "Align.strand", "Align.chromStart"], how="inner")
    # reset index
    dataset_dict["CS_df"] = dataset_dict["CS_df"].reset_index(drop=True)
    dataset_dict["GS_df"] = dataset_dict["GS_df"].reset_index(drop=True)
    
    # Save filtered datasets
    CS_row_idx_dict = cpg_analysis.return_row_idx_by_sgrna(dataset_dict["CS_df"], cpg_analysis.sgrna_list)
    CS_row_num_dict = {sgrna: len(row_idx) for sgrna, row_idx in CS_row_idx_dict.items()}
    
    GS_row_idx_dict = cpg_analysis.return_row_idx_by_sgrna(dataset_dict["GS_df"], cpg_analysis.sgrna_list)
    GS_row_num_dict = {sgrna: len(row_idx) for sgrna, row_idx in GS_row_idx_dict.items()}
    
    reads_ratio_CSGS = cpg_analysis.return_reads_ratio_by_sgrna(CS_row_num_dict, GS_row_num_dict, cpg_analysis.sgrna_list)
    print(reads_ratio_CSGS)
    print(dataset_dict["CS_df"])
    print(dataset_dict["CS_df"].shape)
    print(dataset_dict["GS_df"].shape)
    min_ratio_int = int(min(reads_ratio_CSGS.values())//1)
    print(f"Minimum reads ratio between CHANGE-seq and GUIDE-seq datasets: {min_ratio_int}")
    print(len(CS_row_idx_dict[list(reads_ratio_CSGS.keys())[0]]))
    
    flanking_seq_processor = fasta_handlers.FlankingSequenceProcessor(
        config = config,
        dataset_str = cpg_analysis.dataset_1_str,
        reference_fasta = fasta_path,
        chrom_size = chrom_size)
    flanking_seq_processor.process_all_entries(dataset=dataset_dict["CS_df"])
    
    flanking_seq_processor = fasta_handlers.FlankingSequenceProcessor(
        config = config,
        dataset_str = cpg_analysis.dataset_2_str,
        reference_fasta = fasta_path,
        chrom_size = chrom_size)
    flanking_seq_processor.process_all_entries(dataset=dataset_dict["GS_df"])
    
    # Load fasta files
    CS_flanking_seq = fasta_handlers.load_fasta_file(config["paths"]["fasta_file"][cpg_analysis.dataset_1_str]["CSGS_flanking_r1"])
    GS_flanking_seq = fasta_handlers.load_fasta_file(config["paths"]["fasta_file"][cpg_analysis.dataset_2_str]["CSGS_flanking_r1"])
    
    print(len(CS_flanking_seq["seq"]))
    print(len(GS_flanking_seq["seq"]))
    
    # Calculate CpG O/E ratio
    CS_cpg_matrix = return_cpg_matrix(config=config, sequence_list=CS_flanking_seq["seq"])
    GS_cpg_matrix = return_cpg_matrix(config=config, sequence_list=GS_flanking_seq["seq"])
    print(CS_cpg_matrix.shape, GS_cpg_matrix.shape)
    
    side_cpg_levels = []
    
    for i in range(1):
        sampled_row_idx_CS = cpg_analysis.return_idx_sampling_ratio_times(
            row_idx_dict_1=CS_row_idx_dict,
            row_idx_dict_2=GS_row_idx_dict,
            min_ratio=min_ratio_int,
            sgrna_list=cpg_analysis.sgrna_list,
            random_seed=42 + i
        )
        
        cpg_levels_array = cpg_analysis.calculate_which_cpg_levels_by_sgrna(
            CS_cpg_matrix=CS_cpg_matrix,
            GS_cpg_matrix=GS_cpg_matrix,
            CS_row_idx_dict=CS_row_idx_dict,
            GS_row_idx_dict=GS_row_idx_dict,
            min_ratio=min_ratio_int,
            random_seed=42 + i
        )
        print(cpg_levels_array.shape)
        side_cpg_levels.append(cpg_levels_array)
    
    side_cpg_levels = np.concatenate(side_cpg_levels, axis=0)
    print(side_cpg_levels.shape)
    
    # each column is a sliding window, by columns, perform binomial test
    import scipy.stats as stats
    p_values = []
    for i in range(side_cpg_levels.shape[1]):
        column_data = side_cpg_levels[:, i]
        num_CS = np.sum(column_data == 1)
        num_GS = np.sum(column_data == 0)
        total = num_CS + num_GS
        
        p_value = stats.binomtest(k=num_GS, n=total, p=0.5).pvalue
        p_values.append(p_value)
        
        print(f"Column {i}: num_CS={num_CS}, num_GS={num_GS}, total={total}")
        
    
    print(f"p-values: {p_values}")
        
    