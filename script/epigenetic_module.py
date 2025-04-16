
import sys
sys.path.append("script/")
import os

import config
import data_loader
import utilities_module
import visualize_module


import tqdm
import pandas as pd
import numpy as np



class PreprocessEpigeneticFeatureClass:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, datatype: str="guideseq", assay: str="atac", scope_range: int=5000, bin_size: int=50):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.datatype = datatype
        self.scope_range = scope_range
        self.bin_size = bin_size
        
        self.assay = assay
        self.epigenetic_feature_base_dir = config.epigenetic_base_dir_path
        self.epigenetic_feature_file_path = f"{self.epigenetic_feature_base_dir}/{self.datatype}_{self.assay}"
        os.makedirs(self.epigenetic_feature_base_dir, exist_ok=True)
        os.makedirs(self.epigenetic_feature_file_path, exist_ok=True)
        
        if self.assay == "atac":
            try:
                import pyBigWig
                self.bw = pyBigWig.open(f"{config.gse149361_path}/GSM4498611_ATAC_FE.bdg.bw")
            except:
                print(f"[ERROR]pyBigWig cannot be loaded or ATAC-seq bigwig file does not exist.")
        
        elif self.assay == "h3k4me3":
            try:
                import pyBigWig
                self.bw = pyBigWig.open(f"{config.gse149361_path}/GSM4495703_H3K4me3_FE.bdg.bw")
            except:
                print(f"[ERROR]pyBigWig cannot be loaded or H3K4me3 bigwig file does not exist.")
        
        elif self.assay == "h3k27ac":
            try:
                import pyBigWig
                self.bw = pyBigWig.open(f"{config.gse149361_path}/GSM4495711_H3K27ac_FE.bdg.bw")
            except:
                print(f"[ERROR]pyBigWig cannot be loaded or H3K27ac bigwig file does not exist.")
        
        else:
            sys.exit(f"[ERROR]Invalid assay type.")
    
        
    def load_epigenetic_signal(self, chrom_str: str="chr1", start_coords: int=0, end_coords: int=-1) -> tuple:
        try:
            if end_coords == -1:
                end_coords = self.bw.chroms()[chrom_str]
            signal_array = np.array(self.bw.values(chrom_str, start_coords, end_coords))
            return (signal_array, int(start_coords), int(end_coords))
        except Exception as e1:
            try:
                chrom_length = self.bw.chroms()[chrom_str]
                if start_coords < 0:
                    start_coords = 0
                if end_coords > chrom_length:
                    end_coords = chrom_length
                signal_array = np.array(self.bw.values(chrom_str, start_coords, end_coords))
                return (signal_array, int(start_coords), int(end_coords))
            except Exception as e2:
                signal_zeros_array = np.zeros(end_coords - start_coords)
                return (signal_zeros_array, int(start_coords), int(end_coords))

    def save_epigenetic_features(self):
        # sgRNA, chrom, SiteWindow, Align.strand, Align.chromStart, Align.chromEnd, Align.off-target, Align.sgRNA, Align.#Mismatches, Align.#Bulges, reads
        print(f"[PREPROCESS]Save epigenetic features. Epigenetic type: {self.assay}. Scope range: {self.scope_range}. Bin size: {self.bin_size}.")
        
        train_sgrna_name_list = self.train_test_info["train_names_list"]
        test_sgrna_name_list = self.train_test_info["test_names_list"]
        train_sgrna_seq_list = self.train_test_info["train_seq_list"]
        test_sgrna_seq_list = self.train_test_info["test_seq_list"]
        
        # For train and test data
        for sgrna_name_list, sgrna_seq_list in zip([train_sgrna_name_list, test_sgrna_name_list], [train_sgrna_seq_list, test_sgrna_seq_list]):
            for sgrna_name, sgrna_seq in tqdm.tqdm(zip(sgrna_name_list, sgrna_seq_list), total=len(sgrna_seq_list)):
                dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
                # Split offtarget or non-offtarget
                data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
                data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
                if self.datatype == "changeseq":
                    data_df_sgrna_non_offtarget = data_df_sgrna_non_offtarget.sample(n=1000, random_state=42)

                # Processing for off-target row
                offtarget_epigenetic_matrix = np.zeros((data_df_sgrna_offtarget.shape[0], self.scope_range*2), dtype=np.float32)
                for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                    start_coords = row._4 - self.scope_range
                    end_coords = row._4 + self.scope_range
                    epigenetic_value_array, start_coords_output, end_coords_output = self.load_epigenetic_signal(row.chrom, start_coords, end_coords)
                    if row._3 == "-":
                        epigenetic_value_array = np.flip(epigenetic_value_array)
                    offtarget_epigenetic_matrix[idx][start_coords_output - start_coords : end_coords_output - end_coords + self.scope_range*2] = epigenetic_value_array 
                offtarget_epigenetic_matrix = np.nan_to_num(offtarget_epigenetic_matrix, nan=0)
                offtarget_epigenetic_matrix_ = offtarget_epigenetic_matrix.reshape(offtarget_epigenetic_matrix.shape[0], (self.scope_range*2)//self.bin_size, self.bin_size)
                offtarget_epigenetic_matrix = offtarget_epigenetic_matrix_.mean(axis=2)
                # Save off-target epigenetic feature
                np.save(f"{self.epigenetic_feature_file_path}/{sgrna_name}.offtarget.npy", offtarget_epigenetic_matrix)

                # Processing for non off-target row
                non_offtarget_epigenetic_matrix = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.scope_range*2), dtype=np.float32)
                for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                    start_coords = row._4 - self.scope_range
                    end_coords = row._4 + self.scope_range
                    epigenetic_value_array, start_coords_output, end_coords_output = self.load_epigenetic_signal(row.chrom, start_coords, end_coords)
                    if row._3 == "-":
                        epigenetic_value_array = np.flip(epigenetic_value_array)
                    non_offtarget_epigenetic_matrix[idx][start_coords_output - start_coords : end_coords_output - end_coords + self.scope_range*2] = epigenetic_value_array 
                non_offtarget_epigenetic_matrix = np.nan_to_num(non_offtarget_epigenetic_matrix, nan=0)
                non_offtarget_epigenetic_matrix_ = non_offtarget_epigenetic_matrix.reshape(non_offtarget_epigenetic_matrix.shape[0], (self.scope_range*2)//self.bin_size, self.bin_size)
                non_offtarget_epigenetic_matrix = non_offtarget_epigenetic_matrix_.mean(axis=2)
                # Save off-target epigenetic feature
                np.save(f"{self.epigenetic_feature_file_path}/{sgrna_name}.non-offtarget.npy", non_offtarget_epigenetic_matrix)
        self.bw.close()


class EpigeneticFeatureClass:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, datatype: str="guideseq", assay: str="atac", assays: list=["h3k4me3", "h3k27ac", "atac"], scope_range: int=5000, bin_size: int=50):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.datatype = datatype
        self.scope_range = scope_range
        self.bin_size = bin_size
        
        self.assay = assay
        self.epigenetic_feature_base_dir = config.epigenetic_base_dir_path
        self.epigenetic_feature_file_path = f"{self.epigenetic_feature_base_dir}/{self.datatype}_{self.assay}"
        os.makedirs(self.epigenetic_feature_base_dir, exist_ok=True)
        os.makedirs(self.epigenetic_feature_file_path, exist_ok=True)
        self.assay_list = assays
        
    def check_file_exist(self):
        # For train data
        for sgrna_name in self.train_test_info["train_names_list"]:
            if not os.path.exists(f"{self.epigenetic_feature_file_path}/{sgrna_name}.offtarget.npy"):
                sys.exit(f"[ERROR]{sgrna_name}.offtarget.npy does not exist.")
            if not os.path.exists(f"{self.epigenetic_feature_file_path}/{sgrna_name}.non-offtarget.npy"):
                sys.exit(f"[ERROR]{sgrna_name}.non-offtarget.npy does not exist.")
        # For test data
        for sgrna_name in self.train_test_info["test_names_list"]:
            if not os.path.exists(f"{self.epigenetic_feature_file_path}/{sgrna_name}.offtarget.npy"):
                sys.exit(f"[ERROR]{sgrna_name}.offtarget.npy does not exist.")
            if not os.path.exists(f"{self.epigenetic_feature_file_path}/{sgrna_name}.non-offtarget.npy"):
                sys.exit(f"[ERROR]{sgrna_name}.non-offtarget.npy does not exist.")
    
    def return_epigenetic_feature(self):
        # self.check_file_exist()
        print(f"[PREPROCESS]Load epigenetic features. Epigenetic type: {self.assay} Scope range: {self.scope_range}. Bin size: {self.bin_size}")
        
        train_sgrna_name_list = self.train_test_info["train_names_list"]
        test_sgrna_name_list = self.train_test_info["test_names_list"]
        
        # Load epigenetic feature numpy file
        train_feature_input_collection = []
        test_feature_input_collection = []
        
        for assay_ in self.assay_list:
            # Train input
            train_feature_input_assay = []
            for sgrna_name in tqdm.tqdm(train_sgrna_name_list, total=len(train_sgrna_name_list), desc=f"Load epigenetic feature ({assay_}) for train data"):
                # Load
                offtarget_epigenetic_array = np.load(f"{self.epigenetic_feature_base_dir}/{self.datatype}_{assay_}/{sgrna_name}.offtarget.npy")
                non_offtarget_epigenetic_array = np.load(f"{self.epigenetic_feature_base_dir}/{self.datatype}_{assay_}/{sgrna_name}.non-offtarget.npy")
                # Append
                train_feature_input_assay.append(offtarget_epigenetic_array)
                train_feature_input_assay.append(non_offtarget_epigenetic_array)
            # Concatenate
            train_feature_input_assay = np.concatenate(train_feature_input_assay, axis=0)
            
            train_feature_input_assay = train_feature_input_assay[:, 90:110] # 90:110 -> 1000bp
            
            train_feature_input_collection.append(train_feature_input_assay)

            # Test input
            test_feature_input_assay = []
            for sgrna_name in tqdm.tqdm(test_sgrna_name_list, total=len(test_sgrna_name_list), desc=f"Load epigenetic feature ({assay_}) for test data"):
                # Load
                offtarget_epigenetic_array = np.load(f"{self.epigenetic_feature_base_dir}/{self.datatype}_{assay_}/{sgrna_name}.offtarget.npy")
                non_offtarget_epigenetic_array = np.load(f"{self.epigenetic_feature_base_dir}/{self.datatype}_{assay_}/{sgrna_name}.non-offtarget.npy")
                # Append
                test_feature_input_assay.append(offtarget_epigenetic_array)
                test_feature_input_assay.append(non_offtarget_epigenetic_array)
            # Concatenate
            test_feature_input_assay = np.concatenate(test_feature_input_assay, axis=0)
            
            test_feature_input_assay = test_feature_input_assay[:, 90:110] # 90:110 -> 1000bp
            
            test_feature_input_collection.append(test_feature_input_assay)

        # Add mean value
        for i in range(len(self.assay_list)):
            train_feature_input_collection[i] = np.concatenate([train_feature_input_collection[i], np.mean(train_feature_input_collection[i], axis=1).reshape(-1, 1)], axis=1)
            test_feature_input_collection[i] = np.concatenate([test_feature_input_collection[i], np.mean(test_feature_input_collection[i], axis=1).reshape(-1, 1)], axis=1)
        
        # Z-score normalization
        for i in range(len(self.assay_list)):
            train_feature_input_collection[i] = (train_feature_input_collection[i] - np.mean(train_feature_input_collection[i], axis=0)) / np.std(train_feature_input_collection[i], axis=0)
            test_feature_input_collection[i] = (test_feature_input_collection[i] - np.mean(test_feature_input_collection[i], axis=0)) / np.std(test_feature_input_collection[i], axis=0)
        
        train_feature_input = np.concatenate(train_feature_input_collection, axis=1)
        test_feature_input = np.concatenate(test_feature_input_collection, axis=1)
    
        return {"train_epigenetic_input": train_feature_input, "test_epigenetic_input": test_feature_input, "train_feature_matrix_list": train_feature_input_collection, "test_feature_matrix_list": test_feature_input_collection}


class EpigeneticAnalysisClass:
    """
    訓練、テストとか分けずに、全てのデータを使って、エピジェネティクスの特徴量を分析するクラス
    return_epigenetic_feature -> {Off-target: [numpy, numpy, ...], Non off-target: [numpy, numpy, ...]}
    
    return_normalized_data -> {"guideseq": [numpy, numpy, ...], "changeseq": [numpy, numpy, ...], "inactive": [numpy, numpy, ...]}
    """
    def __init__(self, assay: str="atac", figtype: str="fig_a"):
        self.assay = assay
        self.figtype = figtype
        self.epigenetic_base_dir = config.epigenetic_base_dir_path
        self.epigenetic_fig_path = config.fig_base_path + "/epigenetic"
        os.makedirs(self.epigenetic_fig_path, exist_ok=True)
    
    def load_dataset_df(self):
        dataset_file_path = f"{config.yaish_et_al_data_path}/GUIDEseq/include_on_targets/GUIDEseq_CR_Lazzarotto_2020_dataset.csv"
        dataset_df = pd.read_csv(self.dataset_file_path)
    
    def return_normalized_data(self):
        # sgRNA information
        sgrnas_json = data_loader.load_sgrna_name() 
        
        # Load GUIDE-seq data
        guideseq_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_list.csv")
        guideseq_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in guideseq_sgrna_seq_list]
        guideseq_data = []
        for sgrna_name in guideseq_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/guideseq_{self.assay}/{sgrna_name}.offtarget.npy")
            guideseq_data.append(epigenetic_data)
        guideseq_data = np.concatenate(guideseq_data, axis=0)
        
        # Load CHANGE-seq data
        changeseg_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv")
        changeseg_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in changeseg_sgrna_seq_list]
        changeseq_data = []
        for sgrna_name in changeseg_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/changeseq_{self.assay}/{sgrna_name}.offtarget.npy")
            changeseq_data.append(epigenetic_data)
        changeseq_data = np.concatenate(changeseq_data, axis=0)
        
        # Load inactive OTS data
        changeseg_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv")
        changeseg_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in changeseg_sgrna_seq_list]
        inactive_data = []
        for sgrna_name in changeseg_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/changeseq_{self.assay}/{sgrna_name}.non-offtarget.npy")
            inactive_data.append(epigenetic_data)
        inactive_data = np.concatenate(inactive_data, axis=0)
        
        return {"guideseq": guideseq_data, "changeseq": changeseq_data, "inactive": inactive_data}

    def return_standardized_data_1(self):
        # sgRNA information
        sgrnas_json = data_loader.load_sgrna_name() 
        
        # Load GUIDE-seq data
        guideseq_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_list.csv")
        guideseq_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in guideseq_sgrna_seq_list]
        guideseq_data = []
        for sgrna_name in guideseq_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/guideseq_{self.assay}/{sgrna_name}.offtarget.npy")
            guideseq_data.append(epigenetic_data)
        guideseq_data = np.concatenate(guideseq_data, axis=0)
        
        # Load CHANGE-seq data
        changeseg_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv")
        changeseg_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in changeseg_sgrna_seq_list]
        changeseq_data = []
        for sgrna_name in changeseg_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/changeseq_{self.assay}/{sgrna_name}.offtarget.npy")
            changeseq_data.append(epigenetic_data)
        changeseq_data = np.concatenate(changeseq_data, axis=0)
        
        # Load inactive OTS data
        changeseg_sgrna_seq_list = data_loader.load_sgrna_list(path=f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv")
        changeseg_sgrna_name_list = [sgrnas_json["sgRNAs_name"][sgrnas_json["sgRNAs_seq"].index(sgrna_seq)] for sgrna_seq in changeseg_sgrna_seq_list]
        inactive_data = []
        for sgrna_name in changeseg_sgrna_name_list:
            epigenetic_data = np.load(f"{self.epigenetic_base_dir}/changeseq_{self.assay}/{sgrna_name}.non-offtarget.npy")
            inactive_data.append(epigenetic_data)
        inactive_data = np.concatenate(inactive_data, axis=0)
        
        # 90:110 -> 1000bp
        guideseq_data = guideseq_data[:, 90:110]
        changeseq_data = changeseq_data[:, 90:110]
        inactive_data = inactive_data[:, 90:110]
        
        return {"guideseq": guideseq_data, "changeseq": changeseq_data, "inactive": inactive_data}
        
        


def main():
    assay_name_dict = {"atac": "ATAC", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac"}
    # Legend figure
    visualize_module.line_graph_legend(data_name_dict={"guideseq": "GUIDE-seq OTS", "changeseq": "CHANGE-seq OTS", "inactive": "Inactive OTS"}, save_path=f"{config.fig_base_path}/epigenetic/legend.png")
    for assay in ["atac", "h3k4me3", "h3k27ac"]:
        for figtype in ["fig_a", "fig_b", "fig_c"]:
            epigeneticAnalysisClass = EpigeneticAnalysisClass(assay=assay, figtype=figtype)
            # Legend figure
            if figtype == "fig_a":
                data_array_dict = epigeneticAnalysisClass.return_normalized_data()
                visualize_module.line_graph_chromatin_state(data_array_dict, title=assay_name_dict[assay], ylabel="", scope_range=5000, bin_size=50, divide_value=1000, 
                                                            save_path=f"{epigeneticAnalysisClass.epigenetic_fig_path}/{figtype}_{assay}_chromatin_state.png")
            elif figtype == "fig_b":
                data_array_dict = epigeneticAnalysisClass.return_standardized_data_1()
                visualize_module.line_graph_chromatin_state(data_array_dict, title=assay_name_dict[assay], ylabel="", scope_range=500, bin_size=50, divide_value=1, 
                                                            save_path=f"{epigeneticAnalysisClass.epigenetic_fig_path}/{figtype}_{assay}_chromatin_state.png")

if __name__ == "__main__":
    main()
