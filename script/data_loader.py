
import sys
sys.path.append("script/")

import config

import pandas as pd
import numpy as np
import csv
import json
import os
import tqdm

from datasets import Dataset

try:
    import pyBigWig
except:
    pass



def load_chrom_size() -> dict:
    with open(f"{config.metadata_path}/hg38_size.txt", "r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        chrom_size_dict = {str(row[0]) : int(row[1]) for row in reader}
    return chrom_size_dict


def load_sgrna_name():
    with open(f"{config.metadata_path}/CHANGEseq_sgRNA_seqname.json", 'r') as file:
        sgRNAs_json = json.load(file)
    return sgRNAs_json

def load_sgrna_list(path=str):
    with open(path, "r", newline="") as file:
        reader = csv.reader(file)
        sgrna_list = [row[0] for row in reader]
    return sgrna_list



class DataLoaderClass:
    def __init__(self, fold: int=0, datatype: str="guideseq"):
        if datatype == "changeseq":
            self.dataset_file_path = f"{config.yaish_et_al_data_path}/CHANGEseq/include_on_targets/CHANGEseq_CR_Lazzarotto_2020_dataset.csv"
            self.sgrna_list_path = f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv"
            self.sgrna_fold_path = f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_folds_split.csv"
        elif datatype == "guideseq":
            self.dataset_file_path = f"{config.yaish_et_al_data_path}/GUIDEseq/include_on_targets/GUIDEseq_CR_Lazzarotto_2020_dataset.csv"
            self.sgrna_list_path = f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_list.csv"
            self.sgrna_fold_path = f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_folds_split.csv"
        self.fold = fold
        self.base_index = config.base_index
        self.num_base = len(self.base_index)
        self.seq_len = 24
        self.len_onehot_encode = len(self.base_index) ** 2
        self.kmer = config.kmer
    
    def load_dataset(self, sgrna: str="AAVS1_site_1") -> pd.DataFrame:
        # sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,reads
        if sgrna == "all":
            dataset_df = pd.read_csv(self.dataset_file_path)
            return dataset_df
        else:
            # Select specific sgRNA data
            sgRNAs_json = load_sgrna_name()
            sgrna_seq = str(sgRNAs_json["sgRNAs_seq"][sgRNAs_json["sgRNAs_name"].index(sgrna)])
            dataset_df = pd.read_csv(self.dataset_file_path)
            dataset_df = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            return dataset_df

    def return_train_test_data(self) -> dict:
        '''
        output:
        dict_type data
        train_sgrna_names, train_sgrna_seqs, test_sgrna_names, test_sgrna_seqs
        '''
        # sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,reads
        # Load sgRNA information
        sgRNAs_json = load_sgrna_name()
        sgrna_seq2name_dict = {sgrna_seq:sgrna_name for sgrna_seq, sgrna_name in zip(sgRNAs_json["sgRNAs_seq"], sgRNAs_json["sgRNAs_name"])}
        with open(self.sgrna_list_path, "r", newline="") as file:
            reader = csv.reader(file)
            sgrna_seq_guideseq_list = [row[0] for row in reader]
        fold_sgrna_df = pd.read_csv(self.sgrna_fold_path)

        # Split train and test
        # Test
        test_sgrna_seq_list = [sgrna_seq for sgrna_seq in fold_sgrna_df.iloc[self.fold].values.tolist() if type(sgrna_seq) == str]
        test_sgrna_name_list = [sgrna_seq2name_dict[sgrna_seq] for sgrna_seq in test_sgrna_seq_list]
        # Train
        train_sgrna_seq_list = [sgrna_seq for sgrna_seq in sgrna_seq_guideseq_list if not sgrna_seq in test_sgrna_seq_list]
        train_sgrna_name_list = [sgrna_seq2name_dict[sgrna_seq] for sgrna_seq in train_sgrna_seq_list]

        # Dict for return
        return_dict = {}
        return_dict["train_names_list"] = train_sgrna_name_list
        return_dict["test_names_list"] = test_sgrna_name_list
        return_dict["train_seq_list"] = train_sgrna_seq_list
        return_dict["test_seq_list"] = test_sgrna_seq_list
        
        self.train_test_datalist = return_dict
        self.flag_return_train_test_data = True
        
        return return_dict
    
    def return_test_data_info(self, dataset_df: pd.DataFrame) -> dict:
        '''
        output
        dict_type data
        test_sgrna_names: [start_index, end_index]
        '''
        
        # Check if return_train_test_data is executed.
        if not self.flag_return_train_test_data:
            self.return_train_test_data()
        
        # Count start and end index for each sgRNA in test data
        test_data_index_dict = {}
        start_index = 0
        for sgrna_seq in tqdm.tqdm(self.train_test_datalist["test_seq_list"], total=len(self.train_test_datalist["test_seq_list"]), desc="Load test data info"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            end_index = start_index + dataset_df_sgrna.shape[0]
            test_data_index_dict[sgrna_seq] = (start_index, end_index)
            start_index = end_index
        
        return test_data_index_dict
    
    def seq_to_encode(self, seq1: str, seq2: str) -> np.array:
        encode_array = np.zeros(len(seq1) * self.len_onehot_encode, dtype=np.int8)
        for j, (base1, base2) in enumerate(zip(seq1, seq2)):
            if base1 == "N":
                base1 = base2
            if base2 == "N":
                base2 = base1
            encode_array[j*self.len_onehot_encode + self.base_index[base1]*self.num_base + self.base_index[base2]] = 1
        return encode_array
    
    def seq_to_categorical_encode(self, seq1: str, seq2: str) -> np.array:
        encode_array = np.zeros(len(seq1))
        for j, (base1, base2) in enumerate(zip(seq1, seq2)):
            if base1 == "N":
                base1 = base2
            if base2 == "N":
                base2 = base1
            if base1 == "N" and base2 == "N":
                base1 = "-"
                base2 = "-"
            encode_array[j] = self.base_index[base1]*self.num_base + self.base_index[base2]
        return encode_array
    
    def return_pairseq_onehot(self) -> dict:
        '''
        return
        train_input, test_input
        '''
        train_sgrna_seq_list = self.train_test_datalist["train_seq_list"]
        test_sgrna_seq_list = self.train_test_datalist["test_seq_list"]
        
        dataset_df = self.load_dataset("all")
        
        train_input = []
        test_input = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Load pair seq onehot encoding for train data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len*self.len_onehot_encode), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, self.len_onehot_encode:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            train_input.append(sgrna_encode_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len*self.len_onehot_encode), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, self.len_onehot_encode:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            train_input.append(sgrna_encode_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Load pair seq onehot encoding for test data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len*self.len_onehot_encode), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, self.len_onehot_encode:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            test_input.append(sgrna_encode_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len*self.len_onehot_encode), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, self.len_onehot_encode:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            test_input.append(sgrna_encode_array)
        
        # Input data concatenate
        train_input = np.concatenate(train_input, axis=0)
        test_input = np.concatenate(test_input, axis=0)
        
        return {"train_input": train_input, "test_input": test_input}
    
    def return_pairseq_categorical_onehot(self) -> dict:
        '''
        return
        train_input, test_input
        '''
        train_sgrna_seq_list = self.train_test_datalist["train_seq_list"]
        test_sgrna_seq_list = self.train_test_datalist["test_seq_list"]
        
        dataset_df = self.load_dataset("all")
        
        train_input = []
        test_input = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Load pair seq categorical encoding for train data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len), dtype=np.int16)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_categorical_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, 1:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            train_input.append(sgrna_encode_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len), dtype=np.int16)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_categorical_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, 1:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            train_input.append(sgrna_encode_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Load pair seq categorical encoding for test data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len), dtype=np.int16)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_categorical_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, 1:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            test_input.append(sgrna_encode_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len), dtype=np.int16)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_categorical_encode(row._6, row._7)
                if len(row._7) == self.seq_len-1:
                    sgrna_encode_array[idx, 1:] = encode_array
                elif len(row._7) == self.seq_len:
                    sgrna_encode_array[idx, :] = encode_array
                else:
                    print(["ERROR"])
            test_input.append(sgrna_encode_array)
        
        # Input data concatenate
        train_input = np.concatenate(train_input, axis=0)
        test_input = np.concatenate(test_input, axis=0)
        
        return {"train_input": train_input, "test_input": test_input}
    
    def return_pairseq_categorical_onehot_for_trueot(self, dataset_df) -> dict:
        '''
        return
        train_input(empty), test_input
        '''
        
        train_input = np.zeros((0, self.seq_len), dtype=np.int16)
        test_input = np.zeros((dataset_df.shape[0], self.seq_len), dtype=np.int16)
        
        # for test data
        # Dataset,Cell type,sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,label,Note
        for idx, row in tqdm.tqdm(enumerate(dataset_df.itertuples(index=False)), total=dataset_df.shape[0], desc="Load TrueOT pair seq categorical encoding for test data"):
            encode_array = self.seq_to_categorical_encode(row._8, row._9)
            if len(row._9) == self.seq_len-1:
                test_input[idx, 1:] = encode_array
            elif len(row._9) == self.seq_len:
                test_input[idx, :] = encode_array
            else:
                sys.exit(["ERROR"])
        
        return {"train_input": train_input, "test_input": test_input}
    
            
    def return_label(self) -> dict:
        '''
        return
        train_label, test_label
        '''
        train_sgrna_seq_list = self.train_test_datalist["train_seq_list"]
        test_sgrna_seq_list = self.train_test_datalist["test_seq_list"]
        
        dataset_df = self.load_dataset("all")
        
        train_label = []
        test_label = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Load label for train data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            offtarget_label_array = np.array([1]*data_df_sgrna_offtarget.shape[0], dtype=np.int8)
            train_label.append(offtarget_label_array)
            
            # Processing for non off-target row
            non_offtarget_label_array = np.array([0]*data_df_sgrna_non_offtarget.shape[0], dtype=np.int8)
            train_label.append(non_offtarget_label_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Load label for test data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            offtarget_label_array = np.array([1]*data_df_sgrna_offtarget.shape[0], dtype=np.int8)
            test_label.append(offtarget_label_array)
            
            # Processing for non off-target row
            non_offtarget_label_array = np.array([0]*data_df_sgrna_non_offtarget.shape[0], dtype=np.int8)
            test_label.append(non_offtarget_label_array)
        
        # Train and Test data concatenate
        train_label = np.concatenate(train_label, axis=0)
        test_label = np.concatenate(test_label, axis=0)
        
        return {"train_label": train_label, "test_label": test_label}
    
    
    def return_mismatch(self) -> dict:
        # sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,reads
        '''
        return
        train_miamatch, test_mismatch
        '''
        train_sgrna_seq_list = self.train_test_datalist["train_seq_list"]
        test_sgrna_seq_list = self.train_test_datalist["test_seq_list"]
        
        dataset_df = self.load_dataset("all")
        
        train_mismatch = []
        test_mismatch = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Load mismatch for train data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            offtarget_mismatch_array = np.zeros((data_df_sgrna_offtarget.shape[0], 7), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                mismatch = int(row._8)
                offtarget_mismatch_array[idx, mismatch] = 1
            train_mismatch.append(offtarget_mismatch_array)
            
            # Processing for non off-target row
            non_offtarget_mismatch_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], 7), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                mismatch = int(row._8)
                non_offtarget_mismatch_array[idx, mismatch] = 1
            train_mismatch.append(non_offtarget_mismatch_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Load mismatch for test data"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            offtarget_mismatch_array = np.zeros((data_df_sgrna_offtarget.shape[0], 7), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                mismatch = int(row._8)
                offtarget_mismatch_array[idx, mismatch] = 1
            test_mismatch.append(offtarget_mismatch_array)
            
            # Processing for non off-target row
            non_offtarget_mismatch_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], 7), dtype=np.int8)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                mismatch = int(row._8)
                non_offtarget_mismatch_array[idx, mismatch] = 1
            test_mismatch.append(non_offtarget_mismatch_array)
        
        # Train and Test data concatenate
        train_mismatch = np.concatenate(train_mismatch, axis=0)
        test_mismatch = np.concatenate(test_mismatch, axis=0)
        
        return {"train_mismatch_input": train_mismatch, "test_mismatch_input": test_mismatch}
    
    
    def seq_to_token(self, seq):
        if len(seq) == self.seq_len - 1:
            seq = "-" + seq
        kmers_list = [seq[i:i+self.kmer] for i in range(len(seq)-self.kmer+1)]
        merged_sequence = ' '.join(kmers_list)
        return merged_sequence
    
    def return_dataset_for_dnabert(self) -> dict:
        '''
        return
        dict: train_dataset, test_dataset: targetDNA, sgRNA, label
        '''
        train_sgrna_seq_list = self.train_test_datalist["train_seq_list"]
        test_sgrna_seq_list = self.train_test_datalist["test_seq_list"]
        
        dataset_df = self.load_dataset(sgrna="all")
        
        train_dataset_dict = {"target_dna":[], "sgrna":[], "label":[]}
        test_dataset_dict = {"target_dna":[], "sgrna":[], "label":[]}
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Load train data for DNABERT"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            for row in data_df_sgrna_offtarget.itertuples(index=False):
                target_dna_token = self.seq_to_token(row._6)
                sgrna_token = self.seq_to_token(row._7)
                train_dataset_dict["target_dna"].append(target_dna_token)
                train_dataset_dict["sgrna"].append(sgrna_token)
                train_dataset_dict["label"].append(int(1))
            
            # Processing for non off-target row
            for row in data_df_sgrna_non_offtarget.itertuples(index=False):
                target_dna_token = self.seq_to_token(row._6)
                sgrna_token = self.seq_to_token(row._7)
                train_dataset_dict["target_dna"].append(target_dna_token)
                train_dataset_dict["sgrna"].append(sgrna_token)
                train_dataset_dict["label"].append(int(0))
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Load test data for DNABERT"):
            dataset_df_sgrna = dataset_df[dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            for row in data_df_sgrna_offtarget.itertuples(index=False):
                target_dna_token = self.seq_to_token(row._6)
                sgrna_token = self.seq_to_token(row._7)
                test_dataset_dict["target_dna"].append(target_dna_token)
                test_dataset_dict["sgrna"].append(sgrna_token)
                test_dataset_dict["label"].append(int(1))
            
            # Processing for non off-target row
            for row in data_df_sgrna_non_offtarget.itertuples(index=False):
                target_dna_token = self.seq_to_token(row._6)
                sgrna_token = self.seq_to_token(row._7)
                test_dataset_dict["target_dna"].append(target_dna_token)
                test_dataset_dict["sgrna"].append(sgrna_token)
                test_dataset_dict["label"].append(int(0))
        
        train_dataset = Dataset.from_dict(train_dataset_dict)
        test_dataset = Dataset.from_dict(test_dataset_dict)
        
        return {"train_dataset": train_dataset, "test_dataset": test_dataset}
    
    def return_dataset_for_dnabert_for_trueot(self, dataset_df: pd.DataFrame) -> dict:
        '''
        return
        dict: train_dataset(empty), test_dataset: targetDNA, sgRNA, label
        '''
        
        train_dataset_dict = {"target_dna":[], "sgrna":[], "label":[]}
        test_dataset_dict = {"target_dna":[], "sgrna":[], "label":[]}
        
        # For test data
        # Dataset,Cell type,sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,label,Note
        
        for row in tqdm.tqdm(dataset_df.itertuples(index=False), total=dataset_df.shape[0], desc="Load TrueOT test data for DNABERT"):
            target_dna_token = self.seq_to_token(row._8)
            sgrna_token = self.seq_to_token(row._9)
            test_dataset_dict["target_dna"].append(target_dna_token)
            test_dataset_dict["sgrna"].append(sgrna_token)
            test_dataset_dict["label"].append(int(row.label))
        
        test_dataset = Dataset.from_dict(test_dataset_dict)
        
        return {"train_dataset": train_dataset_dict, "test_dataset": test_dataset}
            
