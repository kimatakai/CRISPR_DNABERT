


import os
import torch
import random
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
from collections import defaultdict

import pysam

import utils.file_handlers as file_handlers
import utils.fasta_handlers as fasta_handlers




class BalancedSampler(Sampler):
    def __init__(self, dataset, majority_rate: float=0.5, seed=None):

        if isinstance(dataset, dict):
            self.labels = dataset["labels"].tolist() if isinstance(dataset["labels"], torch.Tensor) else dataset["labels"]
        elif isinstance(dataset, Dataset):
            self.labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        else:
            self.labels = dataset["labels"].tolist() if isinstance(dataset["labels"], torch.Tensor) else dataset["labels"]
        self.majority_rate = majority_rate
        self.seed = seed
        
        # Classify indices by label
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.label_to_indices[label].append(i)
        
        # Identify minority and majority classes
        if 0 in self.label_to_indices and 1 in self.label_to_indices:
            self.minority_class = 0 if len(self.label_to_indices[0]) < len(self.label_to_indices[1]) else 1
            self.majority_class = 1 - self.minority_class
        else:
            raise ValueError("Dataset must contain both classes 0 and 1.")
        self.minority_indices = self.label_to_indices[self.minority_class]
        self.all_majority_indices = self.label_to_indices[self.majority_class]
        
        # Initialize epoch state
        self.reset_sampler_state()
    
    def reset_sampler_state(self):
        # Pool of majority class indices that have not been sampled in the previous epochs
        self.available_majority_indices = list(self.all_majority_indices)
        if self.seed is not None:
            random.Random(self.seed).shuffle(self.available_majority_indices)
        else:
            random.shuffle(self.available_majority_indices)
        self.used_majority_indices_in_cycle = set() # Indices used in the current epoch
        self.last_epoch_indices = []
        
    def __iter__(self):
        num_majority_samples = len(self.all_majority_indices)
        num_majority_to_sample = int(num_majority_samples * self.majority_rate)

        if len(self.available_majority_indices) < num_majority_to_sample:
            # print(f"Warning: Not enough new majority samples. Resetting available majority pool for {len(self.available_majority_indices)} available vs {num_majority_to_sample} needed.")
            self.reset_sampler_state()
        # Select available majority indices for this epoch
        selected_majority_indices = self.available_majority_indices[:num_majority_to_sample]
        # Record indices selected in this epoch as used, and remove them from the available pool
        self.used_majority_indices_in_cycle.update(selected_majority_indices)
        self.available_majority_indices = self.available_majority_indices[num_majority_to_sample:]
        # Concatenate minority and selected majority indices
        combined_indices = self.minority_indices + selected_majority_indices
        # Shuffle the whole indices
        random.shuffle(combined_indices)
        
        self.last_epoch_indices = combined_indices
        return iter(combined_indices)
    
    def __len__(self):
        if hasattr(self, 'last_epoch_indices') and self.last_epoch_indices:
            return len(self.last_epoch_indices)
        else:
            num_minority_samples = len(self.minority_indices)
            num_majority_samples = len(self.all_majority_indices)
            return num_minority_samples + int(num_majority_samples * self.majority_rate)

    def set_epoch(self, epoch):
        pass  # Placeholder for setting epoch if needed, currently not used


class DataLoaderClass:
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = self.config["paths"]["database_dir"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        
        self.fasta_path = self.database_dir + config["paths"]["reference_genome"]["hg38"]
        self.reference_fa = pysam.FastaFile(self.fasta_path)
        self.chrom_size = fasta_handlers.return_chrom_size(self.reference_fa)
    
    def load_sgrna_list(self) -> list:
        if self.dataset_name in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2", "SchmidBurgk_2020_TTISS"]:
            sgrna_list_path = self.database_dir + self.config["paths"]["off_target_dataset"][self.dataset_name]["sgrna_list"]
            self.included_sgrna_list = file_handlers.load_csv_list(sgrna_list_path)
        else:
            self.included_sgrna_list = None
    
    def load_dataset(self) -> dict:
        self.load_sgrna_list()
        if self.dataset_name == "Lazzarotto_2020_CHANGE_seq":
            dataset = self.return_Lazzarotto_2020_CHANGE_seq_dataset()
            return dataset
        elif self.dataset_name == "Lazzarotto_2020_GUIDE_seq":
            dataset = self.return_Lazzarotto_2020_GUIDE_seq_dataset()
            return dataset
        elif self.dataset_name == "Chen_2017_GUIDE_seq":
            dataset = self.return_Chen_2017_GUIDE_seq_dataset()
            return dataset
        elif self.dataset_name == "Listgarten_2018_GUIDE_seq":
            dataset = self.return_Listgarten_2018_GUIDE_seq_dataset()
            return dataset
        elif self.dataset_name == "Tsai_2015_GUIDE_seq_1":
            dataset = self.return_Tsai_2015_GUIDE_seq_dataset()
            return dataset
        elif self.dataset_name == "Tsai_2015_GUIDE_seq_2":
            dataset = self.return_Tsai_2015_GUIDE_seq_dataset()
            return dataset
        elif self.dataset_name == "SchmidBurgk_2020_TTISS":
            dataset = self.return_SchmidBurgk_2020_TTISS_dataset()
            return dataset

    def load_and_convert_to_dict(self) -> dict:
        dataset_path = self.database_dir + self.config["paths"]["off_target_dataset"][self.dataset_name]["dataset"]
        dataset = file_handlers.load_csv_dataset(dataset_path)
        if self.dataset_name in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2"]:
            dataset = file_handlers.filter_dataset_by_sgrna(dataset, self.included_sgrna_list)
        dataset = file_handlers.filter_chromosome(dataset, self.chrom_size)
        
        # Dataset information
        dataset_dict = self.load_dataset_information(dataset)
        return dataset_dict
    
    def split_dataset(self, dataset_dict: dict, fold) -> dict:
        split_functions = {
            "Lazzarotto_2020_CHANGE_seq": self.split_Lazzarotto_2020_CHANGE_seq_dataset,
            "Lazzarotto_2020_GUIDE_seq": self.split_Lazzarotto_2020_GUIDE_seq_dataset,
            "SchmidBurgk_2020_TTISS": self.split_SchmidBurgk_2020_TTISS_dataset,
            "Chen_2017_GUIDE_seq": self.split_Chen_2017_GUIDE_seq_Listgarten_2018_GUIDE_seq_dataset,
            "Listgarten_2018_GUIDE_seq": self.split_Chen_2017_GUIDE_seq_Listgarten_2018_GUIDE_seq_dataset,
            "Tsai_2015_GUIDE_seq_1": self.split_Tsai_2015_GUIDE_seq_dataset,
            "Tsai_2015_GUIDE_seq_2": self.split_Tsai_2015_GUIDE_seq_dataset
        }
        return split_functions[self.dataset_name](dataset_dict, fold)
    
    def load_dataset_information(self, dataset: pd.DataFrame) -> dict:
        sgrna_list = dataset["sgRNA"].tolist()
        rna_seq_list = dataset["Align.sgRNA"].tolist()
        dna_seq_list = dataset["Align.off-target"].tolist()
        reads_list = dataset["reads"].tolist()
        label_list = [1 if r > 0 else 0 for r in reads_list]
        chrom_list = dataset["chrom"].tolist()
        strand_list = dataset["Align.strand"].tolist()
        start_pos_list = dataset["Align.chromStart"].tolist()
        end_pos_list = dataset["Align.chromEnd"].tolist()
        mismatch_list = dataset["Align.#Mismatches"].tolist()
        bulge_list = dataset["Align.#Bulges"].tolist()
        return {
            "sgrna": sgrna_list,
            "rna_seq": rna_seq_list,
            "dna_seq": dna_seq_list,
            "reads": reads_list,
            "label": label_list,
            "chrom": chrom_list,
            "strand": strand_list,
            "start_pos": start_pos_list,
            "end_pos": end_pos_list,
            "mismatch": mismatch_list,
            "bulge": bulge_list
        }

    def update_dataset(self, dataset_dict: dict, train_sgrna: list=None, test_sgrna: list=None, train_idx: list=None, test_idx: list=None) -> dict:
        dataset_dict.update({
            "train_sgrna": train_sgrna,
            "test_sgrna": test_sgrna,
            "train_idx": train_idx,
            "test_idx": test_idx
        })
        return dataset_dict
    
    
    def split_Lazzarotto_2020_CHANGE_seq_dataset(self, dataset_dict: dict, fold) -> dict:
        if self.config["fold"] == "all":
            dataset_dict = self.update_dataset(dataset_dict=dataset_dict, train_sgrna=self.included_sgrna_list, train_idx=list(range(len(self.included_sgrna_list))))
            return dataset_dict
        else:
            """
            Split sgrna list like train_list->["sgrna1", "sgrna2", ...], test_list->["sgrna6", "sgrna7", ..."]
            """
            split_sgrna_path = self.database_dir + self.config["paths"]["off_target_dataset"][self.dataset_name]["sgrna_split_list"]
            sgrna_fold_list = file_handlers.load_csv_list(split_sgrna_path)  # ['0,1,2,3,4,5', 'sgrna1, sgrna2, ...', 'sgrna6, sgrna7, ...']
            sgrna_fold_list = [row for i, row in enumerate(sgrna_fold_list) if i != 0]  # remove header row ('0,1,2,3,4,5') # len(sgrna_fold_list) == 10 (for fold 0~9)
            
            # Fold 0~9
            if 0 <= self.config["fold"] and self.config["fold"] <= 9:
                # Test sgRNA list
                test_sgrna_fold_list = sgrna_fold_list[self.config["fold"]]
                test_sgrna_fold_list = [sgrna for sgrna in test_sgrna_fold_list.split(",") if sgrna != ""]  # remove empty strings
                test_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in test_sgrna_fold_list]
                # Train sgRNA list
                train_sgrna_fold_list = []
                for i, sgrnas in enumerate(sgrna_fold_list):
                    if i != self.config["fold"]:
                        train_sgrna_fold_list += [sgrna for sgrna in sgrnas.split(",") if sgrna != ""]
                train_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in train_sgrna_fold_list]
                dataset_dict = self.update_dataset(dataset_dict=dataset_dict, train_sgrna=train_sgrna_fold_list, test_sgrna=test_sgrna_fold_list, train_idx=train_idx, test_idx=test_idx)
                return dataset_dict
            
            # Fold10~13
            else:
                # Load sgRNA split list
                new_guideseq_sgrna_list_path = self.database_dir + self.config["paths"]["off_target_dataset"]["Lazzarotto_2020_GUIDE_seq"]["sgrna_list_new"]
                sgrna_fold_list_new = file_handlers.load_csv_list(new_guideseq_sgrna_list_path)  # ["sgrna1", "sgrna2", ..., "sgrna20"]
                sgrna_fold_list_new = [sgrna_fold_list_new[0:5], sgrna_fold_list_new[5:10], sgrna_fold_list_new[10:15], sgrna_fold_list_new[15:20]]
                # Test sgRNA list (sgrna_fold_list_new include sgrnas for fold 10~13)
                test_sgrna_fold_list = sgrna_fold_list_new[self.config["fold"] - 10] # Subtract 10 to get the index for fold 10~13
                test_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in test_sgrna_fold_list]
                # Train sgRNA list
                train_sgrna_fold_list = [i for i in self.included_sgrna_list if i not in test_sgrna_fold_list]
                train_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in train_sgrna_fold_list]
                dataset_dict = self.update_dataset(dataset_dict=dataset_dict, train_sgrna=train_sgrna_fold_list, test_sgrna=test_sgrna_fold_list, train_idx=train_idx, test_idx=test_idx)
                return dataset_dict
    
    def return_Lazzarotto_2020_CHANGE_seq_dataset(self) -> dict:
        # Load Lazzarotto 2020 CHANGE-seq dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split.
        dataset_dict = self.split_Lazzarotto_2020_CHANGE_seq_dataset(dataset_dict, self.config["fold"])
        return dataset_dict

    def split_Lazzarotto_2020_GUIDE_seq_dataset(self, dataset_dict: dict, fold) -> dict:
        # Train / Test split
        if fold == "all":
            dataset_dict = self.update_dataset(
                dataset_dict=dataset_dict, train_sgrna=self.included_sgrna_list, train_idx=list(range(len(dataset_dict["sgrna"])))
            )
        else:
            # File load
            sgrna_split_list_path = self.database_dir + self.config["paths"]["off_target_dataset"][self.dataset_name]["sgrna_split_list"]
            new_sgrna_list_path = self.database_dir + self.config["paths"]["off_target_dataset"][self.dataset_name]["sgrna_list_new"]
            sgrna_split_list = file_handlers.load_csv_list(sgrna_split_list_path)
            new_sgrna_list = file_handlers.load_csv_list(new_sgrna_list_path)
            # Process
            sgrna_split_list = [row for i, row in enumerate(sgrna_split_list) if i != 0]  # remove header row
            sgrna_split_list = [row.split(",") for row in sgrna_split_list]
            sgrna_split_list = [[item for item in row if item != ""] for row in sgrna_split_list]
            new_sgrna_list = [new_sgrna_list[0:5], new_sgrna_list[5:10], new_sgrna_list[10:15], new_sgrna_list[15:20]]
            sgrna_fold_list = sgrna_split_list + new_sgrna_list
            # Fold 0~13
            # Test sgRNA list
            test_sgrna_fold_list = sgrna_fold_list[fold]
            test_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in test_sgrna_fold_list]
            # Train sgRNA list
            train_sgrna_fold_list = [sgrna for sgrna in self.included_sgrna_list if sgrna not in test_sgrna_fold_list]
            train_idx = [i for i, sgrna in enumerate(dataset_dict["sgrna"]) if sgrna in train_sgrna_fold_list]
            dataset_dict = self.update_dataset(dataset_dict=dataset_dict, train_sgrna=train_sgrna_fold_list, test_sgrna=test_sgrna_fold_list, train_idx=train_idx, test_idx=test_idx)
        return dataset_dict
    
    def return_Lazzarotto_2020_GUIDE_seq_dataset(self) -> dict:
        # Load Lazzarotto 2020 GUIDE-seq dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split
        dataset_dict = self.split_Lazzarotto_2020_GUIDE_seq_dataset(dataset_dict, self.config["fold"])
        return dataset_dict

    def split_Chen_2017_GUIDE_seq_Listgarten_2018_GUIDE_seq_dataset(self, dataset_dict: dict, fold) -> dict:
        # Return all dataset in train and test no matter what fold is set.
        sgrna_list = dataset_dict["sgrna"]
        included_sgrna_list = list(set(sgrna_list))
        dataset_dict = self.update_dataset(
            dataset_dict=dataset_dict, train_sgrna=included_sgrna_list, test_sgrna=included_sgrna_list, 
            train_idx=list(range(len(sgrna_list))), test_idx=list(range(len(sgrna_list)))
        )
        return dataset_dict
        
    def return_Chen_2017_GUIDE_seq_dataset(self) -> dict:
        # Load Chen 2017 GUIDE-seq dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split
        dataset_dict = self.split_Chen_2017_GUIDE_seq_Listgarten_2018_GUIDE_seq_dataset(dataset_dict, self.config["fold"])
        return dataset_dict
    
    def return_Listgarten_2018_GUIDE_seq_dataset(self) -> dict:
        # Load Listgarten 2018 GUIDE-seq dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split
        dataset_dict = self.split_Chen_2017_GUIDE_seq_Listgarten_2018_GUIDE_seq_dataset(dataset_dict, self.config["fold"])
        return dataset_dict
    
    def split_Tsai_2015_GUIDE_seq_dataset(self, dataset_dict: dict, fold) -> dict:
        # Return all dataset in train and test no matter what fold is set.
        sgrna_list = dataset_dict["sgrna"]
        included_sgrna_list = list(set(sgrna_list))
        dataset_dict = self.update_dataset(
            dataset_dict=dataset_dict, train_sgrna=included_sgrna_list, test_sgrna=included_sgrna_list, 
            train_idx=list(range(len(sgrna_list))), test_idx=list(range(len(sgrna_list)))
        )
        return dataset_dict
    
    def return_Tsai_2015_GUIDE_seq_dataset(self) -> dict:
        # Load Tsai 2015 GUIDE-seq dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split
        dataset_dict = self.split_Tsai_2015_GUIDE_seq_dataset(dataset_dict, self.config["fold"])
        return dataset_dict
    
    def split_SchmidBurgk_2020_TTISS_dataset(self, dataset_dict: dict, fold) -> dict:
        # Train / Test split
        if fold == "all":
            dataset_dict = self.update_dataset(
                dataset_dict=dataset_dict, train_sgrna=self.included_sgrna_list, train_idx=list(range(len(dataset_dict["sgrna"])))
            )
        else:
            # Split self.included_sgrna_list into ten folds forward
            sgrna_list = dataset_dict["sgrna"]
            splited_sgrna_list = [self.included_sgrna_list[i:i + 6] for i in range(0, len(self.included_sgrna_list), 6)]
            test_sgrna_fold_list = splited_sgrna_list[fold]
            train_sgrna_fold_list = [sgrna for sublist in splited_sgrna_list if sublist != test_sgrna_fold_list for sgrna in sublist]
            train_idx = [i for i, sgrna in enumerate(sgrna_list) if sgrna in train_sgrna_fold_list]
            test_idx = [i for i, sgrna in enumerate(sgrna_list) if sgrna in test_sgrna_fold_list]
            dataset_dict = self.update_dataset(dataset_dict=dataset_dict, train_sgrna=train_sgrna_fold_list, test_sgrna=test_sgrna_fold_list, train_idx=train_idx, test_idx=test_idx)
        return dataset_dict
    
    def return_SchmidBurgk_2020_TTISS_dataset(self) -> dict:
        # Load Schmid-Burgk 2020 TTISS dataset
        dataset_dict = self.load_and_convert_to_dict()
        # Train / Test split
        dataset_dict = self.split_SchmidBurgk_2020_TTISS_dataset(dataset_dict, self.config["fold"])
        return dataset_dict
