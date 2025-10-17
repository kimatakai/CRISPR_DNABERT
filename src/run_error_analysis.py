import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set working directory
working_dir = "/mnt/c/Users/tynkk/home/research/bioinformatics/1_crispr_off_on_target_2025"
# working_dir = "/home/kimata/home/1_crispr_off_on_target_2025"
os.chdir(working_dir)
print(f"Working directory: {os.getcwd()}")

import argparse
import json
import numpy as np
import tqdm
import multiprocessing
from multiprocessing import Pool
import utils.file_handlers as file_handlers
import utils.sequence_module as sequence_module
import utils.check_set as check_set
import utils.error_analysis as error_analysis
import utils.epigenetic_module as epigenetic_module
import models.data_loader as data_loader
import models.result as result

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run Deep Machine Learning.")
parser.add_argument("--config_path", "-cp", type=str, required=True, default="config.yaml", help="Path to the configuration YAML file.")
parser.add_argument("--model", "-m", type=str, required=True, default="DNABERT-2", help="Model to use for training or inference.")
parser.add_argument("--dataset", "-ds", type=str, default="Lazzarotto_2020_CHANGE_seq", help="Dataset to use for training or inference.")
parser.add_argument("--fold", "-f", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13", help="Fold numbers for cross-validation. Separate multiple folds with commas.")
parser.add_argument("--iter", "-i", type=str, default="0", help="Iteration numbers for cross-validation. Separate multiple folds with commas.")
parser.add_argument("--exe_type", "-exe", type=str, default="direct", choices=["direct", "screening-2"], help="Execution type: 'direct' for training or 'screening' for screening mode.")
args = parser.parse_args()
config_path = args.config_path
model_name = args.model
dataset_name = args.dataset
fold_list = args.fold
fold_list = [int(f) for f in fold_list.split(",")]
iter_list = args.iter
iter_list = [int(i) for i in iter_list.split(",")]
if model_name not in ["DNABERT-2", "GRU-Embed", "CRISPR-BERT-2024"]:
    raise ValueError(f"Model {model_name} is not supported.")

config = file_handlers.load_yaml(config_path) # -> dict
config["model_info"]["model_name"] = model_name
config["dataset_name"]["dataset_dl"] = dataset_name
config["random_seed"] = None
config["train"] = False
config["test"] = False
config["exe_type"] = args.exe_type

# config["dataset_name"]["dataset_dl"] = "Lazzarotto_2020_CHANGE_seq"
# config["dataset_name"]["dataset_dl"] = "Lazzarotto_2020_GUIDE_seq"
# config["dataset_name"]["dataset_dl"] = "Chen_2017_GUIDE_seq"
# config["dataset_name"]["dataset_dl"] = "Listgarten_2018_GUIDE_seq"
# config["dataset_name"]["dataset_dl"] = "Lazzarotto_2020_CSGS"

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len)
        return (padded_seq_rna, padded_seq_dna)
    
    def preprocess_inputs(self, dataset: dict) -> dict:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        rna_seq_list = dataset["rna_seq"]
        dna_seq_list = dataset["dna_seq"]
        
        # Prepare the arguments for multiprocessing
        worker_args = [(seq_rna, seq_dna, self.max_pairseq_len) for seq_rna, seq_dna in zip(rna_seq_list, dna_seq_list)]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_alignment_hyphen, worker_args), total=len(worker_args), desc="Processing sequences"))
        
        dataset["padded_rna_seq"] = [seq[0] for seq in _processed_seqs]
        dataset["padded_dna_seq"] = [seq[1] for seq in _processed_seqs]
        return dataset


def main():
    # Load result
    global config
    aggregate_result_dict = result.aggregate_results(config, fold_list, iter_list)
    
    # Load dataset
    DataLoaderClass = data_loader.DataLoaderClass(config)
    dataset = DataLoaderClass.load_dataset()
    dataProcessor = DataProcessor(config)
    # dataset = dataProcessor.preprocess_inputs(data_loader.DataLoaderClass(config).load_dataset())
    
    # Mismatch analysis
    # error_analysis.mismatch_analysis_by_cm(config, dataset, aggregate_result_dict)
    
    # Epigenetic analysis
    for type in ["atac"]:
        config = check_set.set_epigenetic(config, type)
        # Load epigenetic data
        epigenetic_feature_array = epigenetic_module.load_npz(config["signal_array_path_list"])
        error_analysis.epigenetic_analysis_by_cm(config, dataset, epigenetic_feature_array, aggregate_result_dict, type)
    

if __name__ == "__main__":
    main()
