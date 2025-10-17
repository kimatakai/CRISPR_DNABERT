import os
import yaml
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error loading YAML file: {file_path}. Error: {e}")

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run Deep Machine Learning.")
parser.add_argument("--config_path", "-cp", type=str, required=False, default="config.yaml", help="Path to the configuration YAML file.")
args, remaining = parser.parse_known_args()

# Load the configuration file
config = load_yaml(args.config_path)

# Set working directory
working_dir = config["paths"]["working_dir"]
os.chdir(working_dir)
print(f"{os.getcwd()}/src/run_model.py is running.")

import utils.file_handlers as file_handlers
import utils.check_set as check_set
import utils.epigenetic_module as epigenetic_module
import models.data_loader as data_loader

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run Deep Machine Learning.")
parser.add_argument("--dataset", "-ds", type=str, default="Lazzarotto_2020_CHANGE_seq", help="Dataset to use for training or inference.")
parser.add_argument("--type", "-t", type=str, default="atac", help="Type of epigenetic data to process (e.g., 'atac').")
args = parser.parse_args()
dataset_name = args.dataset
type_of_data = args.type

config["dataset_name"]["dataset_current"] = dataset_name
config["type_of_data"] = type_of_data

# config["dataset_name"]["dataset_dl"] = "Lazzarotto_2020_CHANGE_seq"
# config["dataset_name"]["dataset_dl"] = "Lazzarotto_2020_GUIDE_seq"
# config["dataset_name"]["dataset_dl"] = "Chen_2017_GUIDE_seq"
# config["dataset_name"]["dataset_dl"] = "SchmidBurgk_2020_TTISS"

SetPathsEpigeneticClass = check_set.SetPathsEpigenetic(config)
config = SetPathsEpigeneticClass.set_epigenetic_path(type_of_data)

def main():
    DataLoaderClass = data_loader.DataLoaderClass(config)
    dataset = DataLoaderClass.load_dataset()
    bigwig_path_list = config["paths"]["epigenetic"][type_of_data]["bigwig_current"]
    signal_array_path_list = config["paths"]["epigenetic"][type_of_data]["npz_current"]
    for bigwig_path_i, npz_path_i in zip(bigwig_path_list, signal_array_path_list):
        if epigenetic_module.exist_npz_file(config, dataset, npz_path_i) == False:
            epigenetic_module.save_bigwig_data_as_npz(config, dataset, bigwig_path_i, npz_path_i)
    
    
    
    

if __name__ == "__main__":
    main()
