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
parser = argparse.ArgumentParser(description="Preprocess.")
parser.add_argument("--config_path", "-cp", type=str, required=False, default="config.yaml", help="Path to the configuration YAML file.")
args, remaining = parser.parse_known_args()

# Load the configuration file
config = load_yaml(args.config_path)

# Set working directory
working_dir = config["paths"]["working_dir"]
os.chdir(working_dir)
print(f"{os.getcwd()}/src/run_preprocess.py is running.")

import utils.check_set as check_set
import utils.file_handlers as file_handlers
import utils.check_set as check_set
import models.data_loader as data_loader
import models.dnabert_module as dnabert_module
import models.dnabert2_module as dnabert2_module
import models.gru_embed_2024 as gru_embed_2024
import models.crispr_bert_2024 as crispr_bert_2024
import models.crispr_hw_2023 as crispr_hw_2023
import models.crispr_dipoff_2025 as crispr_dipoff_2025
import models.crispr_bert_2025 as crispr_bert_2025

# Rest of Parser
parser.add_argument("--model", "-m", type=check_set.check_model_arg, required=True, default="DNABERT", help="Model to use for training or inference.")
parser.add_argument("--dataset", "-ds", type=str, required=False, default="Lazzarotto_2020_CHANGE_seq", help="Dataset to use for training or inference.")
# dataset: Lazzarotto_2020_CHANGE_seq, Lazzarotto_2020_GUIDE_seq, Chen_2017_GUIDE_seq, Listgarten_2018_GUIDE_seq, SchmidBurgk_2020_TTISS
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset

config["model_info"]["model_name"] = model_name
config["dataset_name"]["dataset_current"] = dataset_name
config["fold"] = "all"

# Path check
SetPathsClass = check_set.SetPaths(config)
config = SetPathsClass.set_seq_preprocess_path()


def main() -> None:
    # Load dataset
    DataLoaderClass = data_loader.DataLoaderClass(config)
    dataset = DataLoaderClass.load_dataset()
    
    if config["model_info"]["model_name"] == "DNABERT":
        DataProcessorDNABERT = dnabert_module.DataProcessorDNABERT(config)
        dataset = DataProcessorDNABERT.preprocess_inputs(dataset)
    
    elif config["model_info"]["model_name"] == "GRU-Embed":
        DataProcessorGRUEmbed = gru_embed_2024.DataProcessorGRUEmbed(config)
        dataset = DataProcessorGRUEmbed.preprocess_inputs(dataset)
        
    elif config["model_info"]["model_name"] == "CRISPR-HW":
        DataProcessorCRISPRHW = crispr_hw_2023.DataProcessorCRISPRHW(config)
        dataset = DataProcessorCRISPRHW.preprocess_inputs(dataset)
    
    elif config["model_info"]["model_name"] == "CRISPR-DIPOFF":
        DataProcessorCrisprDipoff = crispr_dipoff_2025.DataProcessorCrisprDipoff(config)
        dataset = DataProcessorCrisprDipoff.preprocess_inputs(dataset)
    
    elif config["model_info"]["model_name"] == "CRISPR-BERT":
        DataProcessorCRISPRBERT = crispr_bert_2024.DataProcessorCRISPRBERT(config)
        DataProcessorCRISPRBERT.preprocess_inputs(dataset)
        
    elif config["model_info"]["model_name"] == "CrisprBERT":
        DataProcessorCrisprBERT = crispr_bert_2025.DataProcessorCrisprBERT(config)
        DataProcessorCrisprBERT.preprocess_inputs(dataset)

if __name__ == "__main__":
    main()