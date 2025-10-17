import os
import yaml
import argparse
import time
import shap

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
print(f"{os.getcwd()}/src/run_model_interpret.py is running.")

import utils.check_set as check_set
import utils.epigenetic_module as epigenetic_module
import utils.interpretation_module as interpretation_module
import models.data_loader as data_loader
import models.dnabert_module as dnabert_module

# Rest of Parser
parser.add_argument("--folds", "-f", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Fold numbers.")
parser.add_argument("--iters", "-i", type=str, default="0,1,2,3,4", help="Iteration numbers.")
parser.add_argument("--dataset_in_cellula", "-dsc", type=str, default=None, help="Dataset in cellula to use for training or inference. e.g., Lazzarotto_2020_GUIDE_seq")
parser.add_argument("--using_epi_data", "-uepi", type=str, default="", help="atac,h3k27ac,h3k27me3,h3k36me3,h3k4me1,h3k4me3,h3k9me3")
args = parser.parse_args()
config_path = args.config_path
dataset_name_in_cellula = args.dataset_in_cellula
model_name = "DNABERT"
dataset_name_in_vitro = "Lazzarotto_2020_CHANGE_seq"
folds = args.folds
iters = args.iters
with_epigenetic = True
using_epi_data = args.using_epi_data
exe_type = "transfer"

# Set configuration parameters
config["model_info"]["model_name"] = model_name
config["dataset_name"]["dataset_in_cellula"] = dataset_name_in_cellula
config["dataset_name"]["dataset_in_vitro"] = dataset_name_in_vitro
config["dataset_name"]["dataset_current"] = dataset_name_in_cellula
config["folds"] = [int(f) for f in folds.split(",")]
config["iters"] = [int(i) for i in iters.split(",")]
config["with_epigenetic"] = with_epigenetic
config["using_epi_data"] = sorted([str(x) for x in using_epi_data.split(",")]) if using_epi_data != "" else []
config["exe_type"] = exe_type


def main():
    global config
    # Load dataset
    DataLoaderClass = data_loader.DataLoaderClass(config)
    DataLoaderClass.load_sgrna_list()
    dataset_dict = DataLoaderClass.load_and_convert_to_dict() # All data are included
    
    # Load epigenetic data
    # Check and update configuration
    SetPathsShapClass = check_set.SetPathsShap(config)
    config = SetPathsShapClass.set_path()
    SetPathsEpigeneticClass = check_set.SetPathsEpigenetic(config)
    config = SetPathsEpigeneticClass.set_path_for_model(config["using_epi_data"])
    dataset_dict = epigenetic_module.load_epigenetic_feature(config = config, dataset_dict = dataset_dict)
    
    # SHAP analysis
    for fold in config["folds"]:
        for iter in config["iters"]:
            config["fold"] = fold
            config["iter"] = iter
            # data_loader split
            # Load splited dataset
            dataset_dict = DataLoaderClass.split_dataset(dataset_dict, fold)
            DataProcessorDNABERTClass = dnabert_module.DataProcessorDNABERT(config)
            dataset_dict = DataProcessorDNABERTClass.load_inputs(dataset_dict)
            # SHAP value calculation
            DNABERTEpiShapInterpretationClass = dnabert_module.DNABERTEpiShapInterpretationClass(config, fold, iter)
            DNABERTEpiShapInterpretationClass.calculate_shap_values(dataset_dict)
    
    # SHAP importance analysis
    SHAPAnalysisClass = dnabert_module.SHAPAnalysisClass(config)
    SHAPAnalysisClass.calculate_and_plot_aggregated_shap_importance()
    SHAPAnalysisClass.calculate_and_plot_position_importance()
    
    # Attention weights and integrated gradients analysis
    SetPathsBERTAnalysis = check_set.SetPathsBERTAnalysis(config)
    config = SetPathsBERTAnalysis.set_path()
    for fold in config["folds"]:
        for iter in config["iters"]:
            config["fold"] = fold
            config["iter"] = iter
            # data_loader split
            # Load splited dataset
            dataset_dict = DataLoaderClass.split_dataset(dataset_dict, fold)
            DataProcessorDNABERTClass = dnabert_module.DataProcessorDNABERT(config)
            dataset_dict = DataProcessorDNABERTClass.load_inputs(dataset_dict)
            # Attention weights and integrated gradients calculation
            DNABERTEpiBERTAnalysisClass = dnabert_module.DNABERTExplainAnalysis(config, dataset_dict)
            # DNABERTEpiBERTAnalysisClass.attention_weights_analysis()
            DNABERTEpiBERTAnalysisClass.integrated_gradient_analysis()
    IGAnalysisClass = interpretation_module.IGAnalysisClass(config)
    IGAnalysisClass.ig_attribution_umap()
            

if __name__ == "__main__":
    main()

