import os
import yaml
import argparse
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Set up the argument parser for config file
parser = argparse.ArgumentParser(description="Run Deep Machine Learning.")
parser.add_argument("--config_path", "-cp", type=str, required=False, default="config.yaml", help="Path to the configuration YAML file.")
args, remaining = parser.parse_known_args()

# Load configuration
config = load_yaml(args.config_path)

# Set working directory
working_dir = config["paths"]["working_dir"]
os.chdir(working_dir)
print(f"{os.getcwd()}/src/run_result_analysis.py is running.")

import utils.check_set as check_set
import eda.eda_module as eda_module

parser.add_argument("--model", "-m", type=check_set.check_model_arg, required=True, default="DNABERT", help="Model to use for training or inference.")
parser.add_argument("--dataset", "-ds", type=str, default="Lazzarotto_2020_GUIDE_seq", 
                    help="dataset: Lazzarotto_2020_CHANGE_seq, Lazzarotto_2020_GUIDE_seq, Chen_2017_GUIDE_seq, Listgarten_2018_GUIDE_seq, SchmidBurgk_2020_TTISS")
parser.add_argument("--folds", "-f", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Fold numbers.")
parser.add_argument("--iters", "-i", type=str, default="0,1,2,3,4", help="Iteration numbers.")
parser.add_argument("--with_epigenetic", "-epi", action="store_true")
parser.add_argument("--exe_type", "-exe", type=str, default="scratch", choices=["scratch", "transfer", "softt"], help="")
parser.add_argument("--function", "-func", type=str, required=True, 
                    help="analysis_atac_confusion_mm")
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
folds = args.folds
iters = args.iters
with_epigenetic = args.with_epigenetic
exe_type = args.exe_type
function = args.function

# Set configuration parameters
config["model_info"]["model_name"] = model_name
config["dataset_name"]["dataset_current"] = dataset_name
config["folds"] = [int(f) for f in folds.split(",")]
config["iters"] = [int(i) for i in iters.split(",")]
config["with_epigenetic"] = with_epigenetic
config["exe_type"] = exe_type

# Check and update configuration
SetConfigClass = check_set.SetPathsResultAnalysis(config)
config = SetConfigClass.set_path()

def main():
    
    if function == "analysis_atac_confusion_mm":
        EdaEpigeneticAnalysisClass = eda_module.EdaEpigeneticAnalysis(config)
        EdaEpigeneticAnalysisClass.analysis_atac_confusion_mm()
    elif function == "analysis_h3k4me3_confusion_mm":
        EdaEpigeneticAnalysisClass = eda_module.EdaEpigeneticAnalysis(config)
        EdaEpigeneticAnalysisClass.analysis_h3k4me3_confusion_mm()
    elif function == "analysis_h3k27ac_confusion_mm":
        EdaEpigeneticAnalysisClass = eda_module.EdaEpigeneticAnalysis(config)
        EdaEpigeneticAnalysisClass.analysis_h3k27ac_confusion_mm()
    
    
    pass

if __name__ == "__main__":
    main()
