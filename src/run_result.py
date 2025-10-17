import os
import yaml
import argparse
import time

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

import utils.check_set as check_set
import utils.result_module as result_module

# Rest of Parser
parser.add_argument("--models", "-m", type=str, required=False, default="GRU-Embed,CRISPR-BERT,CRISPR-HW,CRISPR-DIPOFF,CrisprBERT,DNABERT", help="Models.")
parser.add_argument("--dataset", "-ds", type=str, default="Lazzarotto_2020_GUIDE_seq", 
                    help="dataset: Lazzarotto_2020_CHANGE_seq, Lazzarotto_2020_GUIDE_seq, Chen_2017_GUIDE_seq, Listgarten_2018_GUIDE_seq, SchmidBurgk_2020_TTISS")
parser.add_argument("--folds", "-f", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Fold numbers.")
parser.add_argument("--iters", "-i", type=str, default="0,1,2,3,4", help="Iteration numbers.")
parser.add_argument("--with_epigenetic", "-epi", action="store_true")
parser.add_argument("--using_epi_data", "-uepi", type=str, default="", help="Type of epigenetic data to use. Options: atac, h3k4me3")
parser.add_argument("--exe_type", "-exe", type=str, default="scratch", choices=["scratch", "transfer", "softt"], help="")
parser.add_argument("--include_epi_transfer", required=False, action="store_true", help="Include epigenetic data in transfer learning.")
args = parser.parse_args()
config_path = args.config_path
models_name = args.models
dataset_name = args.dataset
folds = args.folds
iters = args.iters
with_epigenetic = args.with_epigenetic
using_epi_data = args.using_epi_data
exe_type = args.exe_type
include_epi_transfer = args.include_epi_transfer

# Set configuration parameters
config["model_info"]["models_name"] = [str(model_name) for model_name in models_name.split(",")]
config["dataset_name"]["dataset_current"] = dataset_name
config["folds"] = [int(f) for f in folds.split(",")]
config["iters"] = [int(i) for i in iters.split(",")]
config["with_epigenetic"] = with_epigenetic
config["using_epi_data"] = sorted([str(x) for x in using_epi_data.split(",")])
config["exe_type"] = exe_type
config["include_epi_transfer"] = include_epi_transfer

# Check and update configuration
SetConfigClass = check_set.SetPathsResult(config)
config = SetConfigClass.set_path()


def main() -> None:
    # Results processing for EXCEL workbook
    ResultsWorkbookClass = result_module.ResultsWorkbook(config)
    # Create EXCEL workbook
    ResultsWorkbookClass.create_workbook()
    ResultsWorkbookClass.add_result_aggregation_sheet()
    ResultsWorkbookClass.add_wilcoxon_pvalue_sheet()
    ResultsWorkbookClass.add_confusion_matrix_sheet()
    ResultsWorkbookClass.add_execution_time_sheet()
    
    # # Visualization for figure
    ResultForVisualizationClass = result_module.ResultForVisualization(config)
    ResultForVisualizationClass.plot_box_plot()
    return

if __name__ == "__main__":
    main()