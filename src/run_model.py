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

# Rest of Parser
parser.add_argument("--model", "-m", type=check_set.check_model_arg, required=True, default="DNABERT", help="Model to use for training or inference.")
parser.add_argument("--dataset_in_cellula", "-dsc", type=str, default=None, help="Dataset in cellula to use for training or inference. e.g., Lazzarotto_2020_GUIDE_seq")
parser.add_argument("--dataset_in_vitro", "-dsv", type=str, default=None, help="Dataset in vitro to use for training. e.g., Lazzarotto_2020_CHANGE_seq")
parser.add_argument("--fold", "-f", type=check_set.check_fold_arg, default=0, help="Fold number for cross-validation. Default is 0.")
parser.add_argument("--iter", "-i", type=int, default=0, help="Iteration number for cross-validation. Default is 0.")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--with_epigenetic", "-epi", action="store_true")
parser.add_argument("--using_epi_data", "-uepi", type=str, default="", help="atac,h3k27ac,h3k27me3,h3k36me3,h3k4me1,h3k4me3,h3k9me3")
parser.add_argument("--exe_type", "-exe", type=str, default="scratch", choices=["scratch", "transfer", "softt"], help="")
args = parser.parse_args()
config_path = args.config_path
model_name = args.model
dataset_name_in_cellula = args.dataset_in_cellula
dataset_name_in_vitro = args.dataset_in_vitro
fold = args.fold
iter = args.iter
if_train = args.train
if_test = args.test
with_epigenetic = args.with_epigenetic
using_epi_data = args.using_epi_data
exe_type = args.exe_type

# Set configuration parameters
config["model_info"]["model_name"] = model_name
config["dataset_name"]["dataset_in_cellula"] = dataset_name_in_cellula
config["dataset_name"]["dataset_in_vitro"] = dataset_name_in_vitro
config["fold"] = fold
config["iter"] = iter
config["random_seed"] = iter + 42
config["train"] = if_train
config["test"] = if_test
config["with_epigenetic"] = with_epigenetic
config["using_epi_data"] = sorted([str(x) for x in using_epi_data.split(",")]) if with_epigenetic else []
config["exe_type"] = exe_type

# Check and update configuration
CheckConfigClass = check_set.CheckConfig(config)
config = CheckConfigClass.check_run_model_config()



def main() -> None:
    start_time = time.time()
    # Branch depending on the execution type
    program = config["program"]

    if program == "scratch":
        import models.model_scratch as model_scratch
        model_scratch.model_scratch(config)
    elif program == "transfer":
        import models.model_transfer as model_transfer
        model_transfer.model_transfer(config)
    elif program == "transfer_epi":
        import models.model_transfer_epi as model_transfer_epi
        model_transfer_epi.model_transfer_epi(config)
    elif program == "softt":
        pass
    elif program == "softt_epi":
        pass
    else:
        raise ValueError(f"Unknown execution program: {program}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{os.getcwd()}/src/run_model.py -m {model_name} -dsc {dataset_name_in_cellula} -dsv {dataset_name_in_vitro} -f {fold} -i {iter} -exe {exe_type}")
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()