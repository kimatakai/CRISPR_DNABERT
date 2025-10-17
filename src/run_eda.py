import os

# Set working directory
working_dir = "/mnt/c/Users/tynkk/home/research/bioinformatics/1_crispr_off_on_target_2025"
os.chdir(working_dir)
print(f"Working directory: {os.getcwd()}")

import sys
import argparse
import numpy as np
import utils.file_handlers as file_handlers
import eda.ots_dist_analysis as ots_dist_analysis
import eda.cpg_analysis as cpg_analysis

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run EDA on CRISPR data.")
parser.add_argument("--config_path", "-cp", type=str, required=True, default="config.yaml", help="Path to the configuration YAML file.")
parser.add_argument("--venn_analysis", action="store_true", help="Perform Venn analysis between CHANGE-seq and GUIDE-seq datasets.")
parser.add_argument("--dsb_count", action="store_true", help="Generate DSB histogram for all datasets.")
parser.add_argument("--cpg_eda", action="store_true")
args = parser.parse_args()
config_path = args.config_path
if_venn_analysis = args.venn_analysis
if_dsb_count = args.dsb_count
if_cpg_eda = args.cpg_eda


config = file_handlers.load_yaml(config_path)


def main():
    
    # Perform Venn diagram analysis between Lazzarotto_2020_CHANGE-seq and Lazzarotto_2020_GUIDE-seq datasets
    if if_venn_analysis:
        print("Performing Venn analysis between CHANGE-seq and GUIDE-seq datasets...")
        ots_dist_analysis.venn_analysis_between_CS_GS(config)
    
    # Generate DSB histogram for all datasets
    if if_dsb_count:
        print("Performing DSB count analysis for all datasets...")
        ots_dist_analysis.dsb_count_analysis(config)
    
    # Perform sequence EDA
    if if_cpg_eda:
        print("Performing CpG EDA...")
        cpg_analysis.run_cpg_analysis_CSGS(config)

if __name__ == "__main__":
    main()