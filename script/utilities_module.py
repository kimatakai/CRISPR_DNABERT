

import sys
sys.path.append("script/")
import os

import config



def return_model_weight_path(model_name: str, datatype: str, fold: int, exp_id: int):
    if model_name == "dnabert":
        model_weight_path = f"{config.dnabert_crispr_finetuned_path}/dnabert_crispr_{datatype}_fold{fold}_exp{exp_id}"
    elif model_name == "dnabert-no-pretrain":
        model_weight_path = f"{config.dnabert_crispr_no_pretrain_path}/dnabert_no_pretrain_{datatype}_fold{fold}_exp{exp_id}"
    elif model_name == "dnabert-epi":
        model_weight_path = f"{config.dnabert_epigenetic_model_path}/dnabert_epi_{datatype}_fold{fold}_exp{exp_id}"
    elif model_name == "dnabert-epi-ablation":
        model_weight_path = f"{config.dnabert_epigenetic_model_path}/dnabert_epi_ablation_{datatype}_fold{fold}_exp{exp_id}"
    elif model_name == "crispr-bert":
        model_weight_path = f"{config.crispr_bert_model_path}/crispr_bert_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "gru-embed":
        model_weight_path = f"{config.gru_embed_model_path}/gru_embed_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "crispr-hw":
        model_weight_path = f"{config.crispr_hw_model_path}/crispr_hw_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "crispr-dipoff":
        model_weight_path = f"{config.crispr_dipoff_model_path}/crispr_dipoff_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "crispr-bert-2025":
        model_weight_path = f"{config.crispr_bert_2025_model_path}/crispr_bert_2025_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "fnn":
        model_weight_path = f"{config.fnn_model_path}/epigenetic_False_{datatype}_fold{fold}_exp{exp_id}.pth"
    elif model_name == "fnn-epi":
        model_weight_path = f"{config.fnn_model_path}/epigenetic_True_{datatype}_fold{fold}_exp{exp_id}.pth"
    else:
        sys.exit(f"[ERROR] Invalid argument raised on return_model_weight_path.")
    return model_weight_path


def return_output_probability_path(model_name: str, datatype: str, fold: int, exp_id: int):
    if model_name == "dnabert":
        probabilities_path = f"{config.probabilities_base_dir_path}/dnabert/dnabert_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "dnabert-no-pretrain":
        probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_no_pretrain/dnabert_no_pretrain_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "dnabert-epi":
        probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_epi/dnabert_epi_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "dnabert-epi-ablation":
        probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_epi_ablation/dnabert_epi_ablation_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "crispr-bert":
        probabilities_path = f"{config.probabilities_base_dir_path}/crispr_bert/crispr_bert_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "gru-embed":
        probabilities_path = f"{config.probabilities_base_dir_path}/gru_embed/gruembed_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "crispr-hw":
        probabilities_path = f"{config.probabilities_base_dir_path}/crispr_hw/crispr_hw_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "crispr-dipoff":
        probabilities_path = f"{config.probabilities_base_dir_path}/crispr_dipoff/crispr_dipoff_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "crispr-bert-2025":
        probabilities_path = f"{config.probabilities_base_dir_path}/crispr_bert_2025/crispr_bert_2025_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "fnn":
        probabilities_path = f"{config.probabilities_base_dir_path}/fnn_model/fnn_epigenetic_False_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    elif model_name == "fnn-epi":
        probabilities_path = f"{config.probabilities_base_dir_path}/fnn_model/fnn_epigenetic_True_probabilities_{datatype}_fold{fold}_exp{exp_id}.npy"
    else:
        sys.exit(f"[ERROR] Invalid argument raised on return_output_probability_path.")
    return probabilities_path


def check_model_weight_exist(model_name: str, datatype: str, fold: int, exp_id: int):
    model_weight_path = return_model_weight_path(model_name, datatype, fold, exp_id)
    if not os.path.exists(model_weight_path):
        sys.exit(f"[ERROR] Model weight file does not exist. Run [python3 main.py -d {datatype} -m {model_name} -f {fold} -e {exp_id} --train]")

def check_output_probability_exist(model_name: str, datatype: str, fold: int, exp_id: int):
    probabilities_path = return_output_probability_path(model_name, datatype, fold, exp_id)
    if not os.path.exists(probabilities_path):
        sys.exit(f"[ERROR] Output probability file does not exist. Run [python3 main.py -d {datatype} -m {model_name} -f {fold} -e {exp_id} --test]")



