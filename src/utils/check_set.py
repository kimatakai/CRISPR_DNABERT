
import os
from typing import Union


###################################################################################################


VALID_MODELS = ["DNABERT", "DNABERT-No-Pretrained", "DNABERT-2", "GRU-Embed", "CRISPR-BERT", "CRISPR-HW", "CRISPR-DIPOFF", "CrisprBERT", "Ensemble"]

def check_fold_arg(arg):
    if arg == "all":
        return "all"
    try:
        return int(arg)
    except ValueError:
        raise ValueError(f"Invalid fold argument: {arg}. It should be an integer or 'all'.")

def check_model_arg(arg):
    if arg not in VALID_MODELS:
        raise ValueError(f"Invalid model argument: {arg}. Valid models are {VALID_MODELS}.")
    return arg


###################################################################################################


def check_fold(fold: Union[str, int], dataset_name: str) -> None:
    if dataset_name == "Lazzarotto_2020_CHANGE_seq":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid folds are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "Lazzarotto_2020_GUIDE_seq":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid folds are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "Chen_2017_GUIDE_seq":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid fold are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "Listgarten_2018_GUIDE_seq":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid fold are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "Tsai_2015_GUIDE_seq_1":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid folds are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "Tsai_2015_GUIDE_seq_2":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid folds are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13].')
    elif dataset_name == "SchmidBurgk_2020_TTISS":
        if fold not in ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError(f'Fold {fold} is not valid for dataset {dataset_name}. Valid folds are ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9].')


def check_dataset_valid(dataset_name_in_cellula: str, dataset_name_in_vitro: str) -> None:
    valid_datasets_in_cellula = ["Lazzarotto_2020_GUIDE_seq", "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2", "SchmidBurgk_2020_TTISS", None]
    valid_datasets_in_vitro = ["Lazzarotto_2020_CHANGE_seq", None]
    if dataset_name_in_cellula not in valid_datasets_in_cellula:
        raise ValueError(f"Invalid dataset_in_cellula: {dataset_name_in_cellula}. Valid datasets are {valid_datasets_in_cellula}.")
    if dataset_name_in_vitro not in valid_datasets_in_vitro:
        raise ValueError(f"Invalid dataset_in_vitro: {dataset_name_in_vitro}. Valid datasets are {valid_datasets_in_vitro}.")
    if dataset_name_in_cellula == None and dataset_name_in_vitro is None:
        raise ValueError("At least one dataset must be specified: either dataset_in_cellula or dataset_in_vitro.")


def decide_execution_program(exe_type: str, with_epigenetic: bool) -> str:
    execution_program = {
        ("scratch", False): "scratch",
        ("transfer", False): "transfer",
        ("transfer", True): "transfer_epi",
        ("softt", False): "softt",
        ("softt", True): "softt_epi"
    }
    return execution_program[(exe_type, with_epigenetic)]



###################################################################################################


class CheckConfig:
    def __init__(self, config: dict):
        self.config = config
    
    def check_run_model_config(self) -> dict:
        # Load the needed arguments from config
        dataset_name_in_cellula = self.config["dataset_name"]["dataset_in_cellula"]
        dataset_name_in_vitro = self.config["dataset_name"]["dataset_in_vitro"]
        fold = self.config["fold"]
        with_epigenetic = self.config["with_epigenetic"]
        exe_type = self.config["exe_type"]
        
        # Check
        check_dataset_valid(dataset_name_in_cellula, dataset_name_in_vitro)
        for dataset_name in [dataset_name_in_cellula, dataset_name_in_vitro]:
            check_fold(fold, dataset_name)
        
        # Update config
        program = decide_execution_program(exe_type, with_epigenetic)
        self.config.update({
            "program": program
        })
        return self.config


class SetPaths:
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        self.model_name = config["model_info"]["model_name"]

    def return_seq_input_path(self, dataset_name: str) -> dict:
        # Return the path for sequence input data
        input_dir_path = self.database_dir + self.config["paths"]["input"][dataset_name]
        if not os.path.exists(input_dir_path):
            os.makedirs(input_dir_path, exist_ok=True)
        
        # Check the model type and return the appropriate input path
        if self.model_name in ["DNABERT", "DNABERT-No-Pretrained"]:
            return {"input_path": input_dir_path + f'/DNABERT_dataset'}
        elif self.model_name == "GRU-Embed":
            return {"input_path": input_dir_path + f'/GRU_Embed_input.pt'}
        elif self.model_name == "CRISPR-BERT":
            return {
                "encode_input_path": input_dir_path + f'/CRISPR_BERT_encode_input.pt',
                "token_input_path": input_dir_path + f'/CRISPR_BERT_token_input.pt'
            }
        elif self.model_name == "CRISPR-HW":
            return {"input_path": input_dir_path + f'/CRISPR_HW_input.pt'}
        elif self.model_name == "CRISPR-DIPOFF":
            return {"input_path": input_dir_path + f'/CRISPR_DIPOFF_input.pt'}
        elif self.model_name == "CrisprBERT":
            return {"input_path": input_dir_path + f'/CrisprBERT_input.pt'}
        else:
            raise ValueError("Model is not supported for sequence input path.")
    
    def return_pretrained_model_path_for_preprocess(self) -> None:
        if self.config["model_info"]["model_name"] == "DNABERT":
            pretrained_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
            self.config["model_info"].update({
                "pretrained_model": pretrained_path
            })

    def set_seq_preprocess_path(self) -> dict:
        dataset_name = self.config["dataset_name"]["dataset_current"]
        seq_input_path = self.return_seq_input_path(dataset_name)
        self.config["input_data_paths"] = seq_input_path # input_data_paths > input_path
        # For models which need pretrained model path
        self.return_pretrained_model_path_for_preprocess()
        return self.config
    

class SetTrainAndTest(SetPaths):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def check_train_test(self) -> None:
        if_train = self.config["train"]
        if self.dataset_current in ["Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2"] and if_train:
            raise ValueError(f"Training is not allowed for dataset {self.dataset_current}. Only testing is allowed.")


class SetPathsScratch(SetTrainAndTest):
    def __init__(self, config: dict):
        super().__init__(config)
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.dataset_name_in_cellula = config["dataset_name"]["dataset_in_cellula"]
        self.dataset_name_in_vitro = config["dataset_name"]["dataset_in_vitro"]
    
    def set_dataset_current(self) -> None:
        if self.dataset_name_in_cellula and self.dataset_name_in_vitro:
            raise ValueError("For scratch, only one dataset should be specified: either dataset_in_cellula or dataset_in_vitro.")
        self.dataset_current = self.dataset_name_in_cellula if self.dataset_name_in_cellula else self.dataset_name_in_vitro
        self.config["dataset_name"]["dataset_current"] = self.dataset_current
    
    def set_model_path(self) -> None:
        # Set the model path based on the model name
        if self.model_name == "DNABERT":
            # If the case of model from scratch, consider DNABERT fine-tuned on pair-task as pre-trained model.
            pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
            # target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_current}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "pretrained_model": pretrained_model_path,
                "model_path": model_path
            })
        elif self.model_name == "DNABERT-No-Pretrained":
            # If the case of model from scratch, consider DNABERT pre-trained model as pre-trained model.
            pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned_no_pretrain"] + "/"
            # target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_current}" + f"/scratch_no_pretrained" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/scratch_no_pretrained" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "pretrained_model": pretrained_model_path,
                "model_path": model_path
            })
        elif self.model_name in ["GRU-Embed", "CRISPR-HW", "CRISPR-DIPOFF", "CrisprBERT"]:
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_current}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "model_path": model_path
            })
        elif self.model_name == "CRISPR-BERT":
            # CRISPR-BERT uses base model
            base_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + self.config["paths"]["model"][self.model_name]["base_model"] + "/"
            # Target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_current}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "base_model": base_model_path,
                "model_path": model_path
            })
    
    def set_result_path(self) -> None:
        result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.json"
        prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.npy"
        time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.txt"
        for path in [result_path, prob_path, time_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.config["paths"].update({
            "result_path": result_path,
            "probability_path": prob_path,
            "time_path": time_path
        })
            
    def set_path(self) -> dict:
        # Set current dataset which is used for model training and inference (not pretraining)
        self.set_dataset_current()
        # Set the input data paths for loading the dataset
        self.config.update({"input_data_paths": self.return_seq_input_path(self.dataset_current)}) # input_data_paths > input_path
        # Set model paths
        self.set_model_path()
        # Set result paths
        self.set_result_path()
        # Check train and test
        self.check_train_test()
        
        print("Path of model: ", self.config["model_info"]["model_path"])
        print("Path of result: ", self.config["paths"]["result_path"])
        print("Path of probability: ", self.config["paths"]["probability_path"])
        print("Path of time: ", self.config["paths"]["time_path"])

        return self.config


class SetPathsTransfer(SetTrainAndTest):
    def __init__(self, config: dict):
        super().__init__(config)
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.dataset_name_in_cellula = config["dataset_name"]["dataset_in_cellula"]
        self.dataset_name_in_vitro = config["dataset_name"]["dataset_in_vitro"]
    
    def set_dataset_current(self) -> None:
        if not self.dataset_name_in_cellula or not self.dataset_name_in_vitro:
            raise ValueError("For transfer, both dataset_in_cellula and dataset_in_vitro must be specified.")
        self.dataset_current = self.dataset_name_in_cellula
        self.config["dataset_name"]["dataset_current"] = self.dataset_current
    
    def set_model_path(self) -> None:
        # Set the model path based on the model name
        if self.model_name == "DNABERT":
            # pre-training model fine-tuned on pair-task
            pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
            # pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pretrained"] + "/"
            # In vitro model fine-tuned on in-vitro dataset from scratch.
            if self.dataset_name_in_cellula == "Lazzarotto_2020_GUIDE_seq":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            elif self.dataset_name_in_cellula == "SchmidBurgk_2020_TTISS":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/foldall_iter{self.iter}.pth"
            else: # "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq" not used in vitro model
                in_vitro_model_path = None
            # Check pretrained model path
            if not os.path.exists(pretrained_model_path):
                raise ValueError(f"Pretrained model path does not exist: {pretrained_model_path}. Please run python3 run_model.py -ds {self.dataset_name_in_vitro} -exe scratch.")
            # target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_current}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
                # model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/SchmidBurgk_2020_TTISS" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "pretrained_model": pretrained_model_path,
                "in_vitro_model": in_vitro_model_path,
                "model_path": model_path
            })
        elif self.model_name in ["GRU-Embed", "CRISPR-HW", "CRISPR-DIPOFF", "CrisprBERT"]:
            # If the case of transfer learning, consider GRU-Embed or CrisprBERT fine-tuned on in-vitro dataset from scratch as pre-trained model.
            if self.dataset_name_in_cellula == "Lazzarotto_2020_GUIDE_seq":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            elif self.dataset_name_in_cellula == "SchmidBurgk_2020_TTISS":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/foldall_iter{self.iter}.pth"
            else: # "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq" not used in vitro model
                in_vitro_model_path = None
            # Check in vitro model path
            if in_vitro_model_path != None and not os.path.exists(in_vitro_model_path):
                raise ValueError(f"In vitro model path does not exist: {in_vitro_model_path}. Please run python3 run_model.py -ds {self.dataset_name_in_vitro} -exe scratch.")
            # target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_current}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
                # model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/SchmidBurgk_2020_TTISS" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "in_vitro_model": in_vitro_model_path,
                "model_path": model_path,
            })
        elif self.model_name == "CRISPR-BERT":
            # CRISPR-BERT uses base model
            base_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + self.config["paths"]["model"][self.model_name]["base_model"] + "/"
            # If the case of transfer learning, consider CRISPR-BERT fine-tuned on in-vitro dataset from scratch as in-vitro model.
            if self.dataset_name_in_cellula == "Lazzarotto_2020_GUIDE_seq":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            elif self.dataset_name_in_cellula == "SchmidBurgk_2020_TTISS":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/foldall_iter{self.iter}.pth"
            else: # "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq" not used in vitro model
                in_vitro_model_path = None
            # Check in vitro model path
            if in_vitro_model_path != None and not os.path.exists(in_vitro_model_path):
                raise ValueError(f"In vitro model path does not exist: {in_vitro_model_path}. Please run python3 run_model.py -ds {self.dataset_name_in_vitro} -exe scratch.")
            # Target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + f"/{self.dataset_current}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
                # model_path = self.database_dir + self.config["paths"]["model"][self.model_name]["base_dir"] + "/SchmidBurgk_2020_TTISS" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.config["model_info"].update({
                "base_model": base_model_path,
                "in_vitro_model": in_vitro_model_path,
                "model_path": model_path
            })
    
    def set_result_path(self) -> None:
        result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.json"
        prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.npy"
        time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.txt"
        for path in [result_path, prob_path, time_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.config["paths"].update({
            "result_path": result_path,
            "probability_path": prob_path,
            "time_path": time_path
        })
    
    def set_path(self) -> dict:
        # Set current dataset which is used for model training and inference (not pretraining)
        self.set_dataset_current()
        # Set the input data paths for loading the dataset
        self.config.update({"input_data_paths": self.return_seq_input_path(self.dataset_current)}) # input_data_paths > input_path
        # Set model paths
        self.set_model_path()
        # Set result paths
        self.set_result_path()
        # Check train and test
        self.check_train_test()
        
        print("Path of model: ", self.config["model_info"]["model_path"]) ############## 
        print("Path of in vitro model: ", self.config["model_info"].get("in_vitro_model", "N/A"))
        print("Path of result: ", self.config["paths"]["result_path"])
        print("Path of probability: ", self.config["paths"]["probability_path"])
        print("Path of time: ", self.config["paths"]["time_path"])
        
        return self.config

class SetPathsTransferEpi(SetTrainAndTest):
    def __init__(self, config: dict):
        super().__init__(config)
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.dataset_name_in_cellula = config["dataset_name"]["dataset_in_cellula"]
        self.dataset_name_in_vitro = config["dataset_name"]["dataset_in_vitro"]
        self.using_epi_data = config["using_epi_data"]
        self.using_epi_data_str = "_".join(self.using_epi_data) if self.using_epi_data else "noepi"
    
    def set_dataset_current(self) -> None:
        if not self.dataset_name_in_cellula or not self.dataset_name_in_vitro:
            raise ValueError("For transfer, both dataset_in_cellula and dataset_in_vitro must be specified.")
        self.dataset_current = self.dataset_name_in_cellula
        self.config["dataset_name"]["dataset_current"] = self.dataset_current
    
    def set_model_path(self) -> None:
        # Set the model path based on the model name
        if self.model_name == "DNABERT":
            # pre-training model fine-tuned on pair-task
            pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
            # In vitro model fine-tuned on in-vitro dataset from scratch.
            if self.dataset_name_in_cellula == "Lazzarotto_2020_GUIDE_seq":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{self.fold}_iter{self.iter}.pth"
            elif self.dataset_name_in_cellula == "SchmidBurgk_2020_TTISS":
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/foldall_iter{self.iter}.pth"
            else: # "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq" not used in vitro model
                in_vitro_model_path = None
                
            # Check pretrained model path
            if not os.path.exists(pretrained_model_path):
                raise ValueError(f"Pretrained model path does not exist: {pretrained_model_path}. Please run python3 run_model.py -ds {self.dataset_name_in_vitro} -exe scratch.")
            
            # In-cellula fine-tuned model without epigenetic data
            if self.dataset_current in ["Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                in_cellula_no_epi_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_current}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                in_cellula_no_epi_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.pth"
            # if not os.path.exists(in_cellula_no_epi_model_path):
            #     raise ValueError(f"In-cellula no-epigenetic model path does not exist: {in_cellula_no_epi_model_path}. Please run python3 run_model.py -ds {self.dataset_name_in_cellula} -exe transfer.")
            
            # target model
            if self.dataset_current in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_current}" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{self.fold}_iter{self.iter}.pth"
            else:
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + "/Lazzarotto_2020_GUIDE_seq" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{self.fold}_iter{self.iter}.pth"
                if not os.path.exists(model_path):
                    raise ValueError(f"Model path does not exist: {model_path}. Please run python3 run_model.py -ds Lazzarotto_2020_GUIDE_seq -exe transfer -epi {','.join(self.using_epi_data)}.")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.config["model_info"].update({
                "pretrained_model": pretrained_model_path,
                "in_vitro_model": in_vitro_model_path,
                "in_cellula_no_epi_model": in_cellula_no_epi_model_path,
                "model_path": model_path
            })
    
    def set_result_path(self) -> None:
        result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{self.fold}_iter{self.iter}.json"
        prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{self.fold}_iter{self.iter}.npy"
        time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{self.fold}_iter{self.iter}.txt"
        testing_prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}.npy"
        training_prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_current}" + f"/{self.model_name}" + f"/transfer" + f"/fold{self.fold}_iter{self.iter}_train.npy"
        for path in [result_path, prob_path, time_path, training_prob_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.config["paths"].update({
            "result_path": result_path,
            "probability_path": prob_path,
            "time_path": time_path,
            "training_prob_path": training_prob_path,
            "testing_prob_path": testing_prob_path
        })
    
    def set_path(self) -> dict:
        # Set current dataset which is used for model training and inference (not pretraining)
        self.set_dataset_current()
        # Set the input data paths for loading the dataset
        self.config.update({"input_data_paths": self.return_seq_input_path(self.dataset_current)}) # input_data_paths > input_path
        # Set model paths
        self.set_model_path()
        # Set result paths
        self.set_result_path()
        # Check train and test
        self.check_train_test()
        
        print("Path of model: ", self.config["model_info"]["model_path"])
        print("Path of in vitro model: ", self.config["model_info"].get("in_vitro_model", "N/A"))
        print("Path of result: ", self.config["paths"]["result_path"])
        print("Path of probability: ", self.config["paths"]["probability_path"])
        print("Path of time: ", self.config["paths"]["time_path"])
        
        return self.config


class SetPathsResult:
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        
        self.models_name = config["model_info"]["models_name"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.with_epigenetic = config["with_epigenetic"]
        self.using_epi_data = config["using_epi_data"]
        print("Using epigenetic data: ", self.using_epi_data)
        self.using_epi_data_str = "_".join(self.using_epi_data) if self.using_epi_data else "noepi"
        self.exe_type = config["exe_type"]
        self.program = decide_execution_program(self.exe_type, self.with_epigenetic) # str
        if self.program in ["transfer_epi"]:
            self.program += f"_{self.using_epi_data_str}"
        

    def check_models(self) -> None:
        for model_name in self.models_name:
            if model_name not in VALID_MODELS:
                raise ValueError(f"Invalid model argument: {model_name}. Valid models are {VALID_MODELS}.")
    
    def set_paths_dict(self) -> None:
        prob_path_dict = {}
        result_path_dict = {}
        for model_name in self.models_name:
            prob_path_list = []
            result_path_list = []
            for fold in self.folds_list:
                for iter in self.iters_list:
                    prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_name}" + f"/{model_name}" + f"/{self.program}" + f"/fold{fold}_iter{iter}.npy"
                    result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_name}" + f"/{model_name}" + f"/{self.program}" + f"/fold{fold}_iter{iter}.json"
                    for path in [prob_path, result_path]:
                        if not os.path.exists(path):
                            raise ValueError(f"Path does not exist: {path}.")
                    prob_path_list.append(prob_path)
                    result_path_list.append(result_path)
            prob_path_dict[model_name] = prob_path_list
            result_path_dict[model_name] = result_path_list
        self.config["probability_paths"] = prob_path_dict
        self.config["result_paths"] = result_path_dict
        
    def set_time_path(self) -> None:
        if self.dataset_name in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
            time_path_dict = {}
            for model_name in self.models_name:
                time_path_list = []
                for fold in self.folds_list:
                    for iter in self.iters_list:
                        if model_name == "Ensemble":
                            time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_name}" + f"/DNABERT" + f"/{self.program}" + f"/fold{fold}_iter{iter}.txt"
                        else:
                            time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_name}" + f"/{model_name}" + f"/{self.program}" + f"/fold{fold}_iter{iter}.txt"
                        if not os.path.exists(time_path):
                            raise ValueError(f"Path does not exist: {time_path}.")
                        time_path_list.append(time_path)
                time_path_dict[model_name] = time_path_list
            self.config["time_paths"] = time_path_dict

    def add_epi_model_path(self) -> None:
        if self.config["include_epi_transfer"] and self.program == "transfer":
            self.config["model_info"]["models_name"].append(f"DNABERT-Epi")
            prob_path_list = []
            result_path_list = []
            for fold in self.folds_list:
                for iter in self.iters_list:
                    prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_name}" + f"/DNABERT" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{fold}_iter{iter}.npy"
                    result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_name}" + f"/DNABERT" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{fold}_iter{iter}.json"
                    for path in [prob_path, result_path]:
                        if not os.path.exists(path):
                            raise ValueError(f"Path does not exist: {path}.")
                    prob_path_list.append(prob_path)
                    result_path_list.append(result_path)
            self.config["probability_paths"].update({f"DNABERT-Epi": prob_path_list})
            self.config["result_paths"].update({f"DNABERT-Epi": result_path_list})

    def add_epi_time_path(self) -> None:
        if self.config["include_epi_transfer"] and self.program == "transfer":
            if self.dataset_name in ["Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS"]:
                time_path_list = []
                for fold in self.folds_list:
                    for iter in self.iters_list:
                        time_path = self.database_dir + self.config["paths"]["time"]["base_dir"] + f"/{self.dataset_name}" + f"/DNABERT" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{fold}_iter{iter}.txt"
                        if not os.path.exists(time_path):
                            raise ValueError(f"Path does not exist: {time_path}.")
                        time_path_list.append(time_path)
                # self.config["time_paths"].extend(time_path_list)
                self.config["time_paths"].update({f"DNABERT-Epi": time_path_list})

    def set_excel_path(self) -> None:
        excel_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + "/excel" + f"/{self.program}" + f"/{self.dataset_name}.xlsx"
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        self.config["excel_path"] = excel_path
        print(self.config["excel_path"])
    
    def set_result_fig_path(self) -> None:
        fig_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["figure"] + f"/{self.dataset_name}" + f"/{self.program}"   
        os.makedirs(fig_dir_path, exist_ok=True)
        self.config["fig_dir_path"] = fig_dir_path

    def set_path(self) -> dict:
        # Check models
        self.check_models()
        
        # Set paths information
        self.set_paths_dict()
        # Set time path
        self.set_time_path()
        
        # Add epi model path if needed
        self.add_epi_model_path()
        self.add_epi_time_path()
        
        # Set excel path
        self.set_excel_path()
        
        # Set figure directory path
        self.set_result_fig_path()
        return self.config

class SetPathsEnsembleResult(SetPathsResult):
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        
        self.models_name = config["model_info"]["models_name"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.with_epigenetic = config["with_epigenetic"]
        self.using_epi_data = config["using_epi_data"]
        self.using_epi_data_str = "_".join(self.using_epi_data) if self.using_epi_data else "noepi"
        self.exe_type = config["exe_type"]
        self.program = decide_execution_program(self.exe_type, self.with_epigenetic) # str
        if self.program in ["transfer_epi"]:
            self.program += f"_{self.using_epi_data_str}"
    
    def set_excel_path_ensemble(self) -> None:
        excel_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + "/excel" + f"/ensemble" + f"/{self.dataset_name}.xlsx"
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        self.config["excel_path"] = excel_path
        print("Ensemble Excel file path: ", self.config["excel_path"])
    
    def set_ensemble_prob_result_path(self) -> None:
        prob_path_dict = {}
        result_path_dict = {}
        for fold in self.folds_list:
            for iter in self.iters_list:
                prob_path = self.database_dir + self.config["paths"]["probability"]["base_dir"] + f"/{self.dataset_name}" + f"/Ensemble" + f"/{self.program}" + f"/fold{fold}_iter{iter}.npy"
                result_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/{self.dataset_name}" + f"/Ensemble" + f"/{self.program}" + f"/fold{fold}_iter{iter}.json"
                for path in [prob_path, result_path]:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                prob_path_dict[(fold, iter)] = prob_path
                result_path_dict[(fold, iter)] = result_path
        self.config["probability_paths_ensemble"] = prob_path_dict
        self.config["result_paths_ensemble"] = result_path_dict
    
    def set_path_ensemble(self) -> dict:
        # Check models
        self.check_models()
        # Set paths information
        self.set_paths_dict()
        # Set DNABERT-Epi model path
        self.add_epi_model_path()
        # Set excel path for ensemble
        self.set_excel_path_ensemble()
        # Set ensemble probability and result paths
        self.set_ensemble_prob_result_path()
        print(self.config["model_info"]["models_name"])
        # Set figure directory path
        self.set_result_fig_path()
        return self.config
        

class SetPathsResultAnalysis(SetPathsResult):
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        
        self.model_name = config["model_info"]["model_name"]
        self.models_name = [self.model_name]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.with_epigenetic = config["with_epigenetic"]
        self.exe_type = config["exe_type"]
        self.program = decide_execution_program(self.exe_type, self.with_epigenetic) # str
    
    def set_fig_path(self) -> None:
        fig_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["figure"]        
        os.makedirs(fig_dir_path, exist_ok=True)
        self.config["fig_dir_path"] = fig_dir_path

    def set_path(self) -> dict:
        # Set paths information
        paths_dict = self.return_paths_dict()
        self.config.update({
            "probability_paths": paths_dict["probability_paths"],
            "result_paths": paths_dict["result_paths"]
        })
        self.set_fig_path()
        return self.config
    
class SetPathsEpigenetic:
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        
    def set_epigenetic_path(self, type_of_data: str) -> dict:
        dir_path = self.database_dir + self.config["paths"]["epigenetic"]["base_dir"] + self.config["paths"]["epigenetic"][type_of_data]["base_dir"]
        bw_path_list = [dir_path + _path for _path in self.config["paths"]["epigenetic"][type_of_data]["bigwig"][self.dataset_name]]
        npz_path_list = [dir_path + _path for _path in self.config["paths"]["epigenetic"][type_of_data]["npz"][self.dataset_name]]
        self.config["paths"]["epigenetic"][type_of_data].update({
            "bigwig_current": bw_path_list, "npz_current": npz_path_list
        })
        return self.config
    
    def check_path(self, type_of_data: str) -> None:
        bw_path_list = self.config["paths"]["epigenetic"][type_of_data]["bigwig_current"]
        npz_path_list = self.config["paths"]["epigenetic"][type_of_data]["npz_current"]
        for path in bw_path_list + npz_path_list:
            if not os.path.exists(path):
                raise ValueError(f"Path does not exist: {path}. Please check the epigenetic data paths.")
    
    def set_path_for_model(self, type_of_data_list: list) -> dict:
        for type_of_data in type_of_data_list:
            self.set_epigenetic_path(type_of_data)
            self.check_path(type_of_data)
        return self.config

class SetPathsShap:
    def __init__(self, config: dict):
        self.config = config
        self.database_dir = config["paths"]["database_dir"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        self.dataset_name_in_vitro = config["dataset_name"]["dataset_in_vitro"]
        self.dataset_name_in_cellula = config["dataset_name"]["dataset_in_cellula"]
        self.model_name = config["model_info"]["model_name"]
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.using_epi_data = config["using_epi_data"]
        self.using_epi_data_str = "_".join(self.using_epi_data) if self.using_epi_data else "noepi"
    
    def set_input_path(self) -> None:
        input_dir_path = self.database_dir + self.config["paths"]["input"][self.dataset_name]
        self.config["input_data_paths"] = {}
        self.config["input_data_paths"]["input_path"] = input_dir_path + f'/DNABERT_dataset'

    def set_model_path(self) -> None:
        model_path_dict = {}
        for fold in self.folds_list:
            for iter in self.iters_list:
                pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{fold}_iter{iter}.pth"
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_cellula}" + f"/transfer_epi_{self.using_epi_data_str}" + f"/fold{fold}_iter{iter}.pth"
                model_path_dict[(fold, iter)] = {
                    "pretrained_model": pretrained_model_path,
                    "in_vitro_model": in_vitro_model_path,
                    "model_path": model_path
                }
        self.config["model_info"]["pretrained_model"] = pretrained_model_path
        self.config["model_info"].update(model_path_dict)

    def set_shap_info_path(self) -> None:
        shap_values_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["shap"] + f"/shap_values"
        os.makedirs(shap_values_dir_path, exist_ok=True)
        shap_lanking_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["shap"] + f"/shap_lanking.tsv"
        os.makedirs(os.path.dirname(shap_lanking_path), exist_ok=True)
        self.config["shap_values_dir_path"] = shap_values_dir_path
        self.config["shap_lanking_path"] = shap_lanking_path
    
    def set_shap_figure_path(self) -> None:
        shap_fig_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["figure"] + "/shap"
        os.makedirs(shap_fig_dir_path, exist_ok=True)
        self.config["shap_fig_dir_path"] = shap_fig_dir_path

    def set_path(self) -> dict:
        self.set_input_path()
        self.set_model_path()
        self.set_shap_info_path()
        self.set_shap_figure_path()
        return self.config

class SetPathsBERTAnalysis:
    def __init__(self, config: dict):
        self.config = config
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.database_dir = config["paths"]["database_dir"]
        self.dataset_name = config["dataset_name"]["dataset_current"]
        self.dataset_name_in_vitro = config["dataset_name"]["dataset_in_vitro"]
        self.dataset_name_in_cellula = config["dataset_name"]["dataset_in_cellula"]
    
    def set_input_path(self) -> None:
        input_dir_path = self.database_dir + self.config["paths"]["input"][self.dataset_name]
        self.config["input_data_paths"] = {}
        self.config["input_data_paths"]["input_path"] = input_dir_path + f'/DNABERT_dataset'
    
    def set_model_path(self) -> None:
        model_path_dict = {}
        for fold in self.folds_list:
            for iter in self.iters_list:
                pretrained_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + self.config["paths"]["model"]["DNABERT"]["pair_finetuned"] + "/"
                in_vitro_model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/{self.dataset_name_in_vitro}" + f"/scratch" + f"/fold{fold}_iter{iter}.pth"
                model_path = self.database_dir + self.config["paths"]["model"]["DNABERT"]["base_dir"] + f"/Lazzarotto_2020_GUIDE_seq" + f"/transfer" + f"/fold{fold}_iter{iter}.pth"
                model_path_dict[(fold, iter)] = {
                    "pretrained_model": pretrained_model_path,
                    "in_vitro_model": in_vitro_model_path,
                    "model_path": model_path
                }
        self.config["model_info"]["pretrained_model"] = pretrained_model_path
        self.config["model_info"].update(model_path_dict)
    
    def set_analysis_result_path(self) -> None:
        analysis_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + f"/BERT_analysis_{self.dataset_name}"
        os.makedirs(analysis_dir_path, exist_ok=True)
        self.config["analysis_dir_path"] = analysis_dir_path

    def set_figure_path(self) -> None:
        fig_dir_path = self.database_dir + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["figure"] + f"/BERT_analysis_{self.dataset_name}"
        os.makedirs(fig_dir_path, exist_ok=True)
        self.config["fig_dir_path"] = fig_dir_path
    
    def set_path(self) -> dict:
        self.set_input_path()
        self.set_model_path()
        self.set_analysis_result_path()
        self.set_figure_path()
        return self.config