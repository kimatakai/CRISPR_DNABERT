
import sys
sys.path.append("script/")
sys.path.append("./")
sys.path.append("../")
import os
import warnings
warnings.filterwarnings('ignore')

import config
from script import utilities_module, data_loader, visualize_module, dnabert_module

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tqdm
        
import argparse
parser = argparse.ArgumentParser(description='DNABERT model aalysis')
parser.add_argument("-d", "--datatype", type=str, default="changeseq", choices=["changeseq", "guideseq", "transfer"], help="")
parser.add_argument("-f", "--fold", type=str, default="0,1,2,3,4,5,6,7,8,9", help="")
parser.add_argument("-e", "--exp_id", type=str, default="0,1,2,3,4", help="")
parser.add_argument("--attention_score", action="store_true")

args = parser.parse_args()

# Processing command argument
try:
    fold_list = [int(fold) for fold in args.fold.split(",")]
except:
    sys.exit(f"[ERROR] Invalid fold.")
try:
    exp_id_list = [int(exp_id) for exp_id in args.exp_id.split(",")]
except:
    sys.exit(f"[ERROR] Invalid experiment id.")


class DataLoaderClassForDNABERTAnalysis:
    def __init__(self, datatype: str, fold_list: list, exp_id_list: list):
        self.datatype = datatype
        self.fold_list = fold_list
        self.exp_id_list = exp_id_list
        self.seq_len = 47
        self.kmer = config.kmer
        
        if self.datatype == "changeseq":
            self.dataset_file_path = f"{config.yaish_et_al_data_path}/CHANGEseq/include_on_targets/CHANGEseq_CR_Lazzarotto_2020_dataset.csv"
            self.sgrna_list_path = f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_list.csv"
            self.sgrna_fold_path = f"{config.yaish_et_al_data_path}/CHANGEseq_sgRNAs_folds_split.csv"
        elif self.datatype == "guideseq" or self.datatype == "transfer":
            self.dataset_file_path = f"{config.yaish_et_al_data_path}/GUIDEseq/include_on_targets/GUIDEseq_CR_Lazzarotto_2020_dataset.csv"
            self.sgrna_list_path = f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_list.csv"
            self.sgrna_fold_path = f"{config.yaish_et_al_data_path}/GUIDEseq_sgRNAs_folds_split.csv"
        
    
    def load_dataset(self) -> pd.DataFrame:
        '''
        output:
        pd.DataFrame
        '''
        self.dataset_df = pd.read_csv(self.dataset_file_path)
        self.flag_load_dataset = True
        return self.dataset_df
    
    def load_test_info(self) -> dict:
        '''
        output:
        dict_type data
        test_sgrna_names, test_sgrna_seqs
        '''
        # sgRNA,chrom,SiteWindow,Align.strand,Align.chromStart,Align.chromEnd,Align.off-target,Align.sgRNA,Align.#Mismatches,Align.#Bulges,reads
        # Load sgRNA information
        sgRNAs_json = data_loader.load_sgrna_name()
        sgrna_seq2name_dict = {sgrna_seq:sgrna_name for sgrna_seq, sgrna_name in zip(sgRNAs_json["sgRNAs_seq"], sgRNAs_json["sgRNAs_name"])}
        fold_sgrna_df = pd.read_csv(self.sgrna_fold_path)

        # Test
        self.fold_sgrna_info_dict = {"test_names_list": {}, "test_seq_list": {}}
        for f in self.fold_list:
            test_sgrna_seq_list = [sgrna_seq for sgrna_seq in fold_sgrna_df.iloc[f].values.tolist() if type(sgrna_seq) == str]
            test_sgrna_name_list = [sgrna_seq2name_dict[sgrna_seq] for sgrna_seq in test_sgrna_seq_list]
            self.fold_sgrna_info_dict["test_names_list"][f] = test_sgrna_name_list
            self.fold_sgrna_info_dict["test_seq_list"][f] = test_sgrna_seq_list
        self.flag_fold_sgrna_info_dict = True
        return self.fold_sgrna_info_dict
    
    def seq_to_token(self, seq):
        if len(seq) == self.seq_len - 1:
            seq = "-" + seq
        kmers_list = [seq[i:i+self.kmer] for i in range(len(seq)-self.kmer+1)]
        merged_sequence = ' '.join(kmers_list)
        return merged_sequence
    
    def return_input_data(self, fold, exp_id) -> dict:
        
        if not self.flag_load_dataset:
            self.load_dataset()
        if not self.flag_fold_sgrna_info_dict:
            self.load_test_info()
        
        self.input_dict = {"target_dna": [], "sgrna": []}
        
        # Load dataset
        dataset_df_test = self.dataset_df[self.dataset_df["sgRNA"].isin(self.fold_sgrna_info_dict["test_seq_list"][fold])]
        # Split offtarget or non-offtarget
        data_df_test_offtarget = dataset_df_test[dataset_df_test["reads"] >= 1]
        data_df_test_non_offtarget = dataset_df_test[dataset_df_test["reads"] == 0]
        
        # offtarget
        for row in data_df_test_offtarget.itertuples(index=False):
            target_dna_token = self.seq_to_token(row._6)
            sgrna_token = self.seq_to_token(row._7)
            self.input_dict["target_dna"].append(target_dna_token)
            self.input_dict["sgrna"].append(sgrna_token)
        
        # non-offtarget
        for row in data_df_test_non_offtarget.itertuples(index=False):
            target_dna_token = self.seq_to_token(row._6)
            sgrna_token = self.seq_to_token(row._7)
            self.input_dict["target_dna"].append(target_dna_token)
            self.input_dict["sgrna"].append(sgrna_token)
        
        return self.input_dict
        
    
    def return_predict_label(self, fold, exp_id) -> dict:
        # Load probability output
        probability_path = utilities_module.return_output_probability_path("dnabert", self.datatype, fold, exp_id)
        if not os.path.exists(probability_path):
            sys.exit(f"[ERROR] Probability file does not exist. Run [python3 main.py -d {self.datatype} -m dnabert -f {fold} -e {exp_id} --test]")
        probability_array = np.load(probability_path)
        
        # Convert to predict label
        predict_label_array = np.argmax(probability_array, axis=1)
            
        return predict_label_array
    
    def return_sampled_input_data(self, input_dict, predict_label_array) -> dict:
        # Get label0 (Non off-target) index random sampling (n=100 or less)
        label0_index = np.where(predict_label_array == 0)[0]
        if len(label0_index) > 100:
            label0_index = np.random.choice(label0_index, 100, replace=False)
        # Get label1 (Off-target) index random sampling (n=100 or less)
        label1_index = np.where(predict_label_array == 1)[0]
        if len(label1_index) > 100:
            label1_index = np.random.choice(label1_index, 100, replace=False)
        
        # Sampling offtarget and non-offtarget input token data
        offtarget_input = {"target_dna": [], "sgrna": []}
        non_offtarget_input = {"target_dna": [], "sgrna": []}
        # For off-target
        for i in label1_index:
            offtarget_input["target_dna"].append(input_dict["target_dna"][i])
            offtarget_input["sgrna"].append(input_dict["sgrna"][i])
        # For non-offtarget
        for i in label0_index:
            non_offtarget_input["target_dna"].append(input_dict["target_dna"][i])
            non_offtarget_input["sgrna"].append(input_dict["sgrna"][i])
        
        # Convert to Dataset
        offtarget_input_dataset = Dataset.from_dict(offtarget_input)
        non_offtarget_input_dataset = Dataset.from_dict(non_offtarget_input)
        
        return {"offtarget": offtarget_input_dataset, "non_offtarget": non_offtarget_input_dataset}
    

    def return_attention_weight(self) -> dict:
        offtarget_all_attention_ = []
        non_offtarget_all_attention_ = []
        for f in self.fold_list:
            for e in self.exp_id_list:
                offtarget_attention_array_temp_path = f"{config.attention_weight_base_dir_path}/dnabert_attention_weight_active_{self.datatype}_fold{f}_exp{e}.npy"
                non_offtarget_attention_array_temp_path = f"{config.attention_weight_base_dir_path}/dnabert_attention_weight_inactive_{self.datatype}_fold{f}_exp{e}.npy"
                if not os.path.exists(offtarget_attention_array_temp_path) or not os.path.exists(non_offtarget_attention_array_temp_path):
                    sys.exit(f"[ERROR] Attention weight file does not exist. Run [python3 ./script/dnabert_analysis.py -d {self.datatype} -f {f} -e {e} --attention_score]")
                offtarget_attention_array_temp = np.load(offtarget_attention_array_temp_path) # (12, 47, 47)
                non_offtarget_attention_array_temp = np.load(non_offtarget_attention_array_temp_path) # (12, 47, 47)
                offtarget_all_attention_.append(offtarget_attention_array_temp)
                non_offtarget_all_attention_.append(non_offtarget_attention_array_temp)
        offtarget_all_attention = np.array(offtarget_all_attention_) # (N, 12, 47, 47)
        non_offtarget_all_attention = np.array(non_offtarget_all_attention_) # (N, 12, 47, 47)
        offtarget_all_attention = np.mean(offtarget_all_attention, axis=0) # (12, 47, 47)
        non_offtarget_all_attention = np.mean(non_offtarget_all_attention, axis=0) # (12, 47, 47)
        
        return {"offtarget": offtarget_all_attention, "non_offtarget": non_offtarget_all_attention}
    
    

class DnabertAnalysisClass:
    def __init__(self, offtarget_dataset, non_offtarget_dataset, datatype: str, fold: int, exp_id: int):
        self.offtarget_dataset = offtarget_dataset
        self.non_offtarget_dataset = non_offtarget_dataset
        self.datatype = datatype
        self.fold = fold
        self.exp_id = exp_id
        self.random_seed = config.random_state + exp_id
        
        self.dnabert_crispr_finetuned_path = utilities_module.return_model_weight_path("dnabert", self.datatype, self.fold, self.exp_id)
        if not os.path.exists(self.dnabert_crispr_finetuned_path):
            sys.exit(f"[ERROR] Model weight file does not exist. Run [python3 main.py -d {self.datatype} -m dnabert -f {self.fold} -e {self.exp_id} --train]")
        os.makedirs(config.attention_weight_base_dir_path, exist_ok=True)
        self.attention_weight_active_array_path = f"{config.attention_weight_base_dir_path}/dnabert_attention_weight_active_{self.datatype}_fold{self.fold}_exp{self.exp_id}.npy"
        self.attention_weight_inactive_array_path = f"{config.attention_weight_base_dir_path}/dnabert_attention_weight_inactive_{self.datatype}_fold{self.fold}_exp{self.exp_id}.npy"
        
        self.kmer = config.kmer
        self.token_max_length = 2*(24 - self.kmer + 1) + 3
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def check_exist_attention_weights(self):
        if os.path.exists(self.attention_weight_active_array_path) and os.path.exists(self.attention_weight_inactive_array_path):
            return True

    def save_attention_weights(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.dnabert_crispr_finetuned_path)
        # definition func for tokenizer
        def tokenize_function(examples):
            return self.tokenizer(examples['target_dna'], examples['sgrna'], padding='max_length', truncation=True, max_length=self.token_max_length)
        
        # load fine-tuned model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.dnabert_crispr_finetuned_path,
            num_labels=2,
            output_hidden_states=False,
            output_attentions=True,
            ignore_mismatched_sizes=True
        ).to(self.device)
        model.eval()
        
        # Tokenize
        offtarget_tokenized = self.offtarget_dataset.map(tokenize_function, batched=True)
        non_offtarget_tokenized = self.non_offtarget_dataset.map(tokenize_function, batched=True)
        
        # Convert to torch tensor
        offtarget_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        non_offtarget_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        offtarget_dataset = TensorDataset(
            torch.tensor(offtarget_tokenized['input_ids']),
            torch.tensor(offtarget_tokenized['attention_mask'])
        )
        non_offtarget_dataset = TensorDataset(
            torch.tensor(non_offtarget_tokenized['input_ids']),
            torch.tensor(non_offtarget_tokenized['attention_mask'])
        )
        # DataLoader
        offtarget_dataloader = DataLoader(offtarget_dataset, batch_size=32)
        non_offtarget_dataloader = DataLoader(non_offtarget_dataset, batch_size=32)
        
        # For offtarget
        # Attention weights list
        offtarget_all_attentions = [[] for _ in range(12)]  # Initialize list for 12 layers
        with torch.no_grad():
            for batch in offtarget_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask)
                attentions = outputs.attentions # tuple of 12 layers
                # Collect attentions for each layer
                for i in range(12):
                    offtarget_all_attentions[i].append(attentions[i].cpu().numpy()) # (N, 12, 47, 47)
        # Concatenate attentions for each layer
        offtarget_all_attentions = [np.concatenate(attentions, axis=0) for attentions in offtarget_all_attentions]
        offtarget_all_attentions = np.array(offtarget_all_attentions) # (12, N, 12, 47, 47)
        offtarget_all_attentions = np.mean(offtarget_all_attentions, axis=1) # (12, 12, 47, 47)
        offtarget_all_attentions = np.mean(offtarget_all_attentions, axis=1) # (12, 47, 47)
        
        # For non-offtarget
        # Attention weights list
        non_offtarget_all_attentions = [[] for _ in range(12)]  # Initialize list for 12 layers
        with torch.no_grad():
            for batch in non_offtarget_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask)
                attentions = outputs.attentions # tuple of 12 layers
                # Collect attentions for each layer
                for i in range(12):
                    non_offtarget_all_attentions[i].append(attentions[i].cpu().numpy())
        # Concatenate attentions for each layer
        non_offtarget_all_attentions = [np.concatenate(attentions, axis=0) for attentions in non_offtarget_all_attentions]
        non_offtarget_all_attentions = np.array(non_offtarget_all_attentions) # (12, N, 12, 47, 47)
        non_offtarget_all_attentions = np.mean(non_offtarget_all_attentions, axis=1) # (12, 12, 47, 47)
        non_offtarget_all_attentions = np.mean(non_offtarget_all_attentions, axis=1) # (12, 47, 47)
        
        # Save attention weights
        np.save(self.attention_weight_active_array_path, offtarget_all_attentions)
        np.save(self.attention_weight_inactive_array_path, non_offtarget_all_attentions)
        print(f"[INFO] Attention weights are saved at {self.attention_weight_active_array_path} and {self.attention_weight_inactive_array_path}")
        
        

def main():
    # Attention weight analysis
    if args.attention_score:
        # Check if model weight exist.
        for fold in fold_list:
            for exp_id in exp_id_list:
                utilities_module.check_model_weight_exist(model_name="dnabert", datatype=args.datatype, fold=fold, exp_id=exp_id)
                utilities_module.check_output_probability_exist(model_name="dnabert", datatype=args.datatype, fold=fold, exp_id=exp_id)
        
        # Attention weights
        dataloaderClassForDNABERTAnalysis = DataLoaderClassForDNABERTAnalysis(datatype=args.datatype, fold_list=fold_list, exp_id_list=exp_id_list)
        dataloaderClassForDNABERTAnalysis.load_dataset()
        dataloaderClassForDNABERTAnalysis.load_test_info()
        for fold in fold_list:
            for exp_id in exp_id_list:
                    # Get all input data
                    input_data = dataloaderClassForDNABERTAnalysis.return_input_data(fold, exp_id)
                    # Get predict label
                    predict_label_array = dataloaderClassForDNABERTAnalysis.return_predict_label(fold, exp_id)
                    # Sampling offtarget and non-offtarget input token data
                    input_data_sampled = dataloaderClassForDNABERTAnalysis.return_sampled_input_data(input_data, predict_label_array)
                    
                    # Save attention weights
                    dnabertAnalysisClass = DnabertAnalysisClass(input_data_sampled["offtarget"], input_data_sampled["non_offtarget"], args.datatype, fold, exp_id)
                    # if not dnabertAnalysisClass.check_exist_attention_weights():
                    #     dnabertAnalysisClass.save_attention_weights()

        # Attention weights visualize
        attention_weights_dict = dataloaderClassForDNABERTAnalysis.return_attention_weight() # {(12, 47, 47), (12, 47, 47)}
        # For offtarget
        offtarget_attention_weight_array = attention_weights_dict["offtarget"]
        visualize_module.attention_weight_visualize(offtarget_attention_weight_array, title="DNABERT Attention Weight (Off-target)", 
                                                    save_path=f"{config.fig_base_path}/attention_weight/attention_weight_offtarget.png")
        non_offtarget_attention_weight_array = attention_weights_dict["non_offtarget"]
        visualize_module.attention_weight_visualize(non_offtarget_attention_weight_array, title="DNABERT Attention Weight (Non off-target)", 
                                                    save_path=f"{config.fig_base_path}/attention_weight/attention_weight_non_offtarget.png")
        # Difference
        attention_weight_diff = offtarget_attention_weight_array - non_offtarget_attention_weight_array
        # Implement max min transformation for each layer
        attention_weight_diff_transformed = np.zeros_like(attention_weight_diff)
        for i in range(12):
            max_val = max(attention_weight_diff[i].max(), -attention_weight_diff[i].min())
            attention_weight_diff_transformed[i] = attention_weight_diff[i] / max_val
        visualize_module.attention_weight_visualize(
            attention_weight_diff_transformed, title="DNABERT Attention Weight Difference (Active OTS - Inactive OTS)", 
            save_path=f"{config.fig_base_path}/attention_weight/attention_weight_diff.png")


if __name__ == "__main__":
    main()