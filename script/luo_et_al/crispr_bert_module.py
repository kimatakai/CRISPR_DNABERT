
import sys
sys.path.append("script/")

import config
import utilities_module

import pandas as pd
import numpy as np
import os
import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from transformers import BertModel, BertConfig


class CrisprBertDataProcessClass:
    def __init__(self, DataLoaderClass, dataset_df: pd.DataFrame, train_test_info: dict):
        self.DataLoaderClass = DataLoaderClass
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        
        self.len_seq = 26
        self.len_encode = 7
        
        self.encode_dict_indel = {'AA': [1, 0, 0, 0, 0, 0, 0],'AT': [1, 1, 0, 0, 0, 1, 0],'AG': [1, 0, 1, 0, 0, 1, 0],'AC': [1, 0, 0, 1, 0, 1, 0],
                                  'TA': [1, 1, 0, 0, 0, 0, 1],'TT': [0, 1, 0, 0, 0, 0, 0],'TG': [0, 1, 1, 0, 0, 1, 0],'TC': [0, 1, 0, 1, 0, 1, 0],
                                  'GA': [1, 0, 1, 0, 0, 0, 1],'GT': [0, 1, 1, 0, 0, 0, 1],'GG': [0, 0, 1, 0, 0, 0, 0],'GC': [0, 0, 1, 1, 0, 1, 0],
                                  'CA': [1, 0, 0, 1, 0, 0, 1],'CT': [0, 1, 0, 1, 0, 0, 1],'CG': [0, 0, 1, 1, 0, 0, 1],'CC': [0, 0, 0, 1, 0, 0, 0],
                                  'A-': [1, 0, 0, 0, 1, 1, 0],'T-': [0, 1, 0, 0, 1, 1, 0],'G-': [0, 0, 1, 0, 1, 1, 0],'C-': [0, 0, 0, 1, 1, 1, 0],
                                  '-A': [1, 0, 0, 0, 1, 0, 1],'-T': [0, 1, 0, 0, 1, 0, 1],'-G': [0, 0, 1, 0, 1, 0, 1],'-C': [0, 0, 0, 1, 1, 0, 1],
                                  '--': [0, 0, 0, 0, 0, 0, 0],'NN': [0, 0, 0, 0, 0, 0, 0]}
        self.token_dict = {'[CLS]': 0, '[SEP]': 1,
                           'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
                           'CA': 6, 'CC': 7, 'CG': 8, 'CT': 9,
                           'GA': 10, 'GC': 11, 'GG': 12, 'GT': 13,
                           'TA': 14, 'TC': 15, 'TG': 16, 'TT': 17,
                           'A-':18,'-A':19,'C-':20,'-C':21,'G-':22,
                           '-G':23,'T-':24,'-T':25,'--':26,'NN':27}
    
    
    def seq_to_encoding(self, seq1: str, seq2: str) -> np.array:
        encode_array = np.zeros((self.len_seq, self.len_encode), dtype=np.int8)
        if len(seq1) == 23:
            j = 1
        else:
            j = 0
        for base1, base2 in zip(seq1, seq2):
            if base1 == "N":
                base1 = base2
            if base2 == "N":
                base2 = base1
            encode_array[j+1] = self.encode_dict_indel[base1+base2]
            j += 1
        return encode_array
    
    def seq_to_tokens(self, seq1: str, seq2: str) -> np.array:
        token_array = np.zeros(self.len_seq, dtype=np.int8)
        token_array[0] = self.token_dict["[CLS]"]
        if len(seq1) == 23:
            j = 1
            token_array[1] = self.token_dict["--"]
        else:
            j = 0
        for base1, base2 in zip(seq1, seq2):
            if base1 == "N":
                base1 = base2
            if base2 == "N":
                base2 = base1
            token_array[j+1] = self.token_dict[base1+base2]
            j += 1
        token_array[self.len_seq-1] = self.token_dict["[SEP]"]
        return token_array
    
    def return_input(self) -> dict:
        '''
        return
        train_encode_input: (n, 26, 7), train_token_input: (n, 26), test_encode_input: (n, 26, 7), test_token_input: (n, 26)
        '''
        train_sgrna_seq_list = self.train_test_info["train_seq_list"]
        test_sgrna_seq_list = self.train_test_info["test_seq_list"]
        
        # Initialize
        train_encode_input = []
        train_token_input = []
        
        test_encode_input = []
        test_token_input = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Train data Encoding and Tokenizing"):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq, self.len_encode))
            sgrna_token_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq))
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                token_array = self.seq_to_tokens(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
                sgrna_token_array[idx] = token_array
            train_encode_input.append(sgrna_encode_array)
            train_token_input.append(sgrna_token_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq, self.len_encode))
            sgrna_token_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq))
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                token_array = self.seq_to_tokens(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
                sgrna_token_array[idx] = token_array
            train_encode_input.append(sgrna_encode_array)
            train_token_input.append(sgrna_token_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list)):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq, self.len_encode))
            sgrna_token_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq))
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                token_array = self.seq_to_tokens(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
                sgrna_token_array[idx] = token_array
            test_encode_input.append(sgrna_encode_array)
            test_token_input.append(sgrna_token_array)
            
            # Processing for non off-target row
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq, self.len_encode))
            sgrna_token_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq))
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                token_array = self.seq_to_tokens(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
                sgrna_token_array[idx] = token_array
            test_encode_input.append(sgrna_encode_array)
            test_token_input.append(sgrna_token_array)
        
        # Input data concatenate
        train_encode_input = np.concatenate(train_encode_input, axis=0)
        train_token_input = np.concatenate(train_token_input, axis=0)
        
        test_encode_input = np.concatenate(test_encode_input, axis=0)
        test_token_input = np.concatenate(test_token_input, axis=0)
        
        return {"train_encode_input": train_encode_input, "train_token_input": train_token_input, "test_encode_input": test_encode_input, "test_token_input": test_token_input}
    
    def return_input_for_trueot(self) -> dict:
        
        # Initialize
        train_encode_input = []
        train_token_input = []
        test_encode_input = np.zeros((self.dataset_df.shape[0], self.len_seq, self.len_encode))
        test_token_input = np.zeros((self.dataset_df.shape[0], self.len_seq))
        
        # For test data
        for idx, row in tqdm.tqdm(enumerate(self.dataset_df.itertuples(index=False)), total=self.dataset_df.shape[0], desc="TrueOT data Encoding and Tokenizing"):
            encode_array = self.seq_to_encoding(row._8, row._9)
            token_array = self.seq_to_tokens(row._8, row._9)
            test_encode_input[idx] = encode_array
            test_token_input[idx] = token_array
        
        return {"train_encode_input": train_encode_input, "train_token_input": train_token_input, "test_encode_input": test_encode_input, "test_token_input": test_token_input}


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.encode_inputs = torch.tensor(data_dict['encode_input'], dtype=torch.long)
        self.token_inputs = torch.tensor(data_dict['token_input'], dtype=torch.long)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.long)

    def __len__(self):
        # return length of datasets
        return len(self.labels)

    def __getitem__(self, idx):
        # Get data of index which be specified
        sample = {
            'encode_input': self.encode_inputs[idx],
            'token_input': self.token_inputs[idx],
            'label': self.labels[idx]
        }
        return sample



class CrisprBertModel(nn.Module):
    def __init__(self, bert_model, input_shape):
        super(CrisprBertModel, self).__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(1, input_shape[1]), padding='same')
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=(2, input_shape[1]), padding='same')
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(3, input_shape[1]), padding='same')
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=35, kernel_size=(5, input_shape[1]), padding='same')
        self.gru = nn.GRU(input_size=80*input_shape[1], hidden_size=40, batch_first=True, bidirectional=True)
        self.bert_gru = nn.GRU(input_size=bert_model.config.hidden_size, hidden_size=40, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(input_shape[0]*2*40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.35)

    def forward(self, encode_input, token_input):
        # BERT output
        bert_output = self.bert(input_ids=token_input).last_hidden_state
        bert_output_gru, _ = self.bert_gru(bert_output)  # (batch_size, seq_len, hidden_dim)
        
        # 1D CNN
        encode_input = encode_input.unsqueeze(1)
        # encode_input = encode_input.permute(0, 3, 2, 1) # (batch_size, channels, seq_len, new_dim)
        encode_input = encode_input.float()
        x1 = torch.relu(self.conv1(encode_input))
        x2 = torch.relu(self.conv2(encode_input))
        x3 = torch.relu(self.conv3(encode_input))
        x4 = torch.relu(self.conv4(encode_input))
        x = torch.cat([x1, x2, x3, x4], dim=1)  # (batch_size, channels, seq_len, new_dim)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # Flatten to (batch_size, seq_len, new_dim)  # (256, 26, 560)

        # GRU
        x_gru, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)

        # Weighted Concatenation
        combined_features = 0.2 * x_gru + 0.8 * bert_output_gru

        # Fully connected layer and softmax
        x = torch.flatten(combined_features, start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        output = self.fc_out(x)
        return output


class CrisprBertClass:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, input_dict: dict, label_dict: dict, fold: int, datatype: str, exp_id: int=0, model_datatype: str=None):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.fold = fold
        self.datatype = datatype
        self.exp_id = exp_id
        self.seed = exp_id + config.random_state
        
        self.train_dataset = {"encode_input": input_dict["train_encode_input"], "token_input": input_dict["train_token_input"], "label": label_dict["train_label"]}
        self.train_dataset_temp = {"encode_input": input_dict["train_encode_input"], "token_input": input_dict["train_token_input"], "label": label_dict["train_label"]}
        self.num_negative_samples = len([i for i, label in enumerate(label_dict["train_label"]) if label == 0])
        self.test_dataset = {"encode_input": input_dict["test_encode_input"], "token_input": input_dict["test_token_input"], "label": label_dict["test_label"]}
        
        os.makedirs(config.crispr_bert_model_path, exist_ok=True)
        if self.datatype == "transfer":
            self.pretrained_model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-bert", datatype="changeseq", fold=self.fold, exp_id=self.exp_id)
        else:
            self.pretrained_model_weight_path = None
        if model_datatype:
            self.model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-bert", datatype=model_datatype, fold=self.fold, exp_id=self.exp_id)
        else:
            self.model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-bert", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/crispr_bert/"
        os.makedirs(self.predicted_probabilities_path, exist_ok=True)
        self.probability_array_path = utilities_module.return_output_probability_path(model_name="crispr-bert", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        
        self.bert_config_path = f"{config.crispr_bert_architecture_path}/bert_config.json"
        
        self.len_seq = 26
        self.len_encode = 7
        
        if self.datatype != "transfer":
            self.epochs = 30
        else:
            self.epochs = 15
        self.batch_size = 256
        self.learning_rate = 0.0001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_config = BertConfig.from_json_file(self.bert_config_path)
        self.bert_model = BertModel(self.bert_config).to(self.device)
        self.crispr_bert_model = CrisprBertModel(self.bert_model, input_shape=(self.len_seq, self.len_encode)).to(self.device)
    
    
    def downsampling_dataset(self, sampling_rate: float):
        # Split label 0 and label 1
        label_0_indices = [i for i, label in enumerate(self.train_dataset_temp["label"]) if label == 0]
        label_1_indices = [i for i, label in enumerate(self.train_dataset_temp["label"]) if label == 1]
        # Count negative samples
        sampled_label_0_indices = random.sample(label_0_indices, int(self.num_negative_samples * sampling_rate))
        final_indices = sampled_label_0_indices + label_1_indices
        # temp Dataset for each epoch training
        self.train_dataset_temp = {
            "encode_input": [self.train_dataset_temp["encode_input"][i] for i in final_indices],
            "token_input": [self.train_dataset_temp["token_input"][i] for i in final_indices],
            "label": [self.train_dataset_temp["label"][i] for i in final_indices]
        }
        # Update `train_dataset` by excluding the unsampled label 0 data
        unsampled_label_0_indices = list(set(label_0_indices) - set(sampled_label_0_indices))
        remaining_indices = label_1_indices + unsampled_label_0_indices  # Keep sampled label 1 and unsampled label 0
        # Filter out the remaining data in train_dataset (this will exclude the unsampled label 0 entries)
        self.train_dataset = {
            "encode_input": [self.train_dataset["encode_input"][i] for i in remaining_indices],
            "token_input": [self.train_dataset["token_input"][i] for i in remaining_indices],
            "label": [self.train_dataset["label"][i] for i in remaining_indices]
        }
    
    
    def train_classification_task(self):
        print(f"[TRAIN] CRISPR-BERT model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load pretrained model if transfer learning
        if self.datatype == "transfer":
            if not os.path.exists(self.pretrained_model_weight_path):
                sys.exit(f"[ERROR] Pretrained model ({self.pretrained_model_weight_path}) does not exist.")
            self.crispr_bert_model.load_state_dict(torch.load(self.pretrained_model_weight_path))
        
        # Prepare for training
        self.downsampling_dataset(sampling_rate=0.2)
        train_loader = DataLoader(CustomDataset(self.train_dataset_temp), batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.crispr_bert_model.parameters(), lr=self.learning_rate)
        self.crispr_bert_model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"[Epoch] {epoch+1}"):
                encode_input = batch["encode_input"]
                token_input = batch["token_input"]
                labels = batch["label"].long()
                encode_input, token_input, labels = encode_input.to(self.device), token_input.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = self.crispr_bert_model(encode_input, token_input)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader)}")

        # Save model weights
        torch.save(self.crispr_bert_model.state_dict(), self.model_weight_path)
        

    def test_classification_task(self):
        print(f"[TEST] CRISPR-BERT model prediction. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load model
        if not os.path.exists(self.model_weight_path):
            sys.exit(f"[ERROR] Trained model ({self.model_weight_path}) does not exist.")
            
        # Extract True label
        true_label_np = torch.IntTensor(self.test_dataset["label"]).cpu().numpy()
        
        if not os.path.exists(self.probability_array_path):
            self.crispr_bert_model.load_state_dict(torch.load(self.model_weight_path))
            self.crispr_bert_model.eval()
            
            # Process for test data
            test_loader = DataLoader(CustomDataset(self.test_dataset), batch_size=self.batch_size, shuffle=False)
            
            # Predict
            all_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    encode_input = batch["encode_input"]
                    token_input = batch["token_input"]
                    labels = batch["label"]
                    encode_input, token_input, labels = encode_input.to(self.device), token_input.to(self.device), labels.to(self.device)
                    outputs = self.crispr_bert_model(encode_input, token_input)
                    all_logits.append(outputs)
            all_logits = torch.cat(all_logits, dim=0)
            
            # Logit -> Prob
            probabilities = torch.softmax(all_logits, dim=1)
            probabilities = probabilities.cpu().numpy()
            
            # Save probabilities
            probabilities = probabilities.astype(np.float32)
            np.save(self.probability_array_path, probabilities)
        
        # Load probabilities
        probabilities = np.load(self.probability_array_path)
        
        return (true_label_np, probabilities)



