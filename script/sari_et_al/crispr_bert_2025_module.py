
import sys
sys.path.append("script/")

import config
import utilities_module

import pandas as pd
import numpy as np
import os
import tqdm
import random

from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from transformers import BertModel, BertConfig



def return_seq_token_dict(base_index: dict, kmer: int):
    base_list = list(base_index.keys())
    depth = 2 * kmer
    token_touple_list = list(product(base_list, repeat=depth))
    token_chr_list = ["".join(token_touple) for token_touple in token_touple_list]
    seq_token_dict = {token: i for i, token in enumerate(token_chr_list)}
    return seq_token_dict


class CrisprBert2025DataProcessClass:
    def __init__(self, DataLoaderClass, dataset_df: pd.DataFrame, train_test_info: dict):
        self.DataLoaderClass = DataLoaderClass
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        
        self.seq_len = 24
        self.base_index = config.base_index
        self.kmer = 2 # Default kmer size on CrisprBERT 2025 paper
        self.seq_token_dict = return_seq_token_dict(self.base_index, self.kmer)

    def seq_to_token_id(self, seq1: str, seq2: str) -> np.array:
        """
        Converts two sequences into an array of token IDs based on k-mer combinations.
        Args:
            seq1 (str): The first input sequence (Target DNA).
            seq2 (str): The second input sequence (sgRNA).
        Returns:
            np.array: An array of token IDs corresponding to the k-mer combinations of the input sequences.
        Raises:
            KeyError: If a k-mer combination is not found in the seq_token_dict.
        """
        if len(seq1) == 23:
            seq1 = "-" + seq1
        if len(seq2) == 23:
            seq2 = "-" + seq2
        # Convert sequences to lists for mutable operations
        seq1_list = list(seq1)
        seq2_list = list(seq2)
        # Tokenize the input sequences
        token_id_array = np.zeros(len(seq1) - self.kmer + 1, dtype=int)
        for i in range(len(seq1) - self.kmer + 1):
            for j in range(self.kmer):
                if seq1_list[i+j] == "N":
                    seq1_list[i+j] = seq2_list[i+j]
                if seq2_list[i+j] == "N":
                    seq2_list[i+j] = seq1_list[i+j]
                if seq1_list[i+j] == "N" and seq2_list[i+j] == "N":
                    seq1_list[i+j] = seq2_list[i+j] = "-"
            token = "".join(seq1_list[i:i+self.kmer]) + "".join(seq2_list[i:i+self.kmer])
            token_id_array[i] = self.seq_token_dict[token]
        return token_id_array
    
    def return_input(self) -> dict:
        
        # Get train and test information
        train_sgrna_seq_list = self.train_test_info["train_seq_list"]
        test_sgrna_seq_list = self.train_test_info["test_seq_list"]
        
        # Prepare input data
        train_token_list = []
        test_token_list = []
        
        # For train data
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Train data tokenizing"):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for offtarget data
            sgrna_token_id_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len - self.kmer + 1), dtype=int)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                token_id_array = self.seq_to_token_id(row._6, row._7)
                sgrna_token_id_array[idx] = token_id_array
            train_token_list.append(sgrna_token_id_array)
            
            # Processing for non-offtarget data
            sgrna_token_id_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len - self.kmer + 1), dtype=int)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                token_id_array = self.seq_to_token_id(row._6, row._7)
                sgrna_token_id_array[idx] = token_id_array
            train_token_list.append(sgrna_token_id_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Test data tokenizing"):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Processing for offtarget data
            sgrna_token_id_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.seq_len - self.kmer + 1), dtype=int)
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                token_id_array = self.seq_to_token_id(row._6, row._7)
                sgrna_token_id_array[idx] = token_id_array
            test_token_list.append(sgrna_token_id_array)
            
            # Processing for non-offtarget data
            sgrna_token_id_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.seq_len - self.kmer + 1), dtype=int)
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                token_id_array = self.seq_to_token_id(row._6, row._7)
                sgrna_token_id_array[idx] = token_id_array
            test_token_list.append(sgrna_token_id_array)
        
        train_token_id_input = np.concatenate(train_token_list, axis=0)
        test_token_id_input = np.concatenate(test_token_list, axis=0)
        
        return {"train_input": train_token_id_input, "test_input": test_token_id_input}
    
    def return_input_for_trueot(self):
        
        train_token_array = np.zeros((0, self.seq_len - self.kmer + 1), dtype=int)
        test_token_array = np.zeros((self.dataset_df.shape[0], self.seq_len - self.kmer + 1), dtype=int)
        
        # For test data
        for idx, row in tqdm.tqdm(enumerate(self.dataset_df.itertuples(index=False)), total=self.dataset_df.shape[0], desc="Test data tokenizing"):
            token_id_array = self.seq_to_token_id(row._8, row._9)
            test_token_array[idx] = token_id_array
        
        return {"train_input": train_token_array, "test_input": test_token_array}


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict['input'], dtype=torch.long)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.long)

    def __len__(self):
        # return length of datasets
        return len(self.labels)

    def __getitem__(self, idx):
        # Get data of index which be specified
        sample = {
            'input': self.inputs[idx],
            'label': self.labels[idx]
        }
        return sample


class CrisprBert2025Model(nn.Module):
    def __init__(self, bert_model):
        super(CrisprBert2025Model, self).__init__()
        # BERT model
        self.bert = bert_model
        # Bidirectional LSTM model
        self.bi_lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        # Fully connected layer
        self.fc1 = nn.Linear(64*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 2)
        # Activation function
        self.relu = nn.ReLU()
        # Dropout
        self.dropout05 = nn.Dropout(0.5)
        self.dropout04 = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask=None):
        # BERT input
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = bert_output.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Bi-LSTM
        lstm_output, _ = self.bi_lstm(seq_output) # (batch_size, seq_len, hidden_size*2)
        
        # CLS token
        cls_output = lstm_output[:, 0, :] # (batch_size, hidden_size*2)
        
        # Fully connected layer
        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout05(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout05(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout04(x)
        
        x = self.fc_out(x)
        return x


class CrisprBert2025Class:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, input_dict: dict, label_dict: dict, fold: int, datatype: str, exp_id: int=0, model_datatype: str=None):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.fold = fold
        self.datatype = datatype
        self.exp_id = exp_id
        self.seed = exp_id + config.random_state
        
        self.train_dataset = {"input": input_dict["train_input"], "label": label_dict["train_label"]}
        self.train_dataset_temp = {"input": input_dict["train_input"], "label": label_dict["train_label"]}
        self.test_dataset = {"input": input_dict["test_input"], "label": label_dict["test_label"]}
        self.num_negative_samples = len([i for i, label in enumerate(label_dict["train_label"]) if label == 0])
        
        os.makedirs(config.crispr_bert_2025_model_path, exist_ok=True)
        if self.datatype == "transfer":
            self.pretrained_model_weight_path = utilities_module.return_model_weight_path("crispr-bert-2025", "changeseq", fold=self.fold, exp_id=self.exp_id)
        else:
            self.pretrained_model_weight_path = None
        if model_datatype:
            self.model_weight_path = utilities_module.return_model_weight_path("crispr-bert-2025", model_datatype, self.fold, self.exp_id)
        else:
            self.model_weight_path = utilities_module.return_model_weight_path("crispr-bert-2025", self.datatype, self.fold, self.exp_id)
        self.probabilities_base_path = f"{config.probabilities_base_dir_path}/crispr_bert_2025"
        os.makedirs(self.probabilities_base_path, exist_ok=True)
        self.probability_array_path = utilities_module.return_output_probability_path(model_name="crispr-bert-2025", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        
        self.seq_len = 24
        self.base_index = config.base_index
        self.kmer = 2 # Default kmer size
        
        if self.datatype != "transfer":
            self.epochs = 10 # Default epoch size on CrisprBERT 2025 paper is 400
        else:
            self.epochs = 5
        self.batch_size = 128
        self.learning_rate = 2e-5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bert_config = BertConfig(
            vocab_size = len(self.base_index) ** (2 * self.kmer),
            max_position_embeddings = 25,
            intermediate_size = 256, # Default intermediate size on CrisprBERT 2025 paper is 2048.
            hidden_act = "gelu",
            hidden_size = 64,
            num_attention_heads = 8,
            num_hidden_layers = 6,
            type_vocab_size = 1,
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1,
            num_labels = 2,
        )
        self.bert_model = BertModel(self.bert_config)
        self.crispr_bert_model = CrisprBert2025Model(self.bert_model).to(self.device)


    def downsampling_dataset(self, sampling_rate: float):
        # Split label 0 and label 1
        label_0_indices = [i for i, label in enumerate(self.train_dataset_temp["label"]) if label == 0]
        label_1_indices = [i for i, label in enumerate(self.train_dataset_temp["label"]) if label == 1]
        # Count negative samples
        sampled_label_0_indices = random.sample(label_0_indices, int(self.num_negative_samples * sampling_rate))
        final_indices = sampled_label_0_indices + label_1_indices
        # temp Dataset for each epoch training
        self.train_dataset_temp = {
            "input": [self.train_dataset_temp["input"][i] for i in final_indices],
            "label": [self.train_dataset_temp["label"][i] for i in final_indices]
        }
        # Update `train_dataset` by excluding the unsampled label 0 data
        unsampled_label_0_indices = list(set(label_0_indices) - set(sampled_label_0_indices))
        remaining_indices = label_1_indices + unsampled_label_0_indices  # Keep sampled label 1 and unsampled label 0
        # Filter out the remaining data in train_dataset (this will exclude the unsampled label 0 entries)
        self.train_dataset = {
            "input": [self.train_dataset["input"][i] for i in remaining_indices],
            "label": [self.train_dataset["label"][i] for i in remaining_indices]
        }
    
    
    def train_classification_task(self):
        print(f"[TRAIN] CrisprBERT model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load pretrained model if transfer learning
        if self.datatype == "transfer":
            if not os.path.exists(self.pretrained_model_weight_path):
                sys.exit(f"[ERROR] Pretrained model ({self.pretrained_model_weight_path}) does not exist.")
            self.crispr_bert_model.load_state_dict(torch.load(self.pretrained_model_weight_path))
        
        # Prepare for training
        self.downsampling_dataset(sampling_rate=0.2)
        train_loader = DataLoader(CustomDataset(self.train_dataset_temp), batch_size=self.batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.crispr_bert_model.parameters(), lr=self.learning_rate)
        self.crispr_bert_model.train()
        
        # training
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"[Epoch] {epoch+1}"):
                inputs = batch["input"]
                labels = batch["label"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = self.crispr_bert_model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader)}")
        
        # Save model weights
        torch.save(self.crispr_bert_model.state_dict(), self.model_weight_path)
    
    
    def test_classification_task(self):
        print(f"[TEST] CrisprBERT model prediction. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
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
                    inputs = batch["input"]
                    labels = batch["label"]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.crispr_bert_model(inputs)
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







