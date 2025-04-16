

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
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from torch.utils.data import DataLoader, Dataset



class CrisprDipoffDataProcessClass:
    def __init__(self, DataLoaderClass, dataset_df: pd.DataFrame, train_test_info: dict):
        self.DataLoaderClass = DataLoaderClass
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        
        self.len_seq = 24
        self.seq_dim = 5 # [A, C, G, T, -]
    
    
    def seq_to_encoding(self, seq1: str, seq2: str) -> np.array:
        encode_array = np.zeros((self.len_seq, self.seq_dim))
        if len(seq1) == 23:
            j = 1
        else:
            j = 0
        for base1, base2 in zip(seq1, seq2):
            if base1 == "N":
                base1 = base2
            if base2 == "N":
                base2 = base1
            if base1 == "N" or base2 == "N":
                base1 = base2 = "-"
            encode_array[j][config.base_index[base1]] = 1
            encode_array[j][config.base_index[base2]] = 1
            j += 1
        return encode_array    
    
    
    def return_input(self) -> dict:
        """
        Processes and encodes sgRNA sequences from training and testing datasets.
        This method retrieves sgRNA sequences from the training and testing datasets,
        splits them into offtarget and non-offtarget categories, encodes the sequences,
        and concatenates the encoded arrays.
        Returns:
            dict: A dictionary containing the encoded training and testing inputs with keys:
                - "train_input" (np.ndarray): Encoded training sgRNA sequences.
                - "test_input" (np.ndarray): Encoded testing sgRNA sequences.
        """
        
        # Retrieve sgRNA sequences from training and testing datasets
        train_sgrna_seq_list = self.train_test_info["train_seq_list"]
        test_sgrna_seq_list = self.train_test_info["test_seq_list"]
        
        # Initialize 
        train_input = []
        test_input = []
        
        # For train data 
        for sgrna_seq in tqdm.tqdm(train_sgrna_seq_list, total=len(train_sgrna_seq_list), desc="Train data Encoding"):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Encoding
            # Processing for offtarget
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq, self.seq_dim))
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
            train_input.append(sgrna_encode_array)
            
            # Processing for non-offtarget
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq, self.seq_dim))
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
            train_input.append(sgrna_encode_array)
        
        # For test data
        for sgrna_seq in tqdm.tqdm(test_sgrna_seq_list, total=len(test_sgrna_seq_list), desc="Test data Encoding"):
            dataset_df_sgrna = self.dataset_df[self.dataset_df["sgRNA"] == sgrna_seq]
            # Split offtarget or non-offtarget
            data_df_sgrna_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] >= 1]
            data_df_sgrna_non_offtarget = dataset_df_sgrna[dataset_df_sgrna["reads"] == 0]
            
            # Encoding
            # Processing for offtarget
            sgrna_encode_array = np.zeros((data_df_sgrna_offtarget.shape[0], self.len_seq, self.seq_dim))
            for idx, row in enumerate(data_df_sgrna_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
            test_input.append(sgrna_encode_array)
            
            # Processing for non-offtarget
            sgrna_encode_array = np.zeros((data_df_sgrna_non_offtarget.shape[0], self.len_seq, self.seq_dim))
            for idx, row in enumerate(data_df_sgrna_non_offtarget.itertuples(index=False)):
                encode_array = self.seq_to_encoding(row._6, row._7)
                sgrna_encode_array[idx] = encode_array
            test_input.append(sgrna_encode_array)
        
        # Concatenate
        train_input = np.concatenate(train_input, axis=0)
        test_input = np.concatenate(test_input, axis=0)
        
        return {"train_input": train_input, "test_input": test_input}
    
    def return_input_for_trueot(self):
        
        train_input = []
        test_input = np.zeros((self.dataset_df.shape[0], self.len_seq, self.seq_dim))
        
        # For test data
        for idx, row in tqdm.tqdm(enumerate(self.dataset_df.itertuples(index=False)), total=self.dataset_df.shape[0], desc="Test data Encoding"):
            encode_array = self.seq_to_encoding(row._8, row._9)
            test_input[idx] = encode_array
        
        return {"train_input": train_input, "test_input": test_input}


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict['input'], dtype=torch.float32)
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


class CrisprDipoffModel(nn.Module):
    def __init__(self, device):
        super(CrisprDipoffModel, self).__init__()
        # emb_size=256, hidden_size=128, hidden_layers=3, output=2
        
        self.device = device

        self.emb_size = 5
        self.hidden_size = 512
        self.lstm_layers = 1
        self.bi_lstm = True
        self.reshape = False

        self.number_hidden_layers = 2
        self.dropout_prob = 0.4
        self.hidden_layers = []

        self.hidden_shape = self.hidden_size*2 if self.bi_lstm else self.hidden_size

        self.embedding = None

        self.lstm= nn.LSTM(self.emb_size, self.hidden_size, num_layers=self.lstm_layers,
                            batch_first=True, bidirectional=self.bi_lstm)

        start_size = self.hidden_shape

        self.relu = nn.ReLU()

        for i in range(self.number_hidden_layers):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(start_size, start_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob)))

            start_size = start_size // 2

        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.output = nn.Linear(start_size, 2)


    def forward(self, input):
        dir = 2 
        h = torch.zeros((self.lstm_layers*dir, input.size(0), self.hidden_size), device=self.device)
        c = torch.zeros((self.lstm_layers*dir, input.size(0), self.hidden_size), device=self.device)

        x, _ = self.lstm(input, (h,c))

        x = x[:, -1, :]

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
        
        x = self.output(x)
        return x


class CrisprDipoffClass:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, input_dict: dict, label_dict: dict, fold: int, datatype: str, exp_id: int=0, model_datatype: str=None):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.fold = fold
        self.datatype = datatype
        self.exp_id = exp_id
        self.seed = exp_id + config.random_state
        
        self.train_dataset = {"input": input_dict["train_input"], "label": label_dict["train_label"]}
        self.train_dataset_temp = {"input": input_dict["train_input"], "label": label_dict["train_label"]}
        self.num_negative_samples = len([i for i, label in enumerate(label_dict["train_label"]) if label == 0])
        self.test_dataset = {"input": input_dict["test_input"], "label": label_dict["test_label"]}
        
        os.makedirs(f"{config.crispr_dipoff_model_path}", exist_ok=True)
        if self.datatype == "transfer":
            self.pretrained_model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-dipoff", datatype="changeseq", fold=self.fold, exp_id=self.exp_id)
        else:
            self.pretrained_model_weight_path = None
        if model_datatype:
            self.model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-dipoff", datatype=model_datatype, fold=self.fold, exp_id=self.exp_id)
        else:
            self.model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-dipoff", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/crispr_dipoff/"
        os.makedirs(self.predicted_probabilities_path, exist_ok=True)
        self.probabilities_array_path = utilities_module.return_output_probability_path(model_name="crispr-dipoff", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        
        if self.datatype != "transfer":
            self.epochs = 50
        else:
            self.epochs = 25
        self.batch_size = 64
        self.learning_rate = 0.0001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.crispr_dipoff_model = CrisprDipoffModel(device=self.device).to(self.device)


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
        print(f"[TRAIN] CRISPR-DIPOFF model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENT: {self.exp_id}. {self.device} will be used.")
        
        # Load pretrained model if transfer learning
        if self.datatype == "transfer":
            if not os.path.exists(self.pretrained_model_weight_path):
                sys.exit(f"[ERROR] Pretrained model ({self.pretrained_model_weight_path}) does not exist.")
            self.crispr_dipoff_model.load_state_dict(torch.load(self.pretrained_model_weight_path))
        
        # Prepare for training
        self.downsampling_dataset(sampling_rate=0.2)
        train_loader = DataLoader(CustomDataset(self.train_dataset_temp), batch_size=self.batch_size, shuffle=True)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.crispr_dipoff_model.parameters(), lr=self.learning_rate)
        self.crispr_dipoff_model.train()
        
        # Training
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"[Epoch] {epoch+1}"):
                inputs = batch["input"]
                labels = batch["label"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = self.crispr_dipoff_model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader)}")
        
        # Save model weights
        torch.save(self.crispr_dipoff_model.state_dict(), self.model_weight_path)
    

    def test_classification_task(self):
        print(f"[TEST] CRISPR-DIPOFF model prediction. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENT: {self.exp_id}. {self.device} will be used.")
        # Load model
        if not os.path.exists(self.model_weight_path):
            sys.exit(f"[ERROR] Trained model ({self.model_weight_path}) does not exist.")
            
        # Extract True label
        true_label_np = torch.IntTensor(self.test_dataset["label"]).cpu().numpy()
        
        if not os.path.exists(self.probabilities_array_path):
            self.crispr_dipoff_model.load_state_dict(torch.load(self.model_weight_path))
            self.crispr_dipoff_model.eval()
            
            # Process for test data
            test_loader = DataLoader(CustomDataset(self.test_dataset), batch_size=self.batch_size, shuffle=False)
            
            # Predict
            all_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["input"]
                    labels = batch["label"]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.crispr_dipoff_model(inputs)
                    all_logits.append(outputs)
            all_logits = torch.cat(all_logits, dim=0)
            
            # Logit -> Prob
            probabilities = torch.softmax(all_logits, dim=1)
            probabilities = probabilities.cpu().numpy()
            
            # Save probabilities
            probabilities = probabilities.astype(np.float32)
            np.save(self.probabilities_array_path, probabilities)
        
        # Load probabilities
        probabilities = np.load(self.probabilities_array_path)
        
        return (true_label_np, probabilities)












