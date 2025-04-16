
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
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from torch.utils.data import DataLoader, Dataset



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


class Attention(nn.Module):
    def __init__(self, time_steps, input_dim):
        super(Attention, self).__init__()
        self.time_steps = time_steps
        self.input_dim = input_dim
        
        self.W_x = nn.Linear(input_dim, time_steps)
        self.W_g = nn.Linear(input_dim, time_steps)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        # x: (batch_size, time_steps, input_dim)
        # g: (batch_size, time_steps, input_dim)
        x_proj = self.W_x(x)  # (batch_size, time_steps, time_steps)
        g_proj = self.W_g(g)  # (batch_size, time_steps, time_steps)

        # Combine x and g
        combined = x_proj + g_proj

        # Compute attention weights
        a = self.softmax(combined)  # (batch_size, time_steps, time_steps)
        
        # Transpose a to match the dimensions for batch matrix multiplication
        a = a.permute(0, 2, 1)  # (batch_size, input_dim, time_steps)

        # Apply attention
        output = torch.bmm(a, x)  # (batch_size, time_steps, input_dim)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Residual connection
        out = self.relu(out)
        return out


class CrisprHwModel(nn.Module):
    def __init__(self, vocab_size=25, embed_size=100, maxlen=24):
        super(CrisprHwModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Conv Layers
        self.conv1 = nn.Conv1d(embed_size, 70, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(70)

        self.conv2 = nn.Conv1d(70, 40, kernel_size=6)
        self.bn2 = nn.BatchNorm1d(40)

        # Residual Block
        self.res_block = ResidualBlock(40)
        self.res_bn = nn.BatchNorm1d(40)

        # BiLSTM
        self.bilstm = nn.LSTM(input_size=40, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm_bn = nn.BatchNorm1d(16)

        # Attention Mechanism
        self.conv3 = nn.Conv1d(embed_size, 40, kernel_size=9)
        self.attention = Attention(time_steps=maxlen, input_dim=40)
        self.att_bn = nn.BatchNorm1d(maxlen)

        # Fully Connected Layers
        self.fc1 = nn.Linear(2240, 300)  # Concatenate outputs # 40 + 40 + 60 = 140 -> 140 * 16 = 2240
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(300, 150)
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(150, 2)  # Binary classification (softmax)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, 24)
        x_embed = self.embedding(x)  # (batch_size, 24, embed_size)

        # Conv1D expects (batch_size, channels, seq_len)
        x_conv = x_embed.permute(0, 2, 1)

        # Conv Layers
        x1 = self.conv1(x_conv)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        # Residual Block
        res_out = self.res_block(x2)
        res_out = self.res_bn(res_out)

        # LSTM Pathway
        lstm_input = x2.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        lstm_out, _ = self.bilstm(lstm_input)
        lstm_out = self.bilstm_bn(lstm_out)

        # Attention Pathway
        conv3_out = self.conv3(x_conv)
        conv3_out = conv3_out.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        att_out = self.attention(lstm_input, conv3_out)
        att_out = self.att_bn(att_out)

        # Concatenate
        res_out_flat = res_out.permute(0, 2, 1).contiguous().view(x.size(0), -1)
        lstm_out_flat = lstm_out.contiguous().view(x.size(0), -1)
        att_out_flat = att_out.contiguous().view(x.size(0), -1)

        merged = torch.cat([res_out_flat, lstm_out_flat, att_out_flat], dim=1)

        # Fully Connected Layers
        fc1_out = self.fc1(merged)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout1(fc1_out)

        fc2_out = self.fc2(fc1_out)
        fc2_out = self.relu(fc2_out)
        fc2_out = self.dropout2(fc2_out)

        output = self.out(fc2_out)

        return output


class CrisprHwClass:
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
        
        os.makedirs(config.crispr_hw_model_path, exist_ok=True)
        if self.datatype == "transfer":
            self.pretrained_model_weight_path = utilities_module.return_model_weight_path(model_name="crispr-hw", datatype="changeseq", fold=self.fold, exp_id=self.exp_id)
        else:
            self.pretrained_model_weight_path = None
        if model_datatype:
            self.model_path = utilities_module.return_model_weight_path(model_name="crispr-hw", datatype=model_datatype, fold=self.fold, exp_id=self.exp_id)
        else:
            self.model_path = utilities_module.return_model_weight_path(model_name="crispr-hw", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/crispr_hw/"
        os.makedirs(self.predicted_probabilities_path, exist_ok=True)
        self.probabilities_array_path = utilities_module.return_output_probability_path(model_name="crispr-hw", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        
        self.vocab_size = 25
        self.input_len = 24
        self.embed_dim = 100
        if self.datatype != "transfer":
            self.epochs = 30
        else:
            self.epochs = 15
        self.batch_size = 128
        self.learning_rate = 0.003
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.crispr_hw_model = CrisprHwModel(vocab_size=self.vocab_size, embed_size=self.embed_dim, maxlen=self.input_len).to(self.device)
    
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
        print(f"[TRAIN] CRISPR-HW model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load pretrained model if transfer learning
        if self.datatype == "transfer":
            if not os.path.exists(self.pretrained_model_weight_path):
                sys.exit(f"[ERROR] Pretrained model ({self.pretrained_model_weight_path}) does not exist.")
            self.crispr_hw_model.load_state_dict(torch.load(self.pretrained_model_weight_path))
        
        # Prepare for training
        self.downsampling_dataset(sampling_rate=0.2)
        train_loader = DataLoader(CustomDataset(self.train_dataset_temp), batch_size=self.batch_size, shuffle=True)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.crispr_hw_model.parameters(), lr=self.learning_rate)
        self.crispr_hw_model.train()
        
        # training
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"[Epoch] {epoch+1}"):
                inputs = batch["input"]
                labels = batch["label"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.crispr_hw_model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader)}")
        
        # Save model weights
        torch.save(self.crispr_hw_model.state_dict(), self.model_path)
        
    
    def test_classification_task(self):
        print(f"[TEST] CRISPR-HW model prediction. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        # Load model
        if not os.path.exists(self.model_path):
            sys.exit(f"[ERROR] Trained model ({self.model_path}) does not exist.")
            
        # Extract True label
        true_label_np = torch.IntTensor(self.test_dataset["label"]).cpu().numpy()
        
        if not os.path.exists(self.probabilities_array_path):
            self.crispr_hw_model.load_state_dict(torch.load(self.model_path))
            self.crispr_hw_model.eval()
            
            # Process for test data
            test_loader = DataLoader(CustomDataset(self.test_dataset), batch_size=self.batch_size, shuffle=False)
            
            # Predict
            all_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["input"]
                    labels = batch["label"]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.crispr_hw_model(inputs)
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



