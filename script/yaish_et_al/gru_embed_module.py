
import sys
sys.path.append("script/")

import os
import tqdm

import config
import utilities_module

import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
from torch.utils.data import DataLoader, Dataset
import numpy as np



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

class GRUEmbModel(nn.Module):
    def __init__(self, input_dim=25, embed_dim=44, gru_units=64, dense_units=[128, 64], 
                 embed_dropout=0.2, additional_input_size=None):
        super(GRUEmbModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.additional_input_size = additional_input_size
        
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=self.embed_dim)
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # GRU Layer
        self.gru = nn.GRU(self.embed_dim, self.gru_units, batch_first=True)
        
        # Dense Layers
        self.flatten = nn.Flatten()
        
        # Dense Layers
        input_size = self.gru_units * 24 
        if self.additional_input_size:
            input_size += self.additional_input_size
        
        self.dense_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        for units in self.dense_units:
            self.dense_layers.append(nn.Linear(input_size, units))
            self.activation_layers.append(nn.ReLU())
            input_size = units 
        
        # --- Output Layer ---
        self.output = nn.Linear(self.dense_units[-1], 2)
    
    def forward(self, x, additional_input=None):
        # Embedding
        x = self.embedding(x)
        x = self.embed_dropout(x)
        
        # GRU
        x, _ = self.gru(x)
        x = self.flatten(x)
        
        # Concatenate
        if additional_input is not None:
            x = torch.cat((x, additional_input), dim=1)
        
        # Dense Layers
        for dense, activation in zip(self.dense_layers, self.activation_layers):
            x = activation(dense(x))
        
        # Output Layer
        x = self.output(x)
        return x
    



class GruEmbedClass:
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
        
        os.makedirs(config.gru_embed_model_path, exist_ok=True)
        if self.datatype == "transfer":
            self.pretrained_model_weight_path = utilities_module.return_model_weight_path(model_name="gru-embed", datatype="changeseq", fold=self.fold, exp_id=self.exp_id)
        else:
            self.pretrained_model_weight_path = None
        if model_datatype:
            self.model_path = utilities_module.return_model_weight_path(model_name="gru-embed", datatype=model_datatype, fold=self.fold, exp_id=self.exp_id)
        else:
            self.model_path = utilities_module.return_model_weight_path(model_name="gru-embed", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/gru_embed/"
        os.makedirs(self.predicted_probabilities_path, exist_ok=True)
        self.probability_array_path = utilities_module.return_output_probability_path(model_name="gru-embed", datatype=self.datatype, fold=self.fold, exp_id=self.exp_id)
        
        self.input_dim = 25
        self.input_shape = (24, )
        self.embed_dim = 44
        self.gru_units = 64
        self.dense_units = [128, 64]
        if self.datatype != "transfer":
            self.epochs = 10
        else:
            self.epochs = 5
        self.batch_size = 512
        self.learning_rate = 0.005
        self.embed_dropout = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gru_emb_model = GRUEmbModel(input_dim=self.input_dim, embed_dim=self.embed_dim, gru_units=self.gru_units, dense_units=self.dense_units, embed_dropout=self.embed_dropout).to(self.device)
        
    
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
        print(f"[TRAIN] GRU-Embed model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENT: {self.exp_id}. {self.device} will be used.")
        
        # Load pretrained model if transfer learning
        if self.datatype == "transfer":
            if not os.path.exists(self.pretrained_model_weight_path):
                sys.exit(f"[ERROR] Pretrained model ({self.pretrained_model_weight_path}) does not exist.")
            self.gru_emb_model.load_state_dict(torch.load(self.pretrained_model_weight_path))
        
        # Prepare for training
        self.downsampling_dataset(sampling_rate=0.2)
        train_loader = DataLoader(CustomDataset(self.train_dataset_temp), batch_size=self.batch_size, shuffle=True)
        
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.gru_emb_model.parameters(), lr=self.learning_rate)
        self.gru_emb_model.train()
        
        # Training
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"[Epoch] {epoch+1}"):
                inputs = batch["input"]
                labels = batch["label"]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                with autocast():
                    outputs = self.gru_emb_model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_loader)}")
        
        # Save model weights
        torch.save(self.gru_emb_model.state_dict(), self.model_path)


    def test_classification_task(self):
        print(f"[TEST] GRU-Embed model prediction. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENT: {self.exp_id}. {self.device} will be used.")
        # Load model
        if not os.path.exists(self.model_path):
            sys.exit(f"[ERROR] Trained model ({self.model_path}) does not exist.")
            
        # Extract True label
        true_label_np = torch.IntTensor(self.test_dataset["label"]).cpu().numpy()
        
        if not os.path.exists(self.probability_array_path):
            self.gru_emb_model.load_state_dict(torch.load(self.model_path))
            self.gru_emb_model.eval()
            
            # Process for test data
            test_loader = DataLoader(CustomDataset(self.test_dataset), batch_size=self.batch_size, shuffle=False)
            
            # Predict
            all_logits = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["input"]
                    labels = batch["label"]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.gru_emb_model(inputs)
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
            



