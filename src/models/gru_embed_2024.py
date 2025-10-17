
import os
import time
import tqdm
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import models.data_loader as data_loader
import models.result as result
import utils.sequence_module as sequence_module


BASE_INDEX = {"A": 0, "T": 1, "C": 2, "G": 3, "-": 4, "N": 4} 


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict['inputs'], dtype=torch.int8)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.long)

    def __len__(self):
        # return length of datasets
        return len(self.labels)

    def __getitem__(self, idx):
        # Get data of index which be specified
        sample = {
            'inputs': self.inputs[idx],
            'labels': self.labels[idx]
        }
        return sample


class GRUEmbedModel(nn.Module):
    def __init__(self):
        super(GRUEmbedModel, self).__init__()
        self.input_dim = 25
        self.embed_dim = 44
        self.gru_units = 64
        self.dense_units = [128, 64]
        self.embed_dropout = 0.2
        self.additional_input_size = None
        
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.embed_dim)
        self.embed_dropout = nn.Dropout(self.embed_dropout)
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


class DataProcessorGRUEmbed:
    def __init__(self, config):
        self.config = config
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len)
        return (padded_seq_rna, padded_seq_dna)
    
    @staticmethod
    def _process_pairseq_to_categorical(args) -> list:
        seq_rna, seq_dna = args
        encode_array = []
        for base_rna, base_dna in zip(seq_rna, seq_dna):
            encode_array.append(
                BASE_INDEX[base_rna] * 5 + BASE_INDEX[base_dna]
            )
        return encode_array
    
    def preprocess_inputs(self, dataset: dict) -> None:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Check if the input path exists
        input_path = self.config["input_data_paths"]["input_path"]

        # Input sequence processing
        rna_seq_list = dataset["rna_seq"]
        dna_seq_list = dataset["dna_seq"]
        
        # Prepare the arguments for multiprocessing
        worker_args = [(seq_rna, seq_dna, self.max_pairseq_len) for seq_rna, seq_dna in zip(rna_seq_list, dna_seq_list)]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_alignment_hyphen, worker_args), total=len(worker_args), desc="Processing sequences"))
        
        with Pool(processes=cpu_count) as pool:
            encoding = list(tqdm.tqdm(pool.imap(self._process_pairseq_to_categorical, _processed_seqs), total=len(_processed_seqs), desc="Encoding sequences"))
        
        # Save as Torch tensor
        input_tensor = torch.tensor(encoding, dtype=torch.int8)
        torch.save(input_tensor, input_path)
        print(f"Input tensor saved to {input_path}. Input tensor shape: {input_tensor.shape}")
    
    def load_inputs(self, dataset_dict: dict) -> dict:
        # Torch input file
        input_path = self.config["input_data_paths"]["input_path"]
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist. Please run python3 run_preprocess.py.")
        else:
            input_tensor = torch.load(input_path)

        # Load labels
        label_list = dataset_dict["label"]
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        
        # Convert to PyTorch Dataset and split it into train and test
        if self.config["fold"] == "all":
            dataset_dict["all_dataset"] = CustomDataset({"inputs": input_tensor, "labels": label_tensor})
        else:
            train_idx = dataset_dict["train_idx"]
            test_idx = dataset_dict["test_idx"]
            dataset_dict["train_dataset"] = CustomDataset({"inputs": input_tensor[train_idx], "labels": label_tensor[train_idx]})
            dataset_dict["test_dataset"] = CustomDataset({"inputs": input_tensor[test_idx], "labels": label_tensor[test_idx]})
        return dataset_dict


class GRUEmbedModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict
        
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 10
        self.batch_size = 512
        self.learning_rate = 0.005
        
        # Path information
        self.model_path = config["model_info"]["model_path"]
        self.result_path = config["paths"]["result_path"]
        self.probability_path = config["paths"]["probability_path"]
        self.time_path = config["paths"]["time_path"]
        
        # Device information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()
    
    
    def training_loop(self, model: nn.Module, train_dataloader: DataLoader, 
                      optimizer: torch.optim.Optimizer, criterion: nn.Module) -> nn.Module:
        print("Starting training loop...")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
                inputs = batch['inputs'].to(self.device).long()
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return model
    
    def inference_loop(self, model: nn.Module, test_dataloader: DataLoader) -> dict: # -> {probability: np.array, prediction: np.array}
        model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader), desc="Inference"):
                inputs = batch['inputs'].to(self.device).long()
                
                outputs = model(inputs)
                logits = outputs.cpu().numpy()
                all_logits.append(logits)
        all_logits = np.vstack(all_logits)
        probabilities = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
        predictions = np.argmax(all_logits, axis=1)
        return {"probability": probabilities, "prediction": predictions}

    def train_scratch(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"]
        else:
            train_dataset = self.dataset_dict["train_dataset"]
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

        # Model preparation
        model = GRUEmbedModel().to(self.device)
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion)
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # model.half()
        torch.save(model.state_dict(), self.model_path)
        
        # Save the training time
        end_time = time.time()
        with open(self.time_path, 'w') as f:
            f.write(str(end_time - self.start_time))
    
    def test_scratch(self) -> None:
        # Load dataset
        if self.fold == "all":
            test_dataset = self.dataset_dict["all_dataset"]
        else:
            test_dataset = self.dataset_dict["test_dataset"]
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist. Please run training first.")
        model = GRUEmbedModel().to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        
        # Inference
        inference_result = self.inference_loop(model, test_dataloader)
        
        # Result processing
        probabilities = inference_result["probability"]
        predictions = inference_result["prediction"]
        true_labels = test_dataset.labels.numpy()
        
        # Save the results
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.probability_path), exist_ok=True)
        result_metrics = result.return_metrics(self.fold, self.iter, list(true_labels), list(predictions), list(probabilities))
        result.save_results(result_metrics, self.result_path)
        np.save(self.probability_path, probabilities)

    # Transfer learning
    def train_transfer(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"]
        else:
            train_dataset = self.dataset_dict["train_dataset"]
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

        # Load in vitro model
        in_vitro_model_path = self.config["model_info"]["in_vitro_model"]
        model = GRUEmbedModel().to(self.device)
        model.load_state_dict(torch.load(in_vitro_model_path))
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion)
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # model.half()
        torch.save(model.state_dict(), self.model_path)
        
        # Save the training time
        end_time = time.time()
        with open(self.time_path, 'w') as f:
            f.write(str(end_time - self.start_time))
        
    def test_transfer(self) -> None:
        self.test_scratch()
        