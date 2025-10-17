
import numpy as np
import os
import tqdm
import time
import multiprocessing
from multiprocessing import Pool
from itertools import product
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import models.data_loader as data_loader
import models.result as result
import utils.sequence_module as sequence_module


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


class CrisprDipoff2025Model(nn.Module):
    def __init__(self, device):
        super(CrisprDipoff2025Model, self).__init__()
        self.device = device
        # Define model architecture
        self.emb_size = 5
        self.hidden_size = 512
        self.lstm_layers = 1
        self.bi_lstm = True
        self.reshape = False
        self.number_hidden_layers = 2
        self.dropout_prob = 0.4
        # Define hidden layers
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


class DataProcessorCrisprDipoff:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        self.seq_dim = 5
        self.base_index = {'A':0, 'T':1, 'G':2, 'C':3, '-':4}
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len)
        return (padded_seq_rna, padded_seq_dna)
    
    @staticmethod
    def _seq_to_encoding(args) -> np.ndarray:
        seq_rna, seq_dna, base_index, seq_dim = args
        encode_array = np.zeros((len(seq_rna), seq_dim), dtype=np.int8)
        for i, (rna_base, dna_base) in enumerate(zip(seq_rna, seq_dna)):
            encode_array[i][base_index.get(rna_base, 4)] = 1
            encode_array[i][base_index.get(dna_base, 4)] = 1
        return encode_array

    def preprocess_inputs(self, dataset_dict: dict) -> None:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Input processing
        input_path = self.config["input_data_paths"]["input_path"]
        rna_seq_list = dataset_dict["rna_seq"]
        dna_seq_list = dataset_dict["dna_seq"]
        
        # Prepare the arguments for multiprocessing
        worker_args = [(seq_rna, seq_dna, self.max_pairseq_len) for seq_rna, seq_dna in zip(rna_seq_list, dna_seq_list)]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_alignment_hyphen, worker_args), total=len(worker_args), desc="Processing sequences"))

        # Unzip the processed sequences
        processed_rna_seqs, processed_dna_seqs = zip(*_processed_seqs)
        worker_args = [(seq_rna, seq_dna, self.base_index, self.seq_dim) for seq_rna, seq_dna in zip(processed_rna_seqs, processed_dna_seqs)]
        with Pool(processes=cpu_count) as pool:
            encoded_seqs = list(tqdm.tqdm(pool.imap(self._seq_to_encoding, worker_args), total=len(worker_args), desc="Encoding sequences"))
        
        # Save as torch tensor
        input_tensor = torch.tensor(encoded_seqs, dtype=torch.int8)
        print(f"Input data shape: {input_tensor.shape}")
        torch.save(input_tensor, input_path)
        print(f"Input data saved to {input_path}")
    
    def load_inputs(self, dataset_dict: dict) -> dict:
        # Torch input file
        input_path = self.config["input_data_paths"]["input_path"]
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file {input_path} does not exist. Please run python3 run_preprocess.py.")
        input_tensor = torch.load(input_path)
        
        # Load labels
        label_list = dataset_dict["label"]
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        
        # Convert to PyTorch Dataset and split it into train and test
        if self.config["fold"] == "all":
            dataset_dict["all_dataset"] = CustomDataset({'inputs': input_tensor, 'labels': label_tensor})
        else:
            train_idx = dataset_dict["train_idx"]
            test_idx = dataset_dict["test_idx"]
            dataset_dict["train_dataset"] = CustomDataset({'inputs': input_tensor[train_idx], 'labels': label_tensor[train_idx]})
            dataset_dict["test_dataset"] = CustomDataset({'inputs': input_tensor[test_idx], 'labels': label_tensor[test_idx]})
        return dataset_dict 


class CRISPRDIPOFFModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict

        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 50
        # self.epochs = 1 # For testing
        self.batch_size = 64
        self.learning_rate = 0.0001
    
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
                inputs = batch['inputs'].to(self.device).float()
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
                inputs = batch['inputs'].to(self.device).float()
                
                outputs = model(inputs)
                logits = outputs.cpu().numpy()
                all_logits.append(logits)
        all_logits = np.vstack(all_logits)
        probabilities = torch.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
        predictions = np.argmax(all_logits, axis=1)
        return {"probability": probabilities, "prediction": predictions}
    
    # From scratch training and testing
    def train_scratch(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"]
        else:
            train_dataset = self.dataset_dict["train_dataset"]
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)
        
        # Model preparation
        model = CrisprDipoff2025Model(self.device).to(self.device)
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
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
        model = CrisprDipoff2025Model(self.device).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # Inference loop
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
        
    # Transfer learning training and testing
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
        model = CrisprDipoff2025Model(self.device).to(self.device)
        model.load_state_dict(torch.load(in_vitro_model_path, map_location=self.device))
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion)
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(model.state_dict(), self.model_path)
        
        # Save the training time
        end_time = time.time()
        with open(self.time_path, 'w') as f:
            f.write(str(end_time - self.start_time)) 
    
    def test_transfer(self) -> None:
        self.test_scratch()
        
        
        
        
        