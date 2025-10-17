

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
from transformers import BertModel, BertConfig

import models.data_loader as data_loader
import models.result as result
import utils.sequence_module as sequence_module




class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = torch.tensor(data_dict['inputs'], dtype=torch.int16)
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


class DataProcessorCrisprBERT:
    def __init__(self, config):
        self.config = config
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        
        self.kmer = 2
        self.base_index = {'A':0, 'T':1, 'G':2, 'C':3, '-':4}
        self.seq_token_dict = self.return_seq_token_dict()
    
    def return_seq_token_dict(self):
        base_list = list(self.base_index.keys())
        depth = 2 * self.kmer
        token_touple_list = list(product(base_list, repeat=depth))
        token_chr_list = ["".join(token_touple) for token_touple in token_touple_list]
        seq_token_dict = {token: i for i, token in enumerate(token_chr_list)}
        return seq_token_dict
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len)
        return (padded_seq_rna, padded_seq_dna)
    
    @staticmethod
    def _process_seq_to_token(args) -> np.ndarray:
        padded_seq_rna, padded_seq_dna, seq_token_dict, kmer = args
        token_id_array = np.zeros(len(padded_seq_rna) - kmer + 1, dtype=np.int16)
        for i in range(len(padded_seq_rna) - kmer + 1):
            token = padded_seq_rna[i:i+kmer] + padded_seq_dna[i:i+kmer]
            token_id_array[i] = seq_token_dict.get(token, 0)
        return token_id_array

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
        
        worker_args = [(padded_seq_rna, padded_seq_dna, self.seq_token_dict, self.kmer) for padded_seq_rna, padded_seq_dna in _processed_seqs]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_seq_to_token, worker_args), total=len(worker_args), desc="Tokenizing sequences"))

        # Save as torch tensor
        input_tensor = torch.tensor(_processed_seqs, dtype=torch.int16)
        torch.save(input_tensor, input_path)
        print(f"Input tensor saved to {input_path}. Input tensor shape: {input_tensor.shape}")
        
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


class CrisprBERTModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict
        
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 2e-5
        self.kmer = 2
        self.base_index = {'A':0, 'T':1, 'G':2, 'C':3, '-':4}
        
        # Path information
        self.model_path = config["model_info"]["model_path"]
        self.result_path = config["paths"]["result_path"]
        self.probability_path = config["paths"]["probability_path"]
        self.time_path = config["paths"]["time_path"]
        
        # Device information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()
        
        # Model information
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
        model = CrisprBert2025Model(self.bert_model).to(self.device) # Total parameters: 439,330
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters: {total_params}")

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
        model = CrisprBert2025Model(self.bert_model).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
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
        model = CrisprBert2025Model(self.bert_model).to(self.device)
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