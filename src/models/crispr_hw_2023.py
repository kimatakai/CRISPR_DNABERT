

import os
import time
import numpy as np
import tqdm
import multiprocessing
from multiprocessing import Pool
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

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


class DataProcessorCRISPRHW:
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
    
    def preprocess_inputs(self, dataset_dict: dict) -> None:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Check if the input path exists
        input_path = self.config["input_data_paths"]["input_path"]

        # Input sequence processing
        rna_seq_list = dataset_dict["rna_seq"]
        dna_seq_list = dataset_dict["dna_seq"]
        
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
            print(f"Input tensor loaded from {input_path}. Input tensor shape: {input_tensor.shape}")

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


class CRISPRHWModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict

        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 30
        self.batch_size = 128
        self.learning_rate = 0.003
        self.vocab_size = 25
        self.input_len = config["parameters"]["max_pairseq_len"]
        self.embed_dim = 100

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
        model = CrisprHwModel(vocab_size=self.vocab_size, embed_size=self.embed_dim, maxlen=self.input_len).to(self.device)
        
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
        model = CrisprHwModel(vocab_size=self.vocab_size, embed_size=self.embed_dim, maxlen=self.input_len).to(self.device)
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
        model = CrisprHwModel(vocab_size=self.vocab_size, embed_size=self.embed_dim, maxlen=self.input_len).to(self.device)
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