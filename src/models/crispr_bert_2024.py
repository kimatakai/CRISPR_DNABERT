
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


import pandas as pd
import numpy as np
import os
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertConfig


ENCODE_DICT_INDEL = {
    'AA': np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    'AT': np.array([1, 1, 0, 0, 0, 1, 0], dtype=np.int8),
    'AG': np.array([1, 0, 1, 0, 0, 1, 0], dtype=np.int8),
    'AC': np.array([1, 0, 0, 1, 0, 1, 0], dtype=np.int8),
    'TA': np.array([1, 1, 0, 0, 0, 0, 1], dtype=np.int8),
    'TT': np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
    'TG': np.array([0, 1, 1, 0, 0, 1, 0], dtype=np.int8),
    'TC': np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int8),
    'GA': np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.int8),
    'GT': np.array([0, 1, 1, 0, 0, 0, 1], dtype=np.int8),
    'GG': np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.int8),
    'GC': np.array([0, 0, 1, 1, 0, 1, 0], dtype=np.int8),
    'CA': np.array([1, 0, 0, 1, 0, 0, 1], dtype=np.int8),
    'CT': np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8),
    'CG': np.array([0, 0, 1, 1, 0, 0, 1], dtype=np.int8),
    'CC': np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int8),
    'A-': np.array([1, 0, 0, 0, 1, 1, 0], dtype=np.int8),
    'T-': np.array([0, 1, 0, 0, 1, 1, 0], dtype=np.int8),
    'G-': np.array([0, 0, 1, 0, 1, 1, 0], dtype=np.int8),
    'C-': np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.int8),
    '-A': np.array([1, 0, 0, 0, 1, 0, 1], dtype=np.int8),
    '-T': np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int8),
    '-G': np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.int8),
    '-C': np.array([0, 0, 0, 1, 1, 0, 1], dtype=np.int8),
    '--': np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    'NN': np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
}
TOKEN_DICT = {
    '[CLS]': 0, '[SEP]': 1,
    'AA': 2, 'AC': 3, 'AG': 4, 'AT': 5,
    'CA': 6, 'CC': 7, 'CG': 8, 'CT': 9,
    'GA': 10, 'GC': 11, 'GG': 12, 'GT': 13,
    'TA': 14, 'TC': 15, 'TG': 16, 'TT': 17,
    'A-': 18, '-A': 19, 'C-': 20, '-C': 21, 'G-': 22,
    '-G': 23, 'T-': 24, '-T': 25, '--': 26, 'NN': 27
}



class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.encode_inputs = torch.tensor(data_dict['encode_inputs'], dtype=torch.int8)
        self.token_inputs = torch.tensor(data_dict['token_inputs'], dtype=torch.int8)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.long)

    def __len__(self):
        # return length of datasets
        return len(self.labels)

    def __getitem__(self, idx):
        # Get data of index which be specified
        sample = {
            'encode_inputs': self.encode_inputs[idx],
            'token_inputs': self.token_inputs[idx],
            'labels': self.labels[idx]
        }
        return sample


class CrisprBert2024Model(nn.Module):
    def __init__(self, bert_model, input_shape):
        super(CrisprBert2024Model, self).__init__()
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


class DataProcessorCRISPRBERT:
    def __init__(self, config):
        self.config = config
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        
        self.total_seq_len = self.max_pairseq_len + 2 # +2 for [CLS] and [SEP] tokens
        self.dim_encode = 7
    
    @staticmethod
    def _process_input_pairseq(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len, dim_encode = args
        seq_rna, seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len-2)
        seq_rna, seq_dna, __ = sequence_module.complete_bulge_seq(seq_rna, seq_dna)
        # Encode the sequences to numerical format
        encode_array = np.zeros((max_pairseq_len, dim_encode), dtype=np.int8)
        for i, (rna_base, dna_base) in enumerate(zip(seq_rna, seq_dna)):
            encode_array[i+1] = ENCODE_DICT_INDEL[rna_base + dna_base]
        # Tokenize
        token_array = np.zeros(max_pairseq_len, dtype=np.int8)
        token_array[0] = TOKEN_DICT['[CLS]']
        for j, (rna_base, dna_base) in enumerate(zip(seq_rna, seq_dna)):
            token_array[j+1] = TOKEN_DICT[rna_base + dna_base]
        token_array[-1] = TOKEN_DICT['[SEP]']
        return (encode_array, token_array)
    
    def preprocess_inputs(self, dataset_dict: dict) -> None:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Input paths
        encode_input_path = self.config["input_data_paths"]["encode_input_path"]
        token_input_path = self.config["input_data_paths"]["token_input_path"]
        
        # Input sequence processing
        rna_seq_list = dataset_dict["rna_seq"]
        dna_seq_list = dataset_dict["dna_seq"]

        # Align all pair sequences to the same length (l=self.max_pairseq_len) -> torch.tensor
        worker_args = [(seq_rna, seq_dna, self.total_seq_len, self.dim_encode) for seq_rna, seq_dna in zip(rna_seq_list, dna_seq_list)]
        with Pool(processes=cpu_count) as pool:
            processed_inputs = list(tqdm.tqdm(pool.imap(self._process_input_pairseq, worker_args), total=len(worker_args), desc="Processing inputs"))
        encode_input = [item[0] for item in processed_inputs]
        token_input = [item[1] for item in processed_inputs]
        encode_input = torch.tensor(encode_input, dtype=torch.int8)
        token_input = torch.tensor(token_input, dtype=torch.int8)

        # Save as Torch tensor
        torch.save(encode_input, encode_input_path)
        torch.save(token_input, token_input_path)
        print(f"Encoded input saved to {encode_input_path}")
        print(f"Token input saved to {token_input_path}")
    
    def load_inputs(self, dataset_dict: dict) -> dict:
        # Torch input file
        encode_input_path = self.config["input_data_paths"]["encode_input_path"]
        token_input_path = self.config["input_data_paths"]["token_input_path"]
        for path in [encode_input_path, token_input_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file {path} does not exist. Please run python3 run_preprocess.py.")
        encode_input = torch.load(encode_input_path)
        token_input = torch.load(token_input_path)
        
        # Load labels
        label_list = dataset_dict["label"]
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        
        # Convert to PyTorch Dataset and split it into train and test
        if self.config["fold"] == "all":
            dataset_dict["all_dataset"] = CustomDataset({"encode_inputs": encode_input, "token_inputs": token_input, "labels": label_tensor})
        else:
            train_idx = dataset_dict["train_idx"]
            test_idx = dataset_dict["test_idx"]
            dataset_dict["train_dataset"] = CustomDataset({"encode_inputs": encode_input[train_idx], "token_inputs": token_input[train_idx], "labels": label_tensor[train_idx]})
            dataset_dict["test_dataset"] = CustomDataset({"encode_inputs": encode_input[test_idx], "token_inputs": token_input[test_idx], "labels": label_tensor[test_idx]})
        return dataset_dict



class CRISPRBERTModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict
        
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 30
        self.batch_size = 256
        self.learning_rate = 0.0001
        
        # Path information
        self.base_model_path = config["model_info"]["base_model"]
        self.base_model_config_path = self.base_model_path + "/bert_config.json"
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
                encode_input = batch['encode_inputs'].to(self.device).long()
                token_input = batch['token_inputs'].to(self.device).long()
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = model(encode_input, token_input)
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
                encode_input = batch['encode_inputs'].to(self.device).long()
                token_input = batch['token_inputs'].to(self.device).long()
                
                outputs = model(encode_input, token_input)
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

        # Model preperation
        bert_config = BertConfig.from_json_file(self.base_model_config_path)
        model = CrisprBert2024Model(
            bert_model=BertModel(bert_config).to(self.device),
            input_shape=(self.config["parameters"]["max_pairseq_len"] + 2, 7)
        ).to(self.device)
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
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
        bert_config = BertConfig.from_json_file(self.base_model_config_path)
        model = CrisprBert2024Model(
            bert_model=BertModel(bert_config).to(self.device),
            input_shape=(self.config["parameters"]["max_pairseq_len"] + 2, 7)
        ).to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        
        # Inference Loop
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
    
    # Transfer learning training
    def train_transfer(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"]
        else:
            train_dataset = self.dataset_dict["train_dataset"]
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

        # Load in vitro model
        bert_config = BertConfig.from_json_file(self.base_model_config_path)
        model = CrisprBert2024Model(
            bert_model=BertModel(bert_config).to(self.device),
            input_shape=(self.config["parameters"]["max_pairseq_len"] + 2, 7)
        )
        in_vitro_model_path = self.config["model_info"]["in_vitro_model"]
        model.load_state_dict(torch.load(in_vitro_model_path))
        model = model.to(self.device)
        
        # Training arguments
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
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