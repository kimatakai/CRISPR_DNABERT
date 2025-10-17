

import os
import tqdm
import time
import multiprocessing
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pyarrow as pa
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import Dataset

import utils.sequence_module as sequence_module
import visualization.plot_epigenetic_fig as plot_epigenetic_fig
import visualization.plot_bert_fig as plot_bert_fig
import models.data_loader as data_loader
import models.result as result



def seq_to_token(seq: str, kmer: int=3) -> str:
    kmers_list = [seq[i:i+kmer] for i in range(len(seq) - kmer + 1)]
    merged_sequence = ' '.join(kmers_list)
    return merged_sequence



class DataProcessorDNABERT:
    def __init__(self, config: dict):
        self.config = config
        self.kmer = 3
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        self.token_max_len = 2 * (self.max_pairseq_len - self.kmer + 1) + 3

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(self.config["model_info"]["pretrained_model"]))

        self.input_path = self.config["input_data_paths"]["input_path"]
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna, max_pairseq_len = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=max_pairseq_len)
        return (padded_seq_rna, padded_seq_dna)
    
    @staticmethod
    def _process_seq_to_token(args) -> tuple:
        seq_rna, seq_dna = args
        rna_seq_token = seq_to_token(seq_rna)
        dna_seq_token = seq_to_token(seq_dna)
        return (rna_seq_token, dna_seq_token)
    
    def tokenize_function(self, example):
        return self.tokenizer(example["rna_seq"], example["dna_seq"], padding='max_length', truncation=True, max_length=self.token_max_len)
    
    def nparray_to_list(self, epi_feature: np.ndarray) -> list:
        flat_feature = epi_feature.flatten()
        offsets = np.arange(0, epi_feature.size + 1, epi_feature.shape[1], dtype=int)
        pyarrow_array = pa.ListArray.from_arrays(offsets, flat_feature)
        return pyarrow_array
    
    def mismatch_to_onehot(self, mismatch_list: list) -> list: # 1 -> [0, 1, 0, 0, 0, 0, 0], 3 -> [0, 0, 0, 1, 0, 0, 0]
        onehot_list = []
        for mismatch in tqdm.tqdm(mismatch_list, total=len(mismatch_list), desc="Processing mismatch to one-hot"):
            onehot = [0] * 7
            if 0 <= mismatch <= 6:
                onehot[mismatch] = 1
            onehot_list.append(onehot)
        return onehot_list
    

    def preprocess_inputs(self, dataset_dict: dict) -> None:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Input sequence processing
        rna_seq_list = dataset_dict["rna_seq"]
        dna_seq_list = dataset_dict["dna_seq"]

        # Prepare the arguments for multiprocessing
        worker_args = [(seq_rna, seq_dna, self.max_pairseq_len) for seq_rna, seq_dna in zip(rna_seq_list, dna_seq_list)]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_alignment_hyphen, worker_args), total=len(worker_args), desc="Processing sequences"))
        
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_seq_to_token, _processed_seqs), total=len(_processed_seqs), desc="Converting sequences to tokens"))
        rna_seq_list, dna_seq_list = zip(*_processed_seqs) # [[-GC GCA CAC ACA ...], [-GC GCA CAC ACC ...], ...]

        # Hugging face tokenizer need to be used with Dataset
        input_dataset = Dataset.from_dict({"rna_seq": rna_seq_list, "dna_seq": dna_seq_list})
        tokenized_input_dataset = input_dataset.map(self.tokenize_function, batched=True) # features: ['rna_seq', 'dna_seq', 'input_ids', 'token_type_ids', 'attention_mask']

        # Save the tokenized dataset (input_ids, attention_mask, token_type_ids) as torch tensors. Remove the original sequences.
        os.makedirs(self.input_path, exist_ok=True)
        tokenized_input_dataset = tokenized_input_dataset.remove_columns(["rna_seq", "dna_seq"])
        tokenized_input_dataset.save_to_disk(self.input_path)
        return None
    
    def load_inputs(self, dataset_dict: dict) -> dict:
        # Check if the input files exist
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path {self.input_path} does not exist. Please run python3 run_preprocess.py.")
        
        # Load the input tensors
        data = Dataset.load_from_disk(self.input_path)
        
        # Column list
        columns_to_format = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        
        # Whether content of 'using_epi_data' exists
        if self.config["using_epi_data"]:
            # Add mismatch features and bulges features as one-hot vectors
            data = data.add_column("mismatch", self.mismatch_to_onehot(dataset_dict["mismatch"]))
            columns_to_format.append("mismatch")
            data = data.add_column("bulge", dataset_dict["bulge"])
            columns_to_format.append("bulge")
            # Add epigenetic features
            for type_of_data in self.config["using_epi_data"]:
                epigenetic_feature = dataset_dict["epigenetic_features"][type_of_data]
                data = data.add_column(type_of_data, self.nparray_to_list(epigenetic_feature))
            columns_to_format += self.config["using_epi_data"]
        # Add labels
        data = data.add_column("labels", dataset_dict["label"])
        
        data.set_format(type="torch", columns=columns_to_format)
        
        # Split the dataset to train and test
        if self.config["fold"] == "all":
            dataset_dict["all_dataset"] = data
        else:
            train_idx = dataset_dict["train_idx"]
            test_idx = dataset_dict["test_idx"]
            dataset_dict["train_dataset"] = data.select(train_idx)
            dataset_dict["test_dataset"] = data.select(test_idx)
        return dataset_dict


class DNABERTEpiModule(nn.Module):
    def __init__(self, config: dict, dnabert_base_model: nn.Module, using_epi_data: list=["atac"]):
        super().__init__()
        self.config = config
        self.mismatch_dim = 7
        self.bulge_dim = 1
        self.epi_hidden_dim = 256
        self.using_epi_data = using_epi_data
        self.epi_feature_dim = {}
        for type_of_data in self.using_epi_data:
            self.epi_feature_dim[type_of_data] = self.config["parameters"]["window_size"][type_of_data]*2 // self.config["parameters"]["bin_size"][type_of_data]
        
        # Sequence encoder
        self.dnabert = dnabert_base_model
        self.dnabert_hidden_size = self.dnabert.config.hidden_size # 768

        # Epigenetic feature encoder and gating module
        self.epi_encoders = nn.ModuleDict()
        self.gating_layers = nn.ModuleDict()
        for type_of_data in self.using_epi_data:
            feature_dim = self.epi_feature_dim[type_of_data]
            # Epigenetic feature encoder
            self.epi_encoders[type_of_data] = nn.Sequential(
                nn.Linear(feature_dim, self.epi_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim, self.epi_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 2, self.epi_hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 4, self.epi_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 2, self.epi_hidden_dim)
            )
            # Gating module
            self.gating_layers[type_of_data] = nn.Sequential(
                nn.Linear(self.dnabert_hidden_size + self.mismatch_dim + self.bulge_dim, self.epi_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim, self.epi_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 2, self.epi_hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 4, self.epi_hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.epi_hidden_dim * 2, self.epi_hidden_dim),
                nn.Sigmoid()
            )
            gate_linear_layer = self.gating_layers[type_of_data][-2]
            if gate_linear_layer.bias is not None:
                nn.init.constant_(gate_linear_layer.bias.data, -3.0)
            nn.init.normal_(gate_linear_layer.weight.data, mean=0.0, std=0.02)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.dnabert_hidden_size + self.epi_hidden_dim * len(self.using_epi_data), 2) # Binary classification
        )

    def forward(self, input_ids, attention_mask, token_type_ids, mismatch=None, bulge=None, **epi_features): # **kwargs
        # Fetch sequence features from DNABERT
        outputs = self.dnabert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_size]

        gated_epi_embeddings = []
        gate_output = {}
        for type_of_data in self.using_epi_data:
            if type_of_data in epi_features and epi_features[type_of_data] is not None:
                feature_tensor = epi_features[type_of_data]
                # Encode epigenetic features
                h_epi = self.epi_encoders[type_of_data](feature_tensor)
                # Generate gate from sequence feature ([CLS] embedding + mismatch + bulge)
                gate_input = torch.cat([cls_embedding, mismatch, bulge.unsqueeze(1)], dim=1)
                gate = self.gating_layers[type_of_data](gate_input)
                gate_output[type_of_data] = gate
                # Apply gate
                gated_epi_embeddings.append(h_epi * gate)

        # Concatenate sequence and epigenetic features
        if gated_epi_embeddings:
            final_embedding = torch.cat([cls_embedding] + gated_epi_embeddings, dim=1)
        else:
            final_embedding = cls_embedding
        logits = self.classifier(final_embedding)
        return logits, gate_output

class DNABERTEpiModuleForShapClass(nn.Module):
    def __init__(self, model, using_epi):
        super().__init__()
        self.model = model
        self.using_epi = using_epi

    def forward(self, input_embedding, attention_mask, token_type_ids, mismatch, bulge, *epi_features):
        # Fetch sequence features from DNABERT
        outputs = self.model.dnabert(inputs_embeds=input_embedding, attention_mask=attention_mask.long(), token_type_ids=token_type_ids.long())
        cls_embedding = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_size]
        
        # Encode and gate epigenetic features
        gated_epi_embeddings = []
        gate_output = {}
        for i, type_of_data in enumerate(self.using_epi):
            feature_tensor = epi_features[i]
            # Encode epigenetic features
            h_epi = self.model.epi_encoders[type_of_data](feature_tensor)
            # Generate gate from sequence feature ([CLS] embedding + mismatch + bulge)
            gate_input = torch.cat([cls_embedding, mismatch, bulge.unsqueeze(1)], dim=1)
            gate = self.model.gating_layers[type_of_data](gate_input)
            gate_output[type_of_data] = gate
            # Apply gate
            gated_epi_embeddings.append(h_epi * gate)
        
        # Concatenate sequence and epigenetic features
        if gated_epi_embeddings:
            final_embedding = torch.cat([cls_embedding] + gated_epi_embeddings, dim=1)
        else:
            final_embedding = cls_embedding
        logits = self.model.classifier(final_embedding)
        return logits



class DNABERTModelClass:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict
        
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.epochs = 8
        self.batch_size = 128
        self.learning_rate = 2e-5
        self.k_layer = 6
        
        # Path information
        # self.base_model_path = config["model_info"]["base_model"]
        self.pretrained_model_path = config["model_info"]["pretrained_model"]
        self.model_path = config["model_info"]["model_path"]
        self.result_path = config["paths"]["result_path"]
        self.probability_path = config["paths"]["probability_path"]
        self.time_path = config["paths"]["time_path"]
        
        # Device information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = time.time()
    
    def training_loop(self, model: AutoModelForSequenceClassification, train_dataloader: DataLoader, 
                      optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler.LambdaLR) -> AutoModelForSequenceClassification:
        print("Starting training loop...")
        scaler = torch.cuda.amp.GradScaler()
        with_epigenetic = self.config["with_epigenetic"]
        using_epi = self.config["using_epi_data"]
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{self.epochs}"):
                labels = batch["labels"].to(self.device)
                
                model_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "token_type_ids": batch["token_type_ids"].to(self.device)
                }
                
                if with_epigenetic:
                    model_inputs["mismatch"] = batch["mismatch"].to(self.device).float()
                    model_inputs["bulge"] = batch["bulge"].to(self.device).float()
                    epi_features = {
                        type_of_data: batch[type_of_data].to(self.device) for type_of_data in using_epi if type_of_data in batch
                    }
                    model_inputs.update(epi_features)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(**model_inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        return model
    
    def inference_loop(self, model: AutoModelForSequenceClassification, test_dataloader: DataLoader) -> dict: # -> {probability: np.array, prediction: np.array}
        model.eval()
        use_epi = self.config.get("using_epi_data") and len(self.config["using_epi_data"]) > 0
        
        all_logits = []
        all_gates = {dtype: [] for dtype in self.config["using_epi_data"]} if use_epi else {}
        
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, total=len(test_dataloader), desc="Inference"):
                model_inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                    "token_type_ids": batch["token_type_ids"].to(self.device)
                }

                if use_epi:
                    model_inputs["mismatch"] = batch["mismatch"].to(self.device).float()
                    model_inputs["bulge"] = batch["bulge"].to(self.device).float()
                    for dtype in self.config["using_epi_data"]:
                        if dtype in batch:
                            model_inputs[dtype] = batch[dtype].to(self.device)

                outputs = model(**model_inputs)
                
                gate_outputs = {}
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    gate_outputs = outputs[1]
                else:
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                
                all_logits.append(logits.cpu().numpy())
                if use_epi and gate_outputs:
                    for dtype, gate_tensor in gate_outputs.items():
                        all_gates[dtype].append(gate_tensor.cpu().numpy())
                
        all_logits = np.concatenate(all_logits, axis=0)
        probabilities = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()[:, 1]
        predictions = np.argmax(all_logits, axis=1)
        
        final_results = {
            "probability": probabilities,
            "prediction": predictions,
            "gates": None
        }
        
        if use_epi:
            final_gates = {}
            for dtype, gate_list in all_gates.items():
                final_gates[dtype] = np.concatenate(gate_list, axis=0) if gate_list else None
            final_results["gates"] = final_gates
        return final_results

    def freeze_all(self, model: AutoModelForSequenceClassification) -> AutoModelForSequenceClassification:
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def unfreeze_last_k_layers(self, model: AutoModelForSequenceClassification, k: int=4) -> AutoModelForSequenceClassification:
        for i in range(12 - k, 12):
            for p in getattr(model.bert.encoder.layer, str(i)).parameters():
                p.requires_grad = True
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True
        return model

    def return_trainable_state_dict(self, model: AutoModelForSequenceClassification, k: int=4) -> dict:
        layers_to_save = [f"bert.encoder.layer.{i}" for i in range(12-k, 12)]
        # layers_to_save.append("bert.encoder.layer.0")
        layers_to_save.append("classifier")
        trainable_state_dict = {
            k: v.cpu() for k, v in model.state_dict().items() if any(k.startswith(layer) for layer in layers_to_save)
        }
        return trainable_state_dict
    
    def print_trainable_ratio(self, model: AutoModelForSequenceClassification) -> None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}/{total_params} ({100 * trainable_params / total_params:.2f}%)")
    
    # Scratch training and testing    
    def train_scratch(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"] # Dataset object (input_ids, attention_mask, token_type_ids, labels)
        else:
            train_dataset = self.dataset_dict["train_dataset"] # Dataset object (input_ids, attention_mask, token_type_ids, labels)
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=12, pin_memory=True)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        model = self.freeze_all(model)
        model = self.unfreeze_last_k_layers(model, k=self.k_layer)
        self.print_trainable_ratio(model)
        model.to(self.device)
        
        # Training arguments
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        num_training_steps = self.epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion, scheduler)
        model.half()
        trainable_state_dict = self.return_trainable_state_dict(model, k=self.k_layer)
        print(trainable_state_dict.keys())
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(self.model_path)
        torch.save(trainable_state_dict, self.model_path)
        
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
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path, 
            num_labels=2, 
            ignore_mismatched_sizes=True
        )
        trainable_diff = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model.to(self.device)
        
        # Inference
        inference_results = self.inference_loop(model, test_dataloader)

        # Results processing
        probabilities = inference_results["probability"]
        predictions = inference_results["prediction"]
        true_labels = test_dataset["labels"].numpy()

        # Save the results
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.probability_path), exist_ok=True)
        result_metrics = result.return_metrics(self.fold, self.iter, list(true_labels), list(predictions), list(probabilities))
        # result.save_results(result_metrics, self.result_path)
        # np.save(self.probability_path, probabilities)
    

    # Transfer learning training and testing
    def train_transfer(self) -> None:
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"] # Dataset object (input_ids, attention_mask, token_type_ids, labels)
        else:
            train_dataset = self.dataset_dict["train_dataset"] # Dataset object (input_ids, attention_mask, token_type_ids, labels)
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler)
        
        # Load in-vitro model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        in_vitro_model_path = self.config["model_info"]["in_vitro_model"]
        trainable_diff = torch.load(in_vitro_model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model = self.freeze_all(model)
        model = self.unfreeze_last_k_layers(model, k=self.k_layer)
        self.print_trainable_ratio(model)
        model.to(self.device)
        # model.print_trainable_parameters()
        
        # Training arguments
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        num_training_steps = self.epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion, scheduler)
        model.half()
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        trainable_state_dict = self.return_trainable_state_dict(model, k=self.k_layer)
        torch.save(trainable_state_dict, self.model_path)
        
        # Save the training time
        end_time = time.time()
        with open(self.time_path, 'w') as f:
            f.write(str(end_time - self.start_time))
        
    def test_transfer(self) -> None:
        self.test_scratch()

    
    
class DNABERTEpiModelClass(DNABERTModelClass):
    def __init__(self, config: dict, dataset_dict: dict):
        super().__init__(config, dataset_dict)
        self.using_epi_data = config["using_epi_data"]
        self.training_prob_path = config["paths"]["training_prob_path"]
        self.testing_prob_path = config["paths"]["testing_prob_path"]
        # self.in_cellula_no_epi_model_path = self.config["model_info"]["in_cellula_no_epi_model"]
    
    def return_trainable_state_dict_dynamic(self, model: nn.Module) -> dict:
        trainable_state_dict = {
            name: param.cpu() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
        return trainable_state_dict

    def save_train_data_probabilities(self) -> None:
        if not os.path.exists(self.training_prob_path):
            # Load dataset
            if self.fold == "all":
                train_dataset = self.dataset_dict["all_dataset"]
            else:
                train_dataset = self.dataset_dict["train_dataset"]
            train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)

            # Load model
            print(self.in_cellula_no_epi_model_path)
            if not os.path.exists(self.in_cellula_no_epi_model_path):
                raise ValueError(f"In-cellula no-epigenetic model path does not exist: {self.in_cellula_no_epi_model_path}.")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_path,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            trainable_diff = torch.load(self.in_cellula_no_epi_model_path, map_location="cpu")
            model.load_state_dict(trainable_diff, strict=False)
            model.to(self.device)
            
            # Inference
            using_epi_data = self.config["using_epi_data"]
            self.config["using_epi_data"] = [] # Disable epigenetic data during inference
            inference_results = self.inference_loop(model, train_dataloader)
            self.config["using_epi_data"] = using_epi_data # Restore the original setting
            
            # Get the probabilities on training data
            probabilities = inference_results["probability"]
            
            # Save the probabilities
            np.save(self.training_prob_path, probabilities)
        else:
            print(f"from {self.training_prob_path}.")

    def check_data(self, dataset, probabilities: np.ndarray) -> None:
        print("Checking dataset...")
        epigenetic_features_aggregate_tn = {}
        epigenetic_features_aggregate_fp = {}
        epigenetic_features_aggregate_fn = {}
        epigenetic_features_aggregate_tp = {}
        for mismatch in [0,1,2,3,4,5,6]:
            epigenetic_features_aggregate_tn[mismatch] = []
            epigenetic_features_aggregate_fp[mismatch] = []
            epigenetic_features_aggregate_fn[mismatch] = []
            epigenetic_features_aggregate_tp[mismatch] = []
        mismatches = dataset["mismatch"]
        labels = dataset["labels"]
        atacs = dataset["atac"]
        predictions = np.argmax(probabilities, axis=1) if probabilities.ndim > 1 else (probabilities >= 0.5).astype(int)
        for i in range(len(dataset)):
            mismatch = mismatches[i].item()
            label = labels[i].item()
            prediction = predictions[i]
            atac = atacs[i].numpy()
            if label == 0 and prediction == 0:
                epigenetic_features_aggregate_tn[mismatch].append(atac)
            elif label == 1 and prediction == 1:
                epigenetic_features_aggregate_tp[mismatch].append(atac)
            elif label == 1 and prediction == 0:
                epigenetic_features_aggregate_fn[mismatch].append(atac)
            elif label == 0 and prediction == 1:
                epigenetic_features_aggregate_fp[mismatch].append(atac)
        # for mismatch in [0,1,2,3,4,5,6]:
        #     print(f"Mismatch {mismatch}:")
        #     print(f"  Label 0 count: {len(epigenetic_features_aggregate_label0[mismatch])}")
        #     print(np.mean(epigenetic_features_aggregate_label0[mismatch], axis=0) if epigenetic_features_aggregate_label0[mismatch] else "No data for label 0")
        #     print(f"  Label 1 count: {len(epigenetic_features_aggregate_label1[mismatch])}")
        #     print(np.mean(epigenetic_features_aggregate_label1[mismatch], axis=0) if epigenetic_features_aggregate_label1[mismatch] else "No data for label 1")
        
        for mismatch in [3,4,5,6]:
            tn_array = np.array(epigenetic_features_aggregate_tn[mismatch])
            fp_array = np.array(epigenetic_features_aggregate_fp[mismatch])
            fn_array = np.array(epigenetic_features_aggregate_fn[mismatch])
            tp_array = np.array(epigenetic_features_aggregate_tp[mismatch])
            tn_col = tn_array[:, 65] if tn_array.size > 0 else np.array([])
            fp_col = fp_array[:, 65] if fp_array.size > 0 else np.array([])
            fn_col = fn_array[:, 65] if fn_array.size > 0 else np.array([])
            tp_col = tp_array[:, 65] if tp_array.size > 0 else np.array([])
            min_val = min(np.min(tn_col), np.min(fp_col), np.min(fn_col), np.min(tp_col))
            max_val = max(np.max(tn_col), np.max(fp_col), np.max(fn_col), np.max(tp_col))
            bins = np.linspace(min_val, max_val, 11)
            # bins = np.array([min_val, 0, max_val])
            hist_tn, _ = np.histogram(tn_col, bins=bins)
            hist_fp, _ = np.histogram(fp_col, bins=bins)
            hist_fn, _ = np.histogram(fn_col, bins=bins)
            hist_tp, _ = np.histogram(tp_col, bins=bins)
            print(f"Mismatch {mismatch}, ATAC position 50:")
            print(bins)
            print(f"  True Negative histogram: {hist_tn}")
            print(f"  True Positive histogram: {hist_tp}")
            print(f"  False Negative histogram: {hist_fn}")
            print(f"  False Positive histogram: {hist_fp}")

    # For transfer learning with epigenetic data
    def train_transfer_epi(self) -> None:
        self.start_time = time.time() # Reset start time due to previous preprocessing time
        # Load dataset
        if self.fold == "all":
            train_dataset = self.dataset_dict["all_dataset"]
        else:
            train_dataset = self.dataset_dict["train_dataset"]
        sampler = data_loader.BalancedSampler(dataset=train_dataset, majority_rate=0.2)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=12, pin_memory=True)
        
        # Load in-vitro model
        dnabert_model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        in_vitro_model_path = self.config["model_info"]["in_vitro_model"]
        trainable_diff = torch.load(in_vitro_model_path, map_location="cpu")
        dnabert_model.load_state_dict(trainable_diff, strict=False)
        dnabert_model = self.freeze_all(dnabert_model)
        dnabert_model = self.unfreeze_last_k_layers(dnabert_model, k=6)
        dnabert_base_model = dnabert_model.bert if hasattr(dnabert_model, 'bert') else dnabert_model.roberta
        
        # Prepare DNABERTEpiModule
        model = DNABERTEpiModule(self.config, dnabert_base_model, using_epi_data=self.config["using_epi_data"])
        model = model.to(self.device)

        # Training arguments
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
        optimizer = torch.optim.AdamW([
            {'params': model.dnabert.parameters(), 'lr': self.learning_rate}, # To compare with non-epigenetic model, set same learning rate
            {'params': model.epi_encoders.parameters(), 'lr': 1e-3},
            {'params': model.gating_layers.parameters(), 'lr': 1e-3},
            {'params': model.classifier.parameters(), 'lr': self.learning_rate} # To compare with non-epigenetic model, set same learning rate
        ])
        criterion = torch.nn.CrossEntropyLoss()
        num_training_steps = self.epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
        
        # Training loop
        model = self.training_loop(model, train_dataloader, optimizer, criterion, scheduler)
        model.half()
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        trainable_state_dict = self.return_trainable_state_dict_dynamic(model)
        torch.save(trainable_state_dict, self.model_path)
        
        # Save the training time
        end_time = time.time()
        with open(self.time_path, 'w') as f:
            f.write(str(end_time - self.start_time))
    
    def test_transfer_epi(self) -> None:
        # Load dataset
        if self.fold == "all":
            test_dataset = self.dataset_dict["all_dataset"]
        else:
            test_dataset = self.dataset_dict["test_dataset"]
        test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist. Please run training first.")
        
        dnabert_model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        in_vitro_model_path = self.config["model_info"]["in_vitro_model"]
        trainable_diff = torch.load(in_vitro_model_path, map_location="cpu")
        # for k, v in trainable_diff.items():
        #     print(k, v.shape)
        dnabert_model.load_state_dict(trainable_diff, strict=False)
        dnabert_model = self.freeze_all(dnabert_model)
        dnabert_model = self.unfreeze_last_k_layers(dnabert_model, k=6)
        dnabert_base_model = dnabert_model.bert if hasattr(dnabert_model, 'bert') else dnabert_model.roberta
        
        # Prepare DNABERTEpiModule
        model = DNABERTEpiModule(self.config, dnabert_base_model, using_epi_data=self.config["using_epi_data"])
        trainable_diff = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model.to(self.device)
        
        # Inference
        inference_results = self.inference_loop(model, test_dataloader)

        # Results processing
        probabilities = inference_results["probability"]
        predictions = inference_results["prediction"]
        true_labels = test_dataset["labels"].numpy()

        # Save the results
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.probability_path), exist_ok=True)
        result_metrics = result.return_metrics(self.fold, self.iter, list(true_labels), list(predictions), list(probabilities))
        result.save_results(result_metrics, self.result_path)
        np.save(self.probability_path, probabilities)


class DNABERTEpiShapInterpretationClass:
    def __init__(self, config: dict, fold: int, iter: int):
        self.config = config
        self.fold = fold
        self.iter = iter
        self.background_size = 100
        # Epigenetic info
        self.using_epi = config["using_epi_data"]
        self.epi_name_map = {"atac": "ATAC-seq", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac"}
        # Path
        self.pretrained_model_path = config["model_info"][(fold, iter)]["pretrained_model"]
        self.in_vitro_model_path = config["model_info"][(fold, iter)]["in_vitro_model"]
        self.model_path = config["model_info"][(fold, iter)]["model_path"]
        self.shap_values_path = config["shap_values_dir_path"] # Directory path
        self.shap_fig_dir_path = config["shap_fig_dir_path"] 
        # Device information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    def sampling_balanced_data(self, test_dataset: Dataset) -> dict:
        # Separate positive and negative samples
        labels = test_dataset['labels']
        indices_0 = [i for i, label in enumerate(labels) if label == 0]
        indices_1 = [i for i, label in enumerate(labels) if label == 1]
        num_positive = len(indices_1)
        print(len(indices_0), len(indices_1))
        
        # Choose samples from label 1 data randomly. (n = len(indices_1))
        if num_positive > 200:
            num_positive = 200
        selected_indices_1 = list(np.random.choice(indices_1, size=num_positive, replace=False))
        selected_indices_0 = list(np.random.choice(indices_0, size=num_positive, replace=False))
        
        # Combine the selected indices
        balanced_indices = selected_indices_0 + selected_indices_1
        
        # Create a balanced dataset
        sampled_dataset = test_dataset.select(balanced_indices)
        # Dataset -> dict
        sampled_data_dict = sampled_dataset.to_dict()
        return sampled_data_dict
    
    def set_epi_feature_names(self):
        self.epi_feature_names = []
        for type_of_data in self.using_epi:
            epi_name = self.epi_name_map[type_of_data]
            window_size = self.config["parameters"]["window_size"][type_of_data]
            bin_size = self.config["parameters"]["bin_size"][type_of_data]
            for i in range(-window_size//bin_size, window_size//bin_size):
                self.epi_feature_names.append(f"{epi_name}:{i*bin_size}~{(i+1)*bin_size}bp")
    
    def return_epigenetic_X(self, dataset_dict: dict) -> pd.DataFrame:
        self.set_epi_feature_names()
        X_epi_array = []
        X_epi_array_dict = {}
        for type_of_data in self.using_epi:
            X_epi_array.append(dataset_dict[type_of_data])
            X_epi_array_dict[type_of_data] = np.array(dataset_dict[type_of_data], dtype=np.float16)
        X_epi_array = np.concatenate(X_epi_array, axis=1) # (num_samples, total_epi_features)
        X_epi = pd.DataFrame(X_epi_array, columns=self.epi_feature_names)
        return X_epi, X_epi_array_dict

    def calculate_shap_values(self, dataset_dict: dict):
        # Load dataset
        if self.fold == "all":
            test_dataset = dataset_dict["all_dataset"]
        else:
            test_dataset = dataset_dict["test_dataset"]
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist. Please run training first.")
        dnabert_model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        trainable_diff = torch.load(self.in_vitro_model_path, map_location="cpu")
        dnabert_model.load_state_dict(trainable_diff, strict=False)
        dnabert_base_model = dnabert_model.bert if hasattr(dnabert_model, 'bert') else dnabert_model.roberta
        # Prepare DNABERTEpiModule
        model = DNABERTEpiModule(self.config, dnabert_base_model, using_epi_data=self.config["using_epi_data"])
        trainable_diff = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model.to(self.device)
        emb_layer = model.dnabert.get_input_embeddings()
        model.eval()
        
        # Prepare SHAP explainer
        # Sample balanced data
        balanced_test_dataset = self.sampling_balanced_data(test_dataset)
        X_epi, X_epi_array_dict = self.return_epigenetic_X(balanced_test_dataset)
        # Background data
        background_indices = X_epi.sample(self.background_size, random_state=42).index.tolist()
        # CLS embedding for background data
        background_bert_input = {
            "input_ids": torch.tensor(balanced_test_dataset["input_ids"])[background_indices].to(self.device),
        }
        with torch.no_grad():
            background_emb = emb_layer(background_bert_input["input_ids"]) # (num_background, hidden_size)
        background_data = [
            background_emb,
            torch.tensor(balanced_test_dataset["attention_mask"])[background_indices].to(self.device).float(),
            torch.tensor(balanced_test_dataset["token_type_ids"])[background_indices].to(self.device).float(),
            torch.tensor(balanced_test_dataset["mismatch"])[background_indices].to(self.device).float(),
            torch.tensor(balanced_test_dataset["bulge"])[background_indices].to(self.device).float()
        ]
        for type_of_data in self.using_epi:
            background_data.append(torch.tensor(X_epi_array_dict[type_of_data])[background_indices, :].to(self.device).float())
        # All data
        all_bert_input = {
            "input_ids": torch.tensor(balanced_test_dataset["input_ids"]).to(self.device),
        }
        with torch.no_grad():
            all_emb = emb_layer(all_bert_input["input_ids"]) # (num_samples, hidden_size)
        all_data = [
            all_emb,
            torch.tensor(balanced_test_dataset["attention_mask"]).to(self.device).float(),
            torch.tensor(balanced_test_dataset["token_type_ids"]).to(self.device).float(),
            torch.tensor(balanced_test_dataset["mismatch"]).to(self.device).float(),
            torch.tensor(balanced_test_dataset["bulge"]).to(self.device).float()
        ]
        for type_of_data in self.using_epi:
            all_data.append(torch.tensor(X_epi_array_dict[type_of_data]).to(self.device).float())

        # Calculate SHAP values
        explainer = shap.DeepExplainer(DNABERTEpiModuleForShapClass(model, self.using_epi), background_data)
        shap_values = explainer.shap_values(all_data, check_additivity=False)
        # Extranct SHAP values for epigenetic features (shap_values = [emb_shap, attn_shap, token_shap, mismatch_shap, bulge_shap, epi1_shap, epi2_shap, ...])
        epigenetic_shap_values = []
        for i in range(len(self.using_epi)):
            _epi_shap_values = shap_values[5 + i] # 5 = len([emb_shap, attn_shap, token_shap, mismatch_shap, bulge_shap])
            epigenetic_shap_values.append(_epi_shap_values) # 5 = len([emb_shap, attn_shap, token_shap, mismatch_shap, bulge_shap])
        epigenetic_shap_values = np.concatenate(epigenetic_shap_values, axis=1) # (num_samples, total_epi_features)
        
        # Save SHAP values by label
        for l in [0, 1]:
            _epi_shap_values_l = epigenetic_shap_values[:, :, l]
            # -> pandas
            epi_shap_df = pd.DataFrame(_epi_shap_values_l, columns=self.epi_feature_names)
            epi_shap_df_path = self.shap_values_path + f"/shap_fold{self.fold}_iter{self.iter}_label{l}.tsv"
            epi_shap_df.to_csv(epi_shap_df_path, sep="\t", index=False)
            
        # Summary plot
        for l in [0, 1]:
            shap.summary_plot(epigenetic_shap_values[:, :, l], X_epi, feature_names=self.epi_feature_names, max_display=30, show=False)
            save_path = self.shap_fig_dir_path + f"/shap_summary_fold{self.fold}_iter{self.iter}_label{l}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()


class SHAPAnalysisClass:
    def __init__(self, config: dict):
        self.config = config
        self.folds_list = config["folds"]
        self.iters_list = config["iters"]
        self.using_epi = config["using_epi_data"]
        self.epi_name_map = {"atac": "ATAC-seq", "h3k4me3": "H3K4me3", "h3k27ac": "H3K27ac"}
        self.epi_feature_names = self.set_epi_feature_names()
        self.color_map = {type_of_data: self.config["colors"][type_of_data] for type_of_data in self.using_epi}
        # Path information
        self.shap_values_path = config["shap_values_dir_path"] # Directory path
        self.shap_fig_dir_path = config["shap_fig_dir_path"] # Directory path
    
    def set_epi_feature_names(self) -> list:
        epi_feature_names = []
        for type_of_data in self.using_epi:
            epi_name = self.epi_name_map[type_of_data]
            window_size = self.config["parameters"]["window_size"][type_of_data]
            bin_size = self.config["parameters"]["bin_size"][type_of_data]
            for i in range(-window_size//bin_size, window_size//bin_size):
                epi_feature_names.append(f"{epi_name}:{i*bin_size}~{(i+1)*bin_size}bp")
        return epi_feature_names
    
    def load_shap_values(self) -> dict:
        shap_values = {}
        for fold in self.folds_list:
            for iteration in self.iters_list:
                shap_df_path = self.shap_values_path + f"/shap_fold{fold}_iter{iteration}_label1.tsv"
                shap_df = pd.read_csv(shap_df_path, sep="\t")
                shap_values[(fold, iteration)] = shap_df.values
        return shap_values

    def calculate_and_plot_aggregated_shap_importance(self) -> None:
        shap_values = self.load_shap_values() # {(fold, iter, label): np.array}, np.array: (num_samples, num_features)
        
        # Calculate mean absolute SHAP values across folds and iterations
        mean_abs_shaps = {}
        for fold in self.folds_list:
            for iteration in self.iters_list:
                _mean_abs_shap = np.mean(np.abs(shap_values[(fold, iteration)]), axis=0) # (num_features,)
                mean_abs_shaps[(fold, iteration)] = _mean_abs_shap
        
        # Aggregate across epigenetic mark types
        mark_epu_importance = {}
        for fold in self.folds_list:
            for iteration in self.iters_list:
                mean_abs_shap = mean_abs_shaps[(fold, iteration)]
                start_idx = 0
                for type_of_data in self.using_epi:
                    window_size = self.config["parameters"]["window_size"][type_of_data]
                    bin_size = self.config["parameters"]["bin_size"][type_of_data]
                    num_bins = (window_size // bin_size) * 2
                    end_idx = start_idx + num_bins
                    mark_epu_importance[(fold, iteration, type_of_data)] = np.sum(mean_abs_shap[start_idx:end_idx])
                    start_idx = end_idx
        
        # Aggregate across folds and iterations
        aggregated_importance = {}
        for type_of_data in self.using_epi:
            aggregated_importance[type_of_data] = []
            for fold in self.folds_list:
                for iteration in self.iters_list:
                    aggregated_importance[type_of_data].append(mark_epu_importance[(fold, iteration, type_of_data)])
        
        # Plotting
        save_path = self.shap_fig_dir_path + f"/shap_epigenetic_mark_importance.png"
        plot_epigenetic_fig.plot_epi_mark_shap_importance(aggregated_importance, self.using_epi, self.epi_name_map, save_path)
        
    def calculate_and_plot_position_importance(self) -> None:
        shap_values = self.load_shap_values() # {(fold, iter, label): np.array}, np.array: (num_samples, num_features)
        
        # Calculate mean absolute SHAP values across folds and iterations
        mean_abs_shaps = {}
        for fold in self.folds_list:
            for iteration in self.iters_list:
                _mean_abs_shap = np.mean(np.abs(shap_values[(fold, iteration)]), axis=0) # (num_features,)
                mean_abs_shaps[(fold, iteration)] = _mean_abs_shap
        
        # Aggregate across folds and iterations
        aggregated_position_importance = []
        for fold in self.folds_list:
            for iteration in self.iters_list:
                aggregated_position_importance.append(mean_abs_shaps[(fold, iteration)])
        aggregated_position_importance = np.array(aggregated_position_importance) # (num_folds * num_iters, num_features)
        mean_position_importance = np.mean(aggregated_position_importance, axis=0) # (num_features,)
        std_position_importance = np.std(aggregated_position_importance, axis=0) # (num_features,)
        mark_position_importance = {}
        start_idx = 0
        for type_of_data in self.using_epi:
            window_size = self.config["parameters"]["window_size"][type_of_data]
            bin_size = self.config["parameters"]["bin_size"][type_of_data]
            num_bins = (window_size // bin_size) * 2
            end_idx = start_idx + num_bins
            mark_position_importance[type_of_data] = {
                "mean": mean_position_importance[start_idx:end_idx],
                "std": std_position_importance[start_idx:end_idx]
            }
            start_idx = end_idx
        position_dict = {}
        for type_of_data in self.using_epi:
            window_size = self.config["parameters"]["window_size"][type_of_data]
            bin_size = self.config["parameters"]["bin_size"][type_of_data]
            positions = np.arange(-window_size, window_size, bin_size)
            position_dict[type_of_data] = positions
        
        # print(mark_position_importance["atac"]["mean"].shape, position_dict["atac"].shape)
        # print(mark_position_importance["atac"]["mean"])
        # print(mark_position_importance["atac"]["std"])
        # print(mark_position_importance["h3k4me3"]["mean"])
        # print(mark_position_importance["h3k4me3"]["std"])

        # Plotting
        save_path = self.shap_fig_dir_path + f"/shap_epigenetic_position_importance.png"
        plot_epigenetic_fig.plot_epi_mark_shap_position_importance(
            mark_position_importance, position_dict, self.using_epi, self.epi_name_map, self.color_map, save_path
        )
    
    
class DNABERTExplainAnalysis:
    def __init__(self, config: dict, dataset_dict: dict):
        self.config = config
        self.dataset_dict = dataset_dict
        
        self.fold = config["fold"]
        self.iter = config["iter"]
        self.kmer = 3
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        self.token_max_len = 2 * (self.max_pairseq_len - self.kmer + 1) + 3 # [CLS], [SEP], [SEP]
        self.k_layer = 6
        
        # Path information
        self.pretrained_model_path = config["model_info"][(self.fold, self.iter)]["pretrained_model"]
        self.model_path = config["model_info"][(self.fold, self.iter)]["model_path"]
        self.analysis_result_dir_path = config["analysis_dir_path"]
        self.fig_dir_path = config["fig_dir_path"]
        
        self.hotspot_sgrnas = [
            "GAGTCCGAGCAGAAGAAGAANGG", "GGAATCCCTTCTGCAGCACCNGG", "GGGTGGGGGGAGTTTGCTCCNGG", "GGTGAGTGAGTGTGTGCGTGNGG",
            "GAACACAAAGCATAGACTGCNGG", "GGCACTGCGGCTGGAGGTGGNGG", "GGGAAAGACCCAGCATCCGTNGG",
            "GAGGGTTGCGTTCCTTGAGCNGG", "GAGTCCGAGCAGAAGAAGAANGG", "GCTAGAGTCACAAGTCCCACNGG", "GCTGCTGCTCTGGTTCCTCGNGG", "GGCACAGCGGCATCATTCCGNGG",
            "GGAGTGAGGGAAACGGCCCCNGG", "GGTGAGTGAGTGTGTGCGTGNGG",
            "GATAACTACACCGAGGAAATNGG", "GAGACCCTGCTCAAGGGCCGNGG", "GATTTCCTCCTCGACCACCANGG", "GCACGTGGCCCAGCCTGCTGNGG",
            "GACATTAAAGATAGTCATCTNGG", "GCATTTTCTTCACGGAAACANGG", "GGTACCTATCGATTGTCAGGNGG",
            "GTCACCAATCCTGTCCCTAGNGG", "GTGGTACTGGCCAGCAGCCGNGG", "GCTGCAGAAACAGCAAGCCCNGG", "GGAGAAGGTGGGGGGGTTCCNGG",
            "GATGCTATTCAGGATGCAGTNGG", "GCTGACCCCGCTGGGCAGGCNGG", "GGGATCAGGTGACCCATATTNGG", "GGGGCCACTAGGGACAGGATNGG", "GGGGGGTTCCAGGGCCTGTCNGG",
            "GCTGTGTTTGCGTCTCTCCCNGG", "GAAGCGTGATGACAAAGAGGNGG", "GGGGGTTCCAGGGCCTGTCTNGG", "GGTGACAAGTGTGATCACTTNGG"
        ]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_data(self) -> Dataset:
        # Load dataset
        if self.fold == "all":
            test_dataset = self.dataset_dict["all_dataset"]
        else:
            test_dataset = self.dataset_dict["test_dataset"]
        return test_dataset
    
    def sampling_data_for_attention_weights_analysis(self, test_dataset: Dataset) -> tuple:
        test_idx = self.dataset_dict["test_idx"]
        label_list = test_dataset['labels']
        mismatch_list = [self.dataset_dict["mismatch"][i] for i in test_idx]
        indices_0 = [i for i, label in enumerate(label_list) if label == 0]
        indices_1 = [i for i, (label, mm) in enumerate(zip(label_list, mismatch_list)) if label == 1]
        num_positive = len(indices_1)
        sampled_indices_0 = list(np.random.choice(indices_0, size=num_positive, replace=False))
        return (test_dataset.select(sampled_indices_0), test_dataset.select(indices_1))
    
    def sampling_data_for_ig_analysis(self, test_dataset: Dataset) -> Dataset:
        test_idx = self.dataset_dict["test_idx"]
        label_list = [self.dataset_dict["label"][i] for i in test_idx]
        mismatch_list = [self.dataset_dict["mismatch"][i] for i in test_idx] 
        sgrna_list = [self.dataset_dict["sgrna"][i] for i in test_idx] # len = included sgrna type
        sgrna_set_list = sorted(list(set(sgrna_list)))
        sgrna_set_list_ = []
        data_for_ig = {}
        for _sgrna in sgrna_set_list:
            ontarget_indices = [i for i, (sgrna, mm) in enumerate(zip(sgrna_list, mismatch_list)) if sgrna == _sgrna and mm == 0]
            offtarget_indices = [i for i, (sgrna, _, label) in enumerate(zip(sgrna_list, mismatch_list, label_list)) if sgrna == _sgrna and label == 1]
            sampled_indices = offtarget_indices
            if len(ontarget_indices) != 1 or len(offtarget_indices) <= 1:
                continue
            sgrna_set_list_.append(_sgrna)
            data_for_ig[_sgrna] = {
                "ontarget_index": ontarget_indices[0],
                "ontarget_dataset": test_dataset.select([ontarget_indices[0]]),
                "offtarget_indices": sampled_indices,
                "offtarget_dataset": test_dataset.select(sampled_indices)
            }
        return sgrna_set_list_, data_for_ig
        
    
    def get_attention_weights(self, model: nn.Module, dataloader: DataLoader) -> np.ndarray:
        attention_weights = [[] for _ in range(self.k_layer)]
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                outputs = model(inputs)
                attentions = outputs.attentions  # Tuple of (num_layers, batch_size, num_heads, seq_len, seq_len) = (12, B, 12, 47, 47)
                for i in range(self.k_layer, 12):
                    attention_weights[i - self.k_layer].append(attentions[i].cpu().numpy())
        # Concatenate attention weights from all batches
        all_attention_weights = [np.concatenate(layer_weights, axis=0) for layer_weights in attention_weights]
        all_attention_weights = np.array(all_attention_weights)  # Shape: (k_layer, total_samples, num_heads, seq_len, seq_len) = (6, N, 12, 47, 47)
        all_attention_weights = np.mean(all_attention_weights, axis=2)  # Shape: (k_layer, total_samples, seq_len, seq_len) = (6, N, 47, 47)
        all_attention_weights_mean = np.mean(all_attention_weights, axis=1)  # Shape: (k_layer, seq_len, seq_len) = (6, 47, 47)
        return all_attention_weights, all_attention_weights_mean
    
    def mann_hitney_u_test(self, attention_weights_0: np.ndarray, attention_weights_1: np.ndarray) -> np.ndarray:
        p_values = np.ones((self.k_layer, self.token_max_len, self.token_max_len))  # Initialize p-values array
        for layer in range(self.k_layer):
            for i in range(self.token_max_len):
                for j in range(self.token_max_len):
                    stat, p = mannwhitneyu(attention_weights_0[layer, :, i, j], attention_weights_1[layer, :, i, j], alternative='two-sided')
                    p_values[layer, i, j] = p
        p_values_flat = p_values.flatten()
        rejected, p_values_corrected, *_ = multipletests(p_values_flat, alpha=0.05, method='fdr_bh') # Benjamini-Hochberg correction
        p_values = p_values_corrected.reshape(self.k_layer, self.token_max_len, self.token_max_len)
        return p_values

    def attention_weights_analysis(self):
        # Load dataset
        test_dataset = self.load_data()
        label_0_dataset, label_1_dataset = self.sampling_data_for_attention_weights_analysis(test_dataset)
        test_dataloader_0 = DataLoader(label_0_dataset, batch_size=512, shuffle=False)
        test_dataloader_1 = DataLoader(label_1_dataset, batch_size=512, shuffle=False)
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist. Please run training first.")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path, 
            num_labels=2, 
            ignore_mismatched_sizes=True,
            output_attentions=True
        )
        trainable_diff = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model.to(self.device)
        model.eval()
        
        # Get attention weights
        attention_weights_0, attention_weight_mean_0 = self.get_attention_weights(model, test_dataloader_0)  # Shape: (k_layer, n_samples, seq_len, seq_len), (k_layer, seq_len, seq_len) = (6, 47, 47)
        attention_weights_1, attention_weight_mean_1 = self.get_attention_weights(model, test_dataloader_1)  # Shape: (k_layer, n_samples, seq_len, seq_len), (k_layer, seq_len, seq_len) = (6, 47, 47)
        attention_weights_diff = attention_weight_mean_1 - attention_weight_mean_0  # Shape: (k_layer, seq_len, seq_len) = (6, 47, 47)
        
        # Mann-Whitney U test
        u_test_result = self.mann_hitney_u_test(attention_weights_0, attention_weights_1)  # Shape: (k_layer, seq_len, seq_len) = (6, 47, 47)
        
        # Save analysis figures
        attention_diff_save_path = self.fig_dir_path + f"/attention_weights_diff_fold{self.fold}_iter{self.iter}.png"
        plot_bert_fig.plot_attention_weights(attention_weights_diff, self.max_pairseq_len, self.kmer, 1, mode="diff", save_path=attention_diff_save_path)
        u_test_save_path = self.fig_dir_path + f"/attention_weights_u_test_fold{self.fold}_iter{self.iter}.png"
        plot_bert_fig.plot_attention_weights(u_test_result, self.max_pairseq_len, self.kmer, 1, mode="pvalue", save_path=u_test_save_path)
    
    def compute_ig_token_importance(self, model, input_ids, attention_mask, baseline_token_ids, target_label=1):
        # Hagging face model automaticallly transforms input_ids to embeddings
        # Wrap forward function of embedding layer for Captum
        def forward_func(input_embeds, attention_mask):
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            logits = outputs.logits
            return logits[:, target_label]

        # Get embedding layer
        embedding = model.bert.embeddings.word_embeddings(input_ids.to(self.device))
        baseline_embeddings = model.bert.embeddings.word_embeddings(baseline_token_ids.to(self.device))
        
        ig = IntegratedGradients(forward_func)
        attributions, delta = ig.attribute(inputs=embedding,
                                           baselines=baseline_embeddings,
                                           additional_forward_args=(attention_mask.to(self.device),),
                                           return_convergence_delta=True)
        
        # Attrubution across tokens, sum over embedding dimension
        token_importance = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()  # (batch, seq_len)
        return token_importance, delta.cpu().numpy()
    
    def ig_loop(self, model, dataloader, baseline_token_ids):
        all_token_importance = []
        all_deltas = []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Integrated Gradient Analysis"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_importance, delta = self.compute_ig_token_importance(model, input_ids, attention_mask, baseline_token_ids, target_label=1)
            if token_importance.shape != (self.token_max_len,):
                all_token_importance.append(token_importance) # (batch, seq_len)
            else:
                all_token_importance.append(token_importance[np.newaxis, :]) # (1, seq_len)
            all_deltas.append(delta) # (batch,)
        if len(dataloader) == 1:
            all_token_importance = np.array(all_token_importance).squeeze(0) # (1, n_samples, seq_len) -> (num_samples, seq_len)
        else:
            all_token_importance = np.concatenate(all_token_importance, axis=0)  # (num_samples, seq_len)
        all_deltas = np.concatenate(all_deltas, axis=0)  # (num_samples,)
        return all_token_importance, all_deltas
    
    def u_test_ig_token_importance(self, token_importances: np.ndarray) -> float:
        # token_importances: (num_samples, seq_len)
        dna_token_importance = token_importances[:, self.max_pairseq_len - self.kmer + 2:-1] # (num_samples, seq_len_dna) = (num_samples, 22)
        hotspot_indices = [4, 5, 6, 14, 15, 16, 17]
        other_indices = [i for i in range(dna_token_importance.shape[1]) if i not in hotspot_indices]
        
        observed_hotspot_mean = np.mean(dna_token_importance[:, hotspot_indices])
        observed_other_mean = np.mean(dna_token_importance[:, other_indices])
        observed_effect_size = observed_hotspot_mean - observed_other_mean
        
        # Random comparisons
        random_effet_sizes = []
        all_indices = list(range(dna_token_importance.shape[1]))
        for _ in range(1000):
            np.random.shuffle(all_indices)
            random_hotspot_indices = all_indices[:len(hotspot_indices)]
            random_other_indices = all_indices[len(hotspot_indices):]
            # Randomly selected hotspot vs. other
            random_hotspot_mean = np.mean(dna_token_importance[:, random_hotspot_indices])
            random_other_mean = np.mean(dna_token_importance[:, random_other_indices])
            random_effect_size = random_hotspot_mean - random_other_mean
            random_effet_sizes.append(random_effect_size)
        # Experienced p-value
        k = np.sum(np.array(random_effet_sizes) >= observed_effect_size)
        N = len(random_effet_sizes)
        empirical_p_value = (k + 1) / (N + 1)  # Add one to numerator and denominator to avoid p-value of zero
        return float(empirical_p_value)
    
    def utest_(self, token_importances: np.ndarray) -> float:
        # token_importances: (num_samples, seq_len)
        dna_token_importance = token_importances[:, self.max_pairseq_len - self.kmer + 2:-1] # (num_samples, seq_len_dna) = (num_samples, 22)
        hotspot_indices = [0, 1, 2, 3, 4, 5, 6]
        other_indices = [i for i in range(dna_token_importance.shape[1]) if i not in hotspot_indices]
        
        observed_hotspot_mean = np.mean(dna_token_importance[:, hotspot_indices])
        observed_other_mean = np.mean(dna_token_importance[:, other_indices])
        observed_effect_size = observed_hotspot_mean - observed_other_mean
        
        # Random comparisons
        random_effet_sizes = []
        all_indices = list(range(dna_token_importance.shape[1]))
        for _ in range(1000):
            np.random.shuffle(all_indices)
            random_hotspot_indices = all_indices[:len(hotspot_indices)]
            random_other_indices = all_indices[len(hotspot_indices):]
            # Randomly selected hotspot vs. other
            random_hotspot_mean = np.mean(dna_token_importance[:, random_hotspot_indices])
            random_other_mean = np.mean(dna_token_importance[:, random_other_indices])
            random_effect_size = random_hotspot_mean - random_other_mean
            random_effet_sizes.append(random_effect_size)
        # Experienced p-value
        k = np.sum(np.array(random_effet_sizes) >= observed_effect_size)
        N = len(random_effet_sizes)
        empirical_p_value = (k + 1) / (N + 1)  # Add one to numerator and denominator to avoid p-value of zero
        print(f"U-test p-value: {empirical_p_value}")
        
    def integrated_gradient_analysis(self):
        # Load dataset
        test_dataset = self.load_data()
        sgrna_set, data_for_ig = self.sampling_data_for_ig_analysis(test_dataset)

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} does not exist. Please run training first.")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path, 
            num_labels=2, 
            ignore_mismatched_sizes=True,
            output_attentions=True
        )
        trainable_diff = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(trainable_diff, strict=False)
        model.to(self.device)
        model.eval()
        
        # IG analysis
        all_token_importance = {}
        all_token_importance_samples = []
        hot_spot_sgrnas = []
        for _sgrna in sgrna_set:
            print(f"IG analysis for sgrna: {_sgrna}")
            ontarget_dataset = data_for_ig[_sgrna]["ontarget_dataset"]
            offtarget_dataset = data_for_ig[_sgrna]["offtarget_dataset"]
            # baseline_token_ids = torch.tensor(ontarget_dataset["input_ids"])
            baseline_token_ids = torch.full_like(torch.tensor(ontarget_dataset["input_ids"]), fill_value=0)  # Use [PAD] token as baseline
            data_loader_ = DataLoader(offtarget_dataset, batch_size=8, shuffle=False)
            token_importance, deltas = self.ig_loop(model, data_loader_, baseline_token_ids)  # (num_samples, seq_len), (num_samples,)
            if token_importance.shape[0] != 1:
                token_importance_mean = np.mean(token_importance, axis=0)  # (seq_len,)
                all_token_importance_samples.append(token_importance)  # (num_samples, seq_len)
            else:
                token_importance_mean = token_importance
                all_token_importance_samples.append(token_importance[np.newaxis, :])  # (1, seq_len)
            all_token_importance[_sgrna] = token_importance_mean  # (seq_len,)
            
            token_importance_dna_ = token_importance_mean[self.max_pairseq_len - self.kmer + 1 + 2: -1]
            max_idx = np.argmax(token_importance_dna_)
            if max_idx in [3, 4, 5, 13, 14, 15, 16]:
                hot_spot_sgrnas.append(_sgrna)
            
        all_token_importance_samples = np.concatenate(all_token_importance_samples, axis=0)  # (num_samples, seq_len)
        importance_p_value = self.u_test_ig_token_importance(all_token_importance_samples)
        self.utest_(all_token_importance_samples)
        save_path = self.analysis_result_dir_path + f"/ig_token_importance_fold{self.fold}_iter{self.iter}.npy"
        np.save(save_path, all_token_importance) # (num_samples, seq_len)
        save_path = self.analysis_result_dir_path + f"/ig_token_importance_u_test_pvalue_fold{self.fold}_iter{self.iter}.txt"
        with open(save_path, 'w') as f:
            f.write(str(importance_p_value))
        
        
        
        # Plot
        save_path = self.fig_dir_path + f"/ig_token_importance_fold{self.fold}_iter{self.iter}.png"
        plot_bert_fig.plot_token_importance(all_token_importance, sgrna_set, self.max_pairseq_len, self.kmer, hot_spot_sgrnas, save_path=save_path)
        
        
      
        