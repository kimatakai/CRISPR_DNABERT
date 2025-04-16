


import os
import random
from itertools import product
import tqdm
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


import config
import utilities_module
from sklearn.metrics import confusion_matrix



class CustomTrainer(Trainer):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # num_items_in_batch=Noneを追加（25/02/15）
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.weight is not None:
            # Ensure weights are on the same device as logits
            weight = self.weight
            if self.args.fp16:
                weight = weight.half()
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class CrisprDnaBertClass:
    def __init__(self, dataset_df: pd.DataFrame, train_test_info: dict, fold: int, datatype: str, dataset_dict: dict, exp_id: int=0, no_pretrained: bool=False):
        self.dataset_df = dataset_df
        self.train_test_info = train_test_info
        self.fold = fold
        self.datatype = datatype
        self.exp_id = exp_id
        self.seed = exp_id + config.random_state
        self.train_dataset = dataset_dict["train_dataset"]
        self.test_dataset = dataset_dict["test_dataset"]
        if no_pretrained:
            self.dnabert_pair_finetuned_path = config.dnabert_pair_finetuned_no_pretrain_path
            self.dnabert_crispr_finetuned_path = utilities_module.return_model_weight_path("dnabert-no-pretrain", self.datatype, self.fold, self.exp_id)
            self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_no_pretrain/"
            self.probabilitiy_array_path = utilities_module.return_output_probability_path("dnabert-no-pretrain", self.datatype, self.fold, self.exp_id)
        else:
            self.dnabert_pair_finetuned_path = config.dnabert_pair_finetuned_path
            self.dnabert_crispr_finetuned_path = utilities_module.return_model_weight_path("dnabert", self.datatype, self.fold, self.exp_id)
            self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/dnabert/"
            self.probabilitiy_array_path = utilities_module.return_output_probability_path("dnabert", self.datatype, self.fold, self.exp_id)
        
        if self.datatype == "transfer":
            if no_pretrained:
                self.transfer_pretrained_model_path = utilities_module.return_model_weight_path("dnabert-no-pretrain", "changeseq", self.fold, self.exp_id)
            else:
                self.transfer_pretrained_model_path = utilities_module.return_model_weight_path("dnabert", "changeseq", self.fold, self.exp_id)
        else:
            self.transfer_pretrained_model_path = None
        
        os.makedirs(self.dnabert_crispr_finetuned_path, exist_ok=True)
        os.makedirs(self.predicted_probabilities_path, exist_ok=True)
        
        self.kmer = config.kmer
        self.token_max_length = 2*(24 - self.kmer + 1) + 3
        if self.datatype == "changeseq":
            self.epochs = 5
            self.learning_rate = 2e-5
        elif self.datatype == "guideseq":
            self.epochs = 5
            self.learning_rate = 2e-5
        elif self.datatype == "transfer":
            self.epochs = 2
            self.learning_rate = 2e-5
        self.batch_size = 256
        self.logging_step = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def add_tokens(self):
        # create tokens (3-mer)
        bases = 'ATGC'
        tokens = []
        for i in range(config.kmer):
            for combination in product(bases, repeat=config.kmer - 1):
                token = ''.join(combination[:i] + ('-',) + combination[i:])
                tokens.append(token)
        return tokens
    
    def downsampling_dataset(self, sampling_rate: float):
        # Split label 0 and label 1
        label_0_indices = [i for i, label in enumerate(self.train_dataset_tokenized["label"]) if label == 0]
        label_1_indices = [i for i, label in enumerate(self.train_dataset_tokenized["label"]) if label == 1]
        # Count negative samples
        num_negative_samples = len([i for i, label in enumerate(self.train_dataset["label"]) if label == 0])
        sampled_label_0_indices = random.sample(label_0_indices, int(num_negative_samples * sampling_rate))
        final_indices = sampled_label_0_indices + label_1_indices
        self.train_dataset_temp_indices = final_indices
        # temp Dataset for each epoch training 
        self.train_dataset_temp = self.train_dataset_tokenized.select(final_indices)
        # Update `train_dataset` by excluding the unsampled label 0 data
        unsampled_label_0_indices = list(set(label_0_indices) - set(sampled_label_0_indices))
        remaining_indices = label_1_indices + unsampled_label_0_indices  # Keep sampled label 1 and unsampled label 0
        # Filter out the remaining data in train_dataset (this will exclude the unsampled label 0 entries)
        self.train_dataset_tokenized = self.train_dataset_tokenized.select(remaining_indices)
    
    
    # compute metrics for classification
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions[0].argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    
    
    def train_classification_task(self):
        print(f"[TRAIN] DNABERT model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load tokenizer
        if self.datatype == "transfer":
            self.tokenizer = AutoTokenizer.from_pretrained(self.transfer_pretrained_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.dnabert_pair_finetuned_path)

        # definition func for tokenizer
        def tokenize_function(examples):
            return self.tokenizer(examples['target_dna'], examples['sgrna'], padding='max_length', truncation=True, max_length=self.token_max_length)

        # Tokenize train input sequences
        self.train_dataset_tokenized = self.train_dataset.map(tokenize_function, batched=True)
        
        # Training argumants
        training_args = TrainingArguments(
            output_dir=self.dnabert_crispr_finetuned_path,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            # gradient_accumulation_steps=1,
            lr_scheduler_type="cosine_with_restarts",
            disable_tqdm=False,
            logging_steps=self.logging_step,
            push_to_hub=False,
            log_level="error",
            save_strategy="no",  # This disables checkpoint saving
            fp16=True,
        )
        
        # Load pretrained model
        if self.datatype == "transfer":
            model_path = self.transfer_pretrained_model_path
        else:
            model_path = self.dnabert_pair_finetuned_path
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        ).to(self.device)
        
        # Undersampling negative samples if dataset is GUIDE-seq
        self.downsampling_dataset(sampling_rate=0.2)
            
        # Shuffle Dataset
        self.train_dataset_temp.shuffle(seed=self.seed)
            
        # Trainer setting
        trainer = CustomTrainer(
            model = model,
            tokenizer = self.tokenizer,
            args = training_args,
            compute_metrics = self.compute_metrics,
            train_dataset = self.train_dataset_temp,
        )
        
        # Fine-tuning
        trainer.train()
        
        # Save fine-tuned model
        self.tokenizer.save_pretrained(self.dnabert_crispr_finetuned_path)
        model.save_pretrained(self.dnabert_crispr_finetuned_path)
        
        
    def test_classification_task(self):
        print(f"[TEST] DNABERT model test. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.dnabert_crispr_finetuned_path)
        # definition func for tokenizer
        def tokenize_function(examples):
            return self.tokenizer(examples['target_dna'], examples['sgrna'], padding='max_length', truncation=True, max_length=self.token_max_length)
        
        # load fine-tuned model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.dnabert_crispr_finetuned_path,
            num_labels=2,
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        ).to(self.device)
        model.eval()
        
        # Extract True label
        true_label = self.test_dataset["label"]
        true_label_np = torch.IntTensor(true_label).cpu().numpy()
        
        if not os.path.exists(self.probabilitiy_array_path):
        
            # Tokenize train input sequences
            test_dataset_tokenized = self.test_dataset.map(tokenize_function, batched=True)
            # Convert Dataset to torch type
            try:
                del self.train_dataset
            except:
                pass
            try:
                del self.train_dataset_tokenized
            except:
                pass
            try:
                del self.train_dataset_temp
            except:
                pass
            datasets = TensorDataset(
                torch.tensor(test_dataset_tokenized['input_ids']),
                torch.tensor(test_dataset_tokenized['attention_mask']),
                torch.tensor(test_dataset_tokenized['token_type_ids']),
                torch.tensor(test_dataset_tokenized['label'])
            )
            data_loader = DataLoader(datasets, batch_size=512)
            # prediction
            all_logits = []
            with torch.no_grad():
                for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc="Get Logits"):
                    input_ids, attention_mask, token_type_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    # state = outputs.hidden_states[-1][:, 0, :]
                    logits = outputs.logits
                    all_logits.append(logits)
                    # all_embedding.append(state)
            all_logits = torch.cat(all_logits, dim=0)
            # logits -> prob
            probabilities = torch.softmax(all_logits, dim=1)
            probabilities = probabilities.cpu().numpy()
            # Save probabilities
            probabilities = probabilities.astype(np.float32)
            np.save(self.probabilitiy_array_path, probabilities)
        
        # Load 
        probabilities = np.load(self.probabilitiy_array_path)
        
        return (true_label_np, probabilities)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, epigenetic_input, labels):
        self.tokenized_dataset = tokenized_dataset
        self.input_tokenized = torch.tensor(tokenized_dataset['input_ids'], dtype=torch.long)
        self.attention_mask = torch.tensor(tokenized_dataset['attention_mask'], dtype=torch.long)
        self.token_type_ids = torch.tensor(tokenized_dataset['token_type_ids'], dtype=torch.long)
        self.epigenetic_input = torch.tensor(epigenetic_input, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        input_tokenized = self.input_tokenized[idx]
        attention_mask = self.attention_mask[idx]
        epigenetic_input = self.epigenetic_input[idx]
        token_type_ids = self.token_type_ids[idx]
        label = self.labels[idx]
        sample = {
            'input_ids': input_tokenized,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'epigenetic_input': epigenetic_input,
            'label': label
        }
        return sample

    
# BERT + Epigenetic class
class DnaBertEpigeneticClass(nn.Module):
    def __init__(self, bert_model, epigenetic_dim, ablation: bool=False):
        super(DnaBertEpigeneticClass, self).__init__()
        self.bert_model = bert_model
        self.epigenetic_dim = epigenetic_dim
        self.ablation = ablation
        
        self.classifier = nn.Sequential(
            nn.Linear(32*3 if ablation else 32*4, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
        
        self.bert_fc1 = nn.Linear(768, 32)
        self.bert_dropout1 = nn.Dropout(0.1)
        
        self.epi_fc1 = nn.Linear(epigenetic_dim, 32)
        self.epi_dropout1 = nn.Dropout(0.1)
        
        self.activation = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, token_type_ids, epigenetic_input, softmax_apply=False):
        # BERT embedding
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = bert_output.hidden_states[-1][:, 0, :] # CLS token embedding
        max_embedding = torch.max(bert_output.hidden_states[-1], dim=1)[0] # Max pooling
        mean_embedding = torch.mean(bert_output.hidden_states[-1], dim=1) # Mean pooling
        
        cls_x = self.bert_fc1(cls_embedding)
        cls_x = self.activation(cls_x)
        cls_x = self.bert_dropout1(cls_x)
        
        max_x = self.bert_fc1(max_embedding)
        max_x = self.activation(max_x)
        max_x = self.bert_dropout1(max_x)
        
        mean_x = self.bert_fc1(mean_embedding)
        mean_x = self.activation(mean_x)
        mean_x = self.bert_dropout1(mean_x)
        
        if not self.ablation:
            # Epigenetic input
            epi_x = self.epi_fc1(epigenetic_input)
            epi_x = self.activation(epi_x)
            epi_x = self.epi_dropout1(epi_x)
            # Concatenate BERT embedding and epigenetic input
            x = torch.cat([cls_x, max_x, mean_x, epi_x], dim=1)
        else:
            # Concatenate BERT embedding
            x = torch.cat([cls_x, max_x, mean_x], dim=1)

        x = self.classifier(x)
        if softmax_apply:
            x = torch.softmax(x, dim=1)
        return x

class CrisprDnaBertEpigeneticClass(CrisprDnaBertClass):
    
    def init(self, ablation: bool=False):
        # Hyperparameters
        self.prob_thres_train = 0.05
        self.prob_thres_test = 0.05
        self.epochs = 5
        self.batch_size = 256
        self.learning_rate = 2e-5
        self.n_estimators = 5
        
        # Prepare path information
        if ablation:
            self.dnabert_epi_model_weight_path = utilities_module.return_model_weight_path("dnabert-epi-ablation", self.datatype, self.fold, self.exp_id)
            os.makedirs(config.dnabert_epigenetic_model_path, exist_ok=True)
            os.makedirs(self.dnabert_epi_model_weight_path, exist_ok=True)
            self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_epi_ablation/"
            os.makedirs(self.predicted_probabilities_path, exist_ok=True)
            self.probabilitiy_array_path_ = utilities_module.return_output_probability_path("dnabert-epi-ablation", self.datatype, self.fold, self.exp_id)
        else:
            self.dnabert_epi_model_weight_path = utilities_module.return_model_weight_path("dnabert-epi", self.datatype, self.fold, self.exp_id)
            os.makedirs(config.dnabert_epigenetic_model_path, exist_ok=True)
            os.makedirs(self.dnabert_epi_model_weight_path, exist_ok=True)
            self.predicted_probabilities_path = f"{config.probabilities_base_dir_path}/dnabert_epi/"
            os.makedirs(self.predicted_probabilities_path, exist_ok=True)
            self.probabilitiy_array_path_ = utilities_module.return_output_probability_path("dnabert-epi", self.datatype, self.fold, self.exp_id)
    
    def return_positive_predicted_dataset(self):
        # Input train data to model
        os.makedirs(config.probabilities_base_dir_path + "/dnabert_train/", exist_ok=True)
        self.train_probability_array_path =  config.probabilities_base_dir_path + f"/dnabert_train/probabilities_{self.datatype}_fold{self.fold}_exp{self.exp_id}.npy"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.dnabert_crispr_finetuned_path)
        # Definition func for tokenizer
        def tokenize_function(examples):
            return self.tokenizer(examples['target_dna'], examples['sgrna'], padding='max_length', truncation=True, max_length=self.token_max_length)
        # Load fine-tuned model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.dnabert_crispr_finetuned_path,
            num_labels=2,
            output_hidden_states=True,
            ignore_mismatched_sizes=True).to(self.device)
        model.eval()
        
        # Train dataset tokenized
        self.train_dataset_tokenized = self.train_dataset.map(tokenize_function, batched=True)
        # Downsampling negative samples
        self.downsampling_dataset(sampling_rate=0.2) # -> self.train_dataset_temp
        
        # Extract True label
        true_label = self.train_dataset_temp["label"]
        true_label_np = torch.IntTensor(true_label).cpu().numpy()
        
        # Load probabilities
        if not os.path.exists(self.train_probability_array_path):
            # Get model output
            datasets = TensorDataset(
                torch.tensor(self.train_dataset_temp['input_ids']),
                torch.tensor(self.train_dataset_temp['attention_mask']),
                torch.tensor(self.train_dataset_temp['token_type_ids']),
                torch.tensor(self.train_dataset_temp['label'])
            )
            data_loader = DataLoader(datasets, batch_size=512)
            # Prediction
            all_logits = []
            with torch.no_grad():
                for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
                    input_ids, attention_mask, token_type_ids, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    all_logits.append(logits)
            all_logits = torch.cat(all_logits, dim=0)
            # Logits -> prob
            probabilities = torch.softmax(all_logits, dim=1)
            probabilities = probabilities.cpu().numpy()
            # Save probabilities
            probabilities = probabilities.astype(np.float32)
            np.save(self.train_probability_array_path, probabilities)
        else:
            probabilities = np.load(self.train_probability_array_path)
        
        return (true_label_np, probabilities)
    
    def print_confusion_matrix(self, true_label, predicted_labels):
        # Compute confusion matrix
        cm = confusion_matrix(true_label, predicted_labels)
        print("Confusion Matrix:")
        print(cm)
    
    def extract_positive_predicted_dataset_train(self, train_true_label: np.array, train_predicted_labels: np.array, epigenetic_input_dict: dict, random_state: int=42) -> tuple:
        # Positive predicted index
        train_positive_predicted_index = [i for i, label in enumerate(train_predicted_labels) if label == 1]
        # True positive index
        train_positive_predicted_true_label = train_true_label[train_positive_predicted_index]
        true_positive_index = [i for i, label in enumerate(train_positive_predicted_true_label) if label == 1]
        # False positive index
        false_positive_index = [i for i, label in enumerate(train_positive_predicted_true_label) if label == 0]
        
        # For tokenized dataset
        true_positive_dataset = self.train_dataset_temp.select(train_positive_predicted_index).select(true_positive_index)
        false_positive_dataset = self.train_dataset_temp.select(train_positive_predicted_index).select(false_positive_index)
        train_positive_predicted_dataset = Dataset.from_dict({
            "input_ids": true_positive_dataset["input_ids"] + false_positive_dataset["input_ids"],
            "attention_mask": true_positive_dataset["attention_mask"] + false_positive_dataset["attention_mask"],
            "token_type_ids": true_positive_dataset["token_type_ids"] + false_positive_dataset["token_type_ids"],})
        
        print(train_positive_predicted_dataset["input_ids"][0])
        print(train_positive_predicted_dataset["attention_mask"][0])
        print(train_positive_predicted_dataset["token_type_ids"][0])
        
        # For epigenetic input
        train_epigenetic_input = epigenetic_input_dict["train_epigenetic_input"]
        train_epigenetic_input = train_epigenetic_input[self.train_dataset_temp_indices]
        train_epigenetic_input_positive_predicted = train_epigenetic_input[train_positive_predicted_index]
        true_positive_epigenetic_input = train_epigenetic_input_positive_predicted[true_positive_index]
        false_positive_epigenetic_input = train_epigenetic_input_positive_predicted[false_positive_index]
        print(np.mean(true_positive_epigenetic_input[:20]), np.mean(true_positive_epigenetic_input[:41]), np.mean(true_positive_epigenetic_input[:62]))
        print(np.mean(false_positive_epigenetic_input[:20]), np.mean(false_positive_epigenetic_input[:41]), np.mean(false_positive_epigenetic_input[:62]))
        train_positive_predicted_epigenetic_input = np.concatenate([true_positive_epigenetic_input, false_positive_epigenetic_input], axis=0)
        
        # For labels
        train_positive_predicted_label = np.array([1] * len(true_positive_index) + [0] * len(false_positive_index))
        print(len(true_positive_index), len(false_positive_index))
        
        return (train_positive_predicted_dataset, train_positive_predicted_epigenetic_input, train_positive_predicted_label)
    
    def extract_positive_predicted_dataset_test(self, test_true_label: np.array, test_predicted_labels: np.array, epigenetic_input_dict: dict) -> tuple:
        # Positive predicted index
        test_positive_predicted_index = [i for i, label in enumerate(test_predicted_labels) if label == 1]
        
        # For tokenized dataset
        test_positive_predicted_dataset = self.test_dataset.select(test_positive_predicted_index)
        # tokenize
        test_positive_predicted_dataset = self.tokenizer(
            test_positive_predicted_dataset['target_dna'], 
            test_positive_predicted_dataset['sgrna'], 
            padding='max_length', truncation=True, max_length=self.token_max_length)
        
        # For epigenetic input
        test_epigenetic_input = epigenetic_input_dict["test_epigenetic_input"]
        test_positive_predicted_epigenetic_input = test_epigenetic_input[test_positive_predicted_index]
        
        # For labels
        test_positive_predicted_label = test_true_label[test_positive_predicted_index]
        
        return (test_positive_predicted_dataset, test_positive_predicted_epigenetic_input, test_positive_predicted_label)
        
        
    def train_classification_task(self, epigenetic_input_dict: dict, ablation_mode: bool=False):
        
        print(f"[TRAIN] DNABERT-Epi model training. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        if ablation_mode:
            print("[ABALTION] DNABERT-Epi model training. Epigenetic input will not be used.")
        
        # Prepare
        self.init(ablation=ablation_mode)
        
        # Get train predict from fine-tuned DNABERT model
        train_true_label, train_predicted_probabilities = self.return_positive_predicted_dataset()
        
        # Print confusion matrix
        print(f"Confusion matrix for training dataset.")
        train_predicted_labels_ = np.argmax(train_predicted_probabilities, axis=1)
        self.print_confusion_matrix(train_true_label, train_predicted_labels_)
        train_predicted_labels = (train_predicted_probabilities[:, 1] >= self.prob_thres_train).astype(int)
        self.print_confusion_matrix(train_true_label, train_predicted_labels)

        # Extract positive predicted dataset
        train_positive_predicted_dataset, train_positive_predicted_epigenetic_input, train_positive_predicted_label \
            = self.extract_positive_predicted_dataset_train(train_true_label, train_predicted_labels, epigenetic_input_dict)
        # Dataset and DataLoader
        train_dataset = CustomDataset(
            tokenized_dataset=train_positive_predicted_dataset,
            epigenetic_input=train_positive_predicted_epigenetic_input,
            labels=train_positive_predicted_label)
        
        # Training multiple models for ensemble
        for i in range(self.n_estimators):
            # Random seed setting
            torch.manual_seed(self.exp_id*10+i)
            torch.cuda.manual_seed(self.exp_id*10+i)
            torch.cuda.manual_seed_all(self.exp_id*10+i)
            random.seed(self.exp_id*10+i)
            np.random.seed(self.exp_id*10+i)
            
            print(f"Training {i+1}/{self.n_estimators} model.")
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
            # Load fine-tuned DNABERT model
            bert_model = AutoModelForSequenceClassification.from_pretrained(
                self.dnabert_crispr_finetuned_path,
                num_labels=2,
                output_hidden_states=True,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                # ignore_mismatched_sizes=True
                ).to(self.device)
        
            # Load DNABERT with epigenetic model
            model = DnaBertEpigeneticClass(bert_model, train_positive_predicted_epigenetic_input.shape[1], ablation=ablation_mode).to(self.device)
            model.train()
        
            # Model setting
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Training
            for epoch in range(self.epochs):
                model.train()
                total_loss = 0
                for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    epigenetic_input = batch["epigenetic_input"].to(self.device)
                    labels = batch["label"].long().to(self.device)

                    optimizer.zero_grad()
                    
                    outputs = model(input_ids, attention_mask, token_type_ids, epigenetic_input)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(train_dataloader)}")

            # Early stopping and learning rate scheduler
            torch.save(model.state_dict(), self.dnabert_epi_model_weight_path + f"/model_{i}.pt")
            
            # Release memory
            del model
            torch.cuda.empty_cache()
        
    
    def test_classification_task(self, epigenetic_input_dict: dict, ablation_mode: bool=False):
        
        print(f"[TEST] DNABERT-Epi model test. FOLD: {self.fold}. DATATYPE: {self.datatype}. EXPERIMENTS: {self.exp_id}. {self.device} will be used.")
        if ablation_mode:
            print("[ABALTION] DNABERT-Epi model test. Epigenetic input will not be used.")
        
        # Prepare
        self.init(ablation=ablation_mode)
        
        # Get test predict from fine-tuned DNABERT model
        test_true_label_np, test_predicted_probabilities = super().test_classification_task()
        
        # if not os.path.exists(self.probabilitiy_array_path_):
        if True:
            # Print confusion matrix
            print(f"Confusion matrix for test dataset.")
            # Get predicted labels
            test_predicted_labels_ = np.argmax(test_predicted_probabilities, axis=1)
            self.print_confusion_matrix(test_true_label_np, test_predicted_labels_)
            test_predicted_labels = (test_predicted_probabilities[:, 1] >= self.prob_thres_test).astype(int)
            # self.print_confusion_matrix(test_true_label_np, test_predicted_labels)

            # Extract positive predicted dataset
            test_positive_predicted_dataset, test_positive_predicted_epigenetic_input, test_positive_predicted_label = self.extract_positive_predicted_dataset_test(
                test_true_label_np, test_predicted_labels, epigenetic_input_dict)

            # Dataset and DataLoader
            dataset = CustomDataset(
                tokenized_dataset=test_positive_predicted_dataset,
                epigenetic_input=test_positive_predicted_epigenetic_input,
                labels=test_positive_predicted_label)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            # Bagging probability
            bagging_probabilities = np.zeros((len(test_positive_predicted_label), 2))

            # Ensemble prediction
            for i in tqdm.tqdm(range(self.n_estimators), total=self.n_estimators):
                # Load fine-tuned DNABERT model
                bert_model = AutoModelForSequenceClassification.from_pretrained(
                    self.dnabert_crispr_finetuned_path,
                    num_labels=2,
                    output_hidden_states=True,
                    ignore_mismatched_sizes=True
                    ).to(self.device)

                # Load DNABERT with epigenetic model
                model = DnaBertEpigeneticClass(bert_model, test_positive_predicted_epigenetic_input.shape[1], ablation=ablation_mode).to(self.device)
                model.load_state_dict(torch.load(self.dnabert_epi_model_weight_path + f"/model_{i}.pt"))
                model.eval()

                # Prediction
                test_predicted_probabilities_ = []
                with torch.no_grad():
                    for batch in data_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        token_type_ids = batch["token_type_ids"].to(self.device)
                        epigenetic_input = batch["epigenetic_input"].to(self.device)
                        outputs = model(input_ids, attention_mask, token_type_ids, epigenetic_input, softmax_apply=True)
                        test_predicted_probabilities_.append(outputs.cpu().numpy())
                test_predicted_probabilities_ = np.concatenate(test_predicted_probabilities_, axis=0)

                # Bagging
                bagging_probabilities += test_predicted_probabilities_

            # Average ensemble
            test_predicted_probabilities__ = bagging_probabilities / self.n_estimators
            print(len(test_predicted_probabilities__))
            print(test_predicted_probabilities__[:10])
            
            # Majority voting
            test_positive_predicted_index = [i for i, label in enumerate(test_predicted_labels) if label == 1]
            test_predicted_probabilities[test_positive_predicted_index, :] = test_predicted_probabilities__*0.5 + test_predicted_probabilities[test_positive_predicted_index, :]*0.5
            # test_predicted_probabilities[test_positive_predicted_index, :] = test_predicted_probabilities__

            # Print confusion matrix
            self.print_confusion_matrix(test_true_label_np, np.argmax(test_predicted_probabilities, axis=1))

            # Save probabilities
            np.save(self.probabilitiy_array_path_, test_predicted_probabilities)
        else:
            test_predicted_probabilities = np.load(self.probabilitiy_array_path_)
            self.print_confusion_matrix(test_true_label_np, np.argmax(test_predicted_probabilities, axis=1))
        
        return (test_true_label_np, test_predicted_probabilities)
