
import os
import sys
import yaml
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run Deep Machine Learning.")
parser.add_argument("--config_path", "-cp", type=str, required=False, default="config.yaml", help="Path to the configuration YAML file.")
parser.add_argument('--pretrain', action='store_true', help='Pretrain the model')
args = parser.parse_args()

# Load the configuration file
config = load_yaml(args.config_path)

# Set working directory
working_dir = config["paths"]["working_dir"]
os.chdir(working_dir)
print(f"{os.getcwd()}/src/models/pair_finetuning_dnabert.py is running.")
src_dir = working_dir + "/src/"
sys.path.append(src_dir)
database_dir = config["paths"]["database_dir"]


def set_path(config: dict) -> dict:
    # Path for fine-tuned model
    dnabert_base_path = database_dir + config["paths"]["model"]["DNABERT"]["base_dir"]
    pretrained_model_path = dnabert_base_path + config["paths"]["model"]["DNABERT"]["pretrained"]
    pair_finetuned_path = dnabert_base_path + config["paths"]["model"]["DNABERT"]["pair_finetuned"]
    pair_finetuned_no_pretrain_path = dnabert_base_path + config["paths"]["model"]["DNABERT"]["pair_finetuned_no_pretrain"]
    config["pretrained_model_path"] = pretrained_model_path
    config["pair_finetuned_path"] = pair_finetuned_path
    config["pair_finetuned_no_pretrain_path"] = pair_finetuned_no_pretrain_path
    os.makedirs(pair_finetuned_path, exist_ok=True)
    os.makedirs(pair_finetuned_no_pretrain_path, exist_ok=True)
    return config
config = set_path(config)

# Import necessary modules
import random
import tqdm
import multiprocessing
from multiprocessing import Pool
from itertools import product
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, BertConfig, TrainerCallback, TrainingArguments
import matplotlib.pyplot as plt

import models.data_loader as data_loader
import utils.sequence_module as sequence_module

# Fix random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Set parameters
def set_params(config: dict) -> dict:
    k_mer = 3
    max_length = 2*(24 - k_mer + 1) + 3
    config["k_mer"] = k_mer
    config["max_length"] = max_length
    return config
config = set_params(config)

# Define the function
def plot_loss_curve(config: dict) -> None:
    loss_array_pretrained = np.load(config["pair_finetuned_path"] + '/loss.npy')
    loss_array_pretrained = np.log(loss_array_pretrained + 1)
    loss_array_no_pretrained = np.load(config["pair_finetuned_no_pretrain_path"] + '/loss.npy')
    loss_array_no_pretrained = np.log(loss_array_no_pretrained + 1)
    epochs = np.linspace(0, 5, len(loss_array_pretrained))
    
    # Plotting
    plt.plot(epochs, loss_array_pretrained, label='Pre-trained DNABERT', color='blue')
    plt.plot(epochs, loss_array_no_pretrained, label='DNABERT from scratch', color='orange')
    plt.yscale('log')
    plt.ylabel('Loss (log scale)')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 5)

    os.makedirs(f'{config["pair_finetuned_path"]}/loss_curve', exist_ok=True)
    plt.savefig(f'{config["pair_finetuned_path"]}/loss_curve/loss_curve.png', dpi=300)


# create tokens (3-mer)
def create_tokens(config: dict) -> list:
    k_mer = config["k_mer"]
    bases = 'ATGC'
    tokens = []
    for i in range(k_mer):
        for combination in product(bases, repeat=k_mer - 1):
            token = ''.join(combination[:i] + ('-',) + combination[i:])
            tokens.append(token)
    return tokens


def return_mismatch_positions(seq1, seq2):
    if len(seq1) != 24:
        raise ValueError("The sequences should be 24 bases long including '-' for bulges.")
    if len(seq1) != len(seq2):
        raise ValueError("The sequences must be of the same length.")
    mismatch_list = []
    for base1, base2 in zip(seq1, seq2):
        if base1 == '-' and base2 == '-':
            mismatch_list.append(0)
        elif base1 == '-' or base2 == '-':
            mismatch_list.append(1)
        else:
            mismatch_list.append(0 if base1 == base2 else 1)
    if seq1[0] == '-' and seq2[0] == '-':
        mismatch_list.append(0) # Bulge
    else:
        mismatch_list.append(1)
    return mismatch_list


def seq_to_kmer(sequence, kmer=config["k_mer"]):
    kmers = [sequence[i:i+kmer] for i in range(len(sequence)-kmer+1)]
    merged_sequence = ' '.join(kmers)
    return merged_sequence


class PrepareDatasetClass:
    def __init__(self, config: dict):
        self.config = config
        self.config["fold"] = "all"
        self.token_max_len = config["max_length"]
        self.dataset_names = [
            "Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS",
            "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2"
        ]
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_path"])
        self.bases = 'ATGC'
        self.max_seq_len = config["parameters"]["max_pairseq_len"]
        self.tokens = create_tokens(config)
        for token in self.tokens:
            self.tokenizer.add_tokens([token], special_tokens=True)
    
    @staticmethod
    def _process_alignment_hyphen(args) -> tuple:
        seq_rna, seq_dna = args
        padded_seq_rna, padded_seq_dna = sequence_module.padding_hyphen_to_seq(seq_rna, seq_dna, maxlen=24)
        return (padded_seq_rna, padded_seq_dna)
    
    @staticmethod
    def _return_mismatch_positions(args) -> list:
        seq_rna, seq_dna = args
        mismatches = return_mismatch_positions(seq_rna, seq_dna)
        return mismatches
    
    @staticmethod
    def _process_seq_to_token(args) -> tuple:
        seq_rna, seq_dna = args
        rna_seq_token = seq_to_kmer(seq_rna)
        dna_seq_token = seq_to_kmer(seq_dna)
        return (rna_seq_token, dna_seq_token)
    
    def tokenize_function(self, example):
        return self.tokenizer(example["rna_seq"], example["dna_seq"], padding='max_length', truncation=True, max_length=self.token_max_len)

    def process(self, rna_sequence_list: list, dna_sequence_list: list) -> Dataset:
        # Count the number of CPU cores available
        cpu_count = min(24, multiprocessing.cpu_count() - 2)
        
        # Prepare the arguments for multiprocessing
        worker_args = [(seq_rna, seq_dna) for seq_rna, seq_dna in zip(rna_sequence_list, dna_sequence_list)]
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_alignment_hyphen, worker_args), total=len(worker_args), desc="Processing sequences"))
        
        with Pool(processes=cpu_count) as pool:
            _mismatches = list(tqdm.tqdm(pool.imap(self._return_mismatch_positions, _processed_seqs), total=len(_processed_seqs), desc="Finding mismatch positions"))
            
        with Pool(processes=cpu_count) as pool:
            _processed_seqs = list(tqdm.tqdm(pool.imap(self._process_seq_to_token, _processed_seqs), total=len(_processed_seqs), desc="Converting sequences to tokens"))
        rna_seq_list, dna_seq_list = zip(*_processed_seqs)
            
        # Hugging face tokenizer need to be used with Dataset
        input_dataset = Dataset.from_dict({"rna_seq": rna_seq_list, "dna_seq": dna_seq_list})
        tokenized_input_dataset = input_dataset.map(self.tokenize_function, batched=True) # features: ['rna_seq', 'dna_seq', 'input_ids', 'token_type_ids', 'attention_mask']
        
        tokenized_input_dataset = tokenized_input_dataset.remove_columns(["rna_seq", "dna_seq"])
        tokenized_input_dataset = tokenized_input_dataset.add_column("label", _mismatches)
        
        # Shuffle the dataset
        tokenized_input_dataset = tokenized_input_dataset.shuffle(seed=42)

        return tokenized_input_dataset
    
    def generate_random_sequence_input(self, rna_seq_list: list, n_samples: int) -> dict:
        rna_seqs = []
        dna_seqs = []
        n_sgrna = len(set(rna_seq_list))
        for rna_seq in list(set(rna_seq_list)):
            # Add on-target pair
            rna_seqs.append("-" + rna_seq)
            dna_seqs.append("-" + rna_seq)
            # Generate off-target pairs with random mutations
            for mismatch_count in range(1, 7):
                for _ in range((n_samples/n_sgrna)//6):
                    # Generate random DNA sequence with specified number of mismatches
                    rna_list = list(rna_seq)
                    dna_list = rna_list.copy()
                    # Randomly select positions to introduce mismatches
                    mismatch_positions = random.sample(range(len(rna_seq)), mismatch_count)
                    for pos in mismatch_positions:
                        original_base = rna_list[pos]
                        dna_list[pos] = random.choice([b for b in self.bases if b != original_base])
                    # 90% chance of no bulge, 10% chance of bulge at random position
                    if random.random() > 0.9:
                        dna_list.insert(0, "-")
                        rna_list.insert(0, "-")
                    else:
                        bulge_pos = random.choice(mismatch_positions)
                        if random.random() > 0.5:
                            rna_list.insert(bulge_pos, "-")
                            dna_list.insert(bulge_pos, self.bases[random.randint(0, 3)])
                        else:
                            dna_list.insert(bulge_pos, "-")
                            rna_list.insert(bulge_pos, self.bases[random.randint(0, 3)])
                    rna_seqs.append(''.join(rna_list))
                    dna_seqs.append(''.join(dna_list))
        return {"rna_seq": rna_seqs, "dna_seq": dna_seqs}

    def load_sequence_data(self, if_test=None) -> Dataset:
        dataset_names = [
            "Lazzarotto_2020_CHANGE_seq", "Lazzarotto_2020_GUIDE_seq", "SchmidBurgk_2020_TTISS",
            "Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2"
        ]
        dna_seqs = []
        sgrna_seqs = []
        for dataset_name in dataset_names:
            self.config["dataset_name"]["dataset_current"] = dataset_name
            DataLoaderClass = data_loader.DataLoaderClass(self.config)
            dataset_dict = DataLoaderClass.load_dataset()
            sgrna_list = dataset_dict["sgRNA"]
            # Generate random sequence inputs
            generated_data = self.generate_random_sequence_input(sgrna_list, n_samples=len(dataset_dict["rna_seq"])//10)
            sgrna_seqs[dataset_name].extend(generated_data["rna_seq"])
            dna_seqs[dataset_name].extend(generated_data["dna_seq"])
        dataset = self.process(sgrna_seqs, dna_seqs)
        if if_test:
            dataset = dataset.select(range(200000))
        return dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def evaluate_func(datasets, model, device='cuda'):
    true_labels = datasets['label']
    
    tensor_dataset = TensorDataset(
        torch.tensor(datasets['input_ids']),
        torch.tensor(datasets['attention_mask']),
        torch.tensor(true_labels, dtype=torch.float)
    )
    data_loader = DataLoader(tensor_dataset, batch_size=32)
    
    model.to(device)
    model.eval()
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
            all_labels.append(labels.cpu())
            
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    probabilities = torch.sigmoid(all_logits)
    predictions = (probabilities > 0.5).int()
    
    accuracy = accuracy_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions, average='macro')
    precision = precision_score(all_labels, predictions, average='macro')
    f1 = f1_score(all_labels, predictions, average='macro')
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False): # Add num_items_in_batch=None
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])

    def get_losses(self):
        return np.array(self.losses)
        


class DNABERTModelClass:
    def __init__(self, config: dict):
        self.config = config
        self.pretrained_model_path = config["pretrained_model_path"]
        self.pair_finetuned_path = config["pair_finetuned_path"]
        self.pair_finetuned_no_pretrain_path = config["pair_finetuned_no_pretrain_path"]
        self.save_path = self.return_save_path(if_pretrain=args.pretrain)
        self.max_length = config["max_length"]
        self.k_mer = config["k_mer"]
        self.tokens = create_tokens(config)
        self.epochs = 5
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
        for token in self.tokens:
            self.tokenizer.add_tokens([token], special_tokens=True)
    
    def return_save_path(self, if_pretrain: bool) -> str:
        if if_pretrain:
            print(f'Fine-tuning from pretrained model')
            print(self.pair_finetuned_path)
            return self.pair_finetuned_path
        else:
            print(f'Fine-tuning from scratch')
            print(self.pair_finetuned_no_pretrain_path)
            return self.pair_finetuned_no_pretrain_path
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples['sgRNA'], examples['targetDNA'], padding='max_length', truncation=True, max_length=self.max_length)

    def model_preparation(self, if_pretrain: bool) -> AutoModelForSequenceClassification:
        if if_pretrain:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_path,  # Weights from pretrained model
                num_labels=24 + 1,  
                problem_type="multi_label_classification",
            )
        else:
            # Not loading weights from pretrained model
            # Create a new model architecture with the same configuration as the pretrained model
            config = BertConfig.from_pretrained(f"{self.pretrained_model_path}")
            config.num_labels = 24 + 1
            config.problem_type = "multi_label_classification"
            config.vocab_size = len(self.tokenizer)
            model = AutoModelForSequenceClassification.from_config(
                config=config, # Load model architecture from pretrained model
            )
        model.resize_token_embeddings(len(self.tokenizer))
        return model
    
    def train(self) -> None:
        print(f'Running pair finetuning task for DNABERT')
        # Dataset
        train_datasets = PrepareDatasetClass(self.config).load_sequence_data()
        
        # training arguments
        training_args = TrainingArguments(
            output_dir= self.save_path,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            lr_scheduler_type="cosine_with_restarts",
            disable_tqdm=False,
            logging_steps=4000,
            push_to_hub=False,
            log_level="error",
            save_strategy="no",  # This disables checkpoint saving
            # fp16=True,  # Enable mixed precision training
        )

        # Model preparation
        model = self.model_preparation(if_pretrain=args.pretrain)
        model.to(self.device)
        
        loss_callback = LossRecorderCallback()
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_datasets,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            callbacks=[loss_callback],  # Add the loss callback here
        )
    
        # Fine-Tuning
        trainer.train()

        loss_array = loss_callback.get_losses()
        np.save(f"{self.save_path}/loss.npy", loss_array)

        # save fine-tuned model
        self.tokenizer.save_pretrained(self.save_path)
        model.save_pretrained(self.save_path)
        

    def test(self) -> None:
        print(f'Running evaluation task for DNABERT')
        # Dataset
        test_datasets = PrepareDatasetClass(self.config).load_sequence_data(if_test=True)
        
        # load tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            self.save_path,
            num_labels=24 + 1,
            problem_type="multi_label_classification",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.save_path)

        results = evaluate_func(test_datasets, model)
        
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        

def main() -> None:
    dnabert_model = DNABERTModelClass(config)
    dnabert_model.train()
    dnabert_model.test()
    plot_loss_curve(config)

if __name__ == "__main__":
    main()