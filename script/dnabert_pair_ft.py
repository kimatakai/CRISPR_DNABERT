import os
import random
from itertools import product
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertConfig, TrainerCallback
import matplotlib.pyplot as plt

import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', action='store_true', help='Pretrain the model')
args = parser.parse_args()
from_pretrain = args.pretrain


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fix random seed
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(config.random_state)


###################################################################################################


k_mer = config.kmer


###################################################################################################


current_directory = os.getcwd()
pretrained_model_path = f"{config.dnabert_pretrained_model_path}"
if from_pretrain:
    finetuned_mismatch_path = f"{config.dnabert_pair_finetuned_path}"
else:
    finetuned_mismatch_path = f"{config.dnabert_pair_finetuned_no_pretrain_path}"
os.makedirs(finetuned_mismatch_path, exist_ok=True)
print(f'Fine-tuning model path : {finetuned_mismatch_path}')
print(f'Pretrained model path : {pretrained_model_path}')


###################################################################################################


def plot_loss_curve():
    loss_array_pretrained = np.load(config.dnabert_pair_finetuned_path + '/loss.npy')
    loss_array_no_pretrained = np.load(config.dnabert_pair_finetuned_no_pretrain_path + '/loss.npy')
    epochs = np.linspace(0, 5, len(loss_array_pretrained))
    
    # Plotting
    plt.plot(epochs, loss_array_pretrained, label='Pre-trained DNABERT', color='blue')
    plt.plot(epochs, loss_array_no_pretrained, label='DNABERT from scratch', color='orange')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 5)
    
    os.makedirs(f"{config.fig_base_path}/loss_curve", exist_ok=True)
    plt.savefig(f"{config.fig_base_path}/loss_curve/loss_curve.png", dpi=300)



###################################################################################################


# create tokens (3-mer)
bases = 'ATGC'
tokens = []
for i in range(k_mer):
    for combination in product(bases, repeat=k_mer - 1):
        token = ''.join(combination[:i] + ('-',) + combination[i:])
        tokens.append(token)


###################################################################################################


def return_mismatch_positions(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("The sequences must be of the same length.")
    
    mismatch_list = []
    for base1, base2 in zip(seq1, seq2):
        if base1 == '-' and base2 == '-':
            mismatch_list.append(0)
        elif base1 == '-' or base2 == '-':
            mismatch_list.append(1)
        elif base1 == 'N' or base2 == 'N':
            mismatch_list.append(0)
        else:
            mismatch_list.append(0 if base1 == base2 else 1)
    
    return mismatch_list


def generate_dna_samples(n_samples=10):
    bases = 'ATGC'
    sequence_length = 24
    dna_samples = []
    sgrna_samples = []
    mismatches = []

    for mismatch_count in range(7):  # Mismatch counts from 0 to 6
        for _ in range(n_samples // 7):  # Ensure each mismatch count is equally represented
            # Generate a random DNA sequence
            seq1 = ''.join(random.choice(bases) for _ in range(sequence_length))    # 'AGGCAGACCTGCGGGATGATTCGG'
            seq1_list = list(seq1)  # ['A', 'G', 'G', ... , 'G', 'G']
            seq2_list = list(seq1)  # ['A', 'G', 'G', ... , 'G', 'G']
            
            # Introduce mismatches
            mismatch_positions = random.sample(range(sequence_length), mismatch_count)
            for pos in mismatch_positions:
                original_base = seq2_list[pos]
                # Choose a different base that is not the original
                seq2_list[pos] = random.choice([b for b in bases if b != original_base])
            
            # Sometimes introduce 'N' at random positions
            if random.random() > 0.99:  # 1% chance to introduce 'N'
                n_position = random.randint(0, sequence_length - 1)
                seq2_list[n_position] = 'N'
            
            if random.random() > 0.9 or mismatch_count==0:   # 90% chance to introduce '-' on head position
                seq1_list[0] = '-'
                seq2_list[0] = '-'
            else:
                n_position = random.sample(mismatch_positions, 1)[0]
                seq1_list[n_position] = '-'
            
            seq1 = ''.join(seq1_list)
            seq2 = ''.join(seq2_list)
            
            if random.random() > 0.5:
                dna, sgrna = seq1, seq2
            else:
                dna, sgrna = seq2, seq1
            
            # Adjust mismatch count if 'N' was introduced
            # final_mismatch_count = sum(1 for x, y in zip(seq1, seq2) if x != y and y != 'N')
            mismatches_labels = return_mismatch_positions(dna, sgrna)
            
            dna_samples.append(seq_to_kmer(dna))
            sgrna_samples.append(seq_to_kmer(sgrna))
            mismatches.append(mismatches_labels)
    
    # shuffle data
    indices = list(range(len(dna_samples)))
    random.shuffle(indices)
    shuffled_dna_samples = [dna_samples[i] for i in indices]
    shuffled_sgrna_samples = [sgrna_samples[i] for i in indices]
    shuffled_mismatches = [mismatches[i] for i in indices]
    
    datasets = Dataset.from_dict({'targetDNA' : shuffled_dna_samples, 'sgRNA' : shuffled_sgrna_samples, 'label' : shuffled_mismatches}) 
    
    return datasets


def seq_to_kmer(sequence, kmer=k_mer):
    kmers = [sequence[i:i+kmer] for i in range(len(sequence)-kmer+1)]
    merged_sequence = ' '.join(kmers)
    return merged_sequence

def list_to_datasets(data):
    random.shuffle(data)
    datasets_dict = {'targetDNA' : [], 'sgRNA' : [], 'label' : []}
    for row in data:
        targetdna = seq_to_kmer(row[3])
        sgrna = seq_to_kmer(row[4])
        datasets_dict['targetDNA'].append(targetdna)
        datasets_dict['sgRNA'].append(sgrna)
        mismatch = int(row[6])
        datasets_dict['label'].append(mismatch)
    datasets = Dataset.from_dict(datasets_dict)
    return datasets



###################################################################################################


# compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


###################################################################################################


def evalate_func(datasets, model, device='cuda'):
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


###################################################################################################


def main():
    
    print(f'!!!!!!!!!! Mismatch Prediction Fine-Tuning !!!!!!!!!!')
    if from_pretrain:
        print(f'Fine-tuning from pretrained model')
    else:
        print(f'Fine-tuning from scratch')
    
    # create train data
    datasets = generate_dna_samples(n_samples=500000)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    for token in tokens:
        tokenizer.add_tokens([token], special_tokens=True)
    
    # definition func for tokenizer
    max_length = 2*(24 - k_mer + 1) + 3
    def tokenize_function(examples):
        return tokenizer(examples['targetDNA'], examples['sgRNA'], padding='max_length', truncation=True, max_length=max_length)
    datasets = datasets.map(tokenize_function, batched=True)
    
    print(datasets["targetDNA"][0][1])
    print(f'targetDNA [0]       : {datasets["targetDNA"][0]}')
    print(f'sgRNA [0]           : {datasets["sgRNA"][0]}')
    print(f'input_ids [0]       : {datasets["input_ids"][0]}')
    print(f'token_type_ids [0]  : {datasets["token_type_ids"][0]}')
    print(f'label [0]           : {datasets["label"][0]}')
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=finetuned_mismatch_path,
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        lr_scheduler_type="cosine_with_restarts",
        disable_tqdm=False,
        logging_steps=200,
        push_to_hub=False,
        log_level="error",
        save_strategy="no",  # This disables checkpoint saving
        fp16=True,  # Enable mixed precision training
    )
    
    # load model
    if from_pretrain:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_path,  # Weights from pretrained model
            num_labels=24,  
            problem_type="multi_label_classification",
        ).to(device)
    else:
        # Not loading weights from pretrained model
        # Create a new model architecture with the same configuration as the pretrained model
        config = BertConfig.from_pretrained(f"{pretrained_model_path}")
        config.num_labels = 24
        config.problem_type = "multi_label_classification"
        config.vocab_size = len(tokenizer)
        model = AutoModelForSequenceClassification.from_config(
            config=config, # Load model architecture from pretrained model
        ).to(device)
    model.resize_token_embeddings(len(tokenizer))
    
    loss_callback = LossRecorderCallback()
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[loss_callback],  # Add the loss callback here
    )
    
    # Fine-Tuning
    trainer.train()
    
    loss_array = loss_callback.get_losses()
    np.save(os.path.join(finetuned_mismatch_path, 'loss.npy'), loss_array)
    
    # save fine-tuned model
    tokenizer.save_pretrained(finetuned_mismatch_path)
    model.save_pretrained(finetuned_mismatch_path)
    print(f'Complete saving mismatch prediction task fine-tuned model')
    
    
    datasets = generate_dna_samples(n_samples=50000)
    
    # load tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_mismatch_path,
        num_labels=24,
        problem_type="multi_label_classification",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(finetuned_mismatch_path)
    
    datasets = datasets.map(tokenize_function, batched=True)
    
    results = evalate_func(datasets, model)
    
    for metric in results:
        print(f'{metric} : {results[metric]}')

if __name__ == '__main__':
    main()