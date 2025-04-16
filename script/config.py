
# Random seed
random_state = 42

# Data information
metadata_path = f'./metadata'

# Datasets dir path
yaish_et_al_data_path = "./yaish_et_al/datasets"

# Epigenetic file
gse149361_path = "./gse149361"

# Epigenetic feature
epigenetic_base_dir_path = "./epigenetic_feature"

# Figures path
fig_base_path = "./figures"

# Attention weight path
attention_weight_base_dir_path = "./attention_weight"

# model path
# DNABERT
dnabert_pretrained_model_path = "./model_weight/pretrained_dnabert3"
dnabert_pair_finetuned_path = "./model_weight/pair_ft"
dnabert_pair_finetuned_no_pretrain_path = "./model_weight/pair_ft_no_pretrain"
dnabert_crispr_finetuned_path = "./model_weight/dnabert_crispr"
dnabert_crispr_no_pretrain_path = "./model_weight/dnabert_crispr_no_pretrain"
dnabert_epigenetic_model_path = "./model_weight/dnabert_epi"
# GRU-Emb model
gru_embed_model_path = "./model_weight/gru_embed_model"
# CRISPR-BERT model
crispr_bert_model_path = "./model_weight/crispr_bert_model"
crispr_bert_architecture_path = "./model_weight/crispr_bert_model/uncased_L-2_H-256_A-4"
# CRISPR-HW model
crispr_hw_model_path = "./model_weight/crispr_hw_model"
# CRISPR-DIPOFF model
crispr_dipoff_model_path = "./model_weight/crispr_dipoff_model"
# CrisprBERT model
crispr_bert_2025_model_path = "./model_weight/crispr_bert_2025_model"
# FNN model
epigenetic_model_path = "./model_weight/epigenetic_model"
fnn_model_path = "./model_weight/fnn_model"

# Result path
# Predicted Probability path
probabilities_base_dir_path = "./probability"
# Result path
result_base_dir_path = "./result"




bin_size = 100000
kmer = 3

type_colors = {
    # Datatype color
    "changeseq": "#8a3319", "guideseq": "#65318e", "inactive": "#b8d200",
    
    # Model color
    "dnabert": "#bebebe", "dnabert-epi": "#cc99ff", "dnabert-epi-ablation": "#ffcc66",
    "crispr-bert": "#1F62A6", "gru-embed": "#fec44f", "crispr-hw": "#AECBE6", "crispr-dipoff": "#BD342F", "crispr-bert-2025": "#FC9272",
    "ensemble": "#addd8e"
}


base_index = {'A':0, 'T':1, 'G':2, 'C':3, '-':4}





