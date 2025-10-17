
import utils.check_set as check_set
import models.data_loader as data_loader


import models.dnabert_module as dnabert_module
import models.gru_embed_2024 as gru_embed_2024
import models.crispr_bert_2024 as crispr_bert_2024
import models.crispr_hw_2023 as crispr_hw_2023
import models.crispr_dipoff_2025 as crispr_dipoff_2025
import models.crispr_bert_2025 as crispr_bert_2025



def model_transfer(config: dict) -> None:
    print("Running model in transfer mode...")
    
    # Set paths in configuration
    SetPathsClass = check_set.SetPathsTransfer(config)
    config = SetPathsClass.set_path()
    
    # Load dataset
    DataLoaderClass = data_loader.DataLoaderClass(config)
    dataset = DataLoaderClass.load_dataset()
    
    # Train and test depending on the model
    if_train = config["train"]
    if_test = config["test"]
    if config["model_info"]["model_name"] == "DNABERT":
        DataProcessorDNABERT = dnabert_module.DataProcessorDNABERT(config)
        dataset = DataProcessorDNABERT.load_inputs(dataset)
        DnabertModelClass = dnabert_module.DNABERTModelClass(config, dataset)
        if if_train:
            DnabertModelClass.train_transfer()
        if if_test:
            DnabertModelClass.test_transfer()
            
    elif config["model_info"]["model_name"] == "GRU-Embed":
        DataProcessorGRUEmbed = gru_embed_2024.DataProcessorGRUEmbed(config)
        dataset = DataProcessorGRUEmbed.load_inputs(dataset)
        GRUEmbedModelClass = gru_embed_2024.GRUEmbedModelClass(config, dataset)
        if if_train:
            GRUEmbedModelClass.train_transfer()
        if if_test:
            GRUEmbedModelClass.test_transfer()
            
    elif config["model_info"]["model_name"] == "CRISPR-BERT":
        DataProcessorCRISPRBERT = crispr_bert_2024.DataProcessorCRISPRBERT(config)
        dataset = DataProcessorCRISPRBERT.load_inputs(dataset)
        CRISPRBERTModelClass = crispr_bert_2024.CRISPRBERTModelClass(config, dataset)
        if if_train:
            CRISPRBERTModelClass.train_transfer()
        if if_test:
            CRISPRBERTModelClass.test_transfer()
    
    elif config["model_info"]["model_name"] == "CRISPR-HW":
        DataProcessorCRISPRHW = crispr_hw_2023.DataProcessorCRISPRHW(config)
        dataset = DataProcessorCRISPRHW.load_inputs(dataset)
        CRISPRHWModelClass = crispr_hw_2023.CRISPRHWModelClass(config, dataset)
        if if_train:
            CRISPRHWModelClass.train_transfer()
        if if_test:
            CRISPRHWModelClass.test_transfer()
    
    elif config["model_info"]["model_name"] == "CRISPR-DIPOFF":
        DataProcessorCRISPRDIPOFF = crispr_dipoff_2025.DataProcessorCrisprDipoff(config)
        dataset = DataProcessorCRISPRDIPOFF.load_inputs(dataset)
        print(dataset.keys())
        CRISPRDIPOFFModelClass = crispr_dipoff_2025.CRISPRDIPOFFModelClass(config, dataset)
        if if_train:
            CRISPRDIPOFFModelClass.train_transfer()
        if if_test:
            CRISPRDIPOFFModelClass.test_transfer()
    
    elif config["model_info"]["model_name"] == "CrisprBERT":
        DataProcessorCrisprBERT = crispr_bert_2025.DataProcessorCrisprBERT(config)
        dataset = DataProcessorCrisprBERT.load_inputs(dataset)
        CrisprBERTModelClass = crispr_bert_2025.CrisprBERTModelClass(config, dataset)
        if if_train:
            CrisprBERTModelClass.train_transfer()
        if if_test:
            CrisprBERTModelClass.test_scratch()