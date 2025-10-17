
import utils.check_set as check_set
import models.data_loader as data_loader
import utils.epigenetic_module as epigenetic_module

import models.dnabert_module as dnabert_module
import models.gru_embed_2024 as gru_embed_2024
import models.crispr_bert_2024 as crispr_bert_2024
import models.crispr_hw_2023 as crispr_hw_2023
import models.crispr_bert_2025 as crispr_bert_2025



def model_transfer_epi(config: dict) -> None:
    print("Running model in transfer mode...")
    
    # Set paths in configuration
    SetPathsClass = check_set.SetPathsTransferEpi(config)
    config = SetPathsClass.set_path()
    
    # Load dataset
    DataLoaderClass = data_loader.DataLoaderClass(config)
    dataset_dict = DataLoaderClass.load_dataset()
    
    # Load epigenetic data
    SetPathEpigeneticClass = check_set.SetPathsEpigenetic(config)
    config = SetPathEpigeneticClass.set_path_for_model(config["using_epi_data"])
    dataset_dict = epigenetic_module.load_epigenetic_feature(config = config, dataset_dict = dataset_dict) # dataset["epigenetic_features"][type_of_data]

    # Train and test depending on the model
    if_train = config["train"]
    if_test = config["test"]
    if config["model_info"]["model_name"] == "DNABERT":
        DataProcessorDNABERT = dnabert_module.DataProcessorDNABERT(config)
        dataset_dict = DataProcessorDNABERT.load_inputs(dataset_dict)
        DnabertModelClass = dnabert_module.DNABERTEpiModelClass(config, dataset_dict)
        if if_train:
            DnabertModelClass.train_transfer_epi()
        if if_test:
            DnabertModelClass.test_transfer_epi()