
import sys
sys.path.append("./")
import os
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch


from script import config, data_loader, dnabert_module, epigenetic_module, result_module
from script.yaish_et_al import gru_embed_module
from script.luo_et_al import crispr_bert_module
from script.yang_et_al import crispr_hw_module
from script.toufikuzzaman_et_al import crispr_dipoff_module
from script.sari_et_al import crispr_bert_2025_module


import argparse

parser = argparse.ArgumentParser(description='CRISPR/Cas off-target effect project')
parser.add_argument("-d", "--datatype", type=str, default="changeseq", choices=["changeseq", "guideseq", "transfer"], help="")
parser.add_argument("-f", "--fold", type=int, default=0, help="")
parser.add_argument("-m", "--model", type=str, default="dnabert", help="")
parser.add_argument("-e", "--exp_id", type=int, default=0, help="")
parser.add_argument("--train", action="store_true")
parser.add_argument("--test", action="store_true")

args = parser.parse_args()

def set_seed(seed_value: int=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(config.random_state + args.exp_id)


def main():
    if args.datatype == "changeseq":
        if args.fold >= 10 or args.fold < 0:
            sys.exit(f"[ERROR] fold {args.fold} is invalid.")
        
        # Data Loader
        dataLoaderClass = data_loader.DataLoaderClass(fold=args.fold, datatype=args.datatype)
        dataset_df = dataLoaderClass.load_dataset(sgrna="all")
        train_test_info = dataLoaderClass.return_train_test_data()
        
        # Train and test
        if args.model == "dnabert":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-no-pretrain":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id, no_pretrained=True)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "gru-embed":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            GruEmbedClass = gru_embed_module.GruEmbedClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                GruEmbedClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = GruEmbedClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert":
            crisprBertDataProcessClass = crispr_bert_module.CrisprBertDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBertDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBertClass = crispr_bert_module.CrisprBertClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-hw":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            CrisprHwClass = crispr_hw_module.CrisprHwClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprHwClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprHwClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-dipoff":
            crisprDipoffDataProcessClass = crispr_dipoff_module.CrisprDipoffDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprDipoffDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprDipoffClass = crispr_dipoff_module.CrisprDipoffClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprDipoffClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDipoffClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert-2025":
            crisprBert2025DataProcessClass = crispr_bert_2025_module.CrisprBert2025DataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBert2025DataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBert2025Class = crispr_bert_2025_module.CrisprBert2025Class(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBert2025Class.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBert2025Class.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        else:
            sys.exit(f"[ERROR] model argument is invalid.")
                
    
    elif args.datatype == "guideseq":
        if args.fold >= 10 or args.fold < 0:
            sys.exit(f"[ERROR] fold {args.fold} is invalid.")
        
        # Data Loader
        dataLoaderClass = data_loader.DataLoaderClass(fold=args.fold, datatype=args.datatype)
        dataset_df = dataLoaderClass.load_dataset(sgrna="all")
        train_test_info = dataLoaderClass.return_train_test_data()
        
        # Train and test
        if args.model == "dnabert":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-no-pretrain":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id, no_pretrained=True)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-epi":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            EpigeneticFeatureClass = epigenetic_module.EpigeneticFeatureClass(dataset_df, train_test_info, datatype=args.datatype, assay=None, assays=["h3k4me3", "h3k27ac", "atac"], scope_range=5000, bin_size=50)
            epigenetic_input_dict = EpigeneticFeatureClass.return_epigenetic_feature()
            # DNABERT-epigenetic instance
            CrisprDnaBertEpigeneticClass = dnabert_module.CrisprDnaBertEpigeneticClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertEpigeneticClass.train_classification_task(epigenetic_input_dict)
            if args.test:
                true_label_np, probabilities = CrisprDnaBertEpigeneticClass.test_classification_task(epigenetic_input_dict)
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "gru-embed":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            GruEmbedClass = gru_embed_module.GruEmbedClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                GruEmbedClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = GruEmbedClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert":
            crisprBertDataProcessClass = crispr_bert_module.CrisprBertDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBertDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBertClass = crispr_bert_module.CrisprBertClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-hw":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            CrisprHwClass = crispr_hw_module.CrisprHwClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprHwClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprHwClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-dipoff":
            crisprDipoffDataProcessClass = crispr_dipoff_module.CrisprDipoffDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprDipoffDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprDipoffClass = crispr_dipoff_module.CrisprDipoffClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprDipoffClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDipoffClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert-2025":
            crisprBert2025DataProcessClass = crispr_bert_2025_module.CrisprBert2025DataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBert2025DataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBert2025Class = crispr_bert_2025_module.CrisprBert2025Class(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBert2025Class.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBert2025Class.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        else:
            sys.exit(f"[ERROR] model argument is invalid.")
    
    elif args.datatype == "transfer":
        if args.fold >= 10 or args.fold < 0:
            sys.exit(f"[ERROR] fold {args.fold} is invalid.")
        
        # Data Loader
        dataLoaderClass = data_loader.DataLoaderClass(fold=args.fold, datatype="guideseq")
        dataset_df = dataLoaderClass.load_dataset(sgrna="all")
        train_test_info = dataLoaderClass.return_train_test_data()
        
        if args.model == "dnabert":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-no-pretrain":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            # DNABERT instance
            CrisprDnaBertClass = dnabert_module.CrisprDnaBertClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id, no_pretrained=True)
            if args.train:
                CrisprDnaBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDnaBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-epi":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            EpigeneticFeatureClass = epigenetic_module.EpigeneticFeatureClass(dataset_df, train_test_info, datatype="guideseq", assay=None, assays=["h3k4me3", "h3k27ac", "atac"], scope_range=5000, bin_size=50)
            epigenetic_input_dict = EpigeneticFeatureClass.return_epigenetic_feature()
            # DNABERT-epigenetic instance
            CrisprDnaBertEpigeneticClass = dnabert_module.CrisprDnaBertEpigeneticClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertEpigeneticClass.train_classification_task(epigenetic_input_dict)
            if args.test:
                true_label_np, probabilities = CrisprDnaBertEpigeneticClass.test_classification_task(epigenetic_input_dict)
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "dnabert-epi-ablation":
            dataset_dict = dataLoaderClass.return_dataset_for_dnabert()
            EpigeneticFeatureClass = epigenetic_module.EpigeneticFeatureClass(dataset_df, train_test_info, datatype="guideseq", assay=None, assays=["h3k4me3", "h3k27ac", "atac"], scope_range=5000, bin_size=50)
            epigenetic_input_dict = EpigeneticFeatureClass.return_epigenetic_feature()
            # DNABERT-epigenetic instance
            CrisprDnaBertEpigeneticClass = dnabert_module.CrisprDnaBertEpigeneticClass(dataset_df, train_test_info, fold=args.fold, datatype=args.datatype, dataset_dict=dataset_dict, exp_id=args.exp_id)
            if args.train:
                CrisprDnaBertEpigeneticClass.train_classification_task(epigenetic_input_dict, ablation_mode=True)
            if args.test:
                true_label_np, probabilities = CrisprDnaBertEpigeneticClass.test_classification_task(epigenetic_input_dict, ablation_mode=True)
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "gru-embed":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            GruEmbedClass = gru_embed_module.GruEmbedClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                GruEmbedClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = GruEmbedClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert":
            crisprBertDataProcessClass = crispr_bert_module.CrisprBertDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBertDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBertClass = crispr_bert_module.CrisprBertClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBertClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBertClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)

        elif args.model == "crispr-hw":
            input_dict = dataLoaderClass.return_pairseq_categorical_onehot()
            label_dict = dataLoaderClass.return_label()
            CrisprHwClass = crispr_hw_module.CrisprHwClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprHwClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprHwClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-dipoff":
            crisprDipoffDataProcessClass = crispr_dipoff_module.CrisprDipoffDataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprDipoffDataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprDipoffClass = crispr_dipoff_module.CrisprDipoffClass(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprDipoffClass.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprDipoffClass.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
        
        elif args.model == "crispr-bert-2025":
            crisprBert2025DataProcessClass = crispr_bert_2025_module.CrisprBert2025DataProcessClass(dataLoaderClass, dataset_df, train_test_info)
            input_dict = crisprBert2025DataProcessClass.return_input()
            label_dict = dataLoaderClass.return_label()
            CrisprBert2025Class = crispr_bert_2025_module.CrisprBert2025Class(dataset_df, train_test_info, input_dict, label_dict, fold=args.fold, datatype=args.datatype, exp_id=args.exp_id)
            if args.train:
                CrisprBert2025Class.train_classification_task()
            if args.test:
                true_label_np, probabilities = CrisprBert2025Class.test_classification_task()
                results = result_module.caluculate_metrics(true_label_np, probabilities, args.datatype, show=True)
            
    else:
        sys.exit(f"[ERROR] datatype argument is invalid.")




if __name__ == "__main__":
    main()










