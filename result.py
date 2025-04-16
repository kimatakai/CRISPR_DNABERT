
import sys
sys.path.append("./")
import os
import warnings
warnings.filterwarnings('ignore')


from script import utilities_module, result_module


import argparse
parser = argparse.ArgumentParser(description='CRISPR/Cas off-target effect project')
parser.add_argument("-d", "--datatype", type=str, default="changeseq", choices=["changeseq", "guideseq", "transfer"], help="")
parser.add_argument("-f", "--fold", type=str, default="0,1,2,3,4,5,6,7,8,9", help="")
parser.add_argument("-m", "--model", type=str, default="gru-embed,crispr-bert,crispr-hw,crispr-dipoff,crispr-bert-2025,dnabert", help="")
parser.add_argument("-e", "--exp_id", type=str, default="0,1,2,3,4", help="")
parser.add_argument("--show", action="store_true")
parser.add_argument("--excel", action="store_true")
parser.add_argument("--ensemble", action="store_true")
parser.add_argument("--boxplot", action="store_true")
parser.add_argument("--curve", action="store_true")

args = parser.parse_args()

# Processing command argument
try:
    fold_list = [int(fold) for fold in args.fold.split(",")]
except:
    sys.exit(f"[ERROR] Invalid fold.")
try:
    model_list = [str(model_name) for model_name in args.model.split(",") if model_name in ["dnabert", "dnabert-no-pretrain", "dnabert-epi", "dnabert-epi-ablation", "crispr-bert", "gru-embed", "crispr-hw", "crispr-dipoff", "crispr-bert-2025"]]
except:
    sys.exit(f"[ERROR] Invalid model.")
try:
    exp_id_list = [int(exp_id) for exp_id in args.exp_id.split(",")]
except:
    sys.exit(f"[ERROR] Invalid experiment id.")

def main():
    # Check if model weight exist.
    # for model_name in model_list:
    #     for fold in fold_list:
    #         for exp_id in exp_id_list:
    #             utilities_module.check_model_weight_exist(model_name=model_name, datatype=args.datatype, fold=fold, exp_id=exp_id)
    
    # Check if output probability exist.
    for model_name in model_list:
        for fold in fold_list:
            for exp_id in exp_id_list:
                utilities_module.check_output_probability_exist(model_name=model_name, datatype=args.datatype, fold=fold, exp_id=exp_id)
    
    # Initialize class
    dataloaderClassForResult = result_module.DataLoaderClassForResult(
        datatype=args.datatype, 
        model_list=model_list, fold_list=fold_list, exp_id_list=exp_id_list, metrics_list=['accuracy', 'recall', 'precision', 'specificity', 'FPR', 'FDR', 'f1', 'ROC-AUC', 'PR-AUC', 'mcc'])
    
    # Create EXCEL workbook
    if args.excel:
        dataloaderClassForResult.create_workbooks()
    
    # Preliminaries
    dataloaderClassForResult.load_dataset()
    dataloaderClassForResult.load_test_info()
    dataloaderClassForResult.return_sgrna_index_dict()
    
    # Load true label
    dataloaderClassForResult.return_test_label()
    
    # Load output probability
    dataloaderClassForResult.aggregate_probabilities()
    
    # Aggregate result by metrics
    dataloaderClassForResult.aggregate_result_by_metrics()
    
    # Whole result
    dataloaderClassForResult.whole_result(show=args.show, to_excel=args.excel, ensemble=args.ensemble)
    
    # Wilcoxon test result
    dataloaderClassForResult.wilcoxon_signed_rank_test(show=args.show, to_excel=args.excel, ensemble=args.ensemble)
                
    # Confusion matrix by mismatch
    dataloaderClassForResult.return_mismatch_index_dict()
    dataloaderClassForResult.confusion_matrix(show=args.show, to_excel=args.excel, ensemble=args.ensemble)
    
    # Boxplot for each model
    if args.boxplot:
        dataloaderClassForResult.boxplot_for_result(ensemble=args.ensemble)
    
    if args.curve:
        dataloaderClassForResult.curve_roc_for_result(ensemble=args.ensemble)
    
    
    
    
    
    


if __name__ == "__main__":
    main()




