

import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

import utils.check_set as check_set
import models.data_loader as data_loader



def return_metrics(fold: int, iter: int, all_true_labels: list, all_pred_labels: list, all_probs: list, if_show: bool=True) -> dict:
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average='binary')
    recall = recall_score(all_true_labels, all_pred_labels, average='binary')
    f1 = f1_score(all_true_labels, all_pred_labels, average='binary')
    mcc = matthews_corrcoef(all_true_labels, all_pred_labels)
    roc_auc = roc_auc_score(all_true_labels, all_probs)
    pr_auc = average_precision_score(all_true_labels, all_probs)
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    if if_show:
        print(f"Fold: {fold}, Iteration: {iter}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
    }
    

def save_results(results: dict, result_path: str) -> None:
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    # check if the file is saved correctly
    if os.path.exists(result_path):
        print(f"Results saved to {result_path}")
    else:
        raise FileNotFoundError(f"Failed to save results to {result_path}")
    return None

def aggregate_results(config: dict, fold_list: list, iter_list: list) -> dict:
    # Result vacant
    result_all = {}
    for iter in iter_list:
        result_all[iter] = {}
        for fold in fold_list:
            result_all[iter][fold] = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "mcc": 0, "roc_auc": 0, "pr_auc": 0, "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0}}
    true_labels_all = []
    pred_labels_all = []
    probability_all = []
    confusion_matrix_idx = {"TN": [], "FP": [], "FN": [], "TP": []}
    
    # Load result on each fold and iteration
    for iter_ in iter_list:
        for fold_ in fold_list:
            config_ = config.copy()
            config_["fold"] = fold_
            config_["iter"] = iter_
            config_ = check_set.set_path(config_)  # Update paths
            
            # Load dataset
            DataLoaderClass = data_loader.DataLoaderClass(config_)
            dataset = DataLoaderClass.load_dataset()
            
            # load result
            true_label = np.array(dataset["label"])[dataset["test_idx"]]
            probability = np.load(config_["probability_path"])   # (1000, )
            pred_label = np.where(probability > 0.5, 1, 0)
            with open(config_["result_save_path"]) as f:
                result = json.load(f)
                        
            # Collect results
            for metrics in ["accuracy", "precision", "recall", "f1_score", "mcc", "roc_auc", "pr_auc"]:
                result_all[iter_][fold_][metrics] = float(result[metrics])
            for cm_key, cm_tuple in zip(["TN", "FP", "FN", "TP"], [(0, 0), (0, 1), (1, 0), (1, 1)]):
                result_all[iter_][fold_]["confusion_matrix"][cm_key] = int(result["confusion_matrix"][cm_tuple[0]][cm_tuple[1]])
            true_labels_all.extend(true_label.tolist())
            pred_labels_all.extend(pred_label.tolist())
            probability_all.extend(probability.tolist())
            
            # Collect confusion matrix index
            test_idx = dataset["test_idx"]
            test_idx_mapping = {i: idx for i, idx in enumerate(test_idx)}
            tn_index = np.where((true_label == 0) & (pred_label == 0))[0]
            confusion_matrix_idx["TN"].append(np.array([test_idx_mapping[i] for i in tn_index]))
            fp_index = np.where((true_label == 0) & (pred_label == 1))[0]
            confusion_matrix_idx["FP"].append(np.array([test_idx_mapping[i] for i in fp_index]))
            fn_index = np.where((true_label == 1) & (pred_label == 0))[0]
            confusion_matrix_idx["FN"].append(np.array([test_idx_mapping[i] for i in fn_index]))
            tp_index = np.where((true_label == 1) & (pred_label == 1))[0]
            confusion_matrix_idx["TP"].append(np.array([test_idx_mapping[i] for i in tp_index]))
    
    # Calculate average results
    result_avg = {}
    for metrics in ["accuracy", "precision", "recall", "f1_score", "mcc", "roc_auc", "pr_auc"]:
        metrics_value_temp = []
        for iter_ in iter_list:
            for fold_ in fold_list:
                metrics_value_temp.append(result_all[iter_][fold_][metrics])
        result_avg[metrics] = float(np.mean(metrics_value_temp))
    for cm_key in ["TN", "FP", "FN", "TP"]:
        cm_value_temp_iter = []
        for iter_ in iter_list:
            cm_value_temp_fold = []
            for fold_ in fold_list:
                cm_value_temp_fold.append(result_all[iter_][fold_]["confusion_matrix"][cm_key])
            if config_["dataset_name"]["dataset_dl"] in ["Chen_2017_GUIDE_seq", "Listgarten_2018_GUIDE_seq"]:
                cm_value_temp_iter.append(np.mean(cm_value_temp_fold))
            else:
                cm_value_temp_iter.append(np.sum(cm_value_temp_fold))
        result_avg[cm_key] = float(np.mean(cm_value_temp_iter))

    # for key, value in result_avg.items():
    #     if key != "confusion_matrix":
    #         print(f"{key}: {float(value):.4f}")
    #     else:
    #         for cm_key, cm_value in value.items():
    #             print(f"{cm_key}: {float(cm_value):.4f}")
    
    true_labels_all = np.array(true_labels_all)
    pred_labels_all = np.array(pred_labels_all)
    probability_all = np.array(probability_all)
    
    # Confusion matrix index
    confusion_matrix_idx = {
        "TN": np.concatenate(confusion_matrix_idx["TN"]),
        "FP": np.concatenate(confusion_matrix_idx["FP"]),
        "FN": np.concatenate(confusion_matrix_idx["FN"]),
        "TP": np.concatenate(confusion_matrix_idx["TP"])
    }
    
    print("Completed aggregating results.")
    
    return {
        "result_all": result_all, # dict
        "result_avg": result_avg, # dict
        "true_labels_all": true_labels_all, # np.array
        "pred_labels_all": pred_labels_all, # np.array
        "probability_all": probability_all, # np.array
        "confusion_matrix_idx": confusion_matrix_idx # dict
    }
    

