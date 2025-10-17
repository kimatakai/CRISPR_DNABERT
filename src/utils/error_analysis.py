
import numpy as np

import visualization.plot_mismatch_fig as plot_mismatch_fig
import visualization.plot_epigenetic_fig as plot_epigenetic_fig


def mismatch_analysis_by_cm(config: dict, dataset: dict, result_dict: dict) -> None:
    # Bar plot of the number of mismatches in each category ("TN", "FP", "FN", "TP")
    confusion_matrix_idx = result_dict["confusion_matrix_idx"]
    mismatch_list = dataset["mismatch"] # 0, 1, 2, 3, 4, 5, 6
    bulge_list = dataset["bulge"] # 0, 1
    n_mismatch_cm = {}
    for cm_key in ["TN", "FP", "FN", "TP"]:
        n_mismatch_bulge = {(mismatch, bulge): 0 for mismatch in range(7) for bulge in range(2)}  # (mismatch, bulge)
        for idx in confusion_matrix_idx[cm_key]:
            n_mismatch_bulge[(mismatch_list[idx], bulge_list[idx])] += 1
        n_mismatch_cm[cm_key] = n_mismatch_bulge

    plot_mismatch_fig.plot_mismatch_bar_graph(
        mismatch_data=n_mismatch_cm,
        titles=["True Negative", "False Positive", "False Negative", "True Positive"],
        save_path=config["paths"]["supplementary"][config["dataset_name"]["dataset_dl"]][config["model_info"]["model_name"]] + f'/training_{config["dataset_name"]["training_dataset"]}_mismatch_analysis_by_cm_1.png'
    )
    
    # Mismatch location and type
    rna_seq_list = dataset["padded_rna_seq"]
    dna_seq_list = dataset["padded_dna_seq"]
    mismatch_type_cm = {}
    for cm_key in ["TN", "FP", "FN", "TP"]:
        mismatch_type_cm[cm_key] = {"rna_seq": [], "dna_seq": []}
        for idx in confusion_matrix_idx[cm_key]:
            mismatch_type_cm[cm_key]["rna_seq"].append(rna_seq_list[idx])
            mismatch_type_cm[cm_key]["dna_seq"].append(dna_seq_list[idx])
    
    plot_mismatch_fig.plot_mismatch_frequency(
        mismatch_data=mismatch_type_cm,
        config=config,
        titles=["True Negative", "False Positive", "False Negative", "True Positive"],
        save_path=config["paths"]["supplementary"][config["dataset_name"]["dataset_dl"]][config["model_info"]["model_name"]] + f'/training_{config["dataset_name"]["training_dataset"]}_mismatch_analysis_by_cm_2.png'
    )


def epigenetic_analysis_by_cm(config: dict, dataset: dict, epigenetic_features_array: np.array, result_dict: dict, type: str) -> None:
    # Epigenetic feature analysis by confusion matrix
    confusion_matrix_idx = result_dict["confusion_matrix_idx"]
    epigenetic_feature_cm = {}
    for cm_key in {"TN", "FP", "FN", "TP"}:
        epigenetic_feature_cm[cm_key] = epigenetic_features_array[confusion_matrix_idx[cm_key]]
    
    plot_epigenetic_fig.plot_epigenetic_line_graph_by_cm(
        config=config,
        epigenetic_data=epigenetic_feature_cm,
        type=type,
        save_path=config["paths"]["supplementary"][config["dataset_name"]["dataset_dl"]][config["model_info"]["model_name"]] + f'/training_{config["dataset_name"]["training_dataset"]}_epigenetic_analysis_by_cm_{type}.png'
    )
