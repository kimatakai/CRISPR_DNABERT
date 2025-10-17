
import os
import re
import tqdm
import numpy as np
from scipy.stats import mannwhitneyu

import utils.check_set as check_set
import utils.epigenetic_module as epigenetic_module
import models.data_loader as data_loader
import visualization.plot_epigenetic_fig as plot_epigenetic_fig




class EDAAnalysis:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_info"]["model_name"]
        self.DataLoaderClass = data_loader.DataLoaderClass(self.config)
    
    def load_dataset_dict(self) -> dict:
        self.DataLoaderClass.load_sgrna_list()
        dataset_dict = self.DataLoaderClass.load_and_convert_to_dict()
        return dataset_dict
    
    def aggregate_index(self, dataset_dict: dict) -> dict:
        index_dict = {}
        for fold in tqdm.tqdm(self.config["folds"], total=len(self.config["folds"]), desc="Aggregating indices for folds"):
            _dataset_dict = self.DataLoaderClass.split_dataset(dataset_dict, fold)
            index_dict[fold] = _dataset_dict["test_idx"]
        return index_dict
    
    def return_probability_array(self) -> dict:
        probability_array = {fold: {iter: None for iter in self.config["iters"]} for fold in self.config["folds"]}
        for path in tqdm.tqdm(self.config["probability_paths"][self.model_name], total=len(self.config["probability_paths"][self.model_name]), desc="Loading probability arrays"):
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            match = re.search(r'fold(\d+)_iter(\d+)', path)
            fold = int(match.group(1))
            iter = int(match.group(2))
            prob_array = np.load(path, allow_pickle=True)
            probability_array[fold][iter] = prob_array
        return probability_array
    
    def return_confusion_entry_idx(self, index_dict: dict, true_label: np.ndarray, proba_dict: dict) -> dict: # {fold: {iter: {tn: [], fp: [], fn: [], tp: []}}}
        # index_dict: {fold: [idx, idx, ...], ...}, true_label: [0, 1, 0, 0, 1, ...], proba_dict: {fold: {iter: [0.1, 0.9, 0.2, ...], ...}, ...}
        index_table_corresponding_to_inside_index = {fold: {idx: i for i, idx in enumerate(index_dict[fold])} for fold in self.config["folds"]}
        confusion_entry_idx = {fold: {iter: {"tn": [], "fp": [], "fn": [], "tp": []} for iter in self.config["iters"]} for fold in self.config["folds"]}
        for fold in tqdm.tqdm(self.config["folds"], total=len(self.config["folds"]), desc="Building confusion entry index"):
            for iter in self.config["iters"]:
                for idx in index_dict[fold]:
                    true = true_label[idx]
                    proba = proba_dict[fold][iter][index_table_corresponding_to_inside_index[fold][idx]]
                    if true == 0 and proba < 0.5:
                        confusion_entry_idx[fold][iter]["tn"].append(idx)
                    elif true == 0 and proba >= 0.5:
                        confusion_entry_idx[fold][iter]["fp"].append(idx)
                    elif true == 1 and proba < 0.5:
                        confusion_entry_idx[fold][iter]["fn"].append(idx)
                    elif true == 1 and proba >= 0.5:
                        confusion_entry_idx[fold][iter]["tp"].append(idx)
        return confusion_entry_idx


class EdaEpigeneticAnalysis(EDAAnalysis):
    def __init__(self, config: dict):
        super().__init__(config)
        # Dataset
        self.dataset_dict = self.load_dataset_dict()
        self.n_samples = len(self.dataset_dict["sgrna"])
        self.index_dict = self.aggregate_index(self.dataset_dict)

    def return_epigenetic_data(self, type_of_data: str) -> None:
        for path in self.config["paths"]["epigenetic"][type_of_data]["npz_current"]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        epigenetic_array = epigenetic_module.load_npz(self.config["paths"]["epigenetic"][type_of_data]["npz_current"])
        if epigenetic_array.shape != (self.n_samples, self.window_size*2 // self.bin_size):
            raise ValueError(f"Shape mismatch: expected ({self.n_samples}, {self.window_size*2 // self.bin_size}), but got {epigenetic_array.shape}")
        return epigenetic_array

    def load_epigenetic_data(self, type_of_data: str):
        # Epigenetic information
        self.window_size = self.config["parameters"]["window_size"][type_of_data]
        self.bin_size = self.config["parameters"]["bin_size"][type_of_data]
        # Path information
        self.SetPathEpigeneticClass = check_set.SetPathsEpigenetic(self.config)
        self.config = self.SetPathEpigeneticClass.set_epigenetic_path(type_of_data = type_of_data)
        self.SetPathEpigeneticClass.check_path(type_of_data = type_of_data)
        self.epigenetic_array = self.return_epigenetic_data(type_of_data = type_of_data)

    def analysis_atac_confusion_mm(self):
        # Load epigenetic data
        self.load_epigenetic_data(type_of_data = "atac")
        heatmap_path = self.config["fig_dir_path"] + f'/atac_confusion_mismatch_heatmap_{self.config["dataset_name"]["dataset_current"]}.png'
        self.analysis_common_confusion_mm(heatmap_path = heatmap_path)
    
    def analysis_h3k4me3_confusion_mm(self):
        # Load epigenetic data
        self.load_epigenetic_data(type_of_data = "h3k4me3")
        heatmap_path = self.config["fig_dir_path"] + f'/h3k4me3_confusion_mismatch_heatmap_{self.config["dataset_name"]["dataset_current"]}.png'
        self.analysis_common_confusion_mm(heatmap_path = heatmap_path)
    
    def analysis_h3k27ac_confusion_mm(self):
        # Load epigenetic data
        self.load_epigenetic_data(type_of_data = "h3k27ac")
        heatmap_path = self.config["fig_dir_path"] + f'/h3k27ac_confusion_mismatch_heatmap_{self.config["dataset_name"]["dataset_current"]}.png'
        self.analysis_common_confusion_mm(heatmap_path = heatmap_path)

    def analysis_common_confusion_mm(self, heatmap_path: str = None):
        # Load Probability data
        probability_array_dict = self.return_probability_array()
        
        # Load True label
        true_label_array = np.array(self.dataset_dict["label"])
        print(f"True label shape: {true_label_array.shape}")

        # Load mismatch data
        mismatch_array = np.array(self.dataset_dict["mismatch"])
        print(f"Mismatch array shape: {mismatch_array.shape}")
        
        # Get confusion matrix entry indices
        confusion_entry_idx = self.return_confusion_entry_idx(self.index_dict, true_label_array, probability_array_dict) # {fold: {iter: {tn: [], fp: [], fn: [], tp: []}}}
        
        # Confusion and mismatch index
        n_mismatch = [1, 2, 3, 4, 5, 6]
        confusion_entries = ["tn", "fp", "fn", "tp"]
        confusion_mismatch_idx = {entry: {mm: [] for mm in n_mismatch} for entry in confusion_entries}
        for fold in tqdm.tqdm(self.config["folds"], total=len(self.config["folds"]), desc="Building confusion mismatch index"):
            for iter in self.config["iters"]:
                for entry in confusion_entries:
                    for idx in confusion_entry_idx[fold][iter][entry]:
                        mm = mismatch_array[idx]
                        if mm in n_mismatch:
                            confusion_mismatch_idx[entry][mm].append(idx)
        for entry in confusion_entries:
            for mm in n_mismatch:
                print(f"{entry} mm{mm}: {len(confusion_mismatch_idx[entry][mm])} samples")
        
        # Confusion and mismatch -> Epigenetic data
        confusion_mismatch_epigenetic = {entry: {mm: None for mm in n_mismatch} for entry in confusion_entries}
        for entry in confusion_entries:
            for mm in n_mismatch:
                idx_list = confusion_mismatch_idx[entry][mm]
                if len(idx_list) > 100:
                    confusion_mismatch_epigenetic[entry][mm] = np.nanmean(self.epigenetic_array[idx_list], axis=0)
                else:
                    confusion_mismatch_epigenetic[entry][mm] = np.array([])

        # Plot
        plot_epigenetic_fig.plot_confusion_mismatch_heatmap(
            config = self.config, 
            confusion_mismatch_epigenetic = confusion_mismatch_epigenetic, 
            window_size=self.window_size, bin_size=self.bin_size,
            save_path = heatmap_path)
        