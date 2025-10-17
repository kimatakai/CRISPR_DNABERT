

import os
import numpy as np

import utils.check_set as check_set
import utils.epigenetic_module as epigenetic_module
import utils.sequence_module as sequence_module
import models.data_loader as data_loader
import models.dnabert_module as dnabert_module
import visualization.plot_bert_fig as plot_bert_fig

class IGAnalysisClass:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_list = [
            "Lazzarotto_2020_GUIDE_seq", "Tsai_2015_GUIDE_seq_1", "Tsai_2015_GUIDE_seq_2",
            "Listgarten_2018_GUIDE_seq", "Chen_2017_GUIDE_seq"
        ]
        self.fold = 11
        self.folds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        self.iter = 0
        self.kmer = 3
        self.max_pairseq_len = config["parameters"]["max_pairseq_len"]
        self.base_list = ["A", "C", "T", "G"]
        self.base_dict = {"A":0, "C":1, "T":2, "G":3}
        
        self.hotspot_sgrnas = [
            "GAGTCCGAGCAGAAGAAGAANGG", "GGAATCCCTTCTGCAGCACCNGG", "GGGTGGGGGGAGTTTGCTCCNGG", "GGTGAGTGAGTGTGTGCGTGNGG",
            "GAACACAAAGCATAGACTGCNGG", "GGCACTGCGGCTGGAGGTGGNGG", "GGGAAAGACCCAGCATCCGTNGG",
            "GAGGGTTGCGTTCCTTGAGCNGG", "GAGTCCGAGCAGAAGAAGAANGG", "GCTAGAGTCACAAGTCCCACNGG", "GCTGCTGCTCTGGTTCCTCGNGG", "GGCACAGCGGCATCATTCCGNGG",
            "GGAGTGAGGGAAACGGCCCCNGG", "GGTGAGTGAGTGTGTGCGTGNGG",
            "GATAACTACACCGAGGAAATNGG", "GAGACCCTGCTCAAGGGCCGNGG", "GATTTCCTCCTCGACCACCANGG", "GCACGTGGCCCAGCCTGCTGNGG",
            "GACATTAAAGATAGTCATCTNGG", "GCATTTTCTTCACGGAAACANGG", "GGTACCTATCGATTGTCAGGNGG",
            "GTCACCAATCCTGTCCCTAGNGG", "GTGGTACTGGCCAGCAGCCGNGG", "GCTGCAGAAACAGCAAGCCCNGG", "GGAGAAGGTGGGGGGGTTCCNGG",
            "GATGCTATTCAGGATGCAGTNGG", "GCTGACCCCGCTGGGCAGGCNGG", "GGGATCAGGTGACCCATATTNGG", "GGGGCCACTAGGGACAGGATNGG", "GGGGGGTTCCAGGGCCTGTCNGG",
            "GCTGTGTTTGCGTCTCTCCCNGG", "GAAGCGTGATGACAAAGAGGNGG", "GGGGGTTCCAGGGCCTGTCTNGG", "GGTGACAAGTGTGATCACTTNGG"
        ]
        
        self.database_dir_path = config["paths"]["database_dir"]
        self.analysis_result_dir_path = self.database_dir_path + self.config["paths"]["results"]["base_dir"]
        self.fig_dir_path = self.database_dir_path + self.config["paths"]["results"]["base_dir"] + self.config["paths"]["results"]["figure"]
    
    def extract_test_data(self, dataset_dict: dict):
        test_idx = dataset_dict["test_idx"]
        label_list = [dataset_dict["label"][i] for i in test_idx]
        mismatch_list = [dataset_dict["mismatch"][i] for i in test_idx] 
        sgrna_list = [dataset_dict["sgrna"][i] for i in test_idx] # len = included sgrna type
        rna_seq_list = [dataset_dict["rna_seq"][i] for i in test_idx]
        dna_seq_list = [dataset_dict["dna_seq"][i] for i in test_idx]
        sgrna_set_list = sorted(list(set(sgrna_list)))
        sgrna_set_list_ = []
        seq_data_for_ig = {}
        for _sgrna in sgrna_set_list:
            ontarget_indices = [i for i, (sgrna, mm) in enumerate(zip(sgrna_list, mismatch_list)) if sgrna == _sgrna and mm == 0]
            offtarget_indices = [i for i, (sgrna, _, label) in enumerate(zip(sgrna_list, mismatch_list, label_list)) if sgrna == _sgrna and label == 1]
            sampled_indices = offtarget_indices
            if len(ontarget_indices) != 1 or len(offtarget_indices) <= 1:
                continue
            sgrna_set_list_.append(_sgrna)
            rna_seq_ = [rna_seq_list[i] for i in sampled_indices]
            dna_seq_ = [dna_seq_list[i] for i in sampled_indices]
            seq_data_for_ig[_sgrna] = {
                "rna_seq": rna_seq_,
                "dna_seq": dna_seq_
            }
        return sgrna_set_list_, seq_data_for_ig
    
    def mismatch_pattern_analysis(self, sgrna_set_list_: list, seq_data_for_ig: dict, dataset_name: str, fold: int, iter: int):
        mismatch_array = np.zeros((len(sgrna_set_list_), 12, self.max_pairseq_len)) # A, C, T, G
        for i, _sgrna in enumerate(sgrna_set_list_):
            rna_seqs = seq_data_for_ig[_sgrna]["rna_seq"]
            dna_seqs = seq_data_for_ig[_sgrna]["dna_seq"]
            for r_s, d_s in zip(rna_seqs, dna_seqs):
                padded_r_s, padded_d_s = sequence_module.padding_hyphen_to_seq(r_s, d_s, self.max_pairseq_len)
                for pos, (r_base, d_base) in enumerate(zip(padded_r_s, padded_d_s)):
                    if r_base in self.base_list and d_base in self.base_list:
                        if r_base != d_base:
                            r_idx = self.base_dict[r_base]
                            d_idx = self.base_dict[d_base]
                            mismatch_array[i, r_idx*3 + d_idx, pos] += 1
            mismatch_array[i] = mismatch_array[i] / len(rna_seqs) if len(rna_seqs) > 0 else mismatch_array[i]
        save_path = self.fig_dir_path + f"/BERT_analysis_{dataset_name}" + f"/mismatch_pattern_fold{fold}_iter{iter}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot_bert_fig.plot_mismatch_pattern(mismatch_array, sgrna_set_list_, self.max_pairseq_len, save_path=save_path)
        
    
    def get_data(self):
        all_ig_data_dict = {}
        all_sgrna_set_list = []
        attention_sgrna = []
        for dataset_name in self.dataset_list:
            print(f"Dataset: {dataset_name}")
            self.config["dataset_name"]["dataset_in_cellula"] = dataset_name
            self.config["dataset_name"]["dataset_current"] = dataset_name
            DataLoaderClass = data_loader.DataLoaderClass(self.config)
            DataLoaderClass.load_sgrna_list()
            dataset_dict = DataLoaderClass.load_and_convert_to_dict()
            if dataset_name == "Lazzarotto_2020_GUIDE_seq":
                for f in self.folds:
                    dataset_dict = DataLoaderClass.split_dataset(dataset_dict, f)
                    sgrna_set_list, seq_data_for_ig = self.extract_test_data(dataset_dict)
                    self.mismatch_pattern_analysis(sgrna_set_list, seq_data_for_ig, dataset_name, f, self.iter)
                    token_importance_ = np.load(self.analysis_result_dir_path + f"/BERT_analysis_{dataset_name}" + f"/ig_token_importance_fold{f}_iter{self.iter}.npy", allow_pickle=True).item()
                    for i, sgrna in enumerate(sgrna_set_list):
                        all_ig_data_dict[sgrna] = {
                            "rna_seq": seq_data_for_ig[sgrna]["rna_seq"],
                            "dna_seq": seq_data_for_ig[sgrna]["dna_seq"],
                            "token_importance": token_importance_[sgrna]
                        }
                        all_sgrna_set_list.append(sgrna)
                        # if sgrna in self.hotspot_sgrnas:
                        token_importance_dna_ = token_importance_[sgrna][self.max_pairseq_len - self.kmer + 1 + 2: -1]
                        max_idx = np.argmax(token_importance_dna_)
                        if max_idx in [3, 4, 5]:
                            attention_sgrna.append(1)
                        elif max_idx in [13, 14, 15, 16]:
                            attention_sgrna.append(2)
                        else:
                            attention_sgrna.append(0)
                        # hotspot_base_vec = np.array([0]*3 + [1]*3 + [0]*7 + [1]*4 + [0]*5)
                        # correlation = np.corrcoef(token_importance_dna_, hotspot_base_vec)[0, 1]
                        # attention_sgrna.append(float(correlation))
            else:
                dataset_dict = DataLoaderClass.split_dataset(dataset_dict, self.fold)
                sgrna_set_list, seq_data_for_ig = self.extract_test_data(dataset_dict)
                self.mismatch_pattern_analysis(sgrna_set_list, seq_data_for_ig, dataset_name, self.fold, self.iter)
                token_importance_ = np.load(self.analysis_result_dir_path + f"/BERT_analysis_{dataset_name}" + f"/ig_token_importance_fold{self.fold}_iter{self.iter}.npy", allow_pickle=True).item()
                for i, sgrna in enumerate(sgrna_set_list):
                    all_ig_data_dict[sgrna] = {
                        "rna_seq": seq_data_for_ig[sgrna]["rna_seq"],
                        "dna_seq": seq_data_for_ig[sgrna]["dna_seq"],
                        "token_importance": token_importance_[sgrna]
                    }
                    all_sgrna_set_list.append(sgrna)
                    token_importance_dna_ = token_importance_[sgrna][self.max_pairseq_len - self.kmer + 1 + 2: -1]
                    max_idx = np.argmax(token_importance_dna_)
                    # if sgrna in self.hotspot_sgrnas:
                    if max_idx in [3, 4, 5]:
                        attention_sgrna.append(1)
                    elif max_idx in [13, 14, 15, 16]:
                        attention_sgrna.append(2)
                    else:
                        attention_sgrna.append(0)
                    # hotspot_base_vec = np.array([0]*3 + [1]*3 + [0]*7 + [1]*4 + [0]*5)
                    # correlation = np.corrcoef(token_importance_dna_, hotspot_base_vec)[0, 1]
                    # attention_sgrna.append(float(correlation))
        return all_ig_data_dict, all_sgrna_set_list, attention_sgrna
    

    def ig_attribution_umap(self):
        all_ig_data_dict, all_sgrna_set_list, attention_sgrna = self.get_data()
        all_token_importance = []
        for sgrna in all_sgrna_set_list:
            token_importance = all_ig_data_dict[sgrna]["token_importance"]
            all_token_importance.append(token_importance)
            
        all_token_importance = np.array(all_token_importance)  # shape: (n_sgrna, seq_len)
        all_token_importance = all_token_importance[:, self.max_pairseq_len - self.kmer + 1 + 2: -1]  # remove cls, sep, pad, rnaseq
        print(all_token_importance.shape)
        save_dir_path = self.fig_dir_path + "/BERT_analysis_all_datasets"
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = save_dir_path + "/ig_importance_umap_all_datasets.png"
        plot_bert_fig.plot_ig_importance_umap(all_token_importance, attention_sgrna, save_path=save_path)






