
# DNABERT for CRISPR/Cas9 off-target

This code is developed for: 
[Improved CRISPR/Cas9 Off-target Prediction with DNABERT and Epigenetic Features (Preprint)](https://www.biorxiv.org/content/10.1101/2025.04.16.649101v1)
* This version of the code includes updates and improvements made during the journal revision process.


## Requirements

- Python interpreter == 3.10.13
- numpy == 2.0.2
- pandas == 2.2.3
- scikit-learn == 1.6.1
- torch == 2.5.1
- transformers == 4.48.3


## Baseline models

### GRU-Embed
- **Paper Title:** Generating, modeling and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges
- **GitHub link:** [GRU-Emb](https://github.com/OrensteinLab/CRISPR-Bulge)
- **Reimplementation code:** [./src/models/gru_embed_2024.py](./src/models/gru_embed_2024.py)

### CRISPR-BERT
- **Paper Title:** Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT
- **GitHub link:** [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)
- **Reimplementation code:** [./src/models/crispr_bert_2024.py](./src/models/crispr_bert_2024.py)

### CRISPR-HW
- **Paper Title:** Prediction of CRISPR-Cas9 off-target activities with mismatches and indels based on hybrid neural network
- **GitHub link:** [CRISPR-HW](https://github.com/Yang-k955/CRISPR-HW)
- **Reimplementation code:** [./src/models/crispr_hw_2023.py](./src/models/crispr_hw_2023.py)

### CRISPR-DIPOFF
- **Paper Title:** CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction
- **GitHub link:** [CRISPR-DIPOFF](https://github.com/tzpranto/CRISPR-DIPOFF)
- **Reimplementation code:** [./src/models/crispr_dipoff_2025.py](./src/models/crispr_dipoff_2025.py)

### CrisprBERT
- **Paper Title:** Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding
- **GitHub link:** [CrisprBERT](https://github.com/OSsari/CrisprBERT)
- **Reimplementation code:** [./src/models/crispr_bert_2025.py](./src/models/crispr_bert_2025.py)


## Prerequisites

The following operations need to be performed as part of the prerequisites:

1. **Download the Dataset**
   - Download the dataset from Yaish et al.'s GitHub repository. The file can be downloaded from [this link](https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/files/datasets.zip).
   - Extract the downloaded file.
   - Set the path to the dataset in the `config.yaml` file.
   - **For the Schmid-Burgk 2020 TTISS dataset,** download the raw sequence data from SRA (accession: PRJNA602092).Then, process the data according to the steps outlined in `/data_processing/Schmid-Burgk_2020/implement.md`. This involves downloading FASTQ files, preprocessing, mapping to the reference genome, identifying off-target candidates with SWOffinder, and counting double-strand breaks.

2. **Download Pretrained DNABERT Files**
   - Download the pretrained DNABERT files from HuggingFace. The DNABERT model is available at [this link](https://huggingface.co/zhihan1996/DNA_bert_3).
   - Set the path to the saved DNABERT files in the `config.yaml` file.

3. **Download CRISPR-BERT Configuration Files**
   - Download the CRISPR-BERT model configuration files from [this link](https://github.com/BrokenStringx/CRISPR-BERT/tree/master/weight/bert_weight).
   - Set the path to the saved configuration files in the `config.yaml` file.

4. **Set Other File Paths**
   - Set the paths to other necessary files in `config.yaml` as required.


## Usage

### 1. Fine-tuning DNABERT for Mismatch Prediction Task
To fine-tune DNABERT for the mismatch prediction task, run the following command:
```bash
python3 src/models/pair_finetuning_dnabert.py --pretrain
```

### 2. Data Preprocessing
To preprocess the data for the models, run the following command:
```bash
python3 src/run_preprocess.py --model <model_name> --dataset <dataset_name>
```

### 3. Training and Testing for Off-target Effect Prediction Task
To train and test the model for the off-target effect prediction task, use the following command as an example:
```bash
python3 src/run_model.py --model DNABERT --dataset_in_cellula Lazzarotto_2020_GUIDE_seq --dataset_in_vitro Lazzarotto_2020_CHANGE_seq --fold 0 --iter 0 --train --exe_type transfer
```

#### Arguments:
- `--model`, `-m`: Specifies the deep learning model. Options are `DNABERT`, `DNABERT-No-Pretrained`, `GRU-Embed`, `CRISPR-BERT`, `CRISPR-HW`, `CRISPR-DIPOFF`, `CrisprBERT`.
- `--dataset_in_cellula`, `-dsc`: Specifies the in cellula dataset.
- `--dataset_in_vitro`, `-dsv`: Specifies the in vitro dataset.
- `--foldf`, `-f`: An integer from 0 to 13. This corresponds to the fold number for 10-fold cross-validation. Default is 0.
- `-iter`, `-i`: An integer. Changing this value will change the random seed used in the code, useful for running multiple experiments. Default is 0.
- `--train`: Include this flag to train the model.
- `--test`: Include this flag to test the model.
- `--with_epigenetic`, `-epi`: Include this flag to use epigenetic features.
- `--using_epi_data`, `-uepi`: Specifies the epigenetic data to use (e.g., atac,h3k27ac,h3k4me3).
- `--exe_type`, `-exe`: Specifies the execution type (scratch or transfer).

### 4. Displaying Results from Multiple Models
To display the results from multiple models, use the following command as an example:
```bash
python3 src/run_result.py --models GRU-Embed,CRISPR-BERT,CRISPR-HW,CRISPR-DIPOFF,CrisprBERT,DNABERT --dataset Lazzarotto_2020_GUIDE_seq --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13 --iters 0,1,2,3,4 --exe_type transfer --include_epi_transfer
```

## Contact

Kai Kimata

kkaibioinformatics(at-mark)gmail_domain

October 17 2025

<br>
<br>


# DNABERTを用いたCRISPR/Cas9のオフターゲット予測

本コードは以下の論文向けに実装：
[Improved CRISPR/Cas9 Off-target Prediction with DNABERT and Epigenetic Features](https://www.biorxiv.org/content/10.1101/2025.04.16.649101v1)


## 環境

- Python interpreter == 3.10.13
- numpy == 2.0.2
- pandas == 2.2.3
- scikit-learn == 1.6.1
- torch == 2.5.1
- transformers == 4.48.3


## ベースラインモデル

### GRU-Embed
- **論文タイトル:** Generating, modeling and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges
- **GitHubリンク:** [GRU-Emb](https://github.com/OrensteinLab/CRISPR-Bulge)
- **再実装コード:** `./src/models/gru_embed_2024.py`

### CRISPR-BERT
- **論文タイトル:** Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT
- **GitHubリンク:** [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)
- **再実装コード:** `./src/models/crispr_bert_2024.py`

### CRISPR-HW
- **論文タイトル:** Prediction of CRISPR-Cas9 off-target activities with mismatches and indels based on hybrid neural network
- **GitHubリンク:** [CRISPR-HW](https://github.com/Yang-k955/CRISPR-HW)
- **再実装コード:** `./src/models/crispr_hw_2023.py`

### CRISPR-DIPOFF
- **論文タイトル:** CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction
- **GitHubリンク:** [CRISPR-DIPOFF](https://github.com/tzpranto/CRISPR-DIPOFF)
- **再実装コード:** `./src/models/crispr_dipoff_2025.py`

### CrisprBERT
- **論文タイトル:** Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding
- **GitHubリンク:** [CrisprBERT](https://github.com/OSsari/CrisprBERT)
- **再実装コード:** `./src/models/crispr_bert_2025.py`


## 前準備

前準備として以下の操作を実行：

1. **データセットのダウンロード**
   - Yaish et al.のGitHubリポジトリからデータセットをダウンロードする。ファイルは[本リンク](https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/files/datasets.zip)からダウンロード可能。
   - ダウンロードしたファイルを解凍する。
   - データセットのパスを`config.yaml`ファイルに設定する。
   - `Schmid-Burgk 2020 TTISSデータセットについて`は、SRA（アクセッション: PRJNA602092）から生シーケンスデータをダウンロードする。その後、`/data_processing/Schmid-Burgk_2020/implement.md` に記載されている手順に従ってデータを処理する。これには、FASTQファイルのダウンロード、前処理、リファレンスゲノムへのマッピング、SWOffinderによるオフターゲット候補の同定、二本鎖切断のカウントが含まれる。

2. **事前学習済みDNABERTファイルのダウンロード**
   - HuggingFaceから事前学習済みDNABERTファイルをダウンロードする。DNABERTモデルは[本リンク](https://huggingface.co/zhihan1996/DNA_bert_3)で公開。
   - 保存したDNABERTファイルのパスを`config.yaml`ファイルに設定する。

3. **CRISPR-BERT設定ファイルのダウンロード**
   - CRISPR-BERTモデルの設定ファイルを[本リンク](https://github.com/BrokenStringx/CRISPR-BERT/tree/master/weight/bert_weight)からダウンロード。
   - 保存した設定ファイルのパスを`config.yaml`ファイルに設定する。

4. **その他のファイルパスの設定**
   - 必要なファイルのパスを`config.yaml`ファイルに設定。


## 使い方

### 1. DNABERTのミスマッチ予測タスクでのファインチューニング
DNABERTをミスマッチ予測タスク用にファインチューニング。：
```bash
python3 script/dnabert_pair_ft.py --pretrain
```

### 2. データ前処理
To preprocess the data for the models, run the following command:
```bash
python3 src/run_preprocess.py --model <model_name> --dataset <dataset_name>
```

### 3. オフターゲット効果予測タスクにおける訓練とテスト
オフターゲット効果予測タスクのモデルを訓練およびテスト。コマンド実行例：
```bash
python3 src/run_model.py --model DNABERT --dataset_in_cellula Lazzarotto_2020_GUIDE_seq --dataset_in_vitro Lazzarotto_2020_CHANGE_seq --fold 0 --iter 0 --train --exe_type transfer
```

#### 引数:
- `--model`, `-m`: 深層学習モデルを指定する。選択肢は、`DNABERT`, `DNABERT-No-Pretrained`, `GRU-Embed`, `CRISPR-BERT`, `CRISPR-HW`, `CRISPR-DIPOFF`, `CrisprBERT`。
- `--dataset_in_cellula`, `-dsc`: in cellulaデータセットを指定する。
- `--dataset_in_vitro`, `-dsv`: in vitroデータセットを指定する。
- `--foldf`, `-f`: 0から13の整数。交差検証の交差数に対応する。デフォルトは0。
- `-iter`, `-i`: 整数。この値を変更すると、コード内のランダムシードが変更される。複数回の実験を実施したい場合に使用する。デフォルトは0。
- `--train`: このフラグを含めると、モデルの訓練が実行される。
- `--test`: このフラグを含めると、モデルのテストが実行される。
- `--with_epigenetic`, `-epi`: エピジェネティックな特徴を使用する場合に、このフラグを含める。
- `--using_epi_data`, `-uepi`: 使用するエピジェネティックデータを指定する (例: atac,h3k27ac,h3k4me3)。
- `--exe_type`, `-exe`: 実行タイプを指定する (scratch または transfer)。

### 4. 複数のモデルの結果をまとめて表示
複数のモデルの結果を表示する。コマンド実行例：
```bash
python3 src/run_result.py --models GRU-Embed,CRISPR-BERT,CRISPR-HW,CRISPR-DIPOFF,CrisprBERT,DNABERT --dataset Lazzarotto_2020_GUIDE_seq --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13 --iters 0,1,2,3,4 --exe_type transfer --include_epi_transfer
```


