
# DNABERT for CRISPR/Cas9 off-target

This code is developed for: 
[Improved CRISPR/Cas9 Off-target Prediction with DNABERT and Epigenetic Features](https://www.biorxiv.org/content/10.1101/2025.04.16.649101v1)


## Requirements

- Python interpreter == 3.10.13
- numpy == 2.0.2
- pandas == 2.2.3
- scikit-learn == 1.6.1
- torch == 2.5.1
- transformers == 4.48.3


## Baseline models

### gru-embed
- **Paper Title:** Generating, modeling and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges
- **GitHub link:** [GRU-Emb](https://github.com/OrensteinLab/CRISPR-Bulge)
- **Reimplementation code:** `./script/yaish_et_al/gru_embed_module.py`

### crispr-bert
- **Paper Title:** Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT
- **GitHub link:** [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)
- **Reimplementation code:** `./script/luo_et_al/crispr_bert_module.py`

### crispr-hw
- **Paper Title:** Prediction of CRISPR-Cas9 off-target activities with mismatches and indels based on hybrid neural network
- **GitHub link:** [CRISPR-HW](https://github.com/Yang-k955/CRISPR-HW)
- **Reimplementation code:** `./script/yang_et_al/crispr_hw_module.py`

### crispr-dipoff
- **Paper Title:** CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction
- **GitHub link:** [CRISPR-DIPOFF](https://github.com/tzpranto/CRISPR-DIPOFF)
- **Reimplementation code:** `./script/toufikuzzaman_et_al/crispr_dipoff_module.py`

### crispr-bert-2025
- **Paper Title:** Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding
- **GitHub link:** [CrisprBERT](https://github.com/OSsari/CrisprBERT)
- **Reimplementation code:** `./script/sari_et_al/crispr_bert_2025_module.py`


## Prerequisites

The following operations need to be performed as part of the prerequisites:

1. **Download the Dataset**
   - Download the dataset from Yaish et al.'s GitHub repository. The file can be downloaded from [this link](https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/files/datasets.zip).
   - Extract the downloaded file.
   - Set the path to the dataset in the `yaish_et_al_data_path` variable in `./script/config.py`.

2. **Download Pretrained DNABERT Files**
   - Download the pretrained DNABERT files from HuggingFace. The DNABERT model is available at [this link](https://huggingface.co/zhihan1996/DNA_bert_3).
   - Set the path to the saved DNABERT files in the `dnabert_pretrained_model_path` variable in `./script/config.py`.

3. **Download CRISPR-BERT Configuration Files**
   - Download the CRISPR-BERT model configuration files from [this link](https://github.com/BrokenStringx/CRISPR-BERT/tree/master/weight/bert_weight).
   - Set the path to the saved configuration files in the `crispr_bert_architecture_path` variable in `./script/config.py`.

4. **Set Other File Paths**
   - Set the paths to other necessary files in `./script/config.py` as required.


## Usage

### 1. Fine-tuning DNABERT for Mismatch Prediction Task
To fine-tune DNABERT for the mismatch prediction task, run the following command:
```bash
python3 script/dnabert_pair_ft.py --pretrain
```

### 2. Training and Testing for Off-target Effect Prediction Task
To train and test the model for the off-target effect prediction task, use the following command as an example:
```bash
python3 main.py -f 0 -e 0 -d guideseq -m dnabert --train
```

#### Arguments:
- `-f`, `--fold`: An integer from 0 to 9. This corresponds to the fold number for 10-fold cross-validation. Default is 0.
- `-e`, `--exp_id`: An integer. Changing this value will change the random seed used in the code, useful for running multiple experiments. Default is 0.
- `-d`, `--datatype`: Specifies the dataset. Options are `changeseq`, `guideseq`, `transfer`. Note that `transfer` can only be specified after training with `changeseq`. Default is `changeseq`.
- `-m`, `--model`: Specifies the deep learning model. Options are `gru-embed`, `crispr-bert`, `crispr-hw`, `crispr-dipoff`, `crispr-bert-2025`, `dnabert`, `dnabert-epi`. Note that `dnabert-epi` can only be specified after training with `dnabert`. Default is `dnabert`.
- `--train`: Include this flag to train the model.
- `--test`: Include this flag to test the model.

### 3. Displaying Results from Multiple Models
To display the results from multiple models, use the following command as an example:
```bash
python3 result.py -f 0,1,2,3,4,5,6,7,8,9 -e 0,1,2,3,4 -d guideseq -m gru-embed,crispr-bert,crispr-hw,crispr-dipoff,crispr-bert-2025,dnabert,dnabert-epi --ensemble --excel
```

#### Arguments:
- `-f`, `--fold`: An integer from 0 to 9. This corresponds to the fold number for 10-fold cross-validation. Default is `0,1,2,3,4,5,6,7,8,9`. Enter the folds to be included in the results, separated by commas.
- `-e`, `--exp_id`: An integer. Changing this value will change the random seed used in the code, useful for running multiple experiments. Default is `0,1,2,3,4`. Enter the experiment IDs to be included in the results, separated by commas.
- `-d`, `--datatype`: Specifies the dataset. Options are `changeseq`, `guideseq`, `transfer`. Note that `transfer` can only be specified after training with `changeseq`.
- `-m`, `--model`: Specifies the deep learning model. Options are `gru-embed`, `crispr-bert`, `crispr-hw`, `crispr-dipoff`, `crispr-bert-2025`, `dnabert`, `dnabert-epi`. Note that `dnabert-epi` can only be specified after training with `dnabert`. Enter the models to be included in the results, separated by commas. Default is `gru-embed,crispr-bert,crispr-hw,crispr-dipoff,crispr-bert-2025,dnabert`.
- `--ensemble`: Include this flag to include ensemble results.
- `--show`: Include this flag to display results in the terminal.
- `--excel`: Include this flag to save the results to an Excel file.
- `--boxplot`: Include this flag to save a box plot of the results.
- `--curve`: Include this flag to save the ROC curve and PR curve of the results.


## Explanation slide

[Explanation slide of this research](explanation_slide_EN.pdf)

## Contact

Kai Kimata

kkaibioinformatics(at-mark)gmail_domain

April 07 2025

<br>
<br>


# DNABERTを用いたCRISPR/Cas9のオフターゲット予測

本コードは以下の論文にて開発：
[Improved CRISPR/Cas9 Off-target Prediction with DNABERT and Epigenetic Features](https://www.biorxiv.org/content/10.1101/2025.04.16.649101v1)


## 環境

- Python interpreter == 3.10.13
- numpy == 2.0.2
- pandas == 2.2.3
- scikit-learn == 1.6.1
- torch == 2.5.1
- transformers == 4.48.3


## ベースラインモデル

### gru-embed
- **論文タイトル:** Generating, modeling and evaluating a large-scale set of CRISPR/Cas9 off-target sites with bulges
- **GitHubリンク:** [GRU-Emb](https://github.com/OrensteinLab/CRISPR-Bulge)
- **再実装コード:** `./script/yaish_et_al/gru_embed_module.py`

### crispr-bert
- **論文タイトル:** Interpretable CRISPR/Cas9 off-target activities with mismatches and indels prediction using BERT
- **GitHubリンク:** [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)
- **再実装コード:** `./script/luo_et_al/crispr_bert_module.py`

### crispr-hw
- **論文タイトル:** Prediction of CRISPR-Cas9 off-target activities with mismatches and indels based on hybrid neural network
- **GitHubリンク:** [CRISPR-HW](https://github.com/Yang-k955/CRISPR-HW)
- **再実装コード:** `./script/yang_et_al/crispr_hw_module.py`

### crispr-dipoff
- **論文タイトル:** CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction
- **GitHubリンク:** [CRISPR-DIPOFF](https://github.com/tzpranto/CRISPR-DIPOFF)
- **再実装コード:** `./script/toufikuzzaman_et_al/crispr_dipoff_module.py`

### crispr-bert-2025
- **論文タイトル:** Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding
- **GitHubリンク:** [CrisprBERT](https://github.com/OSsari/CrisprBERT)
- **再実装コード:** `./script/sari_et_al/crispr_bert_2025_module.py`


## 前準備

前準備として以下の操作を実行：

1. **データセットのダウンロード**
   - Yaish et al.のGitHubリポジトリからデータセットをダウンロードする。ファイルは[本リンク](https://github.com/OrensteinLab/CRISPR-Bulge/blob/main/files/datasets.zip)からダウンロード可能。
   - ダウンロードしたファイルを解凍する。
   - データセットのパスを`yaish_et_al_data_path`変数に設定する。設定場所は`./script/config.py`。

2. **事前学習済みDNABERTファイルのダウンロード**
   - HuggingFaceから事前学習済みDNABERTファイルをダウンロードする。DNABERTモデルは[本リンク](https://huggingface.co/zhihan1996/DNA_bert_3)で公開。
   - 保存したDNABERTファイルのパスを`dnabert_pretrained_model_path`変数に設定する。設定場所は`./script/config.py`。

3. **CRISPR-BERT設定ファイルのダウンロード**
   - CRISPR-BERTモデルの設定ファイルを[本リンク](https://github.com/BrokenStringx/CRISPR-BERT/tree/master/weight/bert_weight)からダウンロード。
   - 保存した設定ファイルのパスを`crispr_bert_architecture_path`変数に設定する。設定場所は`./script/config.py`。

4. **その他のファイルパスの設定**
   - 必要なファイルのパスを`./script/config.py`に設定。


## 使い方

### 1. DNABERTのミスマッチ予測タスクでのファインチューニング
DNABERTをミスマッチ予測タスク用にファインチューニング。：
```bash
python3 script/dnabert_pair_ft.py --pretrain
```

### 2. オフターゲット効果予測タスクにおける訓練とテスト
オフターゲット効果予測タスクのモデルを訓練およびテスト。コマンド実行例：
```bash
python3 main.py -f 0 -e 0 -d guideseq -m dnabert --train
```

#### 引数:
- `-f`, `--fold`: 0から9の整数。10分割交差検証の交差数に対応する。デフォルトは0。
- `-e`, `--exp_id`: 整数。この値を変更すると、コード内のランダムシードが変更される。複数回の実験を実施したい場合に使用する。デフォルトは0。
- `-d`, `--datatype`: データセットを指定する。選択肢は`changeseq`, `guideseq`, `transfer`の3つ。ただし、`transfer`は`changeseq`で訓練を実行した後に指定できる。デフォルトは`changeseq`。
- `-m`, `--model`: 深層学習モデルを指定する。選択肢は`gru-embed`, `crispr-bert`, `crispr-hw`, `crispr-dipoff`, `crispr-bert-2025`, `dnabert`, `dnabert-epi`。ただし、`dnabert-epi`は`dnabert`で訓練した後に指定できる。デフォルトは`dnabert`。
- `--train`: このフラグを含めると、モデルの訓練が実行される。
- `--test`: このフラグを含めると、モデルのテストが実行される。

### 3. 複数のモデルの結果をまとめて表示
複数のモデルの結果を表示する。コマンド実行例：
```bash
python3 result.py -f 0,1,2,3,4,5,6,7,8,9 -e 0,1,2,3,4 -d guideseq -m gru-embed,crispr-bert,crispr-hw,crispr-dipoff,crispr-bert-2025,dnabert,dnabert-epi --ensemble --excel
```

#### 引数:
- `-f`, `--fold`: 0から9の整数。10分割交差検証の交差数に対応する。デフォルトは`0,1,2,3,4,5,6,7,8,9`。結果に含めたいfoldをカンマ区切りで入力する。スペースは含めないこと。
- `-e`, `--exp_id`: 整数。この値を変更すると、コード内のランダムシードが変更される。複数回の実験を実施したい場合に使用する。デフォルトは`0,1,2,3,4`。結果に含めたい結果をカンマ区切りで入力する。スペースは含めないこと。
- `-d`, `--datatype`: データセットを指定する。選択肢は`changeseq`, `guideseq`, `transfer`。ただし、`transfer`は`changeseq`で訓練を実行した後に指定できる。
- `-m`, `--model`: 深層学習モデルを指定する。選択肢は`gru-embed`, `crispr-bert`, `crispr-hw`, `crispr-dipoff`, `crispr-bert-2025`, `dnabert`, `dnabert-epi`。ただし、`dnabert-epi`は`dnabert`で訓練した後に指定できる。結果に含めたいモデルをカンマ区切りで入力する。スペースは含めないこと。デフォルトは`gru-embed,crispr-bert,crispr-hw,crispr-dipoff,crispr-bert-2025,dnabert`。
- `--ensemble`: このフラグを含めると、アンサンブルした結果が含められる。
- `--show`: このフラグを含めると、ターミナル上に結果が表示される。
- `--excel`: このフラグを含めると、結果がExcelファイルに保存される。
- `--boxplot`: このフラグを含めると、結果の箱ひげ図が保存される。
- `--curve`: このフラグを含めると、結果のROCカーブとPRカーブが保存される。

## 説明スライド

[本研究の説明スライド](explanation_slide_JP.pdf)
