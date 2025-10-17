import yaml
import pandas as pd


def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_csv_dataset(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, sep=",", header=0, index_col=None)


def load_csv_list(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def filter_dataset_by_sgrna(dataset_df: pd.DataFrame, sgrna_list: list) -> pd.DataFrame:
    filtered_dataset_df = dataset_df[dataset_df["sgRNA"].isin(sgrna_list)]
    return filtered_dataset_df.reset_index(drop=True)


def filter_OTS_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    filtered_dataset_df = dataset_df[dataset_df["reads"] > 0]
    return filtered_dataset_df.reset_index(drop=True)


def filter_chromosome(dataset_df: pd.DataFrame, chrom_size: dict) -> pd.DataFrame:
    filtered_dataset_df = dataset_df[dataset_df["chrom"].isin(chrom_size.keys())]
    return filtered_dataset_df.reset_index(drop=True)
