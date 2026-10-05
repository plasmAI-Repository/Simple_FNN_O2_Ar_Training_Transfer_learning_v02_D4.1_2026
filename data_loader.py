import torch
import json
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MergedDataset(Dataset):
    def __init__(self, csv_file, statsfile, x_columns=None, sep=';'):
        df = pd.read_csv(csv_file, sep=sep)
        with open(statsfile, 'r') as f:
            stats = json.load(f)

        # Min-max normalization for columns present in stats
        for column in df.columns:
            if column in stats:
                min_val = stats[column]['min']
                max_val = stats[column]['max']
                diff = max_val - min_val if max_val != min_val else 1.0
                df[column] = (df[column] - min_val) / diff

        if x_columns is None:
            x_columns = ['Power', 'Pressure']

        self.X_columns = [col for col in x_columns if col in df.columns]
        self.y_columns = [col for col in df.columns if col not in self.X_columns]

        self.X = df[self.X_columns]
        self.y = df[self.y_columns]
        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


class MergedDatasetTest(Dataset):
    def __init__(self, csv_file, stats_file, x_columns=None, sep=';'):
        df = pd.read_csv(csv_file, sep=sep)
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        for column in df.columns:
            if column in stats:
                min_val = stats[column]['min']
                max_val = stats[column]['max']
                diff = max_val - min_val if max_val != min_val else 1.0
                df[column] = (df[column] - min_val) / diff

        if x_columns is None:
            x_columns = ['Power', 'Pressure']

        self.X_columns = [col for col in x_columns if col in df.columns]
        self.y_columns = [col for col in df.columns if col not in self.X_columns]

        self.X = df[self.X_columns]
        self.y = df[self.y_columns]
        self.df = df

    def __getitem__(self, index):
        X = self.X.iloc[index, :].values
        y = self.y.iloc[index, :].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]


def unscale_min_max(data, column_names, scaling_info):
    """
    Reverses the min-max normalization for predicted data arrays.
    """
    unscaled_data = np.copy(data)
    num_columns = data.shape[1]
    for i, column in enumerate(column_names):
        if i >= num_columns:
            print(f"Warning: Index {i} is out of bounds for data with {num_columns} columns.")
            break
        if column in scaling_info:
            min_ = scaling_info[column]['min']
            max_ = scaling_info[column]['max']
            unscaled_data[:, i] = (data[:, i] * (max_ - min_)) + min_
    return unscaled_data