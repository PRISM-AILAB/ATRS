import os
import pandas as pd
import numpy as np
import gzip
import json
import yaml
import random
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


# --------- JSONL Utils ---------

def parse(path):
    """Parses a gzipped jsonl file line by line."""
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    """Converts parse results into a DataFrame."""
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


# --------- Parquet / YAML Utils ---------

def load_parquet(fpath):
    """Loads a parquet file using the pyarrow engine."""
    return pd.read_parquet(fpath, engine="pyarrow")


def save_parquet(df: pd.DataFrame, fpath: str):
    """Saves a DataFrame to a parquet file."""
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df.to_parquet(fpath, engine="pyarrow")


def load_yaml(fpath: str) -> dict:
    """Loads a config.yaml file."""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------- Seed Setting ---------

def set_seed(seed: int):
    """Fixes random seeds for numpy, torch, etc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------- Metric Calculation ---------

def get_metrics(preds, trues):
    """
    Takes predictions (preds) and ground truth (trues).
    Returns MSE, RMSE, MAE, and MAPE (%).
    """
    preds = preds.squeeze()
    trues = trues.squeeze()

    # Convert tensors to numpy if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    
    return mse, rmse, mae, mape