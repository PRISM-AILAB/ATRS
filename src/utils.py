import os
import random

import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def load_parquet(fpath):
    """Read a parquet file into a DataFrame (pyarrow engine)."""
    return pd.read_parquet(fpath, engine="pyarrow")


def save_parquet(df: pd.DataFrame, fpath: str):
    """Write a DataFrame to parquet, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    df.to_parquet(fpath, engine="pyarrow")


def load_yaml(fpath: str) -> dict:
    """Load a YAML config file as a dict."""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Fix random seeds across random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_metrics(preds, trues):
    """Calculate regression metrics: MSE, RMSE, MAE, MAPE."""
    preds = preds.squeeze()
    trues = trues.squeeze()
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    return mse, rmse, mae, mape
