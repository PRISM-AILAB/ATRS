import pandas as pd, numpy as np
import gzip, json, yaml, random, torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


# --------- JSONL용 유틸 ---------
def parse(path):
    """gzip으로 압축된 jsonl 파일 한 줄씩 파싱"""
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    """parse 결과를 DataFrame으로 변환"""
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


# --------- Parquet / YAML 유틸 ---------
def load_parquet(fpath):
    """pyarrow 엔진으로 parquet 파일 로드"""
    return pd.read_parquet(fpath, engine="pyarrow")


def save_parquet(df: pd.DataFrame, fpath: str):
    """DataFrame을 parquet 파일로 저장"""
    df.to_parquet(fpath, engine="pyarrow")


def load_yaml(fpath: str) -> dict:
    """config.yaml 파일 로드"""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------- Seed 고정 ---------
def set_seed(seed: int):
    """numpy, torch 등 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------- 평가 지표 계산 ---------
def get_metrics(preds, trues):
    """
    예측값(preds)과 실제값(trues)을 받아
    MSE, RMSE, MAE, MAPE(%)를 반환.
    """
    preds = preds.squeeze()
    trues = trues.squeeze()

    # 텐서일 경우 numpy로 변환
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    return mse, rmse, mae, mape

