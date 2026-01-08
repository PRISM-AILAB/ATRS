import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# data/
DATA_PATH = os.path.join(ROOT_DIR, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")

# utils / config
UTILS_PATH = os.path.join(ROOT_DIR, "src")

# 모델 저장 폴더
MODEL_PATH = os.path.join(ROOT_DIR, "model")
SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "save")

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
