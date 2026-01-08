# src/data.py

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

from src.path import RAW_PATH, PROCESSED_PATH
from src.utils import save_parquet
from src.ate import ATEExtractor


# ============================================================
# Artifacts
# ============================================================

@dataclass
class W2VArtifacts:
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder

    user_tokenizer: Tokenizer
    item_tokenizer: Tokenizer

    user_embedding_matrix: np.ndarray
    item_embedding_matrix: np.ndarray

    user_vocab_size: int
    item_vocab_size: int

    user_maxlen: int
    item_maxlen: int


# ============================================================
# gz json load
# ============================================================

def load_json_gz(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        df = pd.read_json(path, compression="gzip", lines=True)
        if len(df) > 0:
            return df
    except Exception:
        pass

    df = pd.read_json(path, compression="gzip", lines=False)
    return df


# ============================================================
# W2V + tokenize
# ============================================================

def _fit_w2v(sentences, vector_size: int, window: int, min_count: int, workers: int, seed: int) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
    )


def _tokenize_w2v(
    train_texts,
    test_texts,
    *,
    w2v_model: Word2Vec,
    max_words: int,
    maxlen: int,
    vector_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Tokenizer]:

    def join_tokens(x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        return "" if pd.isna(x) else str(x)

    train_str = [join_tokens(x) for x in train_texts]
    test_str = [join_tokens(x) for x in test_texts]

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_str)

    train_seq = tokenizer.texts_to_sequences(train_str)
    test_seq = tokenizer.texts_to_sequences(test_str)

    train_seq = pad_sequences(train_seq, maxlen=maxlen, padding="post")
    test_seq = pad_sequences(test_seq, maxlen=maxlen, padding="post")

    word_index = tokenizer.word_index
    vocab_size = min(max_words, len(word_index))

    embedding_matrix = np.zeros((vocab_size + 1, vector_size), dtype=np.float32)
    for word, idx in word_index.items():
        if idx > max_words:
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]

    return train_seq, test_seq, embedding_matrix, vocab_size, tokenizer


# ============================================================
# Aspect aggregation (전체 df 기준)
# ============================================================

def _ensure_token_list(x: Any) -> List[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                if "aspect" in v and isinstance(v["aspect"], str):
                    out.append(v["aspect"])
                elif "term" in v and isinstance(v["term"], str):
                    out.append(v["term"])
            else:
                out.append(str(v))
        return out
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _aggregate_aspects_by_id(
    df: pd.DataFrame,
    id_col: str,
    aspect_col: str,
    *,
    unique: bool = True,
) -> Dict[Any, List[str]]:
    tmp = df[[id_col, aspect_col]].copy()
    tmp[aspect_col] = tmp[aspect_col].apply(_ensure_token_list)

    grouped = tmp.groupby(id_col)[aspect_col].apply(list)  # List[List[str]]

    mapping: Dict[Any, List[str]] = {}
    for _id, lists in grouped.items():
        flat = [tok for sub in lists for tok in sub]
        if unique:
            seen = set()
            uniq = []
            for t in flat:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            mapping[_id] = uniq
        else:
            mapping[_id] = flat
    return mapping


def _apply_aspect_mapping(
    df: pd.DataFrame,
    id_col: str,
    mapping: Dict[Any, List[str]],
    out_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[id_col].map(mapping)
    df[out_col] = df[out_col].apply(lambda x: x if isinstance(x, list) else [])
    return df


# ============================================================
# DataLoader
# ============================================================

class DataLoader:
    """
    (원본 코드 스타일 유지)
    - raw: .json.gz
    - ATE (리뷰 단위) -> aspect 컬럼 생성
    - 전체 df 기준으로 user/item별 aspect 모아서 user_aspect_set/item_aspect_set 생성
    - 그 다음 train_test_split
    - df_train/df_test에서 user_aspect_set/item_aspect_set 그대로 W2V/Tokenizer 입력
    """

    def __init__(
        self,
        fname: str,
        *,
        raw_ext: str = "json.gz",
        test_size: float = 0.2,
        random_state: int = 42,

        # columns
        user_id_col: str = "user_id",
        item_id_col: str = "parent_asin",
        rating_col: str = "rating",

        # ATE
        run_ate: bool = True,
        text_col: str = "pyabsa_clean_text",
        raw_text_col: str = "review_text",
        aspect_col: str = "aspect",
        ate_result_dir: str = "output_results",
        ate_device: str = "cuda:0",

        # aggregation output cols
        user_aspect_col: str = "user_aspect_set",
        item_aspect_col: str = "item_aspect_set",
        agg_unique: bool = True,

        # W2V
        w2v_vector_size: int = 300,
        w2v_window: int = 5,
        w2v_min_count: int = 1,
        w2v_workers: int = 4,

        # Tokenizer
        max_words: int = 42904,
        user_maxlen: int = 45,
        item_maxlen: int = 602,
    ):
        self.fname = fname
        self.raw_ext = raw_ext
        self.test_size = test_size
        self.random_state = random_state

        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col

        self.run_ate = run_ate
        self.text_col = text_col
        self.aspect_col = aspect_col
        self.ate_result_dir = ate_result_dir
        self.ate_device = ate_device

        self.user_aspect_col = user_aspect_col
        self.item_aspect_col = item_aspect_col
        self.agg_unique = agg_unique

        self.w2v_vector_size = w2v_vector_size
        self.w2v_window = w2v_window
        self.w2v_min_count = w2v_min_count
        self.w2v_workers = w2v_workers

        self.max_words = max_words
        self.user_maxlen = user_maxlen
        self.item_maxlen = item_maxlen
        
        self.use_ate = run_ate
        self.text_col = text_col
        self.raw_text_col = raw_text_col

        # 1) raw load
        df_raw = self._load_raw()

       # 2) required columns
        for c in [self.user_id_col, self.item_id_col, self.rating_col]:
            if c not in df_raw.columns:
                raise KeyError(f"Required column not found: {c}")

        # 3) ATE (aspect column 기준으로 실행 여부 판단)
        if self.use_ate and self.aspect_col not in df_raw.columns:
            print(
                f"[DataLoader] `{self.aspect_col}` not found. "
                f"Running ATE from `{self.raw_text_col}`..."
            )

            ate = ATEExtractor(
                device=self.ate_device
            )

            df_raw = ate.run(
                df=df_raw,
                text_col=self.raw_text_col,   # 원본 텍스트
                aspect_col=self.aspect_col,   # 보통 "aspect"
            )
        else:
            print(
                f"[DataLoader] `{self.aspect_col}` exists. "
                f"Skipping ATE."
            )

        # 3) ATE
        df_feat = self._run_ate_if_needed(df_raw)

        # 4) aggregation 먼저 (전체 df 기준) 
        df_feat = self._build_review_sets_global(df_feat)

        # 5) encode IDs
        df_idx = self._encode_ids(df_feat)

        # 6) split (train/test)
        self.train, self.test = self._data_split(df_idx)

        # 7) W2V + sequences (train/test)
        self.artifacts, self.seqs = self._make_w2v_and_sequences(self.train, self.test)

        # 8) save processed
        self._save_processed()

    def _load_raw(self) -> pd.DataFrame:
        raw_path = os.path.join(RAW_PATH, f"{self.fname}.{self.raw_ext}")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        ext = self.raw_ext.lower()
        if ext in ["json.gz", "jsonl.gz"]:
            df = load_json_gz(raw_path)
        elif ext in ["parquet"]:
            df = pd.read_parquet(raw_path)
        else:
            raise ValueError(f"Unsupported raw_ext: {self.raw_ext}")

        # ==================================================
        # Column mapping (raw schema → code schema)
        # ==================================================
        COLUMN_MAP = {
            "review_text": "reviewText",
            "review_stars": "overall",
            "rest_id": "parent_asin",
        }

        for raw_col, std_col in COLUMN_MAP.items():
            if raw_col in df.columns and std_col not in df.columns:
                df[std_col] = df[raw_col]
        # rating 통일
        if "rating" not in df.columns:
            if "overall" in df.columns:
                df["rating"] = df["overall"]
            elif "review_stars" in df.columns:
                df["rating"] = df["review_stars"]

        print(f"[DataLoader] Raw loaded: {df.shape} from {raw_path}")
        return df


    def _run_ate_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not self.run_ate:
            print("[DataLoader] Skip ATE (run_ate=False).")
            return df

        if self.aspect_col in df.columns:
            print("[DataLoader] Skip ATE (aspect column already exists).")
            return df

        ate = ATEExtractor(
            checkpoint="english",
            auto_device=False,
            device=self.ate_device,
            cal_perplexity=False,
            result_dir=self.ate_result_dir,
        )

        df = ate.run(
            df=df,
            text_col=self.text_col,
            aspect_col=self.aspect_col,
            print_result=False,
            pred_sentiment=False,
            save_result=True,
        )
        print(f"[DataLoader] ATE done. aspect added: {df.shape}")
        return df

    def _build_review_sets_global(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.aspect_col not in df.columns:
            raise KeyError(f"`{self.aspect_col}` not found. ATE 결과가 필요해.")

        user_map = _aggregate_aspects_by_id(
            df, id_col=self.user_id_col, aspect_col=self.aspect_col, unique=self.agg_unique
        )
        item_map = _aggregate_aspects_by_id(
            df, id_col=self.item_id_col, aspect_col=self.aspect_col, unique=self.agg_unique
        )

        df = _apply_aspect_mapping(df, self.user_id_col, user_map, self.user_aspect_col)
        df = _apply_aspect_mapping(df, self.item_id_col, item_map, self.item_aspect_col)

        print("[DataLoader] Built review sets globally (same as your original pipeline intent).")
        return df

    def _encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        df["user_idx"] = self.user_encoder.fit_transform(df[self.user_id_col].astype(str))
        df["item_idx"] = self.item_encoder.fit_transform(df[self.item_id_col].astype(str))

        self.num_users = int(df["user_idx"].nunique())
        self.num_items = int(df["item_idx"].nunique())

        print(f"[DataLoader] Encoded users={self.num_users}, items={self.num_items}")
        return df

    def _data_split(self, df: pd.DataFrame):
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        print(f"[DataLoader] Train: {train.shape}")
        print(f"[DataLoader] Test:  {test.shape}")
        return train, test

    def _make_w2v_and_sequences(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Tuple[W2VArtifacts, Dict[str, np.ndarray]]:

        for col in [self.user_aspect_col, self.item_aspect_col]:
            if col not in train.columns:
                raise KeyError(f"Required aggregated column missing: {col}")

        user_w2v = _fit_w2v(
            sentences=train[self.user_aspect_col].tolist(),
            vector_size=self.w2v_vector_size,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            workers=self.w2v_workers,
            seed=self.random_state,
        )

        item_w2v = _fit_w2v(
            sentences=train[self.item_aspect_col].tolist(),
            vector_size=self.w2v_vector_size,
            window=self.w2v_window,
            min_count=self.w2v_min_count,
            workers=self.w2v_workers,
            seed=self.random_state,
        )

        user_train_seq, user_test_seq, user_emb_mat, user_vocab, user_tok = _tokenize_w2v(
            train[self.user_aspect_col].tolist(),
            test[self.user_aspect_col].tolist(),
            w2v_model=user_w2v,
            max_words=self.max_words,
            maxlen=self.user_maxlen,
            vector_size=self.w2v_vector_size,
        )

        item_train_seq, item_test_seq, item_emb_mat, item_vocab, item_tok = _tokenize_w2v(
            train[self.item_aspect_col].tolist(),
            test[self.item_aspect_col].tolist(),
            w2v_model=item_w2v,
            max_words=self.max_words,
            maxlen=self.item_maxlen,
            vector_size=self.w2v_vector_size,
        )

        artifacts = W2VArtifacts(
            user_encoder=self.user_encoder,
            item_encoder=self.item_encoder,
            user_tokenizer=user_tok,
            item_tokenizer=item_tok,
            user_embedding_matrix=user_emb_mat,
            item_embedding_matrix=item_emb_mat,
            user_vocab_size=user_vocab,
            item_vocab_size=item_vocab,
            user_maxlen=self.user_maxlen,
            item_maxlen=self.item_maxlen,
        )

        seqs = {
            "user_train_seq": user_train_seq,
            "user_test_seq": user_test_seq,
            "item_train_seq": item_train_seq,
            "item_test_seq": item_test_seq,

            "y_train": train[self.rating_col].to_numpy(dtype=np.float32),
            "y_test": test[self.rating_col].to_numpy(dtype=np.float32),

            "user_id_train": train["user_idx"].to_numpy(dtype=np.int32),
            "user_id_test": test["user_idx"].to_numpy(dtype=np.int32),

            "item_id_train": train["item_idx"].to_numpy(dtype=np.int32),
            "item_id_test": test["item_idx"].to_numpy(dtype=np.int32),
        }

        print("[DataLoader] W2V + sequences prepared.")
        print(f"  - user_emb_mat: {user_emb_mat.shape}, user_vocab_size={user_vocab}, user_maxlen={self.user_maxlen}")
        print(f"  - item_emb_mat: {item_emb_mat.shape}, item_vocab_size={item_vocab}, item_maxlen={self.item_maxlen}")

        return artifacts, seqs

    def _save_processed(self):
        os.makedirs(PROCESSED_PATH, exist_ok=True)

        train_path = os.path.join(PROCESSED_PATH, f"{self.fname}_train.parquet")
        test_path = os.path.join(PROCESSED_PATH, f"{self.fname}_test.parquet")

        save_parquet(self.train, train_path)
        save_parquet(self.test, test_path)

        artifacts_path = os.path.join(PROCESSED_PATH, f"{self.fname}_w2v_artifacts.pkl")
        seqs_path = os.path.join(PROCESSED_PATH, f"{self.fname}_seqs.pkl")

        with open(artifacts_path, "wb") as f:
            pickle.dump(self.artifacts, f)

        with open(seqs_path, "wb") as f:
            pickle.dump(self.seqs, f)

        print(f"[DataLoader] Saved parquet to {PROCESSED_PATH}")
        print(f"[DataLoader] Saved artifacts: {artifacts_path}")
        print(f"[DataLoader] Saved seqs:      {seqs_path}")
