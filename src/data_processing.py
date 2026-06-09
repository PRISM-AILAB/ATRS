import os
from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.aspect_extraction import ATExtractor
from src.path import PROCESSED_PATH, RAW_PATH
from src.text_cleaning import clean_text, k_core_filter
from src.utils import load_json_gz, load_parquet, save_parquet


# ---- Aspect tokenizer ----------------------------------------------------

@dataclass
class SimpleTokenizer:
    """Whitespace tokenizer mapping words to integer indices (0 = padding, 1 = OOV)."""

    oov_token: str = "<OOV>"
    word_index: dict[str, int] = field(default_factory=dict)

    def fit_on_texts(self, texts: list[str]) -> None:
        """Build vocab from whitespace-tokenized strings."""
        counts: dict[str, int] = {}
        for text in texts:
            for w in text.split():
                counts[w] = counts.get(w, 0) + 1

        current_idx = 1
        if self.oov_token:
            self.word_index[self.oov_token] = current_idx
            current_idx += 1

        for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            if w not in self.word_index:
                self.word_index[w] = current_idx
                current_idx += 1

    def texts_to_sequences(self, texts: list[str]) -> list[list[int]]:
        """Convert strings to lists of vocab indices (unseen → OOV)."""
        oov_idx = self.word_index.get(self.oov_token)
        seqs = []
        for text in texts:
            seq = [i for w in text.split()
                   if (i := self.word_index.get(w, oov_idx)) is not None]
            seqs.append(seq)
        return seqs


# ---- Training-input artifacts schema ------------------------------------

@dataclass
class W2VArtifacts:
    """Container holding everything the RS module needs at model build time."""

    num_users: int
    num_items: int
    user_tokenizer: SimpleTokenizer
    item_tokenizer: SimpleTokenizer
    user_embedding_matrix: np.ndarray
    item_embedding_matrix: np.ndarray
    user_vocab_size: int
    item_vocab_size: int
    user_aspect_maxlen: int
    item_aspect_maxlen: int


# ---- DataProcessor (pipeline orchestrator) ------------------------------

@dataclass
class DataProcessor:
    """End-to-end data pipeline: raw JSONL → clean → ATE (cached); split → W2V → seqs (in-memory)."""

    fname: str
    raw_ext: str
    test_size: float
    random_state: int
    k_core: int
    aspect_length_percentile: float
    w2v_vector_size: int
    w2v_window: int
    w2v_min_count: int

    USER_ID_COL: ClassVar[str] = "user_id"
    ITEM_ID_COL: ClassVar[str] = "parent_asin"
    RATING_COL: ClassVar[str] = "rating"
    RAW_TEXT_COL: ClassVar[str] = "text"
    CLEAN_TEXT_COL: ClassVar[str] = "clean_text"
    ASPECT_COL: ClassVar[str] = "aspect"
    USER_ASPECT_COL: ClassVar[str] = "user_aspect_set"
    ITEM_ASPECT_COL: ClassVar[str] = "item_aspect_set"
    W2V_WORKERS: ClassVar[int] = 4

    COLUMN_ALIASES: ClassVar[dict] = {
        "review_text": "RAW_TEXT_COL",
    }

    def __post_init__(self):
        self.raw_path           = os.path.join(RAW_PATH, f"{self.fname}.{self.raw_ext}")
        self.preprocessed_path  = os.path.join(PROCESSED_PATH, f"{self.fname}_preprocessed.parquet")
        self.aspects_path       = os.path.join(PROCESSED_PATH, f"{self.fname}_aspects.parquet")

        self.w2v_params = {
            "vector_size": self.w2v_vector_size,
            "window": self.w2v_window,
            "min_count": self.w2v_min_count,
            "workers": self.W2V_WORKERS,
            "seed": self.random_state,
        }

    # ---- cache checks

    def _aspects_exists(self) -> bool:
        """True iff the aspect-set parquet exists on disk."""
        return os.path.exists(self.aspects_path)

    def _preprocessed_exists(self) -> bool:
        """True iff the cleaned + k-core parquet exists on disk."""
        return os.path.exists(self.preprocessed_path)

    # ---- pipeline stages

    def _load_and_normalize(self) -> pd.DataFrame:
        """Load raw file, map column aliases, drop rows missing critical fields."""
        df = load_json_gz(self.raw_path)
        print(f"[Stats] Raw rows: {len(df):,}")

        if "verified_purchase" not in df.columns:
            raise KeyError("Raw data missing required column: 'verified_purchase'")
        before = len(df)
        df = df[df["verified_purchase"] == True].reset_index(drop=True)   # noqa: E712 — pandas mask needs ==
        print(f"[Stats] Dropped {before - len(df):,} unverified rows; remaining {len(df):,}")

        for src_col, attr_name in self.COLUMN_ALIASES.items():
            dst_col = getattr(self, attr_name)
            if src_col in df.columns and dst_col not in df.columns:
                df[dst_col] = df[src_col]

        return df.dropna(subset=[self.USER_ID_COL, self.ITEM_ID_COL, self.RAW_TEXT_COL])

    def _clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean review text, drop empty rows, apply k-core filter."""
        tqdm.pandas(desc="Review text cleaning")
        print("[DataProcessor] Cleaning 'text' → 'clean_text'")
        df[self.CLEAN_TEXT_COL] = df[self.RAW_TEXT_COL].progress_apply(clean_text)
        df = df[df[self.CLEAN_TEXT_COL].str.len() > 0]
        df = k_core_filter(df, self.USER_ID_COL, self.ITEM_ID_COL, k=self.k_core)
        print(f"[Stats] After cleaning & {self.k_core}-core: {len(df):,}")
        return df

    def _attach_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run ATE if aspect column missing; ensure list type."""
        if self.ASPECT_COL not in df.columns:
            print("[DataProcessor] Running ATE...")
            ate = ATExtractor()
            df = ate.run(df=df, text_col=self.CLEAN_TEXT_COL, aspect_col=self.ASPECT_COL, save_result=True)
        df[self.ASPECT_COL] = df[self.ASPECT_COL].apply(lambda x: x if isinstance(x, list) else [])
        return df

    def _aggregate_aspect_sets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach per-user/item aspect sequences; drop rows with empty sets."""
        print("[DataProcessor] Aggregating aspect sets...")
        flatten = lambda s: list(chain.from_iterable(s))
        df[self.USER_ASPECT_COL] = df[self.USER_ID_COL].map(
            df.groupby(self.USER_ID_COL)[self.ASPECT_COL].apply(flatten)
        )
        df[self.ITEM_ASPECT_COL] = df[self.ITEM_ID_COL].map(
            df.groupby(self.ITEM_ID_COL)[self.ASPECT_COL].apply(flatten)
        )

        before = len(df)
        df = df[(df[self.USER_ASPECT_COL].str.len() > 0)
                & (df[self.ITEM_ASPECT_COL].str.len() > 0)]
        print(f"[Stats] After aspect-set drop: {len(df):,} (dropped {before - len(df):,} empty-aspect rows)")
        return df

    def _fit_and_apply_id_encoders(self, df: pd.DataFrame) -> tuple[int, int]:
        """Encode user/item IDs to integers; return (num_users, num_items)."""
        user_encoder = LabelEncoder().fit(df[self.USER_ID_COL].unique())
        item_encoder = LabelEncoder().fit(df[self.ITEM_ID_COL].unique())
        df["user_idx"] = user_encoder.transform(df[self.USER_ID_COL])
        df["item_idx"] = item_encoder.transform(df[self.ITEM_ID_COL])
        return len(user_encoder.classes_), len(item_encoder.classes_)

    def _train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Shuffle-split into train/test copies."""
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        return train.copy(), test.copy()

    # ---- W2V / sequence building

    def _build_w2v_artifacts(self, train: pd.DataFrame, num_users: int, num_items: int) -> W2VArtifacts:
        """Build tokenizers, Word2Vec embeddings, and aspect maxlens from the train split."""
        print("[DataProcessor] Building W2V artifacts (tokenizers, Word2Vec, embeddings)...")
        user_tokenizer, user_emb, user_vocab_size = self._build_one_side_w2v(
            train, self.USER_ID_COL, self.USER_ASPECT_COL,
        )
        item_tokenizer, item_emb, item_vocab_size = self._build_one_side_w2v(
            train, self.ITEM_ID_COL, self.ITEM_ASPECT_COL,
        )
        user_aspect_maxlen, item_aspect_maxlen = self._compute_aspect_maxlens(train)

        print(f"[DataProcessor] User vocab size: {user_vocab_size} / item vocab size: {item_vocab_size}")
        print(f"[DataProcessor] Max aspects user: {user_aspect_maxlen} / item: {item_aspect_maxlen}")

        return W2VArtifacts(
            num_users=num_users, num_items=num_items,
            user_tokenizer=user_tokenizer, item_tokenizer=item_tokenizer,
            user_embedding_matrix=user_emb, item_embedding_matrix=item_emb,
            user_vocab_size=user_vocab_size, item_vocab_size=item_vocab_size,
            user_aspect_maxlen=user_aspect_maxlen, item_aspect_maxlen=item_aspect_maxlen,
        )

    def _build_one_side_w2v(self, train: pd.DataFrame, id_col: str,
                            aspect_col: str) -> tuple[SimpleTokenizer, np.ndarray, int]:
        """Fit tokenizer + Word2Vec for one side (user or item); return (tokenizer, embedding, vocab_size)."""
        vec_size = self.w2v_params["vector_size"]
        corpus = train[aspect_col].apply(" ".join).tolist()
        tokenizer = SimpleTokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(corpus)
        vocab_size = len(tokenizer.word_index) + 1
        sentences = (
            train.drop_duplicates(subset=[id_col])[aspect_col]
                 .apply(lambda lst: list(dict.fromkeys(lst))).tolist()
        )
        w2v = Word2Vec(sentences=sentences, **self.w2v_params)
        emb = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for w, i in tokenizer.word_index.items():
            if w in w2v.wv:
                emb[i] = w2v.wv[w]
        return tokenizer, emb, vocab_size

    def _compute_aspect_maxlens(self, train: pd.DataFrame) -> tuple[int, int]:
        """p-th percentile of per-user/item aspect set sizes on train."""
        p = float(self.aspect_length_percentile)
        return (
            int(np.percentile(train[self.USER_ASPECT_COL].str.len(), p)),
            int(np.percentile(train[self.ITEM_ASPECT_COL].str.len(), p)),
        )

    def _attach_seq_columns(self, train: pd.DataFrame, test: pd.DataFrame, artifacts: W2VArtifacts) -> None:
        """Add padded user/item seq columns to train and test in place."""
        sides = [
            (self.USER_ASPECT_COL, artifacts.user_aspect_maxlen, artifacts.user_tokenizer, "user_seq"),
            (self.ITEM_ASPECT_COL, artifacts.item_aspect_maxlen, artifacts.item_tokenizer, "item_seq"),
        ]
        for df in (train, test):
            for aspect_col, maxlen, tokenizer, out_col in sides:
                texts = df[aspect_col].apply(" ".join).tolist()
                seqs = tokenizer.texts_to_sequences(texts)
                padded = np.zeros((len(seqs), maxlen), dtype=np.int64)
                for i, s in enumerate(seqs):
                    if s:
                        trunc = s[:maxlen]
                        padded[i, :len(trunc)] = trunc
                df[out_col] = padded.tolist()

    def _build_seqs(self, train: pd.DataFrame, test: pd.DataFrame) -> dict:
        """Materialize numpy arrays from train/test dfs for the data loader."""
        return {
            "user_id_train":  train["user_idx"].values.astype(np.int64),
            "item_id_train":  train["item_idx"].values.astype(np.int64),
            "y_train":        train[self.RATING_COL].values.astype(np.float32),
            "user_id_test":   test["user_idx"].values.astype(np.int64),
            "item_id_test":   test["item_idx"].values.astype(np.int64),
            "y_test":         test[self.RATING_COL].values.astype(np.float32),
            "user_seq_train": np.asarray(train["user_seq"].tolist(), dtype=np.int64),
            "item_seq_train": np.asarray(train["item_seq"].tolist(), dtype=np.int64),
            "user_seq_test":  np.asarray(test["user_seq"].tolist(),  dtype=np.int64),
            "item_seq_test":  np.asarray(test["item_seq"].tolist(),  dtype=np.int64),
        }

    # ---- driver

    def run(self) -> tuple[W2VArtifacts, dict]:
        """Cache-resumable through aspects; split + W2V + seqs run in-memory each call."""
        print(f"\n{'=' * 10} Data Processing {'=' * 10}")
        print("[DataProcessor] External resources required on first use: "
              "NLTK corpora (stopwords/wordnet/omw-1.4) and the PyABSA English ATE checkpoint.")

        if self._aspects_exists():
            print(f"[DataProcessor] Resuming from aspects checkpoint: {self.aspects_path}")
            df = load_parquet(self.aspects_path)
        else:
            if self._preprocessed_exists():
                print(f"[DataProcessor] Resuming from preprocessed checkpoint: {self.preprocessed_path}")
                df = load_parquet(self.preprocessed_path)
            else:
                df = self._load_and_normalize()
                df = self._clean_and_filter(df)
                save_parquet(df, self.preprocessed_path)
                print(f"[DataProcessor] Saved preprocessed parquet: {self.preprocessed_path}")

            df = self._attach_aspects(df)
            df = self._aggregate_aspect_sets(df)
            save_parquet(df, self.aspects_path)
            print(f"[DataProcessor] Saved aspect-set parquet: {self.aspects_path}")

        num_users, num_items = self._fit_and_apply_id_encoders(df)
        train, test = self._train_test_split(df)
        artifacts = self._build_w2v_artifacts(train, num_users, num_items)
        self._attach_seq_columns(train, test, artifacts)
        print(f"[Stats] Split sizes: train={len(train):,}, test={len(test):,}")
        print("[DataProcessor] Processing complete.")

        return artifacts, self._build_seqs(train, test)


# ---- Torch Dataset / DataLoader -----------------------------------------

@dataclass
class RecommenderDataset(Dataset):
    """Map-style dataset for (user_id, item_id, user_seq, item_seq, label)."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_seq: np.ndarray
    item_seq: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "user_seq": torch.tensor(self.user_seq[idx], dtype=torch.long),
            "item_seq": torch.tensor(self.item_seq[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def get_data_loader(args: dict, user_ids: np.ndarray, item_ids: np.ndarray,
                    user_seq: np.ndarray, item_seq: np.ndarray, labels: np.ndarray,
                    shuffle: bool = True) -> DataLoader:
    """Wrap arrays in `RecommenderDataset` and return a torch DataLoader."""
    dataset = RecommenderDataset(user_ids, item_ids, user_seq, item_seq, labels)
    return DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
