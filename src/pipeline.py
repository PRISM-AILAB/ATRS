import os
import json
import pickle
from dataclasses import dataclass, field
from itertools import chain
from typing import ClassVar, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

from src.path import RAW_PATH, PROCESSED_PATH, ATE_RESULT_PATH
from src.utils import load_parquet, save_parquet
from src.preprocessing import load_json_gz, clean_text, k_core_filter
from src.aspect_extraction import ATExtractor

# Internal pipeline column names (produced by DataProcessor, consumed by load_processed_data).
USER_IDX_COL = "user_idx"
ITEM_IDX_COL = "item_idx"
USER_SEQ_COL = "user_seq"
ITEM_SEQ_COL = "item_seq"


# ---- Cache metadata helpers ---------------------------------------------

def _meta_path(cache_path: str) -> str:
    return cache_path + ".meta.json"


def _cache_hit(cache_path: str, expected: dict) -> bool:
    """True iff cache file exists and its sidecar `.meta.json` equals `expected`."""
    if not os.path.exists(cache_path) or not os.path.exists(_meta_path(cache_path)):
        return False
    with open(_meta_path(cache_path), "r") as f:
        return json.load(f) == expected


def _save_meta(cache_path: str, meta: dict):
    with open(_meta_path(cache_path), "w") as f:
        json.dump(meta, f, sort_keys=True, indent=2)


# ---- 1. Tokenizer --------------------------------------------------------

@dataclass
class SimpleTokenizer:
    """Whitespace tokenizer mapping words to integer indices (0 = padding, 1 = OOV)."""

    oov_token: str = "<OOV>"
    word_index: dict = field(default_factory=dict)

    def fit_on_texts(self, texts: List[str]):
        """Build vocab from whitespace-tokenized strings."""
        counts: dict = {}
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

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert strings to lists of vocab indices (unseen → OOV)."""
        oov_idx = self.word_index.get(self.oov_token)
        seqs = []
        for text in texts:
            seq = [i for w in text.split()
                   if (i := self.word_index.get(w, oov_idx)) is not None]
            seqs.append(seq)
        return seqs


# ---- 2. Artifacts schema -------------------------------------------------

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


# ---- 3. DataProcessor (pipeline orchestrator) ----------------------------

@dataclass
class DataProcessor:
    """End-to-end data pipeline."""

    fname: str
    raw_ext: str = "json.gz"
    test_size: float = 0.2
    random_state: int = 42

    user_id_col: str = "user_id"
    item_id_col: str = "parent_asin"
    rating_col: str = "rating"

    clean_text_col: str = "clean_text"
    raw_text_col: str = "text"
    aspect_col: str = "aspect"
    ate_result_dir: str = ATE_RESULT_PATH
    ate_device: str = "cuda:0"

    user_aspect_col: str = "user_aspect_set"
    item_aspect_col: str = "item_aspect_set"
    aspect_length_percentile: float = 90.0

    w2v_vector_size: int = 300
    w2v_window: int = 5
    w2v_min_count: int = 1
    w2v_workers: int = 4

    COLUMN_ALIASES: ClassVar[dict] = {
        "user_id":     "user_id_col",
        "parent_asin": "item_id_col",
        "review_text": "raw_text_col",
        "rating":      "rating_col",
    }

    def __post_init__(self):
        # Per-fname output roots (created on demand).
        self.dataset_dir = os.path.join(PROCESSED_PATH, self.fname)
        self.ate_dataset_dir = os.path.join(self.ate_result_dir, self.fname)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.ate_dataset_dir, exist_ok=True)

        self.w2v_params = {
            "vector_size": self.w2v_vector_size,
            "window": self.w2v_window,
            "min_count": self.w2v_min_count,
            "workers": self.w2v_workers,
            "seed": self.random_state,
        }

    # ---- cache metadata (upstream params cascade into downstream signatures)

    def _meta_preprocessed(self) -> dict:
        return {
            "stage": "preprocessed",
            "fname": self.fname,
            "raw_ext": self.raw_ext,
            "user_id_col": self.user_id_col,
            "item_id_col": self.item_id_col,
            "raw_text_col": self.raw_text_col,
            "clean_text_col": self.clean_text_col,
            "rating_col": self.rating_col,
        }

    def _meta_aspects(self) -> dict:
        return {
            **self._meta_preprocessed(),
            "stage": "aspects",
            "aspect_col": self.aspect_col,
            "user_aspect_col": self.user_aspect_col,
            "item_aspect_col": self.item_aspect_col,
        }

    def _meta_splits(self) -> dict:
        return {
            **self._meta_aspects(),
            "stage": "splits",
            "test_size": float(self.test_size),
            "random_state": int(self.random_state),
            "aspect_length_percentile": float(self.aspect_length_percentile),
            "w2v_vector_size": int(self.w2v_vector_size),
            "w2v_window": int(self.w2v_window),
            "w2v_min_count": int(self.w2v_min_count),
        }

    def run(self):
        """Run the full pipeline; short-circuits at the outermost cache layer if all metadata matches."""
        print(f"\n{'='*10} Data Processing {'='*10}")
        print("[DataProcessor] External resources required on first use: "
              "NLTK corpora (stopwords/wordnet/omw-1.4) and the PyABSA English ATE checkpoint.")

        train_path = os.path.join(self.dataset_dir, "train.parquet")
        test_path = os.path.join(self.dataset_dir, "test.parquet")
        splits_meta = self._meta_splits()
        if _cache_hit(train_path, splits_meta) and _cache_hit(test_path, splits_meta):
            print("[DataProcessor] Train/test parquet caches match config; skipping full pipeline.")
            return

        df = self._load_aspects_or_build()
        num_users, num_items = self._fit_and_apply_id_encoders(df)
        train, test = self._split(df)
        artifacts = self._build_artifacts(train, num_users, num_items)
        self._attach_seq_columns(train, test, artifacts)
        self._save_outputs(train, test)
        print("[DataProcessor] Processing Complete.")

    # ---- cached preprocessing / aspect-set stages

    def _load_preprocessed_or_build(self) -> pd.DataFrame:
        """Return clean+5-core DataFrame; use `preprocessed.parquet` cache if config-signature matches."""
        cache_path = os.path.join(self.dataset_dir, "preprocessed.parquet")
        meta = self._meta_preprocessed()
        if _cache_hit(cache_path, meta):
            print(f"[DataProcessor] Loading cached preprocessed parquet: {cache_path}")
            return load_parquet(cache_path)
        df = self._load_and_normalize()
        df = self._clean_and_filter(df)
        save_parquet(df, cache_path)
        _save_meta(cache_path, meta)
        print(f"[DataProcessor] Saved preprocessed parquet: {cache_path}")
        return df

    def _load_aspects_or_build(self) -> pd.DataFrame:
        """Return aspect-set DataFrame; use `aspects.parquet` cache if config-signature matches."""
        cache_path = os.path.join(self.dataset_dir, "aspects.parquet")
        meta = self._meta_aspects()
        if _cache_hit(cache_path, meta):
            print(f"[DataProcessor] Loading cached aspect-set parquet: {cache_path}")
            return load_parquet(cache_path)
        df = self._load_preprocessed_or_build()
        df = self._attach_aspects(df)
        df = self._aggregate_aspect_sets(df)
        save_parquet(df, cache_path)
        _save_meta(cache_path, meta)
        print(f"[DataProcessor] Saved aspect-set parquet: {cache_path}")
        return df

    # ---- pipeline stages

    def _load_and_normalize(self) -> pd.DataFrame:
        """Load raw file, map column aliases, drop rows missing critical fields."""
        df = load_json_gz(os.path.join(RAW_PATH, f"{self.fname}.{self.raw_ext}"))
        print(f"[Stats] Raw Data: {len(df):,}")

        for src_col, attr_name in self.COLUMN_ALIASES.items():
            dst_col = getattr(self, attr_name)
            if src_col in df.columns and dst_col not in df.columns:
                df[dst_col] = df[src_col]

        return df.dropna(subset=[self.user_id_col, self.item_id_col, self.raw_text_col])

    def _clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean review text, drop empty rows, apply 5-core filter."""
        tqdm.pandas()
        print("[DataProcessor] Cleaning text...")
        df[self.clean_text_col] = df[self.raw_text_col].progress_apply(clean_text)
        df = df[df[self.clean_text_col].str.len() > 0]
        df = k_core_filter(df, self.user_id_col, self.item_id_col, k=5)
        print(f"[Stats] After Cleaning & 5-Core: {len(df):,}")
        return df

    def _attach_aspects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run ATE if aspect column missing; ensure list type."""
        if self.aspect_col not in df.columns:
            print("[DataProcessor] Running ATE...")
            ate = ATExtractor(result_dir=self.ate_dataset_dir, device=self.ate_device)
            df = ate.run(df=df, text_col=self.clean_text_col, aspect_col=self.aspect_col, save_result=True)
        df[self.aspect_col] = df[self.aspect_col].apply(lambda x: x if isinstance(x, list) else [])
        return df

    def _aggregate_aspect_sets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach per-user/item aspect sequences; drop rows with empty sets."""
        print("[DataProcessor] Aggregating Aspect Sets...")
        flatten = lambda s: list(chain.from_iterable(s))
        df[self.user_aspect_col] = df[self.user_id_col].map(
            df.groupby(self.user_id_col)[self.aspect_col].apply(flatten)
        )
        df[self.item_aspect_col] = df[self.item_id_col].map(
            df.groupby(self.item_id_col)[self.aspect_col].apply(flatten)
        )

        before = len(df)
        df = df[(df[self.user_aspect_col].str.len() > 0)
                & (df[self.item_aspect_col].str.len() > 0)]
        print(f"[Stats] After Aspect-set Drop: {len(df):,} (dropped {before - len(df):,} empty-aspect rows)")
        return df

    def _fit_and_apply_id_encoders(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Encode user/item IDs to integers; return (num_users, num_items)."""
        user_encoder = LabelEncoder().fit(df[self.user_id_col].unique())
        item_encoder = LabelEncoder().fit(df[self.item_id_col].unique())
        df[USER_IDX_COL] = user_encoder.transform(df[self.user_id_col])
        df[ITEM_IDX_COL] = item_encoder.transform(df[self.item_id_col])
        return len(user_encoder.classes_), len(item_encoder.classes_)

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Shuffle-split into train/test copies."""
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        return train.copy(), test.copy()

    # ---- artifact / sequence building

    def _build_artifacts(self, train: pd.DataFrame, num_users: int, num_items: int) -> W2VArtifacts:
        """Build (or load from cache) tokenizers, Word2Vec embeddings, and aspect maxlens."""
        cache_path = os.path.join(self.dataset_dir, "w2v_cache.pkl")
        meta = self._meta_splits()

        if _cache_hit(cache_path, meta):
            print(f"[DataProcessor] Loading cached W2V artifacts: {cache_path}")
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            user_tokenizer = cache["user_tokenizer"]
            item_tokenizer = cache["item_tokenizer"]
            user_emb = cache["user_embedding"]
            item_emb = cache["item_embedding"]
            user_vocab_size = cache["user_vocab_size"]
            item_vocab_size = cache["item_vocab_size"]
            user_aspect_maxlen = cache["user_aspect_maxlen"]
            item_aspect_maxlen = cache["item_aspect_maxlen"]
        else:
            print("[DataProcessor] Building W2V artifacts (tokenizers, Word2Vec, embeddings)...")
            vec_size = self.w2v_params["vector_size"]

            # User side
            user_corpus = train[self.user_aspect_col].apply(" ".join).tolist()
            user_tokenizer = SimpleTokenizer(oov_token="<OOV>")
            user_tokenizer.fit_on_texts(user_corpus)
            user_vocab_size = len(user_tokenizer.word_index) + 1
            user_sentences = (
                train.drop_duplicates(subset=[self.user_id_col])[self.user_aspect_col]
                     .apply(lambda lst: list(dict.fromkeys(lst))).tolist()
            )
            w2v_user = Word2Vec(sentences=user_sentences, **self.w2v_params)
            user_emb = np.zeros((user_vocab_size, vec_size), dtype=np.float32)
            for w, i in user_tokenizer.word_index.items():
                if w in w2v_user.wv:
                    user_emb[i] = w2v_user.wv[w]

            # Item side
            item_corpus = train[self.item_aspect_col].apply(" ".join).tolist()
            item_tokenizer = SimpleTokenizer(oov_token="<OOV>")
            item_tokenizer.fit_on_texts(item_corpus)
            item_vocab_size = len(item_tokenizer.word_index) + 1
            item_sentences = (
                train.drop_duplicates(subset=[self.item_id_col])[self.item_aspect_col]
                     .apply(lambda lst: list(dict.fromkeys(lst))).tolist()
            )
            w2v_item = Word2Vec(sentences=item_sentences, **self.w2v_params)
            item_emb = np.zeros((item_vocab_size, vec_size), dtype=np.float32)
            for w, i in item_tokenizer.word_index.items():
                if w in w2v_item.wv:
                    item_emb[i] = w2v_item.wv[w]

            user_aspect_maxlen, item_aspect_maxlen = self._compute_aspect_maxlens(train)

            with open(cache_path, "wb") as f:
                pickle.dump({
                    "user_tokenizer": user_tokenizer,
                    "item_tokenizer": item_tokenizer,
                    "user_embedding": user_emb,
                    "item_embedding": item_emb,
                    "user_vocab_size": user_vocab_size,
                    "item_vocab_size": item_vocab_size,
                    "user_aspect_maxlen": user_aspect_maxlen,
                    "item_aspect_maxlen": item_aspect_maxlen,
                }, f)
            _save_meta(cache_path, meta)

        print(f"   [Dynamic] User Vocab Size: {user_vocab_size} / Item Vocab Size: {item_vocab_size}")
        print(f"   [Dynamic] Max Aspects User: {user_aspect_maxlen} / Item: {item_aspect_maxlen}")

        return W2VArtifacts(
            num_users=num_users, num_items=num_items,
            user_tokenizer=user_tokenizer, item_tokenizer=item_tokenizer,
            user_embedding_matrix=user_emb, item_embedding_matrix=item_emb,
            user_vocab_size=user_vocab_size, item_vocab_size=item_vocab_size,
            user_aspect_maxlen=user_aspect_maxlen, item_aspect_maxlen=item_aspect_maxlen,
        )

    def _compute_aspect_maxlens(self, train: pd.DataFrame) -> Tuple[int, int]:
        """p-th percentile of per-user/item aspect set sizes on train."""
        p = float(self.aspect_length_percentile)
        return (
            int(np.percentile(train[self.user_aspect_col].str.len(), p)),
            int(np.percentile(train[self.item_aspect_col].str.len(), p)),
        )

    def _attach_seq_columns(self, train: pd.DataFrame, test: pd.DataFrame, artifacts: W2VArtifacts):
        """Add padded user/item seq columns to train and test in place."""
        sides = [
            (self.user_aspect_col, artifacts.user_aspect_maxlen, artifacts.user_tokenizer, USER_SEQ_COL),
            (self.item_aspect_col, artifacts.item_aspect_maxlen, artifacts.item_tokenizer, ITEM_SEQ_COL),
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

    # ---- persistence

    def _save_outputs(self, train: pd.DataFrame, test: pd.DataFrame):
        """Persist train/test parquet with sidecar metadata (W2V cached separately in `_build_artifacts`)."""
        train_path = os.path.join(self.dataset_dir, "train.parquet")
        test_path = os.path.join(self.dataset_dir, "test.parquet")
        save_parquet(train, train_path)
        save_parquet(test, test_path)
        meta = self._meta_splits()
        _save_meta(train_path, meta)
        _save_meta(test_path, meta)


# ---- 4. Load API (for model training) ------------------------------------

def load_processed_data(fname: str, rating_col: str = "rating") -> Tuple[W2VArtifacts, dict]:
    """Load (W2VArtifacts, seqs) from persisted train/test parquet and W2V cache."""
    dataset_dir = os.path.join(PROCESSED_PATH, fname)
    train = load_parquet(os.path.join(dataset_dir, "train.parquet"))
    test  = load_parquet(os.path.join(dataset_dir, "test.parquet"))

    with open(os.path.join(dataset_dir, "w2v_cache.pkl"), "rb") as f:
        cache = pickle.load(f)

    artifacts = W2VArtifacts(
        num_users=int(max(train[USER_IDX_COL].max(), test[USER_IDX_COL].max())) + 1,
        num_items=int(max(train[ITEM_IDX_COL].max(), test[ITEM_IDX_COL].max())) + 1,
        user_tokenizer=cache["user_tokenizer"],
        item_tokenizer=cache["item_tokenizer"],
        user_embedding_matrix=cache["user_embedding"],
        item_embedding_matrix=cache["item_embedding"],
        user_vocab_size=cache["user_vocab_size"],
        item_vocab_size=cache["item_vocab_size"],
        user_aspect_maxlen=cache["user_aspect_maxlen"],
        item_aspect_maxlen=cache["item_aspect_maxlen"],
    )

    seqs = {
        "user_id_train":  train[USER_IDX_COL].values.astype(np.int64),
        "item_id_train":  train[ITEM_IDX_COL].values.astype(np.int64),
        "y_train":        train[rating_col].values.astype(np.float32),
        "user_id_test":   test[USER_IDX_COL].values.astype(np.int64),
        "item_id_test":   test[ITEM_IDX_COL].values.astype(np.int64),
        "y_test":         test[rating_col].values.astype(np.float32),
        "user_seq_train": np.asarray(train[USER_SEQ_COL].tolist(), dtype=np.int64),
        "item_seq_train": np.asarray(train[ITEM_SEQ_COL].tolist(), dtype=np.int64),
        "user_seq_test":  np.asarray(test[USER_SEQ_COL].tolist(),  dtype=np.int64),
        "item_seq_test":  np.asarray(test[ITEM_SEQ_COL].tolist(),  dtype=np.int64),
    }
    return artifacts, seqs


# ---- 5. Torch Dataset / DataLoader ---------------------------------------

@dataclass(eq=False, repr=False)
class RecommenderDataset(Dataset):
    """Map-style Dataset for (user_id, item_id, user_seq, item_seq, label)."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_seq: np.ndarray
    item_seq: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "user_seq": torch.tensor(self.user_seq[idx], dtype=torch.long),
            "item_seq": torch.tensor(self.item_seq[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def get_data_loader(args: dict, user_ids, item_ids, user_seq, item_seq, labels,
                    shuffle: bool = True) -> DataLoader:
    """Wrap arrays in `RecommenderDataset` and return a torch DataLoader."""
    dataset = RecommenderDataset(user_ids, item_ids, user_seq, item_seq, labels)
    return DataLoader(dataset, batch_size=args.get("batch_size", 128), shuffle=shuffle, num_workers=0)
