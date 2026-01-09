import os
import re
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


from src.path import RAW_PATH, PROCESSED_PATH
from src.utils import save_parquet
from src.ate import ATEExtractor

tqdm.pandas()

# ============================================================
# 0. Tokenization & Padding
# ============================================================

class SimpleTokenizer:
    """
    A lightweight replacement for Keras Tokenizer.
    Maps words to integers (1-based index). 0 is reserved for padding.
    """
    def __init__(self, oov_token="<OOV>"):
        self.oov_token = oov_token
        self.word_index = {} # word -> idx
        self.index_word = {} # idx -> word
        self.word_counts = {}
        self.num_words = None # Not strictly enforced during fit, used later

    def fit_on_texts(self, texts: List[str]):
        """Builds vocabulary from list of strings."""
        for text in texts:
            words = text.split()
            for w in words:
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                
        # Sort by frequency (descending) usually, but simple iteration is fine for basic map
        # Reserve 0 for padding
        current_idx = 1
        
        # Add OOV token first if it exists
        if self.oov_token:
            self.word_index[self.oov_token] = current_idx
            self.index_word[current_idx] = self.oov_token
            current_idx += 1
            
        # Add other words
        # Sorting by frequency is better to keep common words if we truncate vocab later
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for w, count in sorted_words:
            if w not in self.word_index:
                self.word_index[w] = current_idx
                self.index_word[current_idx] = w
                current_idx += 1

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Transforms strings to list of integers."""
        seqs = []
        oov_idx = self.word_index.get(self.oov_token, None)
        
        for text in texts:
            words = text.split()
            seq = []
            for w in words:
                idx = self.word_index.get(w, oov_idx)
                if idx is not None:
                    seq.append(idx)
            seqs.append(seq)
        return seqs

def pad_sequences_numpy(sequences: List[List[int]], maxlen: int, padding='post', truncating='post', value=0):
    """
    NumPy implementation of pad_sequences.
    Returns: np.ndarray of shape (len(sequences), maxlen)
    """
    num_samples = len(sequences)
    # PyTorch Embedding expects int64 (Long) usually
    x = np.full((num_samples, maxlen), value, dtype=np.int64)
    
    for idx, s in enumerate(sequences):
        if not s:
            continue  # empty sequence
            
        # Truncate
        if truncating == 'pre':
            trunc = s[-maxlen:]
        else: # 'post'
            trunc = s[:maxlen]
            
        # Pad
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        else: # 'pre'
            x[idx, -len(trunc):] = trunc
            
    return x


# ============================================================
# 1. Artifacts
# ============================================================

@dataclass
class W2VArtifacts:
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder
    
    # Use custom SimpleTokenizer
    user_tokenizer: SimpleTokenizer
    item_tokenizer: SimpleTokenizer
    
    user_embedding_matrix: np.ndarray
    item_embedding_matrix: np.ndarray
    
    # Dynamic Stats
    aspect_vocab_size: int
    user_aspect_len: int
    item_aspect_len: int


# ============================================================
# 2. Helper Functions
# ============================================================

def load_json_gz(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_json(path, compression="gzip", lines=True)
        if len(df) > 0: return df
    except Exception: pass
    try:
        df = pd.read_json(path, compression="gzip", lines=False)
        return df
    except Exception as e:
        print(f"[Error] Failed to load json: {e}")
        return pd.DataFrame()

def clean_text_func(txt):
    if pd.isna(txt) or txt is None: return None
    txt = str(txt)
    # Regex cleaning (same as before)
    txt = re.sub(r"[^\w\s.,!?'-]", '', txt)
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'\s+([?.!,])', r'\1', txt)
    txt = txt.strip()
    return txt if txt else None


# ============================================================
# 3. DataLoader Class (Pure PyTorch Ready)
# ============================================================

class DataLoader:
    def __init__(
        self,
        fname: str,
        raw_ext: str = "json.gz",
        test_size: float = 0.2,
        random_state: int = 42,
        
        # Columns
        user_id_col: str = "user_id",
        item_id_col: str = "parent_asin",
        rating_col: str = "rating",
        
        # ATE
        run_ate: bool = True,
        text_col: str = "clean_text",
        raw_text_col: str = "text",
        aspect_col: str = "aspect",
        ate_result_dir: str = "output_results",
        ate_device: str = "cuda:0",

        # Aggregation
        user_aspect_col: str = "user_aspect_set",
        item_aspect_col: str = "item_aspect_set",
        agg_unique: bool = True,

        # W2V Params
        w2v_vector_size: int = 300,
        w2v_window: int = 5,
        w2v_min_count: int = 1,
        w2v_workers: int = 4,
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
        self.raw_text_col = raw_text_col
        self.aspect_col = aspect_col
        self.ate_result_dir = ate_result_dir
        self.ate_device = ate_device
        
        self.user_aspect_col = user_aspect_col
        self.item_aspect_col = item_aspect_col
        self.agg_unique = agg_unique
        
        self.w2v_params = {
            "vector_size": w2v_vector_size,
            "window": w2v_window,
            "min_count": w2v_min_count,
            "workers": w2v_workers,
            "seed": random_state
        }

        self.process()

    def process(self):
        print(f"\n{'='*10} Data Processing (Pure PyTorch) {'='*10}")
        
        # 1. Load
        raw_path = os.path.join(RAW_PATH, f"{self.fname}.{self.raw_ext}")
        df = load_json_gz(raw_path)
        print(f"[Stats] Raw Data: {len(df):,}")

        # Basic Filter
        if "user_id" in df.columns and self.user_id_col not in df.columns: df[self.user_id_col] = df["user_id"]
        if "parent_asin" in df.columns and self.item_id_col not in df.columns: df[self.item_id_col] = df["parent_asin"]
        if "text" in df.columns and self.raw_text_col not in df.columns: df[self.raw_text_col] = df["text"]
        if "rating" in df.columns and self.rating_col not in df.columns: df[self.rating_col] = df["rating"]

        df = df.dropna(subset=[self.user_id_col, self.item_id_col, self.raw_text_col])
        df = df.drop_duplicates(subset=[self.user_id_col, self.item_id_col])
        
        # 2. Text Clean
        print("[DataLoader] Cleaning text...")
        df[self.text_col] = df[self.raw_text_col].progress_apply(clean_text_func)
        df = df[df[self.text_col].str.len() > 0]

        # 3. K-Core Filter
        df = self._k_core_filter(df, k=5)
        print(f"[Stats] After Cleaning & 5-Core: {len(df):,}")

        # 4. ATE
        if self.run_ate and self.aspect_col not in df.columns:
            print("[DataLoader] Running ATE...")
            ate = ATEExtractor(result_dir=self.ate_result_dir, device=self.ate_device)
            df = ate.run(df=df, text_col=self.text_col, aspect_col=self.aspect_col, save_result=True)
        elif self.aspect_col not in df.columns:
            df[self.aspect_col] = [[] for _ in range(len(df))]

        # 5. Aggregate
        print("[DataLoader] Aggregating Aspect Sets...")
        df[self.aspect_col] = df[self.aspect_col].apply(lambda x: x if isinstance(x, list) else [])
        
        user_map = df.groupby(self.user_id_col)[self.aspect_col].sum()
        item_map = df.groupby(self.item_id_col)[self.aspect_col].sum()

        if self.agg_unique:
            user_map = user_map.apply(lambda x: list(set(x)))
            item_map = item_map.apply(lambda x: list(set(x)))

        df[self.user_aspect_col] = df[self.user_id_col].map(user_map)
        df[self.item_aspect_col] = df[self.item_id_col].map(item_map)

        # 6. Split
        self.train, self.test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        
        # 7. Tokenizer & W2V
        artifacts, seqs = self._build_vocab_and_vectors()
        
        # 8. Save
        self._save_pickles(artifacts, seqs)
        self._save_processed()
        print("[DataLoader] Processing Complete.")

    def _k_core_filter(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        while True:
            u_cnt = df[self.user_id_col].value_counts()
            i_cnt = df[self.item_id_col].value_counts()
            valid_u = u_cnt[u_cnt >= k].index
            valid_i = i_cnt[i_cnt >= k].index
            old_len = len(df)
            df = df[df[self.user_id_col].isin(valid_u) & df[self.item_id_col].isin(valid_i)]
            if len(df) == old_len: break
        return df

    def _build_vocab_and_vectors(self):
        print("[DataLoader] Calculating Stats & W2V (Custom Tokenizer)...")
        
        # 1. Label Encoding
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()
        all_u = pd.concat([self.train[self.user_id_col], self.test[self.user_id_col]]).unique()
        all_i = pd.concat([self.train[self.item_id_col], self.test[self.item_id_col]]).unique()
        user_enc.fit(all_u)
        item_enc.fit(all_i)
        
        self.train['user_idx'] = user_enc.transform(self.train[self.user_id_col])
        self.train['item_idx'] = item_enc.transform(self.train[self.item_id_col])
        self.test['user_idx'] = user_enc.transform(self.test[self.user_id_col])
        self.test['item_idx'] = item_enc.transform(self.test[self.item_id_col])
        
        # 2. Tokenizer (Using Custom SimpleTokenizer)
        def to_str(lst): return " ".join(lst)
        
        all_aspect_texts = pd.concat([
            self.train[self.user_aspect_col].apply(to_str),
            self.train[self.item_aspect_col].apply(to_str),
            self.test[self.user_aspect_col].apply(to_str),
            self.test[self.item_aspect_col].apply(to_str)
        ]).astype(str).tolist()
        
        # Initialize and Fit Custom Tokenizer
        tokenizer = SimpleTokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(all_aspect_texts)
        
        vocab_size = len(tokenizer.word_index) + 1 # +1 for padding(0)
        print(f"   [Dynamic] Unique Aspects (Vocab Size): {vocab_size}")
        
        # 3. Word2Vec
        sentences = [t.split() for t in all_aspect_texts]
        w2v = Word2Vec(sentences=sentences, **self.w2v_params)
        
        emb_mat = np.zeros((vocab_size, self.w2v_params['vector_size']), dtype=np.float32)
        for w, i in tokenizer.word_index.items():
            if w in w2v.wv:
                emb_mat[i] = w2v.wv[w]

        # 4. Max Length (Dynamic)
        def get_lens(df, col):
            return df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)

        u_lens = pd.concat([get_lens(self.train, self.user_aspect_col), get_lens(self.test, self.user_aspect_col)])
        i_lens = pd.concat([get_lens(self.train, self.item_aspect_col), get_lens(self.test, self.item_aspect_col)])

        user_max_len = int(u_lens.max())
        item_max_len = int(i_lens.max())
        print(f"   [Dynamic] Max Aspects User: {user_max_len} / Item: {item_max_len}")

        # 5. Padding (Using custom numpy padding)
        def get_padded(df, col, maxlen):
            texts = df[col].apply(to_str).astype(str).tolist()
            seqs = tokenizer.texts_to_sequences(texts)
            # Use custom padding function
            return pad_sequences_numpy(seqs, maxlen=maxlen, padding='post', truncating='post')

        seqs_dict = {
            "user_id_train": self.train["user_idx"].values.astype(np.int64), # PyTorch likes int64
            "item_id_train": self.train["item_idx"].values.astype(np.int64),
            "y_train": self.train[self.rating_col].values.astype(np.float32),
            
            "user_id_test": self.test["user_idx"].values.astype(np.int64),
            "item_id_test": self.test["item_idx"].values.astype(np.int64),
            "y_test": self.test[self.rating_col].values.astype(np.float32),
            
            "user_train_seq": get_padded(self.train, self.user_aspect_col, user_max_len),
            "item_train_seq": get_padded(self.train, self.item_aspect_col, item_max_len),
            "user_test_seq": get_padded(self.test, self.user_aspect_col, user_max_len),
            "item_test_seq": get_padded(self.test, self.item_aspect_col, item_max_len),
        }
        
        artifacts = W2VArtifacts(
            user_encoder=user_enc, item_encoder=item_enc,
            user_tokenizer=tokenizer, item_tokenizer=tokenizer,
            user_embedding_matrix=emb_mat, item_embedding_matrix=emb_mat,
            
            aspect_vocab_size=vocab_size,
            user_aspect_len=user_max_len,
            item_aspect_len=item_max_len
        )
        
        return artifacts, seqs_dict

    def _save_pickles(self, artifacts, seqs):
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        with open(os.path.join(PROCESSED_PATH, f"{self.fname}_w2v_artifacts.pkl"), "wb") as f:
            pickle.dump(artifacts, f)
        with open(os.path.join(PROCESSED_PATH, f"{self.fname}_seqs.pkl"), "wb") as f:
            pickle.dump(seqs, f)

    def _save_processed(self):
        save_parquet(self.train, os.path.join(PROCESSED_PATH, f"{self.fname}_train.parquet"))
        save_parquet(self.test, os.path.join(PROCESSED_PATH, f"{self.fname}_test.parquet"))
