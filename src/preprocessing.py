import html
import re
from typing import Optional

import contractions
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resources() -> None:
    """Download NLTK resources on first run (idempotent)."""
    for resource, lookup in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(lookup)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()
_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

_URL_RE = re.compile(r"http\S+|www\.\S+")
_CONTROL_RE = re.compile(r"[\x00-\x1F\x7F]")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _is_mostly_english(txt: str, threshold: float = 0.9) -> bool:
    """True if ≥ `threshold` of chars are ASCII."""
    if not txt:
        return False
    return sum(1 for c in txt if ord(c) < 128) / len(txt) >= threshold


def clean_text(txt: Optional[str]) -> Optional[str]:
    """Paper Sec 4.1 preprocessing (HTML/URL strip, lowercase, contractions, stopwords, lemmatize)."""
    if txt is None or pd.isna(txt):
        return None
    txt = BeautifulSoup(str(txt), "html.parser").get_text()
    txt = html.unescape(txt)
    txt = _URL_RE.sub("[URL]", txt)
    if not _is_mostly_english(txt):
        return None
    txt = contractions.fix(txt.lower())
    txt = _PUNCT_RE.sub(" ", _CONTROL_RE.sub(" ", txt))
    tokens = [_LEMMATIZER.lemmatize(w) for w in txt.split() if w not in _STOPWORDS]
    txt = _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()
    return txt if txt else None


def k_core_filter(df: pd.DataFrame, user_col: str, item_col: str, k: int) -> pd.DataFrame:
    """Drop users/items with fewer than `k` interactions until convergence."""
    while True:
        u_cnt = df[user_col].value_counts()
        i_cnt = df[item_col].value_counts()
        old_len = len(df)
        df = df[df[user_col].isin(u_cnt[u_cnt >= k].index)
                & df[item_col].isin(i_cnt[i_cnt >= k].index)]
        if len(df) == old_len:
            break
    return df
