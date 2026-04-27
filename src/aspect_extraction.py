import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.path import PYABSA_WORKDIR


@dataclass
class ATExtractor:
    """Wrapper around PyABSA's pretrained ATE pipeline with DataFrame-index recovery."""

    device: str = "cuda"
    checkpoint: str = "english"
    auto_device: bool = False
    cal_perplexity: bool = False

    def __post_init__(self):
        # PyABSA hardcodes "./checkpoints.json" + result-JSON to CWD; chdir contains them.
        os.makedirs(PYABSA_WORKDIR, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(PYABSA_WORKDIR)
            from pyabsa import ATEPCCheckpointManager
            self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
                checkpoint=self.checkpoint,
                auto_device=self.auto_device,
                device=self.device,
                cal_perplexity=self.cal_perplexity,
            )
        finally:
            os.chdir(cwd)

    @staticmethod
    def _make_marked_texts(df: pd.DataFrame, text_col: str) -> list[str]:
        """Prepend the row index to each text as '<idx> [SEP] <text>'."""
        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")
        texts = df[text_col].fillna("").astype(str)
        return (df.index.astype(str) + " [SEP] " + texts).tolist()

    def extract(self, df: pd.DataFrame, text_col: str, *,
                print_result: bool = False, pred_sentiment: bool = False,
                save_result: bool = True) -> None:
        """Run aspect term extraction on every row of `df[text_col]`."""
        cwd = os.getcwd()
        try:
            os.chdir(PYABSA_WORKDIR)
            self.aspect_extractor.extract_aspect(
                inference_source=self._make_marked_texts(df, text_col=text_col),
                print_result=print_result,
                pred_sentiment=pred_sentiment,
                save_result=save_result,
                result_save_path=".",
            )
        finally:
            os.chdir(cwd)

    @staticmethod
    def _safe_split_sentence(sentence: str) -> tuple[Optional[int], str]:
        """Split PyABSA 'sentence' back into (row_index, text), tolerating whitespace around [SEP]."""
        if sentence is None:
            return None, ""
        s = str(sentence)
        parts = re.split(r"\s*\[\s*SEP\s*\]\s*", s, maxsplit=1)
        if len(parts) == 2:
            try:
                return int(parts[0].strip()), parts[1]
            except Exception:
                return None, s
        return None, s

    @staticmethod
    def _load_results(json_paths: list[str]) -> pd.DataFrame:
        """Merge one or more PyABSA result JSON files into a single DataFrame."""
        all_data = []
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data if isinstance(data, list) else [data])
        return pd.DataFrame(all_data)

    def _results_to_aspect_df(self, df_ate: pd.DataFrame) -> pd.DataFrame:
        """Recover original row index; return a DataFrame with only the `aspect` column."""
        if "sentence" not in df_ate.columns:
            raise KeyError("ATE results must contain 'sentence' column.")
        if "aspect" not in df_ate.columns:
            raise KeyError("ATE results must contain 'aspect' column.")

        tmp = df_ate.copy()
        tmp["recovered_index"] = tmp["sentence"].apply(lambda s: self._safe_split_sentence(s)[0])
        tmp = tmp.dropna(subset=["recovered_index"]).copy()
        tmp["recovered_index"] = tmp["recovered_index"].astype(int)
        tmp = tmp.set_index("recovered_index")
        tmp.index.name = None
        return tmp[["aspect"]].copy()

    @staticmethod
    def _merge_aspects(df: pd.DataFrame, df_aspect: pd.DataFrame, *,
                       aspect_col: str = "aspect") -> pd.DataFrame:
        """Left-join the extracted aspect column back onto the original DataFrame."""
        if aspect_col not in df_aspect.columns:
            raise KeyError(f"{aspect_col} column not found in df_aspect.")
        return df.merge(df_aspect[[aspect_col]], left_index=True, right_index=True, how="left")

    def run(self, df: pd.DataFrame, text_col: str, *,
            aspect_col: str = "aspect",
            print_result: bool = False,
            pred_sentiment: bool = False,
            save_result: bool = True) -> pd.DataFrame:
        """End-to-end ATE pipeline: extract → load JSON → recover index → merge."""
        self.extract(df=df, text_col=text_col, print_result=print_result,
                     pred_sentiment=pred_sentiment, save_result=save_result)

        json_paths = [
            os.path.join(PYABSA_WORKDIR, fn) for fn in os.listdir(PYABSA_WORKDIR)
            if fn.lower().endswith(".json") and "atepc" in fn.lower()
        ]

        if not json_paths:
            print(f"[ATExtractor] Warning: no ATE JSON results found in {PYABSA_WORKDIR}.")
            df[aspect_col] = [[] for _ in range(len(df))]
            return df

        df_aspect = self._results_to_aspect_df(self._load_results(json_paths))
        df_aspect = df_aspect.rename(columns={"aspect": aspect_col})
        return self._merge_aspects(df, df_aspect, aspect_col=aspect_col)
