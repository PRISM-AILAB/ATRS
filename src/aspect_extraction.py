import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from pyabsa import ATEPCCheckpointManager


@dataclass
class ATExtractor:
    """Wrapper around PyABSA's pretrained ATE pipeline with DataFrame-index recovery."""

    checkpoint: str = "english"
    auto_device: bool = False
    device: str = "cuda:0"
    cal_perplexity: bool = False
    result_dir: str = "output_results"

    def __post_init__(self):
        os.makedirs(self.result_dir, exist_ok=True)
        self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint=self.checkpoint,
            auto_device=self.auto_device,
            device=self.device,
            cal_perplexity=self.cal_perplexity,
        )

    @staticmethod
    def _make_marked_texts(df: pd.DataFrame, text_col: str) -> List[str]:
        """Prepend the row index to each text as '<idx> [SEP] <text>'."""
        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")
        texts = df[text_col].fillna("").astype(str)
        return (df.index.astype(str) + " [SEP] " + texts).tolist()

    def extract(self, df: pd.DataFrame, text_col: str, *,
                print_result: bool = False, pred_sentiment: bool = False,
                save_result: bool = True) -> None:
        """Run aspect term extraction on every row of `df[text_col]`."""
        self.aspect_extractor.extract_aspect(
            inference_source=self._make_marked_texts(df, text_col=text_col),
            print_result=print_result,
            pred_sentiment=pred_sentiment,
            save_result=save_result,
            result_save_path=self.result_dir,
        )

    @staticmethod
    def _safe_split_sentence(sentence: str) -> Tuple[Optional[int], str]:
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
    def load_results(json_paths: List[str]) -> pd.DataFrame:
        """Merge one or more PyABSA result JSON files into a single DataFrame."""
        all_data = []
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data if isinstance(data, list) else [data])
        return pd.DataFrame(all_data)

    def results_to_aspect_df(self, df_ate: pd.DataFrame) -> pd.DataFrame:
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
    def merge_aspects(df: pd.DataFrame, df_aspect: pd.DataFrame, *,
                      aspect_col: str = "aspect") -> pd.DataFrame:
        """Left-join the extracted aspect column back onto the original DataFrame."""
        if aspect_col not in df_aspect.columns:
            raise KeyError(f"{aspect_col} column not found in df_aspect.")
        return df.merge(df_aspect[[aspect_col]], left_index=True, right_index=True, how="left")

    def run(self, df: pd.DataFrame, text_col: str, *,
            result_json_paths: Optional[List[str]] = None,
            aspect_col: str = "aspect",
            print_result: bool = False,
            pred_sentiment: bool = False,
            save_result: bool = True) -> pd.DataFrame:
        """End-to-end ATE pipeline: extract → load JSON → recover index → merge."""
        self.extract(df=df, text_col=text_col, print_result=print_result,
                     pred_sentiment=pred_sentiment, save_result=save_result)

        if result_json_paths is None:
            result_json_paths = (
                [os.path.join(self.result_dir, fn) for fn in os.listdir(self.result_dir)
                 if fn.lower().endswith(".json") and "atepc" in fn.lower()]
                if os.path.exists(self.result_dir) else []
            )

        if not result_json_paths:
            print(f"Warning: No ATE JSON results found in {self.result_dir}.")
            df[aspect_col] = [[] for _ in range(len(df))]
            return df

        df_aspect = self.results_to_aspect_df(self.load_results(result_json_paths))
        df_aspect = df_aspect.rename(columns={"aspect": aspect_col})
        return self.merge_aspects(df, df_aspect, aspect_col=aspect_col)
