import os
import re
import json
from typing import List, Optional, Tuple

import pandas as pd
from pyabsa import ATEPCCheckpointManager

class ATEExtractor:
    """
    PyABSA ATE (Aspect Term Extraction) wrapper.
    It marks input text with "<idx> [SEP] <text>" to preserve the original DataFrame index.
    """

    def __init__(
        self,
        checkpoint: str = "english",
        auto_device: bool = False,
        device: str = "cuda:0",
        cal_perplexity: bool = False,
        result_dir: str = "output_results",
    ):
        self.checkpoint = checkpoint
        self.auto_device = auto_device
        self.device = device
        self.cal_perplexity = cal_perplexity
        self.result_dir = result_dir

        os.makedirs(self.result_dir, exist_ok=True)

        self.aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint=self.checkpoint,
            auto_device=self.auto_device,
            device=self.device,
            cal_perplexity=self.cal_perplexity,
        )

    @staticmethod
    def _make_marked_texts(df: pd.DataFrame, text_col: str) -> List[str]:
        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")

        # Prepend index to track rows: "index [SEP] text"
        texts = df[text_col].fillna("").astype(str)
        marked = df.index.astype(str) + " [SEP] " + texts
        return marked.tolist()

    def extract(
        self,
        df: pd.DataFrame,
        text_col: str,
        *,
        print_result: bool = False,
        pred_sentiment: bool = False,
        save_result: bool = True,
    ) -> None:
        """Run ATE extraction."""
        texts = self._make_marked_texts(df, text_col=text_col)

        self.aspect_extractor.extract_aspect(
            inference_source=texts,
            print_result=print_result,
            pred_sentiment=pred_sentiment,
            save_result=save_result,
            result_save_path=self.result_dir,
        )

    @staticmethod
    def _safe_split_sentence(sentence: str) -> Tuple[Optional[int], str]:
        """
        Robust split for PyABSA output.
        PyABSA might alter spaces around [SEP] (e.g., ' [ SEP ] ', '[SEP]').
        """
        if sentence is None:
            return None, ""

        s = str(sentence)
        
        # Regex to handle variable spacing around [SEP]
        pattern = r"\s*\[\s*SEP\s*\]\s*"
        parts = re.split(pattern, s, maxsplit=1)

        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1]
            try:
                return int(left), right
            except Exception:
                return None, s

        return None, s

    @staticmethod
    def load_results(json_paths: List[str]) -> pd.DataFrame:
        """Load and merge JSON results."""
        all_data = []
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)

        return pd.DataFrame(all_data)

    def results_to_aspect_df(self, df_ate: pd.DataFrame) -> pd.DataFrame:
        """Recover original index from 'sentence' column and extract aspects."""
        if "sentence" not in df_ate.columns:
            raise KeyError("ATE results must contain 'sentence' column.")

        tmp = df_ate.copy()
        
        # Split index and text
        recovered = tmp["sentence"].apply(self._safe_split_sentence)
        tmp["recovered_index"] = recovered.apply(lambda x: x[0])
        
        # Filter valid rows
        tmp = tmp.dropna(subset=["recovered_index"]).copy()
        tmp["recovered_index"] = tmp["recovered_index"].astype(int)

        tmp = tmp.set_index("recovered_index")
        tmp.index.name = None

        if "aspect" not in tmp.columns:
            raise KeyError("ATE results must contain 'aspect' column.")

        return tmp[["aspect"]].copy()

    @staticmethod
    def merge_aspects(
        df: pd.DataFrame,
        df_aspect: pd.DataFrame,
        *,
        aspect_col: str = "aspect",
    ) -> pd.DataFrame:
        """Left join aspects to the original DataFrame."""
        if aspect_col not in df_aspect.columns:
            raise KeyError(f"{aspect_col} column not found in df_aspect.")

        out = df.copy()
        out = out.merge(df_aspect[[aspect_col]], left_index=True, right_index=True, how="left")
        return out

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        *,
        result_json_paths: Optional[List[str]] = None,
        aspect_col: str = "aspect",
        print_result: bool = False,
        pred_sentiment: bool = False,
        save_result: bool = True,
    ) -> pd.DataFrame:
        """Full pipeline: Extract -> Load -> Merge."""
        
        # 1. Extract
        self.extract(
            df=df,
            text_col=text_col,
            print_result=print_result,
            pred_sentiment=pred_sentiment,
            save_result=save_result,
        )

        # 2. Find Result JSONs
        if result_json_paths is None:
            # PyABSA usually saves in the current directory or result_dir with specific naming
            cwd = os.getcwd()
            result_json_paths = [
                os.path.join(cwd, fn)
                for fn in os.listdir(cwd)
                if fn.lower().endswith(".json") and "atepc" in fn.lower()
            ]

        if not result_json_paths:
            # Fallback: check result_dir if cwd has nothing
            if os.path.exists(self.result_dir):
                result_json_paths = [
                    os.path.join(self.result_dir, fn)
                    for fn in os.listdir(self.result_dir)
                    if fn.lower().endswith(".json") and "atepc" in fn.lower()
                ]

        if not result_json_paths:
            print(f"Warning: No ATE JSON results found in {os.getcwd()} or {self.result_dir}.")
            # Return original df with empty aspect list to prevent crash
            df[aspect_col] = [[] for _ in range(len(df))]
            return df

        # 3. Load & Process
        df_ate = self.load_results(result_json_paths)
        df_aspect = self.results_to_aspect_df(df_ate)
        df_aspect = df_aspect.rename(columns={"aspect": aspect_col})

        # 4. Merge
        df_all = self.merge_aspects(df, df_aspect, aspect_col=aspect_col)
        return df_all