# src/ate.py

import os
import re
import json
from typing import List, Optional, Tuple

import pandas as pd
from pyabsa import ATEPCCheckpointManager


class ATEExtractor:
    """
    PyABSA ATE(Aspect Term Extraction) wrapper.

    - 원본 df index를 보존하기 위해: "<idx> [SEP] <text>" 형태로 marked_text 생성
    - PyABSA 결과 json의 sentence 필드에는 보통 "<idx> [ SEP ] <text>" 처럼 변형되어 저장될 수 있어
      공백 유무/형태 차이를 고려해 robust split을 제공.
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

        # index 보존 + 결측 방지
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
        """
        ATE 실행. save_result=True면 result_dir 아래에 json 파일(들)이 저장됨.
        """
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
        PyABSA 결과의 sentence는 ' [ SEP ] ' 또는 '[SEP]' 등으로 저장될 수 있어
        다양한 변형을 허용해서 idx / text 복구.
        """
        if sentence is None:
            return None, ""

        s = str(sentence)

        # 다양한 SEP 표기를 모두 허용:
        #  - " [ SEP ] "
        #  - "[ SEP ]"
        #  - "[SEP]"
        #  - " [SEP] "
        # 등등을 커버
        pattern = r"\s*\[\s*SEP\s*\]\s*"
        parts = re.split(pattern, s, maxsplit=1)

        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1]
            try:
                return int(left), right
            except Exception:
                return None, s

        # 분리 실패 시
        return None, s

    @staticmethod
    def load_results(json_paths: List[str]) -> pd.DataFrame:
        """
        PyABSA 결과 json들을 읽어서 하나의 df로 합침.
        """
        all_data = []
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    # 혹시 dict 형태로 저장된 경우 대비
                    all_data.append(data)

        df_ate = pd.DataFrame(all_data)
        return df_ate

    def results_to_aspect_df(self, df_ate: pd.DataFrame) -> pd.DataFrame:
        """
        df_ate에서 sentence 기반 index 복구 후, aspect 컬럼만 남긴 DF 생성.
        """
        if "sentence" not in df_ate.columns:
            raise KeyError("ATE results must contain 'sentence' column.")

        tmp = df_ate.copy()

        recovered = tmp["sentence"].apply(self._safe_split_sentence)
        tmp["recovered_index"] = recovered.apply(lambda x: x[0])
        tmp["recovered_text"] = recovered.apply(lambda x: x[1])

        # index 복구 성공한 행만 사용
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
        """
        원본 df(index) 기준으로 df_aspect를 left join해서 aspect 컬럼 추가.
        """
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

        # 1) ATE 실행
        self.extract(
            df=df,
            text_col=text_col,
            print_result=print_result,
            pred_sentiment=pred_sentiment,
            save_result=save_result,
        )

        # 2) 결과 json 경로 결정
        if result_json_paths is None:
            # 현재 작업 디렉토리 기준으로 FAST_LCF 결과 json 탐색
            cwd = os.getcwd()
            result_json_paths = [
                os.path.join(cwd, fn)
                for fn in os.listdir(cwd)
                if fn.lower().endswith(".json")
                and "atepc" in fn.lower()
            ]

        if not result_json_paths:
            raise FileNotFoundError(
                f"No json results found. result_dir={self.result_dir}"
            )

        # 3) json -> df_ate -> df_aspect
        df_ate = self.load_results(result_json_paths)
        df_aspect = self.results_to_aspect_df(df_ate)
        df_aspect = df_aspect.rename(columns={"aspect": aspect_col})

        # 4) merge
        df_all = self.merge_aspects(df, df_aspect, aspect_col=aspect_col)
        return df_all
