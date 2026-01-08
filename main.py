# main.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH, UTILS_PATH
from src.utils import load_yaml, load_parquet, set_seed, get_metrics

from src.data import DataLoader
from model.proposed import (
    build_proposed,
    get_data_loader,
    proposed_trainer,
    proposed_tester,
    SelfAttentionBlock,  # load_model 시 custom object
)


def _load_pickles(fname: str):
    artifacts_path = os.path.join(PROCESSED_PATH, f"{fname}_w2v_artifacts.pkl")
    seqs_path = os.path.join(PROCESSED_PATH, f"{fname}_seqs.pkl")

    if not (os.path.exists(artifacts_path) and os.path.exists(seqs_path)):
        raise FileNotFoundError(
            f"Processed pickle not found:\n- {artifacts_path}\n- {seqs_path}"
        )

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    with open(seqs_path, "rb") as f:
        seqs = pickle.load(f)

    return artifacts, seqs


if __name__ == "__main__":
    # 1) config 로드
    CONFIG_FPATH = os.path.join(UTILS_PATH, "config.yaml")
    cfg = load_yaml(CONFIG_FPATH)

    dargs = cfg.get("data", {})
    args = cfg.get("args", {})

    FNAME = dargs.get("fname")
    args["fname"] = FNAME

    seed = cfg.get("seed", 42)
    set_seed(seed)

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # 2) processed train/test parquet 존재 여부 확인 (이번 data.py는 train/test 저장)
    train_path = os.path.join(PROCESSED_PATH, f"{FNAME}_train.parquet")
    test_path = os.path.join(PROCESSED_PATH, f"{FNAME}_test.parquet")

    need_processing = not (os.path.exists(train_path) and os.path.exists(test_path))

    if need_processing:
        print("[main] Processed parquet not found. Running DataLoader from raw (.json.gz)...")
        _ = DataLoader(
            fname=FNAME,
            raw_ext=dargs.get("raw_ext", "json.gz"),
            test_size=dargs.get("test_size", 0.2),
            random_state=seed,
            # ATE / column 옵션은 config에 있으면 그대로 연결
            run_ate=dargs.get("run_ate", True),
            text_col=dargs.get("text_col", "pyabsa_clean_text"),
            aspect_col=dargs.get("aspect_col", "aspect"),
            ate_result_dir=dargs.get("ate_result_dir", "output_results"),
            ate_device=dargs.get("ate_device", "cuda:0"),
            # aggregation output cols
            user_aspect_col=dargs.get("user_aspect_col", "user_aspect_set"),
            item_aspect_col=dargs.get("item_aspect_col", "item_aspect_set"),
            agg_unique=dargs.get("agg_unique", True),
            # w2v/tokenizer params
            max_words=dargs.get("max_words", 42904),
            user_maxlen=dargs.get("user_maxlen", 45),
            item_maxlen=dargs.get("item_maxlen", 602),
        )
        # DataLoader 내부에서 parquet/pickle 저장까지 수행
    else:
        print("[main] Found processed parquet. Skipping raw processing.")

    # 3) parquet 로드(분석/확인용)
    train_df = load_parquet(train_path)
    test_df = load_parquet(test_path)

    print("[main] train:", train_df.shape)
    print("[main] test :", test_df.shape)

    # 4) artifacts/seqs 로드 (학습 입력은 여기서 가져옴)
    artifacts, seqs = _load_pickles(FNAME)

    # ---- 원래 코드 validation_split=0.125 유지: train arrays에서 val 분리
    val_ratio = args.get("val_ratio", 0.125)

    u_tr = seqs["user_id_train"]
    i_tr = seqs["item_id_train"]
    us_tr = seqs["user_train_seq"]
    it_tr = seqs["item_train_seq"]
    y_tr = seqs["y_train"]

    u_tr, u_val, i_tr, i_val, us_tr, us_val, it_tr, it_val, y_tr, y_val = train_test_split(
        u_tr, i_tr, us_tr, it_tr, y_tr,
        test_size=val_ratio,
        random_state=seed,
    )

    # test arrays
    u_te = seqs["user_id_test"]
    i_te = seqs["item_id_test"]
    us_te = seqs["user_test_seq"]
    it_te = seqs["item_test_seq"]
    y_te = seqs["y_test"]

    # 5) tf.data.Dataset 생성 (입력 4개)
    train_loader = get_data_loader(args, u_tr, i_tr, us_tr, it_tr, y_tr, shuffle=True)
    val_loader = get_data_loader(args, u_val, i_val, us_val, it_val, y_val, shuffle=False)
    test_loader = get_data_loader(args, u_te, i_te, us_te, it_te, y_te, shuffle=False)

    # 6) 모델 생성 (Proposed)
    model = build_proposed(
        num_users = artifacts.user_encoder.classes_.shape[0],
        num_items = artifacts.item_encoder.classes_.shape[0],
        user_vocab_size=artifacts.user_vocab_size,
        item_vocab_size=artifacts.item_vocab_size,
        user_maxlen=artifacts.user_maxlen,
        item_maxlen=artifacts.item_maxlen,
        user_embedding_matrix=artifacts.user_embedding_matrix,
        item_embedding_matrix=artifacts.item_embedding_matrix,
        num_heads=args.get("num_heads", 8),
        id_dim=args.get("id_dim", 128),
        dropout=args.get("dropout", 0.1),
        learning_rate=args.get("lr", 1e-3),
        cnn_filters=args.get("cnn_filters", 100),
        cnn_kernel_size=args.get("cnn_kernel_size", 5),
        ffn_dim=args.get("ffn_dim", 2048),
        model_name="Proposed",
    )
    model.summary()

    # 7) 학습
    best_model_path = os.path.join(SAVE_MODEL_PATH, f"{FNAME}_Best_Model.keras")

    proposed_trainer(
        args={
            **args,
            "num_epochs": args.get("num_epochs", 100),
            "patience": args.get("patience", 5),
        },
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=best_model_path,
    )

    # 8) Best 모델 로드 + 테스트
    best_model = keras.models.load_model(
    best_model_path,
    custom_objects={"SelfAttentionBlock": SelfAttentionBlock},
    safe_mode=False,
    )


    test_preds, test_trues = proposed_tester(
        args=args,
        model=best_model,
        test_loader=test_loader,
    )

    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    print(f"[TEST] RMSE={rmse:.5f}  MSE={mse:.5f}  MAE={mae:.5f}  MAPE={mape:.3f}%")
