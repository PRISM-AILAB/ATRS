import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH, UTILS_PATH
from src.utils import load_yaml, load_parquet, set_seed, get_metrics

# Updated imports for PyTorch
from src.data import DataLoader
from model.proposed import (
    ProposedModel,
    get_data_loader,
    proposed_trainer,
    proposed_tester,
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
    # 1) Load Config
    CONFIG_FPATH = os.path.join(UTILS_PATH, "config.yaml")
    cfg = load_yaml(CONFIG_FPATH)

    dargs = cfg.get("data", {})
    args = cfg.get("args", {})

    FNAME = dargs.get("fname")
    args["fname"] = FNAME

    seed = cfg.get("seed", 42)
    set_seed(seed)
    
    # Check Device (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Device: {device}")

    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # 2) Check if Processed Data Exists
    train_path = os.path.join(PROCESSED_PATH, f"{FNAME}_train.parquet")
    test_path = os.path.join(PROCESSED_PATH, f"{FNAME}_test.parquet")

    need_processing = not (os.path.exists(train_path) and os.path.exists(test_path))

    if need_processing:
        print("[main] Processed parquet not found. Running DataLoader...")
        # Note: 'max_words' and 'maxlen' removed as they are now dynamic
        _ = DataLoader(
            fname=FNAME,
            raw_ext=dargs.get("raw_ext", "json.gz"),
            test_size=dargs.get("test_size", 0.2),
            random_state=seed,
            
            run_ate=dargs.get("run_ate", True),
            text_col=dargs.get("text_col", "clean_text"),
            aspect_col=dargs.get("aspect_col", "aspect"),
            ate_result_dir=dargs.get("ate_result_dir", "output_results"),
            ate_device=dargs.get("ate_device", "cuda:0"),
            
            user_aspect_col=dargs.get("user_aspect_col", "user_aspect_set"),
            item_aspect_col=dargs.get("item_aspect_col", "item_aspect_set"),
            agg_unique=dargs.get("agg_unique", True),
            
            # W2V params can be passed here if needed
        )
    else:
        print("[main] Found processed parquet. Skipping raw processing.")

    # 3) Load Data (For verification/logging)
    train_df = load_parquet(train_path)
    test_df = load_parquet(test_path)

    print("[main] Train shape:", train_df.shape)
    print("[main] Test shape :", test_df.shape)

    # 4) Load Artifacts & Sequences
    artifacts, seqs = _load_pickles(FNAME)
    
    # --- Validation Split ---
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

    # Test Arrays
    u_te = seqs["user_id_test"]
    i_te = seqs["item_id_test"]
    us_te = seqs["user_test_seq"]
    it_te = seqs["item_test_seq"]
    y_te = seqs["y_test"]

    # 5) Create PyTorch DataLoaders
    train_loader = get_data_loader(args, u_tr, i_tr, us_tr, it_tr, y_tr, shuffle=True)
    val_loader = get_data_loader(args, u_val, i_val, us_val, it_val, y_val, shuffle=False)
    test_loader = get_data_loader(args, u_te, i_te, us_te, it_te, y_te, shuffle=False)

    # 6) Build Model (ProposedModel)
    print("[main] Building PyTorch Model...")
    
    # Retrieve dynamic stats from artifacts
    # Note: Using new attribute names from data.py
    vocab_size = artifacts.aspect_vocab_size
    user_maxlen = artifacts.user_aspect_len
    item_maxlen = artifacts.item_aspect_len
    
    print(f"   - Aspect Vocab Size: {vocab_size}")
    print(f"   - User Max Len: {user_maxlen}")
    print(f"   - Item Max Len: {item_maxlen}")

    model = ProposedModel(
        num_users=len(artifacts.user_encoder.classes_),
        num_items=len(artifacts.item_encoder.classes_),
        
        user_vocab_size=vocab_size,
        item_vocab_size=vocab_size, # Shared vocab
        
        user_maxlen=user_maxlen,
        item_maxlen=item_maxlen,
        
        user_embedding_matrix=artifacts.user_embedding_matrix,
        item_embedding_matrix=artifacts.item_embedding_matrix,
        
        num_heads=args.get("num_heads", 8),
        id_dim=args.get("id_dim", 128),
        dropout=args.get("dropout", 0.1),
        cnn_filters=args.get("cnn_filters", 100),
        cnn_kernel_size=args.get("cnn_kernel_size", 5),
        ffn_dim=args.get("ffn_dim", 2048),
    )

    # 7) Training
    # Change extension from .keras to .pth for PyTorch
    best_model_path = os.path.join(SAVE_MODEL_PATH, f"{FNAME}_Best_Model.pth")

    model = proposed_trainer(
        args={
            **args,
            "num_epochs": args.get("num_epochs", 100),
            "patience": args.get("patience", 5),
            "lr": args.get("lr", 1e-3),
        },
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=best_model_path,
        device=device
    )

    # 8) Testing
    # The trainer returns the model with best weights loaded, so we can test directly
    test_preds, test_trues = proposed_tester(
        args=args,
        model=model,
        test_loader=test_loader,
        device=device
    )

    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    print(f"[TEST] RMSE={rmse:.5f}  MSE={mse:.5f}  MAE={mae:.5f}  MAPE={mape:.3f}%")