import os

import torch
from sklearn.model_selection import train_test_split

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH, UTILS_PATH, ATE_RESULT_PATH
from src.utils import load_yaml, load_parquet, set_seed, get_metrics
from src.pipeline import DataProcessor, W2VArtifacts, get_data_loader, load_processed_data
from model.proposed import ProposedModel, proposed_trainer, proposed_tester


def split_train_val_test(seqs: dict, val_ratio: float, seed: int) -> dict:
    """Carve a validation split out of the training arrays and bundle all three splits."""
    (
        user_id_train, user_id_val,
        item_id_train, item_id_val,
        user_seq_train, user_seq_val,
        item_seq_train, item_seq_val,
        y_train, y_val,
    ) = train_test_split(
        seqs["user_id_train"], seqs["item_id_train"],
        seqs["user_seq_train"], seqs["item_seq_train"],
        seqs["y_train"],
        test_size=val_ratio, random_state=seed,
    )
    return {
        "train": (user_id_train, item_id_train, user_seq_train, item_seq_train, y_train),
        "val":   (user_id_val,   item_id_val,   user_seq_val,   item_seq_val,   y_val),
        "test":  (seqs["user_id_test"], seqs["item_id_test"],
                  seqs["user_seq_test"], seqs["item_seq_test"], seqs["y_test"]),
    }


def build_loaders(splits: dict, args: dict):
    """Wrap each split tuple in a torch DataLoader."""
    return (
        get_data_loader(args, *splits["train"], shuffle=True),
        get_data_loader(args, *splits["val"], shuffle=False),
        get_data_loader(args, *splits["test"], shuffle=False),
    )


def build_model(artifacts: W2VArtifacts, args: dict) -> ProposedModel:
    """Instantiate `ProposedModel` from the persisted artifacts and config args."""
    return ProposedModel(
        num_users=artifacts.num_users,
        num_items=artifacts.num_items,
        user_vocab_size=artifacts.user_vocab_size,
        item_vocab_size=artifacts.item_vocab_size,
        user_aspect_maxlen=artifacts.user_aspect_maxlen,
        item_aspect_maxlen=artifacts.item_aspect_maxlen,
        user_embedding_matrix=artifacts.user_embedding_matrix,
        item_embedding_matrix=artifacts.item_embedding_matrix,
        num_heads=args.get("num_heads", 12),
        id_dim=args.get("id_dim", 128),
        dropout=args.get("dropout", 0.1),
        cnn_filters=args.get("cnn_filters", 100),
        cnn_kernel_size=args.get("cnn_kernel_size", 5),
        ffn_dim=args.get("ffn_dim", 2048),
    )


def run_data_processing(dargs: dict, args: dict, seed: int, fname: str):
    """Invoke the DataProcessor pipeline (internal caches decide what to skip)."""
    DataProcessor(
        fname=fname,
        raw_ext=dargs.get("raw_ext", "json.gz"),
        test_size=dargs.get("test_size", 0.2),
        random_state=seed,
        clean_text_col=dargs.get("clean_text_col", "clean_text"),
        aspect_col=dargs.get("aspect_col", "aspect"),
        ate_result_dir=dargs.get("ate_result_dir", ATE_RESULT_PATH),
        ate_device=dargs.get("ate_device", "cuda:0"),
        user_aspect_col=dargs.get("user_aspect_col", "user_aspect_set"),
        item_aspect_col=dargs.get("item_aspect_col", "item_aspect_set"),
        aspect_length_percentile=dargs.get("aspect_length_percentile", 90),
        w2v_vector_size=args.get("w2v_vector_size", 300),
        w2v_window=args.get("w2v_window", 5),
        w2v_min_count=args.get("w2v_min_count", 1),
    ).run()


def main():
    cfg = load_yaml(os.path.join(UTILS_PATH, "config.yaml"))
    dargs = cfg.get("data", {})
    args = cfg.get("args", {})
    fname = dargs.get("fname")
    args["fname"] = fname

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Device: {device}")

    run_data_processing(dargs, args, seed, fname)

    dataset_dir = os.path.join(PROCESSED_PATH, fname)
    train_df = load_parquet(os.path.join(dataset_dir, "train.parquet"))
    test_df = load_parquet(os.path.join(dataset_dir, "test.parquet"))
    print("[main] Train shape:", train_df.shape)
    print("[main] Test shape :", test_df.shape)

    artifacts, seqs = load_processed_data(fname)
    splits = split_train_val_test(seqs, args.get("val_ratio", 0.125), seed)
    train_loader, val_loader, test_loader = build_loaders(splits, args)

    print("[main] Building PyTorch Model...")
    print(f"   - User Vocab Size: {artifacts.user_vocab_size} / Item Vocab Size: {artifacts.item_vocab_size}")
    print(f"   - User Max Len:    {artifacts.user_aspect_maxlen} / Item Max Len:    {artifacts.item_aspect_maxlen}")

    model = build_model(artifacts, args)
    save_dir = os.path.join(SAVE_MODEL_PATH, fname)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best.pth")
    model = proposed_trainer(
        args=args, model=model,
        train_loader=train_loader, val_loader=val_loader,
        best_model_path=best_model_path, device=device,
    )

    test_preds, test_trues = proposed_tester(model, test_loader, device=device)
    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    print(f"[TEST] RMSE={rmse:.4f}  MSE={mse:.4f}  MAE={mae:.4f}  MAPE={mape:.3f}%")


if __name__ == "__main__":
    main()
