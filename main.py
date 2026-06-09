import os

# Import sklearn before torch — Windows cu121+MKL segfaults pd.read_json otherwise.
from sklearn.model_selection import train_test_split
import torch

from model.atrs import ATRS, predict, train
from src.data_processing import DataProcessor, W2VArtifacts, get_data_loader
from src.path import SAVE_MODEL_PATH, SRC_PATH
from src.utils import get_metrics, load_yaml, set_seed


def run_data_processing(dargs: dict, args: dict, seed: int, fname: str) -> tuple[W2VArtifacts, dict]:
    """Run the DataProcessor pipeline; returns in-memory artifacts and seqs (no on-disk split cache)."""
    return DataProcessor(
        fname=fname,
        raw_ext=dargs["raw_ext"],
        test_size=dargs["test_size"],
        random_state=seed,
        k_core=dargs["k_core"],
        aspect_length_percentile=dargs["aspect_length_percentile"],
        w2v_vector_size=args["w2v_vector_size"],
        w2v_window=args["w2v_window"],
        w2v_min_count=args["w2v_min_count"],
    ).run()


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


def build_loaders(args: dict, seqs: dict, seed: int) -> tuple:
    """Carve val out of train and wrap each split in a torch DataLoader."""
    splits = split_train_val_test(seqs, args["val_ratio"], seed)
    print(f"[Main] Train shape: ({len(splits['train'][0]):,}, *)")
    print(f"[Main] Val shape  : ({len(splits['val'][0]):,}, *)")
    print(f"[Main] Test shape : ({len(splits['test'][0]):,}, *)")
    return (
        get_data_loader(args, *splits["train"], shuffle=True),
        get_data_loader(args, *splits["val"],   shuffle=False),
        get_data_loader(args, *splits["test"],  shuffle=False),
    )


def build_model(artifacts: W2VArtifacts, args: dict) -> ATRS:
    """Instantiate the ATRS model from the persisted artifacts and config args."""
    return ATRS(
        num_users=artifacts.num_users,
        num_items=artifacts.num_items,
        user_vocab_size=artifacts.user_vocab_size,
        item_vocab_size=artifacts.item_vocab_size,
        user_aspect_maxlen=artifacts.user_aspect_maxlen,
        item_aspect_maxlen=artifacts.item_aspect_maxlen,
        user_embedding_matrix=artifacts.user_embedding_matrix,
        item_embedding_matrix=artifacts.item_embedding_matrix,
        num_heads=args["num_heads"],
        id_dim=args["id_dim"],
        dropout=args["dropout"],
        cnn_filters=args["cnn_filters"],
        cnn_kernel_size=args["cnn_kernel_size"],
        ffn_dim=args["ffn_dim"],
    )


def resolve_device(requested: str) -> str:
    """Honor the configured device, falling back to CPU if CUDA was requested but unavailable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[Main] CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    cfg = load_yaml(os.path.join(SRC_PATH, "config.yaml"))
    dargs = cfg["data"]
    args = cfg["args"]
    fname = dargs["fname"]

    seed = cfg["seed"]
    train_seed = cfg["train_seed"]
    device = resolve_device(args["device"])
    print(f"[Main] Device: {device}")

    # Data split, val carve, and W2V use the fixed `seed` so re-runs share one benchmark.
    set_seed(seed)
    artifacts, seqs = run_data_processing(dargs, args, seed, fname)
    train_loader, val_loader, test_loader = build_loaders(args, seqs, seed)

    print("[Main] Building PyTorch model...")
    print(f"[Main] User vocab size: {artifacts.user_vocab_size} / item vocab size: {artifacts.item_vocab_size}")
    print(f"[Main] User max len:    {artifacts.user_aspect_maxlen} / item max len:    {artifacts.item_aspect_maxlen}")

    # Model init / dropout / shuffle use `train_seed`; edit it in config and re-run for another run.
    set_seed(train_seed)
    model = build_model(artifacts, args)

    save_dir = os.path.join(SAVE_MODEL_PATH, fname)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_seed{train_seed}.pth")
    model = train(
        args=args, model=model,
        train_loader=train_loader, val_loader=val_loader,
        best_model_path=best_model_path, device=device,
    )

    test_preds, test_trues = predict(model, test_loader, device=device)
    mae, mse, rmse, mape = get_metrics(test_preds, test_trues)
    print(f"[Test] (train_seed={train_seed}) MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}  MAPE={mape:.3f}%")


if __name__ == "__main__":
    main()
