import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils import get_metrics


# ---- Aspect-term encoder ------------------------------------------------

class AspectEncoder(nn.Module):
    """Aspect-term encoder: W2V → Conv1d → max-pool → projection."""

    def __init__(self, embedding_matrix: np.ndarray, cnn_filters: int,
                 kernel_size: int, out_dim: int, dropout: float):
        super().__init__()
        emb_dim = embedding_matrix.shape[1]
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=True, padding_idx=0
        )
        self.conv1d = nn.Conv1d(emb_dim, cnn_filters, kernel_size=kernel_size)
        self.projection = nn.Linear(cnn_filters, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.word_embedding(seq).permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.dropout(self.projection(x))


# ---- Self-attention block ------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + residual FFN."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.mha(x, x, x)
        out1 = self.ln1(x + self.dropout(attn))
        ff = self.ffn(out1)
        return self.ln2(out1 + self.dropout(ff))


# ---- ATRS model ----------------------------------------------------------

class ATRS(nn.Module):
    """ATRS — Aspect Term-aware Recommender System: Aspect Term Extraction Module (upstream, src/aspect_extraction.py) + RS Module (CNN aspect encoder + ID embedding + self-attention + rating MLP)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_vocab_size: int,
        item_vocab_size: int,
        user_aspect_maxlen: int,
        item_aspect_maxlen: int,
        user_embedding_matrix: np.ndarray,
        item_embedding_matrix: np.ndarray,
        num_heads: int,
        id_dim: int,
        dropout: float,
        cnn_filters: int,
        cnn_kernel_size: int,
        ffn_dim: int,
    ):
        super().__init__()

        head_dim = max(1, id_dim // num_heads)
        self.attn_dim = head_dim * num_heads

        if user_embedding_matrix.shape[0] != user_vocab_size:
            raise ValueError(
                f"user_embedding_matrix rows ({user_embedding_matrix.shape[0]}) != user_vocab_size ({user_vocab_size})."
            )
        if item_embedding_matrix.shape[0] != item_vocab_size:
            raise ValueError(
                f"item_embedding_matrix rows ({item_embedding_matrix.shape[0]}) != item_vocab_size ({item_vocab_size})."
            )

        self.num_users = num_users
        self.num_items = num_items
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.user_aspect_maxlen = user_aspect_maxlen
        self.item_aspect_maxlen = item_aspect_maxlen
        self.id_dim = id_dim
        self.num_heads = num_heads

        # ---- 1. Aspect-term encoders (CNN over W2V) --------------------
        self.user_aspect_encoder = AspectEncoder(user_embedding_matrix, cnn_filters, cnn_kernel_size, id_dim, dropout)
        self.item_aspect_encoder = AspectEncoder(item_embedding_matrix, cnn_filters, cnn_kernel_size, id_dim, dropout)

        # ---- 2. ID embeddings --------------------------------
        self.user_id_embedding = nn.Embedding(num_users, id_dim)
        self.item_id_embedding = nn.Embedding(num_items, id_dim)

        # ---- 3. Concat-and-project ---------------------------
        self.user_project = nn.Linear(2 * id_dim, self.attn_dim)
        self.item_project = nn.Linear(2 * id_dim, self.attn_dim)

        # ---- 4. Self-attention blocks -----------------------
        self.user_self_attention = SelfAttentionBlock(self.attn_dim, num_heads, ffn_dim, dropout)
        self.item_self_attention = SelfAttentionBlock(self.attn_dim, num_heads, ffn_dim, dropout)

        # ---- 5. Rating MLP ---------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(2 * self.attn_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        user_id  = inputs["user_id"]
        item_id  = inputs["item_id"]
        user_seq = inputs["user_seq"]
        item_seq = inputs["item_seq"]

        at_u = self.user_aspect_encoder(user_seq)
        e_u  = self.user_id_embedding(user_id)
        z_u  = self.user_project(torch.cat([at_u, e_u], dim=-1)).unsqueeze(1)
        F_u  = self.user_self_attention(z_u).squeeze(1)

        at_v = self.item_aspect_encoder(item_seq)
        e_v  = self.item_id_embedding(item_id)
        z_v  = self.item_project(torch.cat([at_v, e_v], dim=-1)).unsqueeze(1)
        F_v  = self.item_self_attention(z_v).squeeze(1)

        O = torch.cat([F_u, F_v], dim=1)
        return self.regressor(O)


# ---- Training / evaluation helpers ---------------------------------------

def _unpack_batch(batch: dict, device: str) -> tuple[dict, torch.Tensor]:
    """Pop `label` from the batch dict and move every tensor to `device`; return (inputs, label)."""
    label = batch.pop("label").to(device)
    inputs = {k: v.to(device) for k, v in batch.items()}
    return inputs, label


def _train_one_epoch(model: nn.Module, loader: DataLoader,
                     optimizer: optim.Optimizer, criterion: nn.Module,
                     device: str) -> float:
    """One training epoch; returns average batch loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs, label = _unpack_batch(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """One validation pass; returns (avg_loss, preds, trues)."""
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, label = _unpack_batch(batch, device)
            output = model(inputs)
            total_loss += criterion(output, label.unsqueeze(1)).item()
            preds.append(output.cpu().numpy().flatten())
            trues.append(label.cpu().numpy().flatten())
    return total_loss / len(loader), np.concatenate(preds), np.concatenate(trues)


# ---- Optimizer dispatch --------------------------------------------------

OPTIMIZERS = {
    "adam": optim.Adam,
}


def _build_optimizer(args: dict, model: nn.Module) -> optim.Optimizer:
    """Resolve `args["optimizer"]` against the OPTIMIZERS registry; raise on unknown name."""
    name = args["optimizer"].lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{name}'. Supported: {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name](model.parameters(), lr=args["lr"])


# ---- Public trainer / predictor ------------------------------------------

def train(
    args: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_model_path: str,
    device: str,
) -> nn.Module:
    """Adam + MSE with early stopping; returns model reloaded from best checkpoint."""
    model.to(device)

    optimizer = _build_optimizer(args, model)
    criterion = nn.MSELoss()
    epochs   = args["num_epochs"]
    patience = args["patience"]

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"[Trainer] Start training on {device}...")
    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_preds, val_trues = _eval_one_epoch(model, val_loader, criterion, device)
        val_mae, val_mse, val_rmse, _ = get_metrics(val_preds, val_trues)
        print(f"[Trainer] Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
              f"Val MAE={val_mae:.4f}  Val MSE={val_mse:.4f}  Val RMSE={val_rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("[Trainer] Saved best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[Trainer] Early stopping triggered.")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on `loader`; return concatenated (preds, trues) arrays."""
    model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, label = _unpack_batch(batch, device)
            preds.append(model(inputs).cpu().numpy().flatten())
            trues.append(label.cpu().numpy().flatten())
    return np.concatenate(preds), np.concatenate(trues)
