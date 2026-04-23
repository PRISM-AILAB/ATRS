from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import get_metrics


# ---- Sub-modules ---------------------------------------------------------

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


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + residual FFN (Eqs 5–10)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, _ = self.multihead_attention(x, x, x)                     # Eq (7) / (8)
        out1 = self.layer_norm_1(x + self.attention_dropout(attn))
        ff = self.feed_forward(out1)
        return self.layer_norm_2(out1 + self.ffn_dropout(ff))            # Eq (9) / (10)


# ---- ATRS model ----------------------------------------------------------

class ProposedModel(nn.Module):
    """ATRS — Aspect Term-aware Recommender System (paper Sec 3)."""

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
        num_heads: int = 12,
        id_dim: int = 128,
        dropout: float = 0.1,
        cnn_filters: int = 100,
        cnn_kernel_size: int = 5,
        ffn_dim: int = 2048,
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

        self.user_aspect_encoder = AspectEncoder(user_embedding_matrix, cnn_filters, cnn_kernel_size, id_dim, dropout)
        self.item_aspect_encoder = AspectEncoder(item_embedding_matrix, cnn_filters, cnn_kernel_size, id_dim, dropout)

        # Eqs (1)–(2)
        self.user_id_embedding = nn.Embedding(num_users, id_dim)
        self.item_id_embedding = nn.Embedding(num_items, id_dim)

        # Eqs (3)–(4)
        self.user_project = nn.Linear(2 * id_dim, self.attn_dim)
        self.item_project = nn.Linear(2 * id_dim, self.attn_dim)

        # Eqs (5)–(10)
        self.user_self_attention = SelfAttentionBlock(self.attn_dim, num_heads, ffn_dim, dropout)
        self.item_self_attention = SelfAttentionBlock(self.attn_dim, num_heads, ffn_dim, dropout)

        # Eqs (11)–(12): MLP rating predictor
        self.rating_predictor = nn.Sequential(
            nn.Linear(2 * self.attn_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        user_seq: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> torch.Tensor:
        at_u = self.user_aspect_encoder(user_seq)
        e_u = self.user_id_embedding(user_id)                                  # Eq (2)
        z_u = self.user_project(torch.cat([at_u, e_u], dim=-1)).unsqueeze(1)   # Eq (3)
        F_u = self.user_self_attention(z_u).squeeze(1)                         # Eqs (7), (9)

        at_v = self.item_aspect_encoder(item_seq)
        e_v = self.item_id_embedding(item_id)                                  # Eq (1)
        z_v = self.item_project(torch.cat([at_v, e_v], dim=-1)).unsqueeze(1)   # Eq (4)
        F_v = self.item_self_attention(z_v).squeeze(1)                         # Eqs (8), (10)

        O = torch.cat([F_u, F_v], dim=1)                                       # Eq (11)
        return self.rating_predictor(O)                                        # Eq (12)


# ---- Training / evaluation helpers ---------------------------------------

def _unpack_batch(batch: dict, device: str):
    """Move batch tensors to `device`."""
    return (
        batch["user_id"].to(device),
        batch["item_id"].to(device),
        batch["user_seq"].to(device),
        batch["item_seq"].to(device),
        batch["label"].to(device),
    )


def _train_one_epoch(model: nn.Module, loader: DataLoader,
                     optimizer: torch.optim.Optimizer, criterion: nn.Module,
                     device: str) -> float:
    """One training epoch; returns average batch loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        uid, iid, useq, iseq, label = _unpack_batch(batch, device)
        optimizer.zero_grad()
        loss = criterion(model(uid, iid, useq, iseq), label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _predict(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference; return concatenated (preds, trues) arrays."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            uid, iid, useq, iseq, label = _unpack_batch(batch, device)
            preds.append(model(uid, iid, useq, iseq).cpu().numpy().flatten())
            trues.append(label.cpu().numpy().flatten())
    return np.concatenate(preds), np.concatenate(trues)


# ---- Public trainer / tester ---------------------------------------------

def proposed_trainer(
    args: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_model_path: str,
    device: str = "cuda",
) -> nn.Module:
    """Adam + MSE (Eq 14) with early-stopping; returns model reloaded from best checkpoint."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001))
    criterion = nn.MSELoss()
    epochs = args.get("num_epochs", 100)
    patience = args.get("patience", 5)

    best_val_mse = float("inf")
    patience_counter = 0

    print(f"Start Training on {device}...")
    for epoch in range(epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_trues = _predict(model, val_loader, device)
        val_mse, val_rmse, val_mae, _ = get_metrics(val_preds, val_trues)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val MSE={val_mse:.4f}  MAE={val_mae:.4f}  RMSE={val_rmse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early Stopping Triggered")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def proposed_tester(model: nn.Module, test_loader: DataLoader,
                    device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on test loader; return (preds, trues)."""
    model.to(device)
    return _predict(model, test_loader, device)
