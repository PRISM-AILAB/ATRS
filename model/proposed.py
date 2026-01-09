import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

# ============================================================
# 0) Blocks
# ============================================================

class PointWiseFFN(nn.Module):
    def __init__(self, d_model: int, dff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """
    Transformer-style self-attention block:
    MHA -> Add&Norm -> FFN -> Add&Norm
    """
    def __init__(self, num_heads: int, key_dim: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        # PyTorch MultiheadAttention expects embed_dim. 
        # In the TF code, key_dim was passed. Assuming embed_dim = key_dim here based on TF logic.
        self.mha = nn.MultiheadAttention(embed_dim=key_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(key_dim)
        self.ln2 = nn.LayerNorm(key_dim)
        self.ffn = PointWiseFFN(key_dim, dff)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, SeqLen, Dim)
        
        # Self Attention
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout(attn_output)
        out1 = self.ln1(x + attn_output)

        # FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_ffn(ffn_output)
        out2 = self.ln2(out1 + ffn_output)

        return out2


# ============================================================
# 1) Proposed Model
# ============================================================

class ProposedModel(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_vocab_size: int,
        item_vocab_size: int,
        user_maxlen: int,
        item_maxlen: int,
        user_embedding_matrix: np.ndarray,
        item_embedding_matrix: np.ndarray,
        num_heads: int = 8,
        id_dim: int = 128,
        dropout: float = 0.1,
        cnn_filters: int = 100,
        cnn_kernel_size: int = 5,
        ffn_dim: int = 2048,
    ):
        super().__init__()

        # --- User Text Branch ---
        self.user_text_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(user_embedding_matrix), freeze=True, padding_idx=0
        )
        self.user_cnn = nn.Conv1d(
            in_channels=user_embedding_matrix.shape[1], 
            out_channels=cnn_filters, 
            kernel_size=cnn_kernel_size
        )
        self.user_aspect_proj = nn.Linear(cnn_filters, id_dim)
        self.user_aspect_dropout = nn.Dropout(dropout)

        # --- User ID Branch ---
        self.user_id_emb = nn.Embedding(num_users, id_dim)

        # --- Item Text Branch ---
        self.item_text_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(item_embedding_matrix), freeze=True, padding_idx=0
        )
        self.item_cnn = nn.Conv1d(
            in_channels=item_embedding_matrix.shape[1], 
            out_channels=cnn_filters, 
            kernel_size=cnn_kernel_size
        )
        self.item_aspect_proj = nn.Linear(cnn_filters, id_dim)
        self.item_aspect_dropout = nn.Dropout(dropout)

        # --- Item ID Branch ---
        self.item_id_emb = nn.Embedding(num_items, id_dim)

        # --- Projection for Attention ---
        # Logic from TF: Concatenate (Aspect, ID) -> Linear -> Attention
        concat_dim = id_dim * 2 
        
        if id_dim % num_heads != 0:
            raise ValueError(f"id_dim ({id_dim}) must be divisible by num_heads ({num_heads}).")
        
        # In TF code, it projected to key_dim (id_dim // num_heads). 
        # CAUTION: Standard Transformer preserves dim. TF code reduced it. 
        # We follow TF logic: output dim = id_dim // num_heads.
        self.key_dim = id_dim // num_heads

        self.user_project = nn.Linear(concat_dim, self.key_dim)
        self.user_proj_dropout = nn.Dropout(dropout)
        
        self.item_project = nn.Linear(concat_dim, self.key_dim)
        self.item_proj_dropout = nn.Dropout(dropout)

        # --- Shared Self Attention ---
        self.sab = SelfAttentionBlock(
            num_heads=num_heads, 
            key_dim=self.key_dim, 
            dff=ffn_dim, 
            dropout_rate=dropout
        )

        # --- MLP ---
        # Input to MLP is Concat(User_Attn_Out, Item_Attn_Out)
        mlp_input_dim = self.key_dim * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU() # TF used ReLU output
        )

    def forward(self, user_id, item_id, user_seq, item_seq):
        # 1) User Text Branch
        u_txt = self.user_text_emb(user_seq)  # (B, L, Emb)
        u_txt = u_txt.permute(0, 2, 1)        # (B, Emb, L) for Conv1d
        u_txt = F.relu(self.user_cnn(u_txt))  # (B, Filters, L_new)
        # Global Max Pooling
        u_txt = F.adaptive_max_pool1d(u_txt, 1).squeeze(2) # (B, Filters)
        u_txt = self.user_aspect_proj(u_txt)
        u_txt = self.user_aspect_dropout(u_txt)

        # 2) User ID Branch
        u_id = self.user_id_emb(user_id).squeeze(1) # (B, ID_Dim)

        # 3) Item Text Branch
        i_txt = self.item_text_emb(item_seq)  # (B, L, Emb)
        i_txt = i_txt.permute(0, 2, 1)        # (B, Emb, L)
        i_txt = F.relu(self.item_cnn(i_txt))
        i_txt = F.adaptive_max_pool1d(i_txt, 1).squeeze(2)
        i_txt = self.item_aspect_proj(i_txt)
        i_txt = self.item_aspect_dropout(i_txt)

        # 4) Item ID Branch
        i_id = self.item_id_emb(item_id).squeeze(1) # (B, ID_Dim)

        # 5) Concatenate & Project
        u_vec = torch.cat([u_txt, u_id], dim=1)
        i_vec = torch.cat([i_txt, i_id], dim=1)

        u_vec = self.user_proj_dropout(self.user_project(u_vec))
        i_vec = self.item_proj_dropout(self.item_project(i_vec))

        # 6) Self Attention
        # Expand dims to (B, 1, D) for Attention
        u_vec = u_vec.unsqueeze(1)
        i_vec = i_vec.unsqueeze(1)

        u_att = self.sab(u_vec)
        i_att = self.sab(i_vec)

        # Flatten
        u_att = u_att.view(u_att.size(0), -1)
        i_att = i_att.view(i_att.size(0), -1)

        # 7) Final Fusion & Output
        final_vec = torch.cat([u_att, i_att], dim=1)
        output = self.mlp(final_vec)
        
        return output


# ============================================================
# 2) Dataset & DataLoader
# ============================================================

class RecommenderDataset(Dataset):
    def __init__(self, user_ids, item_ids, user_seq, item_seq, labels):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.user_seq = user_seq
        self.item_seq = item_seq
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "item_id": torch.tensor(self.item_ids[idx], dtype=torch.long),
            "user_seq": torch.tensor(self.user_seq[idx], dtype=torch.long),
            "item_seq": torch.tensor(self.item_seq[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def get_data_loader(
    args: dict,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_seq: np.ndarray,
    item_seq: np.ndarray,
    labels: np.ndarray,
    shuffle: bool = True,
):
    dataset = RecommenderDataset(user_ids, item_ids, user_seq, item_seq, labels)
    batch_size = args.get("batch_size", 128)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ============================================================
# 3) Trainer & Tester
# ============================================================

def proposed_trainer(
    args: dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_model_path: str,
    device: str = "cuda"
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.get("lr", 0.001))
    criterion = nn.MSELoss()

    epochs = args.get("num_epochs", 100)
    patience = args.get("patience", 5)
    
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Start Training on {device}...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            uid = batch["user_id"].to(device)
            iid = batch["item_id"].to(device)
            useq = batch["user_seq"].to(device)
            iseq = batch["item_seq"].to(device)
            label = batch["label"].to(device).unsqueeze(1) # (B, 1)

            pred = model(uid, iid, useq, iseq)
            loss = criterion(pred, label)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                uid = batch["user_id"].to(device)
                iid = batch["item_id"].to(device)
                useq = batch["user_seq"].to(device)
                iseq = batch["item_seq"].to(device)
                label = batch["label"].to(device).unsqueeze(1)

                pred = model(uid, iid, useq, iseq)
                loss = criterion(pred, label)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved Best Model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  -> Early Stopping Triggered")
                break

    # Load best weights
    model.load_state_dict(torch.load(best_model_path))
    return model


def proposed_tester(
    args: dict,
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for batch in test_loader:
            uid = batch["user_id"].to(device)
            iid = batch["item_id"].to(device)
            useq = batch["user_seq"].to(device)
            iseq = batch["item_seq"].to(device)
            label = batch["label"].to(device)

            pred = model(uid, iid, useq, iseq)
            
            preds_list.append(pred.cpu().numpy().flatten())
            trues_list.append(label.cpu().numpy().flatten())

    return np.concatenate(preds_list), np.concatenate(trues_list)