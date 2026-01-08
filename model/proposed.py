# model/proposed.py

import numpy as np
import tensorflow as tf

from typing import Dict, Tuple, Optional

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Flatten,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
    Concatenate,
    Lambda,
    Layer,
    MultiHeadAttention,
    LayerNormalization,
    Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 0) Blocks
# ============================================================

def point_wise_feed_forward_network(d_model: int, dff: int) -> Sequential:
    return Sequential(
        [
            Dense(dff, activation="relu"),
            Dense(d_model, activation="linear"),
        ],
        name="PointWiseFFN",
    )


class SelfAttentionBlock(Layer):
    """
    Transformer-style self-attention block:
    MHA -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        dff: int,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            name="MHA",
        )
        self.dropout = Dropout(dropout_rate, name="Dropout")
        self.add = Add(name="Add")
        self.ln = LayerNormalization(epsilon=epsilon, name="LayerNorm")
        self.ffn = point_wise_feed_forward_network(key_dim, dff)

    def call(self, inputs, training=None):
        # inputs: (B, T, D)
        x = inputs

        att = self.mha(x, x, x)
        att = self.dropout(att, training=training)
        x = self.add([x, att])
        x = self.ln(x)

        ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out, training=training)
        x = self.add([x, ffn_out])
        x = self.ln(x)

        return x


# ============================================================
# 1) Proposed Model (MyModel)
# ============================================================

def build_proposed(
    num_users: int,
    num_items: int,
    user_vocab_size: int,
    item_vocab_size: int,
    user_maxlen: int,
    item_maxlen: int,
    user_embedding_matrix: np.ndarray,
    item_embedding_matrix: np.ndarray,
    *,
    num_heads: int = 8,
    id_dim: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    cnn_filters: int = 100,
    cnn_kernel_size: int = 5,
    ffn_dim: int = 2048,
    model_name: str = "Proposed",
) -> Model:
    """
    Inputs (4):
      - user_id_input: (B, 1)
      - item_id_input: (B, 1)
      - user_aspect_input: (B, user_maxlen)
      - item_aspect_input: (B, item_maxlen)

    Output:
      - rating prediction: (B, 1)
    """

    # --- 1) user text: w2v embedding -> CNN -> pooling -> linear
    user_aspect_input = Input(shape=(user_maxlen,), name="Input_UserText")
    user_aspect_embedding = Embedding(
        input_dim=user_vocab_size + 1,
        output_dim=user_embedding_matrix.shape[1],
        weights=[user_embedding_matrix],
        input_length=user_maxlen,
        trainable=False,
        name="Embedding_UserText",
    )(user_aspect_input)

    user_cnn = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        activation="relu",
        name="User_CNN",
    )(user_aspect_embedding)

    user_aspect_vec = GlobalMaxPooling1D(name="User_GlobalMaxPooling")(user_cnn)
    user_aspect_vec = Dense(id_dim, activation="linear", name="User_Aspect_Linear")(user_aspect_vec)
    user_aspect_vec = Dropout(dropout, name="Dropout_UserAspect")(user_aspect_vec)

    # --- 2) user id embedding
    user_id_input = Input(shape=(1,), name="user_id_input")
    user_id_emb = Embedding(input_dim=num_users, output_dim=id_dim, name="user_id_emb")(user_id_input)
    user_id_vec = Flatten(name="UserIDFlatten")(user_id_emb)

    # --- 3) item text: w2v embedding -> CNN -> pooling -> linear
    item_aspect_input = Input(shape=(item_maxlen,), name="Input_ItemText")
    item_aspect_embedding = Embedding(
        input_dim=item_vocab_size + 1,
        output_dim=item_embedding_matrix.shape[1],
        weights=[item_embedding_matrix],
        input_length=item_maxlen,
        trainable=False,
        name="Embedding_ItemText",
    )(item_aspect_input)

    item_cnn = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        activation="relu",
        name="Item_CNN",
    )(item_aspect_embedding)

    item_aspect_vec = GlobalMaxPooling1D(name="Item_GlobalMaxPooling")(item_cnn)
    item_aspect_vec = Dense(id_dim, activation="linear", name="Item_Aspect_Linear")(item_aspect_vec)
    item_aspect_vec = Dropout(dropout, name="Dropout_ItemAspect")(item_aspect_vec)

    # --- 4) item id embedding
    item_id_input = Input(shape=(1,), name="item_id_input")
    item_id_emb = Embedding(input_dim=num_items, output_dim=id_dim, name="item_id_emb")(item_id_input)
    item_id_vec = Flatten(name="ItemIDFlatten")(item_id_emb)

    # --- 5) concat (user) / concat (item)
    user_vec = Concatenate(name="Concat_UserReviewID")([user_aspect_vec, user_id_vec])
    item_vec = Concatenate(name="Concat_ItemReviewID")([item_aspect_vec, item_id_vec])

    # --- 6) self-attention on projected vectors
    if id_dim % num_heads != 0:
        raise ValueError(f"id_dim({id_dim}) must be divisible by num_heads({num_heads}).")

    key_dim = id_dim // num_heads  

    user_vec = Dense(units=key_dim, activation="linear", name="A_project")(user_vec)
    user_vec = Dropout(dropout, name="Dropout_Aproject")(user_vec)
    item_vec = Dense(units=key_dim, activation="linear", name="B_project")(item_vec)
    item_vec = Dropout(dropout, name="Dropout_Bproject")(item_vec)

    # (B, D) -> (B, 1, D)
    user_vec = Lambda(lambda x: tf.expand_dims(x, axis=1), name="Expand_User")(user_vec)
    item_vec = Lambda(lambda x: tf.expand_dims(x, axis=1), name="Expand_Item")(item_vec)

    sab = SelfAttentionBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        dff=ffn_dim,
        dropout_rate=dropout,
        name="SelfAttentionBlock",
    )

    user_att = sab(user_vec)
    item_att = sab(item_vec)

    user_att = Flatten(name="Flatten_UserAtt")(user_att)
    item_att = Flatten(name="Flatten_ItemAtt")(item_att)

    # --- 7) final fusion
    final = Concatenate(name="Concat_UserItem")([user_att, item_att])

    # --- 8) prediction MLP
    dense = Dense(128, activation="linear", name="Dense_128")(final)
    dense = Dropout(dropout, name="Dropout_128")(dense)
    dense = Dense(64, activation="linear", name="Dense_64")(dense)
    dense = Dropout(dropout, name="Dropout_64")(dense)
    output = Dense(1, activation="relu", name="Output")(dense)

    model = Model(
        inputs=[user_id_input, item_id_input, user_aspect_input, item_aspect_input],
        outputs=output,
        name=model_name,
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error", "mean_squared_error"],
    )

    return model


# ============================================================
# 2) tf.data.Dataset 생성: get_data_loader
# ============================================================

def get_data_loader(
    args: dict,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    user_seq: np.ndarray,
    item_seq: np.ndarray,
    labels: np.ndarray,
    *,
    shuffle: bool = True,
) -> tf.data.Dataset:

    x_dict = {
        "user_id_input": np.asarray(user_ids).astype("int32"),
        "item_id_input": np.asarray(item_ids).astype("int32"),
        "Input_UserText": np.asarray(user_seq).astype("int32"),
        "Input_ItemText": np.asarray(item_seq).astype("int32"),
    }
    y = np.asarray(labels).astype("float32")

    batch_size = args.get("batch_size", 128)
    seed = args.get("seed", 42)

    ds = tf.data.Dataset.from_tensor_slices((x_dict, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 3) Trainer & Tester
# ============================================================

def proposed_trainer(
    args: dict,
    model: Model,
    train_loader: tf.data.Dataset,
    val_loader: tf.data.Dataset,
    best_model_path: str,
):
    epochs = args.get("num_epochs", 100)
    patience = args.get("patience", 5)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def proposed_tester(
    args: dict,
    model: Model,
    test_loader: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = model.predict(test_loader).reshape(-1)

    trues_list = []
    for _, y in test_loader:
        trues_list.append(y.numpy())
    trues = np.concatenate(trues_list, axis=0).reshape(-1)

    return preds, trues
