import tensorflow as tf
import numpy as np

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Self-attention
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.002))(res)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer(input_width, n_features, head_size=32, num_heads=2, ff_dim=32, num_transformer_blocks=2, mlp_units=[16], dropout=0.3):
    """Model Time-Series Transformer - Dioptimasi Anti-Overfitting"""
    inputs = tf.keras.layers.Input(shape=(input_width, n_features))
    
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in mlp_units:
        x = tf.keras.layers.Dense(dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        
    outputs = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.models.Model(inputs, outputs)

class R2Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            val_preds, val_labels = [], []
            for inputs, labels in self.val_dataset:
                preds = self.model.predict(inputs, verbose=0)
                val_preds.append(preds)
                val_labels.append(labels.numpy())
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            ss_res = np.sum((val_labels - val_preds)**2)
            ss_tot = np.sum((val_labels - np.mean(val_labels))**2)
            r2 = 1 - (ss_res / ss_tot)
            print(f"Epoch {epoch}: val_loss={logs['val_loss']:.4f}, val_R2={r2:.4f}")
