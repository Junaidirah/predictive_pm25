import tensorflow as tf
import numpy as np

def build_lstm(input_width, n_features):
    """Model Murni LSTM"""
    inputs = tf.keras.layers.Input(shape=(input_width, n_features))
    
    # LSTM Layer: Kurangi units jadi 32, tambah L2 regularization
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32, 
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.002),
            recurrent_regularizer=tf.keras.regularizers.l2(0.002)
        )
    )(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Dense Layer: Kurangi units jadi 16, tambah L2 & Dropout ekstra
    x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output Layer
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
