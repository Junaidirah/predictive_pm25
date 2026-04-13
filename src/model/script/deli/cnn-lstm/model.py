import tensorflow as tf
import numpy as np

def build_strong_lstm(input_width, n_features):
    inputs = tf.keras.layers.Input(shape=(input_width, n_features))
    
    # CNN Layer
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM Layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Dense Layer
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
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
