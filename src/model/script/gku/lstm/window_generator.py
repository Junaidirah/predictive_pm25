import tensorflow as tf
import pandas as pd
import numpy as np


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 feature_columns, label_columns):
        # Simpan dataframe asli (sudah dinormalisasi)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Kolom fitur dan label
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.all_columns = feature_columns + label_columns

        # Parameter windowing
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :len(self.feature_columns)]
        labels = features[:, self.labels_slice, len(self.feature_columns):]

        # Set shape eksplisit agar TF graph-mode mengenal dimensi
        inputs.set_shape([None, self.input_width, len(self.feature_columns)])
        labels.set_shape([None, self.label_width, len(self.label_columns)])

        # Squeeze ke [batch, label_width] agar cocok dengan output Dense(1) model
        # Tanpa ini, label shape [batch,1,1] vs model output [batch,1] → loss mismatch
        labels = tf.squeeze(labels, axis=-1)   # [batch, label_width, 1] → [batch, label_width]

        return inputs, labels

    def make_dataset(self, data, shuffle=True):
        data_array = data[self.all_columns].values.astype(np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data_array,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=32,
        )
        return ds.map(self.split_window)

    def plot_df(self, data_type='train'):
        """Mengambil satu batch dan menampilkan input/label dalam bentuk DataFrame."""
        if data_type == 'train':
            ds = self.train
        elif data_type == 'val':
            ds = self.val
        else:
            ds = self.test

        inputs, labels = next(iter(ds))
        # Ambil contoh pertama dari batch
        example_input = inputs[0].numpy()
        example_label = labels[0].numpy()

        df_input = pd.DataFrame(example_input, columns=self.feature_columns)
        df_input['Type'] = 'Input'

        df_label = pd.DataFrame(example_label, columns=self.label_columns)
        df_label['Type'] = 'Target'

        return pd.concat([df_input, df_label], axis=0).reset_index(drop=True)

    @property
    def train(self):
        return self.make_dataset(self.train_df, shuffle=True)

    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Column names: {self.all_columns}'])