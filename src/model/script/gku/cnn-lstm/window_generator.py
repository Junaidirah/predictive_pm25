import numpy as np
import tensorflow as tf

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 feature_columns, label_columns):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift + label_width - 1
        self.all_columns = feature_columns + label_columns

    def split_window(self, features):
        inputs = features[:, :self.input_width, :len(self.feature_columns)]
        label_start = self.input_width + self.shift - 1
        labels = features[:, label_start:label_start + self.label_width, len(self.feature_columns):]
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

    @property
    def train(self):
        return self.make_dataset(self.train_df, shuffle=True)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df, shuffle=False)
        
    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)