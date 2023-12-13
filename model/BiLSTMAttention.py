import tensorflow as tf
from model.AttentionLayer import AttentionLayers


class BiLSTMAttentionModel(tf.keras.models.Model):
    def __init__(self, num_units, num_label, max_length, lr, embedding_dim):
        super().__init__()

        self.num_units = num_units
        self.num_label = num_label
        self.lr = lr

        self.input_ly = tf.keras.layers.InputLayer(input_shape=(max_length, embedding_dim))
        self.lstm_fw = tf.keras.layers.LSTM(units=self.num_units, return_sequences=True, return_state=True)
        self.lstm_bw = tf.keras.layers.LSTM(units=self.num_units, go_backwards=True, return_sequences=True,
                                            return_state=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm_fw, backward_layer=self.lstm_bw)
        self.attn_ly = AttentionLayers(self.num_units)
        self.dense = tf.keras.layers.Dense(self.num_label)

    def call(self, x):
        seq = self.input_ly(x)
        output, fw_h, fw_c, bw_h, bw_c = self.bi_lstm(x)
        h_s = tf.keras.layers.Concatenate()([fw_h, bw_h])
        c_s = tf.keras.layers.Concatenate()([fw_c, bw_c])

        self.context, self.attn_weights = self.attn_ly(h_s, c_s, output)
        seq = tf.reduce_sum(self.context, 1)
        seq = tf.nn.tanh(seq)
        out = self.dense(seq)

        return out
