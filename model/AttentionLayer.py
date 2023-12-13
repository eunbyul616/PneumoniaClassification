import tensorflow as tf


class AttentionLayers(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units

    def build(self, input_shape):
        self.w1 = tf.keras.layers.Dense(self.num_units)
        self.w2 = tf.keras.layers.Dense(self.num_units)
        self.v = tf.keras.layers.Dense(1)

    def call(self, h_s, c_s, output):
        query = tf.concat([h_s, c_s], axis=-1)
        query = tf.keras.layers.RepeatVector(output.shape[2])(query)
        perm = tf.keras.layers.Permute((2, 1))(output)

        score = tf.nn.tanh(self.w1(perm) + self.w2(query))
        score = self.v(score)
        score = tf.keras.layers.Permute((2, 1))(score)

        attn_weights = tf.nn.softmax(score)
        context = attn_weights * output

        return context, attn_weights
