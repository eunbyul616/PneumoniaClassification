import argparse
import tensorflow as tf
import os
from model import BiLSTMAttentionModel
from utils import DataPreprocessing
import json


class Train:
    def __init__(self, config):
        self.config = config
        train_data = DataPreprocessing(self.config)
        train_x, train_y = train_data.get_preprocessed_data(file=config['dataset']['raw']['train_file'])
        train_y = tf.keras.utils.to_categorical(train_y)
        print('Train Data Shape: ', train_x.shape, train_y.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.train_dataset = train_dataset.shuffle(len(train_x)).batch(self.config['dataset']['batch_size'],
                                                                       drop_remainder=True)

        self.model = BiLSTMAttentionModel(num_units=self.config['hyper_params']['num_units'],
                                          num_label=self.config['hyper_params']['num_label'],
                                          max_length=self.config['hyper_params']['max_length'],
                                          lr=self.config['hyper_params']['lr'],
                                          embedding_dim=self.config['hyper_params']['embedding_dim'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['hyper_params']['lr'])
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.train_acc_metric = tf.keras.metrics.CategoricalCrossentropy()

        self.checkpoint_dir = config['train']['save_dir']
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, config['train']['checkpoint_prefix'])
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)

        return loss_value

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.config['metrics'])

    def train(self):
        for epoch in range(self.config['train']['epochs']):
            print("\nEpoch: %d" % (epoch,))

            for step, (x_batch, y_batch) in enumerate(self.train_dataset):
                loss_value = self.train_step(x_batch, y_batch)

                if step % 200 == 0:
                    print("Training loss %d: %.4f" % (step, float(loss_value)))

            train_acc = self.train_acc_metric.result()
            print("Training Accurary: %.4f" % (float(train_acc),))

            self.train_acc_metric.reset_states()

            if (epoch + 1) % 50 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print("Save Model: %d" % (epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default='config.json')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    model_train = Train(config=config)
    model_train.compile_model()
    model_train.train()
