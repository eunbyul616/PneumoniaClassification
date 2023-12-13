import argparse
import tensorflow as tf
from utils import DataPreprocessing
import json


class Test:
    def __init__(self, config, model):
        self.config = config
        self.model = tf.keras.saving.load_model(model)

        test_data = DataPreprocessing(self.config)
        test_x, test_y = test_data.get_preprocessed_data(file=config['dataset']['raw']['test_file'])
        test_y = tf.keras.utils.to_categorical(test_y)
        print('Test Data Shape: ', test_x.shape, test_y.shape)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        self.test_dataset = test_dataset.shuffle(len(test_x)).batch(self.config['dataset']['batch_size'],
                                                                    drop_remainder=True)

        self.acc_metric = tf.keras.metrics.CategoricalCrossentropy()

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        self.acc_metric.update_state(y, val_logits)

    def evaluate(self):
        for x_batch, y_batch in self.test_dataset:
            self.test_step(x_batch, y_batch)

        acc = self.acc_metric.result()
        self.acc_metric.reset_states()
        print("Test Accuracy: %.4f" % (float(acc),))

    def predict(self, input_data):
        pred = self.model.predict(input_data)
        return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=False, default='config.json')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    model_test = Test(config=config, model=config['test']['model_dir'])
    model_test.evaluate()
