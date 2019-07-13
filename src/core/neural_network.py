import math
import time
import keras
import pickle
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation

# learning rate decay


class NeuralNetwork:
    def __init__(self, usr_config: Config, aux_config: Config):
        pass

    def _generate_training_data(self, seed_file_min: int, seed_file_max: int) -> Tuple[np.array, np.array]:
        """
        Generates the training data

        Parameters
        ----------
        seed_file_min: int
            Lower bound of seed
        seed_file_max: int
            Upper bound of byte range

        Returns
        -------
        seed: Numpy2D
            Input to the deep neural network. Represent the byte sequence of the seeds parsed into the DNN
        bitmap: Numpy2D
            Invoked branchs after having parsed the input seeds to the target program
        """
        # Initialize the seeds and bitmap to zero
        n_rows: int = seed_file_max - seed_file_min
        seed: Numpy2D = np.zeros(
            (n_rows, self.aux_config.core.MAX_FILE_SIZE))
        bitmap: Numpy2D = np.zeros(
            (n_rows, self.aux_config.core.MAX_BITMAP_SIZE))

        # Populate the input tensor.
        # One row for every seed file. For seed files smaller than the
        # MAX_FILE_SIZE, those extra bytes will be zero.
        for i in range(seed_file_min, seed_file_max):
            tmp = open(self.seed_list[i], 'r').read()
            ln = len(tmp)
            if ln < MAX_FILE_SIZE:
                tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
            seed[i - seed_file_min] = [ord(byte_str) for byte_str in list(tmp)]

        # Populate the output tensor.
        # One row for every seed file. But, this time the columns will the
        # invoked branches indices.
        for i in range(seed_file_min, seed_file_max):
            file_name = "./bitmaps/" + \
                self.seed_list[i].split('/')[-1] + ".npy"
            bitmap[i - seed_file_min] = np.load(file_name)

        return seed, bitmap

    def vectorize_file(self, fl: pathlib.PosixPath) -> Numpy2D:
        """
        Generate a vector representation of a file.

        Parameters
        ----------
        fl: pathlib.PosixPath
            Path to the file

        Returns
        -------
        seed: Numpy2D
            A 2D numpy array with 1 row of float vector.
        """
        seed: Numpy2D = np.zeros((1, MAX_FILE_SIZE))
        tmp = open(fl, 'r').read()
        ln = len(tmp)
        if ln < self.aux_config.core.MAX_FILE_SIZE:
            tmp = tmp + (MAX_FILE_SIZE - ln) * '\0'
        seed[0] = [ord(byte_str) for byte_str in list(tmp)]
        seed = seed.astype('float32') / 255
        return seed


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.7
    epochs_drop = 10.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print(step_decay(len(self.losses)))

# compute jaccard accuracy for multiple label


def accur_1(y_true, y_pred):
    y_true = tf.round(y_true)
    pred = tf.round(y_pred)
    summ = tf.constant(MAX_BITMAP_SIZE, dtype=tf.float32)
    wrong_num = tf.subtract(summ, tf.reduce_sum(
        tf.cast(tf.equal(y_true, pred), tf.float32), axis=-1))
    right_1_num = tf.reduce_sum(tf.cast(tf.logical_and(
        tf.cast(y_true, tf.bool), tf.cast(pred, tf.bool)), tf.float32), axis=-1)
    ret = K.mean(tf.divide(right_1_num, tf.add(right_1_num, wrong_num)))
    return ret


def train_generate(batch_size):
    global seed_list
    while 1:
        np.random.shuffle(seed_list)
        # load a batch of training data
        for i in range(0, NUM_SEEDS, batch_size):
            # load full batch
            if (i + batch_size) > NUM_SEEDS:
                x, y = generate_training_data(i, NUM_SEEDS)
                x = x.astype('float32') / 255
            # load remaining data for last batch
            else:
                x, y = generate_training_data(i, i + batch_size)
                x = x.astype('float32') / 255
            yield (x, y)


def build_model():
    batch_size = 32
    num_classes = MAX_BITMAP_SIZE
    epochs = 50

    model = Sequential()
    model.add(Dense(4096, input_dim=MAX_FILE_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.adam(lr=0.0001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[accur_1])
    model.summary()

    return model


def train(model):
    loss_history = LossHistory()
    lrate = keras.callbacks.LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate]
    model.fit_generator(train_generate(16),
                        steps_per_epoch=(NUM_SEEDS / 16 + 1),
                        epochs=100,
                        verbose=1, callbacks=callbacks_list)
    # Save model and weights
    model.save_weights("hard_label.h5")
