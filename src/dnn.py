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

#learning rate decay


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

#compute jaccard accuracy for multiple label


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
        for i in range(0, SPLIT_RATIO, batch_size):
            # load full batch
            if (i + batch_size) > SPLIT_RATIO:
                x, y = generate_training_data(i, SPLIT_RATIO)
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
                        steps_per_epoch=(SPLIT_RATIO / 16 + 1),
                        epochs=100,
                        verbose=1, callbacks=callbacks_list)
    # Save model and weights
    model.save_weights("hard_label.h5")
