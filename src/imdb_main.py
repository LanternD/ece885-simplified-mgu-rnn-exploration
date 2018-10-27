from __future__ import print_function
import os
import time
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from MGUModel import MGUBasicModel, MGUVariantModel, MGUVariant4Model
from keras.models import load_model
from keras.datasets import imdb
import numpy as np

__author__ = 'Deliang Yang'

CLS_NUM = 1  # total classes, ten written digit
max_features = 20000  # vocabulary size


def model_def_load(function_flag, model_name):

    if function_flag == 0:
        # Define new model. Model will not be trained unless function flag is set to 1 and rerun the code.
        print('Define new model:', model_name)
        hidden_units = 100

        model = Sequential()
        model.add(Embedding(max_features, 128))
        if model_name == 'basic':
            mgu_basic = MGUBasicModel(implementation=1, output_dim=hidden_units,
                                      activation='tanh')

            model.add(mgu_basic)
        elif model_name == 'variant':
            mgu_variant = MGUVariantModel(implementation=1, output_dim=hidden_units,
                                          activation='tanh')
            model.add(mgu_variant)
        elif model_name == 'lstm':
            model.add(LSTM(implementation=1, output_dim=hidden_units,
                           activation='tanh'))

        elif model_name == 'variant4':
            mgu_variant4 = MGUVariant4Model(implementation=1, output_dim=hidden_units,
                                            activation='tanh', input_shape=(28, 28))
            model.add(mgu_variant4)
        else:
            print('Invalid argument, exit.')
            return None

        model.add(Dense(CLS_NUM, activation='sigmoid'))
        # my_sgd = SGD(lr=0.03)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.save('./' + model_name + '_imdb_model_weights.h5')
        return None
    elif function_flag == 1:
        # load model that is already defined
        print('Loading existing model:', model_name)
        if model_name == 'basic':
            custom_layer = {'MGUBasicModel': MGUBasicModel}
        elif model_name == 'variant':
            custom_layer = {'MGUVariantModel': MGUVariantModel}
        elif model_name == 'variant4':
            custom_layer = {'MGUVariant4Model': MGUVariant4Model}
        else:
            custom_layer = None
        model = load_model('./' + model_name + '_imdb_model_weights.h5', custom_objects=custom_layer)
        return model


def train_test_process(model, model_name):

    batch_size = 128
    nb_epochs = 100
    maxlen = 100

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    # convert class vectors to binary class matrices

    csv_path = './result_output/' + model_name + '_imdb_log.log'
    csv_logger = callbacks.CSVLogger(csv_path, append=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs,
              verbose=2, validation_data=(X_test, y_test), callbacks=[csv_logger])

    model.save('./' + model_name + '_imdb_model_weights.h5')


def run():

    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. Time stamp: ' + current_time)

    model_name = 'variant4'
    model = model_def_load(function_flag=1, model_name=model_name)

    if model is not None:
        print(model.summary())
        train_test_process(model, model_name)

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time-start_time) + ' s')


if __name__ == '__main__':
    run()
