import os
import time
from pathlib import Path
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.optimizers import SGD, Adadelta, RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks
from MGUModel import MGUBasicModel, MGUVariantModel, MGUVariant4Model
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing import image as ki
from PIL import Image

__author__ = 'Deliang Yang'

CLS_NUM = 10  # total classes, ten written digit


def model_def_load(function_flag, model_name):

    if function_flag == 0:
        # Define new model. Model will not be trained unless function flag is set to 1 and rerun the code.
        print('Define new model:', model_name)
        hidden_units = 100

        model = Sequential()
        if model_name == 'basic':
            mgu_basic = MGUBasicModel(implementation=1, output_dim=hidden_units,
                                      activation='tanh', input_shape=(28, 28))

            model.add(mgu_basic)
        elif model_name == 'variant':
            mgu_variant = MGUVariantModel(implementation=1, output_dim=hidden_units,
                                          activation='tanh', input_shape=(28, 28))
            model.add(mgu_variant)
        elif model_name == 'lstm':
            model.add(LSTM(implementation=1, output_dim=hidden_units,
                           activation='tanh', input_shape=(28, 28)))

        elif model_name == 'variant4':
            mgu_variant4 = MGUVariant4Model(implementation=1, output_dim=hidden_units,
                                            activation='tanh', input_shape=(28, 28))
            model.add(mgu_variant4)
        else:
            print('Invalid argument, exit.')
            return None
        my_sgd = SGD(lr=0.03)
        model.add(Dense(CLS_NUM))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=my_sgd,
                      metrics=['accuracy'])
        model.save('./' + model_name + '_mnist_model_weights.h5')
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
        model = load_model('./' + model_name + '_mnist_model_weights.h5', custom_objects=custom_layer)
        return model


def train_test_process(model, model_name):

    batch_size = 128
    nb_epochs = 100

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, CLS_NUM)
    Y_test = np_utils.to_categorical(y_test, CLS_NUM)

    csv_path = './result_output/' + model_name + '_mnist_log.log'
    csv_logger = callbacks.CSVLogger(csv_path, append=True)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
              verbose=2, validation_data=(X_test, Y_test), callbacks=[csv_logger])

    model.save('./' + model_name + '_mnist_model_weights.h5')


def run():

    start_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print('Task begins. Time stamp: ' + current_time)

    model_name = 'variant4'
    model = model_def_load(function_flag=0, model_name=model_name)

    if model is not None:
        print(model.summary())
        train_test_process(model, model_name)

    end_time = time.time()
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))

    print(current_time)
    print('Total execution time: ' + '%.3f' % (end_time-start_time) + ' s')


if __name__ == '__main__':
    run()
