import os
import time
from pathlib import Path
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Adagrad
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

WORK_DIR = './'  # '/mnt/home/yangdeli/ece885/final_proj/'
PKL_DST_PATH = 'F:/NIST_data/nist_subset/'# WORK_DIR + 'data/'
FILE_PATH = 'F:/NIST_data/by_class/'
CLS_NUM = 62


class DataPreprocessor(object):

    def __init__(self):
        self.data = []
        self.dirs = []
        self.file_list = []
        self.file_num_list = []
        self.train_total_num = 0
        self.X_train = []
        self.Y_train = []

    def dir_list_gen(self):
        self.dirs = os.listdir(FILE_PATH)
        # print('Number of subdirectories in ' + FILE_PATH + ': ' + str(len(self.dirs)))

    def get_file_list(self):
        self.dir_list_gen()

        self.file_list = []
        for dirss in self.dirs:
            file_path = FILE_PATH + dirss + '/train_' + dirss  # e.g.: F:\NIST_data\by_class\4b\train_4b
            files = []
            for root, dirs, files in os.walk(file_path):
                print('Number of files in PATH \'' + file_path + '\': ' + str(len(files)))
            self.file_list.append(files)
            self.file_num_list.append(len(files))  # record the num of jpgs in each folder.
        print(len(self.file_list))
        self.train_total_num = sum(self.file_num_list)

    def load_pic_to_pkl(self):
        # train-test split, extract a subset of NIST
        print('Number of classes: ', len(self.dirs))
        # print(self.file_list)
        train_img_all = []
        test_img_all = []
        train_tag_all = []
        test_tag_all = []
        for i in range(0, len(self.dirs)):

            dir_name = self.dirs[i]
            print('Class:', dir_name)
            # get picture
            file_path = FILE_PATH + dir_name + '/train_' + dir_name  # e.g.: F:\NIST_data\by_class\4b\train_4b
            for root, dirs, files in os.walk(file_path):
                print('Number of files in PATH \'' + file_path + '\': ' + str(len(files)))
            # files_in_dir.append(files)
            files_in_dir = files

            train_size = 1024
            test_size = 256  # int(1 / 5 * pic_num)
            # traverse all the classes
            file_path_queue = []
            for pic in files_in_dir[: 5 * test_size]:
                file_path_queue.append(FILE_PATH + dir_name + '/train_' + dir_name + '/' + pic)
            pic_num = len(file_path_queue)

            # print('Class:', dir_name, 'total pics:', pic_num, 'train size:', train_size, 'test size:', test_size)

            for img_name in file_path_queue[:-test_size]:
                img = ki.load_img(img_name, grayscale=False, target_size=(64, 64))
                xx_trn = np.array(img)
                single_layer = xx_trn[:, :, 0]
                # G = xx_trn[:, :, 1]
                # B = xx_trn[:, :, 2]
                # img = Image.fromarray(single_layer, 'P')
                # img.show()
                # print(single_layer.shape)
                train_img_all.append(single_layer)
                train_tag_all.append([i])
            for img_name in file_path_queue[-test_size:]:
                img = ki.load_img(img_name, grayscale=False, target_size=(64, 64))
                xx_tst = np.array(img)
                single_layer = xx_tst[:, :, 0]
                test_img_all.append(single_layer)
                test_tag_all.append([i])

        X_train = np.asarray(train_img_all)
        X_test = np.asarray(test_img_all)
        Y_train = np.asarray(train_tag_all)
        Y_test = np.asarray(test_tag_all)

        print('Total train size:', X_train.shape, Y_train.shape)
        print('Total test size:', X_test.shape, Y_test.shape)

        data_set = ((X_train, Y_train), (X_test, Y_test))
        f_save_pkl = open(PKL_DST_PATH + 'train_test_64_128.pkl', 'wb')
        pickle.dump(data_set, f_save_pkl, pickle.HIGHEST_PROTOCOL)
        f_save_pkl.close()

    def save_all_test_pkl(self):
        self.dir_list_gen()

        x_test = []
        y_test = []

        for cls in self.dirs:
            with open(PKL_DST_PATH + cls + '_64.pkl', 'rb') as f_pkl:
                all_data = pickle.load(f_pkl)
                x_test += list(all_data[1][0].reshape((all_data[1][0].shape[0], 64, 64)))
                y_test += list(all_data[1][1])
                print(all_data[1][0].shape)
                # self.y_test = np_utils.to_categorical(self.y_test, CLS_NUM)
                del all_data
                f_pkl.close()
        print(len(y_test))
        x_test = np.asarray(x_test)
        test_pkl_tp = (x_test, y_test)

        f_save_pkl = open(PKL_DST_PATH + 'test.pkl', 'wb')
        pickle.dump(test_pkl_tp, f_save_pkl, pickle.HIGHEST_PROTOCOL)
        f_save_pkl.close()

    def normalizer(self):
        self.X_train = self.X_train.astype('float32')
        self.X_train /= 255

    def run(self):
        # self.dir_list_gen()

        self.dir_list_gen()
        print('Subfolder list:', self.dirs)
        self.load_pic_to_pkl()
        # self.save_all_test_pkl()
        # print(self.X_train.shape)
        # self.normalizer()'


class NISTProcessor(object):

    def __init__(self, model_name):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = Sequential()
        self.model_name = model_name
        self.model_path = WORK_DIR + self.model_name + '_nist_model_weights_v4.h5'
        self.dirs = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                     '41', '42', '43', '44', '45', '46', '47', '48', '49', '4a',
                     '4b', '4c', '4d', '4e', '4f', '50', '51', '52', '53', '54',
                     '55', '56', '57', '58', '59', '5a', '61', '62', '63', '64',
                     '65', '66', '67', '68', '69', '6a', '6b', '6c', '6d', '6e',
                     '6f', '70', '71', '72', '73', '74', '75', '76', '77', '78',
                     '79', '7a']

    def load_train_pkl(self, class_hex):
        # load only one pkl file.
        with open(PKL_DST_PATH + class_hex + '_64.pkl', 'rb') as f_pkl:
            all_data = pickle.load(f_pkl)
            self.x_train = all_data[0][0]#.reshape((all_data[0][0].shape[0], 64, 64))
            self.y_train = all_data[0][1]

            self.x_train = self.x_train.astype('float32')
            self.x_train /= 255

            # print(self.x_train[0].tolist())
            # img = Image.fromarray(self.x_train[0], 'P')
            # img.show()

            self.y_train = np_utils.to_categorical(self.y_train, CLS_NUM)  # convert to one-hot vector
            print('Class Hex:', class_hex)
            print('Training samples:', self.x_train.shape[0])
            # print(self.y_train.shape)
            del all_data
            f_pkl.close()

    def load_test_set(self):
        self.x_test = None
        self.y_test = None

        with open(PKL_DST_PATH + 'test.pkl', 'rb') as f_pkl:
            all_data = pickle.load(f_pkl)
            self.x_test = all_data[0].reshape((all_data[0].shape[0], 64, 64))
            self.y_test = np.asarray(all_data[1])
            self.y_test = np_utils.to_categorical(self.y_test, CLS_NUM)
            del all_data
            print('Test samples:', self.x_test.shape[0])
            f_pkl.close()

    def load_train_test_pkl(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        with open(PKL_DST_PATH + 'train_test_64_128.pkl', 'rb') as f_pkl:
            all_data = pickle.load(f_pkl)
            self.x_train = all_data[0][0]#.reshape((all_data[0][0].shape[0], 64, 64, 1))
            self.y_train = all_data[0][1]
            self.x_test = all_data[1][0]#.reshape((all_data[0][0].shape[0], 64, 64, 1))
            self.y_test = all_data[1][1]

            # img = Image.fromarray(self.x_train[0], 'P')
            # img.show()

            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255

            print(self.x_train.shape)

            self.y_train = np_utils.to_categorical(self.y_train, CLS_NUM)  # convert to one-hot vector
            self.y_test = np_utils.to_categorical(self.y_test, CLS_NUM)
            print('Training samples:', self.x_train.shape[0])
            # print(self.y_train.shape)
            del all_data
            f_pkl.close()

    def define_netowrk(self):
        self.load_train_test_pkl()
        kernel_size = (3, 3)
        self.model = Sequential()
        # self.model.add(Dense(200,
        #                      activation='tanh', input_shape=(64, 64)))

        # self.model.add(Convolution2D(64, kernel_size,
        #                      input_shape=self.x_train.shape[1:]))#(64, 64)
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D())
        if self.model_name == 'lstm':

            self.model.add(LSTM(implementation=1, units=100,
                                activation='tanh', input_shape=(64, 64)))

        elif self.model_name == 'basic':
            mgu_basic = MGUBasicModel(implementation=1, units=100,
                                      activation='tanh',
                                      input_shape=(64, 64))#self.x_train.shape[1:]
            self.model.add(mgu_basic)
        elif self.model_name == 'variant': 
            mgu_variant = MGUVariantModel(implementation=1, units=100,
                                          activation='tanh',
                                          input_shape=(64, 64))
            self.model.add(mgu_variant)            

        elif self.model_name == 'variant4':
            mgu_variant4 = MGUVariant4Model(implementation=1, units=100,
                                            activation='tanh', input_shape=(64, 64))
            self.model.add(mgu_variant4)
        self.model.add(Dense(CLS_NUM))
        self.model.add(Activation('softmax'))
        my_optimizer = Adadelta()# RMSprop(lr=0.005)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=my_optimizer,
                           metrics=['accuracy'])
        self.model.save(self.model_path)
        print('Model built successfully, saved to local disk.')
        del self.model

    def train_test_process(self):
        print('>> Training Phase')
        self.load_train_test_pkl()
        # input image dimensions
        self.model = None
        if self.model_name == 'basic':
            # load weights
            self.model = load_model(self.model_path,
                                    custom_objects={'MGUBasicModel': MGUBasicModel})
        elif self.model_name == 'variant':
            self.model = load_model(self.model_path,
                                    custom_objects={'MGUVariantModel': MGUVariantModel})
        elif self.model_name == 'lstm':
            self.model = load_model(self.model_path)
        elif self.model_name == 'variant4':
            self.model = load_model(self.model_path,
                                    custom_objects={'MGUVariant4Model': MGUVariant4Model})
        else:
            print('Error: Invalid input argument.')

        print(self.model.summary())

        # print(self.x_train.shape[0], 'train samples')

        csv_path = WORK_DIR + 'result_output/' + self.model_name + '_nist_log_v4.log'
        csv_logger = callbacks.CSVLogger(csv_path, append=False)

        self.model.fit(self.x_train, self.y_train,
                       batch_size=128,
                       epochs=100,
                       verbose=2,
                       validation_data=(self.x_test, self.y_test),
                       callbacks=[csv_logger])
        
        self.model.save(self.model_path)
