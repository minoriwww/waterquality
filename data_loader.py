# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd
import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="" #Prevent TensorFlow from accessing the GPU? [duplicate]

from PIL import Image
import cv2
import csv
import re
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
import time

from KFold import *
from bootstraping import MultiClassficationBootstraping
import CNN_Regression

data_path = './Images'

# img_height = 3024
# img_width = 4032


img_height = 224
img_width = 224
nb_channels = num_channels =3 #1

BATCH_SIZE = 4 #8 16 1
RANDOM_STATE = 42
time_str = str(time.time())
BIN_SIZE = 20#20
MAX_LIM = 200#200
MIN_LIM = 20

IS_SCALE = False
nb_epochs = 120

###########################################################################
# processing training data
###########################################################################

def pre_training_data(is_scaler=True, is_categorical=False, bin_method='bins', is_1hot_categ=True):
    u"""
    return train, label, vali_train, vali_label,X_test,y_test
    type: ndarray
    """
    # only test  image3
    X_train, X_val, y_train, y_val = None, None, None, None
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images2')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images3')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images4')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images6')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images7_high_range')
    # X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images8_high_range')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images9_200_300')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images5')

    # X_test, X_val, y_test_origin, y_val_origin = train_test_split(X_val, y_val_origin, test_size=0.5, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.6, shuffle=True)
    print("train shape", str(X_train.shape))
    print("test shape",str(X_test.shape))
    print("val shape",str(X_val.shape))


    if is_scaler:

        # scaler = preprocessing.MaxAbsScaler() MaxAbsScaler
        scaler = preprocessing.MaxAbsScaler()#StandardScaler
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_train = y_train.flatten()

        joblib.dump(scaler, 'MaxAbsScaler.pkl')

        # scaler_val = preprocessing.MaxAbsScaler()
        y_val = scaler.transform(y_val.reshape(-1, 1))
        y_val = y_val.flatten()


        y_test = scaler.transform(y_test.reshape(-1, 1))
        y_test = y_test.flatten()

        # joblib.dump(scaler_val, 'MaxAbsScaler_vali.pkl')

    if is_categorical:
        y_test = np.array([processing_y(i, default_method=bin_method) for i in y_test])
        y_train = np.array([processing_y(i, default_method=bin_method) for i in y_train])
        y_val = np.array([processing_y(i, default_method=bin_method) for i in y_val])
        is_categorical = False
        print(y_val)
        print(y_val.shape)
        if len(np.unique(y_train)) >= 2 and is_1hot_categ == True:
            is_categorical = True
            y_train = keras.utils.to_categorical(y_train)
            y_val = keras.utils.to_categorical(y_val)
            y_test = keras.utils.to_categorical(y_test)

    # print(y_test)###
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # print(X_train.shape)###
    # print(X_val.shape)###
    # print(X_test.shape)###
    # print(y_test)###
    # (train,label),(vali_train,vali_label) = CNN_Regression.load_data()
    train = X_train
    label = y_train
    vali_train = X_val
    vali_label = y_val

    return train, label, vali_train, vali_label,X_test,y_test



def post_training_data(y_test, y_pred, is_scaler=True ):
    if not is_scaler:
        y_test = y_test
        y_pred = y_pred
    else:
        scaler_val = joblib.load('MaxAbsScaler.pkl')
        y_test, y_pred = transform_y(y_test, y_pred, scaler_val)

    np.save('y_pred.npy', y_pred)
    np.save('y_test_transformed.npy', y_test)
    print(y_pred.shape)###
    print(y_test.shape)###
    y_pred = y_pred.flatten()
    print(y_pred)###
    print(y_test)###
    # y_test = y_test.flatten()

    # if is_categorical:
    #     y_pred = y_pred.reshape(-1,2)
    #     y_test = y_test[:,0]
    #     y_pred = y_pred[:,0]

    ########### margin cut off ###########
    y_pred[np.where(y_pred>500)  ] = 500
    y_pred[np.where(y_pred<0) ] = 0
    return y_pred, y_test


BIN_SIZE = 20#20
MAX_LIM = 200#200
MIN_LIM = 20
nb_bins = (MAX_LIM - MIN_LIM)/BIN_SIZE

IS_SCALE = False
nb_epochs = 1000

def save_result(name,y_pred,y_true):
    with open(name,"a") as csvfile:
        writer = csv.writer(csvfile)
        y_diff = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
        y = [y_pred, y_true, y_diff]
        mse = 0.0
        for num in range(len(y_diff)):
            mse += pow(y_diff[num],2) / len(y_diff)
        rmse = mse^(0.5)
        print('rmse = ', rmse)
        writer.writerows(y)

def processing_y(value_float, default_method='bins', bin_size=BIN_SIZE, max_lim=MAX_LIM, min_lim=MIN_LIM):
    """
    default_method='bins': devide the range equally
    0 - min_lim: label 0
    min_lim - max_lim: label int(value_float/bin_size)+1
    >= max_lim: label is number of bins + 1

    default_method='threebins': devide the range to 3 bins
    0 - min_lim: label 0
    min_lim - max_lim: label 1
    >= max_lim: label is 2

    return:
    y after binning (e.g. 200->0.8)
    """
    if default_method is None:
        return value_float

    if default_method == 'bins':
        nb_bins = (max_lim - min_lim)/bin_size
        if value_float >= max_lim: return nb_bins+1
        if value_float < min_lim: return 0
        return int(value_float/bin_size)+1

    if default_method == 'bins_reverse':
        return float(bin_size * value_float)

    if default_method == 'threebins':
        if value_float >= max_lim: return 2
        if value_float < min_lim: return 0
        return 1

def load_data_200(data=None, vali_data=None, label=None,
    vali_label=None,
    data_path = './Images',
    test_size_ratio=0.2
    ):
    """return
    img_height, img_width, nb_channels, train_data_len, self.val_data_len
    (channel last)
    save:
    ---
    self.X_train = None
    self.y_train = None
    self.train_data_len = 0

    self.X_val = None
    self.y_val = None
    self.val_data_len = 0

    """
    ################################# equal size bin number ###########################
    # nb_class_factor = 51 # nbclass = 4 200
    nb_class_factor = 200
    #######################################
    num_images = 1

    for lists in os.listdir(data_path):
        sub_path = os.path.join(data_path, lists)
        # print(sub_path)
        if os.path.isfile(sub_path):
            num_images += 1

    X = np.zeros((num_images, img_height, img_width, nb_channels), dtype=np.float32)
    y = np.ones(num_images, dtype=np.float32)

    g = os.walk(data_path)
    for path,dir_list,file_list in g:
        per_image_Rmean = []
        per_image_Gmean = []
        per_image_Bmean = []
        for j, file_name in enumerate(file_list, 0):
            print(file_name)
            img = cv2.imread(os.path.join(path, file_name))
            img = cv2.resize(img, (img_height, img_width))
            per_image_Bmean.append(np.mean(img[:,:,0]))
            per_image_Gmean.append(np.mean(img[:,:,1]))
            per_image_Rmean.append(np.mean(img[:,:,2]))

            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = np.array(img)
            X[j] = img
            y[j] = float(int(file_name[:3]))

        # R_mean = np.mean(per_image_Rmean)
        # G_mean = np.mean(per_image_Gmean)
        # B_mean = np.mean(per_image_Bmean)
        # X[..., 0] = X[..., 0] - float(B_mean)
        # X[..., 1] = X[..., 1] - float(G_mean)
        # X[..., 2] = X[..., 2] - float(R_mean)

    if data is None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
        print("concatenate with data of last state")
        X_train = np.concatenate((data, X_train), axis=0)
        X_val = np.concatenate((vali_data, X_val), axis=0)
        y_train = np.concatenate((label, y_train), axis=0)
        y_val = np.concatenate((vali_label, y_val), axis=0)
        # print(y_train.shape)
        # print(X_train.shape)
    return X_train, X_val, y_train, y_val

def transform_y(y_test, y_pred, scaler_test):
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    y_test = scaler_test.inverse_transform(y_test)
    y_pred = scaler_test.inverse_transform(y_pred)
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    return y_test, y_pred


def generator_alldata(X_train):
    '''
    return a generator
    '''
    datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # samplewise_center = True,
            # samplewise_std_normalization=True,
            # rotation_range=4,
            width_shift_range=0.01,
            height_shift_range=0.01,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range = 0.01,
            fill_mode='nearest'
            # validation_split = 0.2
        )
    datagen_val = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        # samplewise_center = True,
        # samplewise_std_normalization=True,
        # rotation_range=4,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range = 0.01,
        fill_mode='nearest'
        # validation_split = 0.2
    )
    datagen.fit(X_train)
    # X_train, X_val, y_train, y_val

    # return datagen.flow(X_train, y_train, batch_size = batch_size)
    return datagen


###########################################################################
#  data preprocessing K fold
###########################################################################


def kfold_gen(fold = 2, path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/"], is_scale=True):

    (train,label) = CNN_Regression.load_kfold(path=path, channel=3, new_size = (img_width, img_height))
    # scaler = preprocessing.MaxAbsScaler()
    # train = scaler.fit_transform(train)
    scaler_val = None
    if is_scale:
        scaler_val = preprocessing.MaxAbsScaler()
        label = scaler_val.fit_transform(label.reshape(-1, 1))
        label = label.flatten()

    # train, vali_train, label, vali_label = train_test_split(train,label, test_size=0.3)
    kf = KFold(train, label, fold, 3)
    # kf = MultiClassficationBootstraping(train, label, fold )
    print(path)
    if len(path) > 1 or path == './Images/':
        kf.get_fold_list(start = 2, end = 2+2*len(path), label=100)
    else:
        kf.get_fold_list(start = 0, end = 2*len(path), label=100)
    # print(list(kf.split(train)))
    for i in range(0, fold):
        #yield  train, label, _, _ = kf.getItem(i)
        train, label, vali_train, vali_label = kf.getItem(i)
        yield [train, label, vali_train, vali_label, scaler_val]

def kfold_gen_sklearn(fold = 2, path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/"], is_scale=True):
    (train,label) = CNN_Regression.load_kfold(path=path, channel=3, new_size = (img_width, img_height)) #ndarray, ndaray
    kf = sklearnKF(n_splits=fold)
    train,label = shuffle(train,label)

    for train_index, test_index in kf.split(train):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_x, test_x = train[train_index], train[test_index]
        y, test_y = label[train_index], label[test_index]
        yield [train_x, y, test_x, test_y, None]

def training_kfold(fold = 2, mode = "all_empty", model_fn = VGG, path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/"], is_scale=True):

    if mode == "all_empty": # using all image set in the same time
        path_list = [path] # reduce for loop iteration time to 1
    else:
        path_list = np.array(path)
        path_list = path_list.reshape(-1,1)

    if mode != "all_empty":

        kfold_iterator = []

        for subpath in path_list:
            kfold_iterator.append(list(kfold_gen_sklearn(fold=fold, path=subpath, is_scale=is_scale)))
        kfold_iterator = np.array(kfold_iterator)
        for fold_id in range(fold):
            train_arr, label_arr, vali_train_arr, vali_label_arr = [], [], [], []

            for i_subpath in range(len(path_list)):
                train, label, vali_train, vali_label, scaler_val = kfold_iterator[i_subpath, fold_id]
                train_arr.append(train)
                label_arr.append(label)
                vali_train_arr.append(vali_train)
                vali_label_arr.append(vali_label)

            print(train_arr[0].shape)

            train = np.concatenate([i for i in train_arr], axis=0)
            label = np.concatenate([i for i in label_arr], axis=0)
            vali_train = np.concatenate([i for i in vali_train_arr], axis=0)
            vali_label = np.concatenate([i for i in vali_label_arr], axis=0)

            y_pred, y_test = run_model(train, label, vali_train, vali_label, model_fn = model_fn, fold=fold_id)
            if is_scale: y_test, y_pred = transform_y(y_test, y_pred, scaler_test)
            CNN_Regression.save_result('./result/regress_bins_'+str(fold_id)+"fold_"+model_fn.__name__+time_str+'.csv',y_pred,y_test)
            train, label, vali_train, vali_label, scaler_val = None, None, None, None, None
    if mode == "all_empty":
        for subpath in path_list:
            for i,[train, label, vali_train, vali_label, scaler_val] in enumerate(kfold_gen(fold=fold, path=subpath)):
                print(vali_label)
                y_pred,y_test = run_model(train, label, vali_train, vali_label, model_fn = model_fn,fold=i)

    return y_pred,y_test



class DataLoader:
    """Data Loader class. As a simple case, the model is tried on TinyImageNet. For larger datasets,
    you may need to adapt this class to use the Tensorflow Dataset API"""

    def __init__(self, batch_size, shuffle=False):  #
        self.X_train = None
        self.X_mean = None
        self.y_train = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        self.X_test = None
        self.y_test = None
        self.test_data_len = 0

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.scaler = None

    def cut_image(self, img_path = "", width = img_width/2.0, height = img_height/2.0):
        # img = Image.open(img_path).convert('L')
        img = Image.open(img_path)
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img4 = img.crop(
            (
                half_the_width - width,
                half_the_height - height,
                half_the_width + width,
                half_the_height + height
            )
        )
        # img4.save(img_path)
        return np.array(img4)

    def load_data_200(self, data_path = './Images', test_size_ratio=0.25):
        """return
        img_height, img_width, num_channels, self.train_data_len, self.val_data_len
        (channel last)
        save:
        ---
        self.X_train = None
        self.y_train = None
        self.train_data_len = 0

        self.X_val = None
        self.y_val = None
        self.val_data_len = 0

        """
        nb_class_factor = 51 # nbclass = 4 200
        #######################################
        num_images = 1
        flag = True
        print(type(self.X_train))
        if type(self.X_train) == 'numpy.ndarray' :
            flag = False
        for lists in os.listdir(data_path):
            sub_path = os.path.join(data_path, lists)
            # print(sub_path)
            if os.path.isfile(sub_path):
                num_images += 1

        X = np.zeros((num_images, img_height, img_width, num_channels), dtype=np.float32)
        y = np.ones(num_images, dtype=np.float32)

        g = os.walk(data_path)
        for path,dir_list,file_list in g:
            for j, file_name in enumerate(file_list, 1):
                print(file_name)
                img = imread(os.path.join(path, file_name))
                # img = self.cut_image(img_path = os.path.join(path, file_name))
                # dsize=(54, 140) as it takes x then y, where as a numpy array shows shape as y then x (y is number of rows and x is number of columns)
                img = resize(img, (img_height, img_width))
                # X[j] = img.transpose(0, 1, 2)
                X[j] = img
                # y[j] = int(int(file_name[:3])/nb_class_factor)
                y[j] = float(file_name[:3])



        print(y.shape)
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MaxAbsScale()

        y = scaler.fit_transform(y.reshape(-1, 1))
        y = y.flatten()
        self.scaler = scaler



        if flag == True:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
            self.X_train = np.concatenate((self.X_train, X_train), axis=0)
            self.X_val = np.concatenate((self.X_val, X_val), axis=0)
            self.y_train = np.concatenate((self.y_train, y_train), axis=0)
            self.y_val = np.concatenate((self.y_val, y_val), axis=0)
            print(self.y_train.shape)
            print(self.X_train.shape)


        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]
        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def load_data(self):
        # First load wnids
        wnids_file = os.path.join(data_path, 'wnids.txt')
        with open(wnids_file, 'r') as wnids_f:
            wnids = [x.strip() for x in wnids_f.readlines()]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Use words.txt to get names for each class
        words_file = os.path.join(data_path, 'words.txt')
        with open(words_file, 'r') as words_f:
            wnid_to_words = dict(line.split('\t') for line in words_f.readlines())

            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = [w.strip() for w in words.split(',')]

        # class_names = [wnid_to_words[wnid] for wnid in wnids]

        # Next load training data
        print("Loading Train Data...")
        X_train = []
        y_train = []

        for i, wnid in enumerate(wnids):
            if (i + 1) % 2 == 0:
                print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
            boxes_file = os.path.join(data_path, 'train', wnid, '%s_boxes.txt' % wnid)
            with open(boxes_file, 'r') as boxes_f:
                filenames = [x.strip() for x in boxes_f.readlines()]
            num_images = len(filenames)

            X_train_block = np.zeros((num_images, img_height, img_width, num_channels), dtype=np.float32)
            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)

            for j, img_file in enumerate(filenames):
                img_file = os.path.join(data_path, 'train', wnid, 'images', img_file)
                img = imread(img_file)

                if img.ndim == 2:
                    ## grayscale file
                    img.shape = (img_height, img_width, 1)

                X_train_block[j] = img.transpose(0, 1, 2)   # Save every image into the train set

            X_train.append(X_train_block)
            y_train.append(y_train_block)

        # We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        self.X_train = X_train
        self.y_train = y_train


        # Next load validation data
        print("Loading Validation Data...")
        val_anno_file = os.path.join(data_path, 'val', 'val_annotations.txt')
        with open(val_anno_file, 'r') as f:
            img_files = []
            val_wnids = []

            for line in f.readlines():
                line = line.strip('\n')
                # Select only validation images in chosen wnids set
                if line.split()[1] in wnids:    # Find the label
                    img_file, wnid = line.split('\t')[:2]
                    img_files.append(img_file)
                    val_wnids.append(wnid)

            num_val = len(img_files)
            X_val = np.zeros((num_val, img_height, img_width, num_channels), dtype=np.float32)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

            for i, img_file in enumerate(img_files):
                img_file = os.path.join(data_path, 'val', 'images', img_file)
                img = imread(img_file)
                if img.ndim == 2:
                    img.shape = (img_height, img_width, 1)

                X_val[i] = img.transpose(0, 1, 2)

        self.X_val = X_val
        self.y_val = y_val
        self.train_data_len = self.X_train.shape[0]
        self.val_data_len = self.X_val.shape[0]

        return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

    def generate_batch(self, type='train'):

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            samplewise_center = True,
            samplewise_std_normalization=True,
            # rotation_range=4,
            width_shift_range=0.01,
            height_shift_range=0.01,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range = 0.01,
            fill_mode='nearest'
            # validation_split = 0.2
        )
        datagen_val = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            samplewise_center = True,
            samplewise_std_normalization=True,
            # rotation_range=4,
            width_shift_range=0.01,
            height_shift_range=0.01,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range = 0.01,
            fill_mode='nearest'
            # validation_split = 0.2
        )

        # print(y.shape)
        # datagen.fit(self.X_train, augment=True)
        # datagen_val.fit(self.X_val, augment=True)
        datagen.fit(self.X_train)
        datagen_val.fit(self.X_val)
        # self.X_train, self.X_val, self.y_train, self.y_val
        if type == 'train':
            return datagen.flow(self.X_train, self.y_train, batch_size = self.batch_size)
        if type == 'test':
            return datagen.flow(self.X_val, self.y_val, batch_size = self.batch_size)
        if type == 'val':
            return datagen.flow(self.X_val, self.y_val, batch_size = self.batch_size)

    # def generate_batch(self, type='train'):
    #     """Generate batch from X_train/X_test and y_train/y_test using a python DataGenerator"""
    #     if type == 'train':
    #         # Training time!
    #         new_epoch = True
    #         start_idx = 0
    #         mask = None
    #         while True:
    #             if new_epoch:
    #                 start_idx = 0
    #                 if self.shuffle:
    #                     mask = np.random.choice(self.train_data_len, self.train_data_len, replace=False)
    #                 else:
    #                     mask = np.arange(self.train_data_len)
    #                 new_epoch = False

    #             # Batch mask selection
    #             X_batch = self.X_train[mask[start_idx:start_idx + self.batch_size]]
    #             y_batch = self.y_train[mask[start_idx:start_idx + self.batch_size]]
    #             start_idx += self.batch_size

    #             # Reset everything after the end of an epoch
    #             if start_idx >= self.train_data_len:
    #                 new_epoch = True
    #                 mask = None
    #             yield X_batch, y_batch
    #     elif type == 'test':
    #         # Testing time!
    #         start_idx = 0
    #         while True:
    #             # Batch mask selection
    #             X_batch = self.X_test[start_idx:start_idx + self.batch_size]
    #             y_batch = self.y_test[start_idx:start_idx + self.batch_size]
    #             start_idx += self.batch_size

    #             # Reset everything
    #             if start_idx >= self.test_data_len:
    #                 start_idx = 0
    #             yield X_batch, y_batch
    #     elif type == 'val':
    #         # Testing time!
    #         start_idx = 0
    #         while True:
    #             # Batch mask selection
    #             X_batch = self.X_val[start_idx:start_idx + self.batch_size]
    #             y_batch = self.y_val[start_idx:start_idx + self.batch_size]
    #             start_idx += self.batch_size

    #             # Reset everything
    #             if start_idx >= self.val_data_len:
    #                 start_idx = 0
    #             yield X_batch, y_batch
    #     else:
    #         raise ValueError("Please select a type from \'train\', \'val\', or \'test\'")



    # def load_data_default(self):
    #     # This method is an example of loading a dataset. Change it to suit your needs..
    #     import matplotlib.pyplot as plt
    #     # For going in the same experiment as the paper. Resizing the input image data to 224x224 is done.
    #     train_data = np.array([plt.imread('./data/0.png')], dtype=np.float32)
    #     self.X_train = train_data
    #     self.y_train = np.array([283], dtype=np.int32)

    #     val_data = np.array([plt.imread('./data/0.png')], dtype=np.float32)
    #     self.X_val = val_data
    #     self.y_val = np.array([283])

    #     self.train_data_len = self.X_train.shape[0]
    #     self.val_data_len = self.X_val.shape[0]
    #     img_height = 224
    #     img_width = 224
    #     num_channels = 3
    #     return img_height, img_width, num_channels, self.train_data_len, self.val_data_len

if __name__ == '__main__':
    train, label, vali_train, vali_label,X_test,y_test = pre_training_data(is_scaler=False, is_categorical=False)
    print("label", label)
    print("y_test", y_test)

