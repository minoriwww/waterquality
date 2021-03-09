
import numpy as np
import os


import random
import string

from imageio import imread
from skimage.transform import rescale, resize, downscale_local_mean
import CNN_Regression

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
# config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.8
# tf.Session(config=config)

config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth=True
tf.Session(config=config)
import numpy as np
import keras
import csv
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras import initializers
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
import time

from KFold import *
img_height = 512
img_width = 512

num_channels = 3 #1

data_batch_size = 8 #8 16 1

def preprocessing_y(value_float, default_method=None, bin_size=20, max_lim=200, min_lim=0):
    if not default_method:
        return value_float
    if default_method == 'bins':
        nb_bins = (max_lim - min_lim)/bin_size
        if value_float>=max_lim: return nb_bins
        if value_float<min_lim: return -1
        return int(value_float/bin_size)


def load_data_200(data, vali_data, label,
    vali_label,
    data_path = './Images',
    test_size_ratio=0.3
    ):
    """return
    img_height, img_width, num_channels, train_data_len, self.val_data_len
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
    print(type(data))

    if data is not None :
        flag = False
        print(flag)
    else: print(flag)
    for lists in os.listdir(data_path):
        sub_path = os.path.join(data_path, lists)
        # print(sub_path)
        if os.path.isfile(sub_path):
            num_images += 1

    X = np.zeros((num_images, img_height, img_width, num_channels), dtype=np.float32)
    y = np.ones(num_images, dtype=np.float32)

    g = os.walk(data_path)
    for path,dir_list,file_list in g:
        for j, file_name in enumerate(file_list, 0):
            print(file_name)
            img = imread(os.path.join(path, file_name))
            img = resize(img, (img_height, img_width))
            X[j] = img
            y[j] = preprocessing_y(float(file_name[:3]))

    print(y.shape)
    # StandardScaler()
    # scaler = preprocessing.MaxAbsScaler()

    # y = scaler.fit_transform(y.reshape(-1, 1))
    # y = y.flatten()
    # scaler = scaler

    if flag == True:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size_ratio, shuffle=True)
        print("concatenate")
        X_train = np.concatenate((data, X_train), axis=0)
        X_val = np.concatenate((vali_data, X_val), axis=0)
        y_train = np.concatenate((label, y_train), axis=0)
        y_val = np.concatenate((vali_label, y_val), axis=0)
        print(y_train.shape)
        print(X_train.shape)

    return X_train, X_val, y_train, y_val

def dnn_model(data,label,vali_data,vali_label):
    # create model
    model = Sequential()
    model.add(BatchNormalization(input_shape=data.shape[1:]))
    model.add(Dense(128, activation='selu',init='he_normal', kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.05)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02), activity_regularizer=regularizers.l1(0.01)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Dense(256,  activation='selu',init='he_normal', kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.05)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1,init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation("linear"))#sigmod

    sgd = optimizers.SGD(lr=1e-6, decay=1e-9, momentum=0.9, nesterov=True)
    ada = optimizers.Adadelta(lr=1e-3)

    # Compile model
    model.summary()
    # model.compile(loss='mean_squared_error', optimizer=sgd)
    model.compile(loss='mean_squared_error', optimizer=ada,metrics=['mse'])
    # if os.path.exists("./dnn_weights.h5"):
    #     print("using weight")
    #     model.load_weights("./dnn_weights.h5")
    return model

def VGG(train,label,vali_train,vali_label):
    print(train.shape[1:])
    model = Sequential()

    VGGmodel = keras.applications.vgg16.VGG16(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=train.shape[1:]
        , pooling=None
        # , classes=1000
        )
    print(VGGmodel.output_shape[1:])


    # model.add(Flatten(input_shape=VGGmodel.output_shape[1:]))
    model.add(Flatten())
    model.add(Dropout(0.5))

    # model.add(Dense(128, kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(0.2)))
    # model.add(BatchNormalization())
    # model.add(Activation('selu'))
    # model.add(Dense(64, kernel_initializer='he_normal', use_bias=True))
    # model.add(BatchNormalization())
    # model.add(Activation('selu'))
    model.add(Dense(1, kernel_initializer='he_normal'))

    # model.add(Activation('sigmoid'))
    model.add(BatchNormalization())
    model.add(Activation('linear')) #softmax

    model = Model( inputs=VGGmodel.input , outputs=model(VGGmodel.output) )
    # model = Model( inputs=inputs, outputs=result )


    model.compile(loss='mean_squared_error', #mse binary_crossentropy Dice-coefficient loss function vs cross-entropy
        optimizer=optimizers.Adam(lr=1e-4),
        metrics=['mse'])
    print(model.summary())

    return model

def training_kfold(fold = 2, model_fn = VGG):

    (train,label) = CNN_Regression.load_kfold(path=["./Images/","./Images2/","./Images3/"], channel=3, new_size = (img_width, img_height))

    # scaler = preprocessing.MaxAbsScaler()

    # train = scaler.fit_transform(train)


    scaler_val = preprocessing.MaxAbsScaler()

    label = scaler_val.fit_transform(label.reshape(-1, 1))
    label = label.flatten()


    fold = 2
    kf = KFold(train, label, fold, 3)
    for i in range(0, fold):
        train, label, vali_train, vali_label = kf.getItem(i)
        model = model_fn(train,label,vali_train,vali_label)
        model_checkpoint = ModelCheckpoint('./modelWights/weights'+model_fn.__name__+'.h5', monitor='val_loss', save_best_only=True)
        history = model.fit(train, label, batch_size = data_batch_size, epochs=200, validation_data=(vali_train,vali_label), callbacks=[model_checkpoint])

        model.save('./modelWights/regression_model'+model_fn.__name__+'.h5')

        y_pred = model.predict(vali_train, batch_size=data_batch_size, verbose=1)

        # y_pred = np.load('imgs_mask_test.npy')
        vali_label = vali_label.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

        vali_label = scaler_val.inverse_transform(vali_label)
        y_pred = scaler_val.inverse_transform(y_pred)
        vali_label = vali_label.flatten()
        y_pred = y_pred.flatten()
        np.save('y_pred.npy', y_pred)
        np.save('vali_label_transformed.npy', vali_label)

        CNN_Regression.save_result('./result/regress_'+model_fn.__name__+str(time.time())+'.csv',y_pred,vali_label)



def training(model_fn = VGG):
    # only test  image3
    X_train, X_val, y_train, y_val = None, None, None, None
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images2')
    X_train, X_val, y_train, y_val = load_data_200(X_train, X_val, y_train, y_val, data_path = './Images3')

    # X_test, X_val_, y_test, y_val_ = load_data_200(None, None, None, None, data_path = './Images3', test_size_ratio=0.01)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.6, shuffle=True)

    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MaxAbsScaler()

    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_train = y_train.flatten()

    scaler_val = preprocessing.MaxAbsScaler()

    y_val = scaler_val.fit_transform(y_val.reshape(-1, 1))
    y_val = y_val.flatten()


    scaler_test = preprocessing.MaxAbsScaler()

    y_test = scaler_test.fit_transform(y_test.reshape(-1, 1))
    y_test = y_test.flatten()

    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # X_train = np.load('X_train.npy')
    # X_val = np.load('X_val.npy')
    # y_train = np.load('y_train.npy')
    # y_val = np.load('y_val.npy')

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    # (train,label),(vali_train,vali_label) = CNN_Regression.load_data()
    train = X_train
    label = y_train
    vali_train = X_val
    vali_label = y_val
    model = model_fn(train,label,vali_train,vali_label)
    # model.load_weights('./modelWights/weightsVGG.h5')
    model_checkpoint = ModelCheckpoint('./modelWights/weights'+model_fn.__name__+'.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(train, label, batch_size = data_batch_size, epochs=200, validation_data=(vali_train,vali_label), callbacks=[model_checkpoint])

    model.save('./modelWights/regression_model'+model_fn.__name__+'.h5')

    y_pred = model.predict(X_test, batch_size=data_batch_size, verbose=1)

    # y_pred = np.load('imgs_mask_test.npy')
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)

    y_test = scaler_test.inverse_transform(y_test)
    y_pred = scaler_test.inverse_transform(y_pred)
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()
    np.save('y_pred.npy', y_pred)
    np.save('y_test_transformed.npy', y_test)

    CNN_Regression.save_result('./result/regress_'+model_fn.__name__+str(time.time())+'.csv',y_pred,y_test)


if __name__ == '__main__':

    model = keras.models.load_model('./modelWights/regression_modelVGG.h5')
    model.load_weights('./modelWights/weightsVGG.h5')
    X_test, X_val, y_test, y_val = None, None, None, None
    X_test, X_val, y_test, y_val = load_data_200(X_test, X_val, y_test, y_val, data_path = './Images4', test_size_ratio=0)


    scaler = preprocessing.MaxAbsScaler()

    y_test = scaler.fit_transform(y_test.reshape(-1, 1))
    y_test = y_test.flatten()

    y_pred = model.predict(X_test)

    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = y_test.flatten()
    y_pred = y_pred.flatten()


    CNN_Regression.save_result('1234.csv',y_pred,y_test)
