
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import tensorflow as tf
config=tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5
tf.Session(config=config)
import cv2
import keras
import csv
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import to_categorical
from keras import initializers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import random
import string


def img_collect(pathlist,path):
    data = []
    label = []
    for p in pathlist:
        src=os.path.join(path,p)
        img=cv2.imread(src)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        data.append(img)
        label.append(int(p[0:3]))
    return data,label


def save_result(name,y_pred,y_true):
    with open(name,"w") as csvfile:
        writer = csv.writer(csvfile)
        y_diff = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
        y = [y_pred, y_true, y_diff]
        mse = 0.0
        for num in range(len(y_diff)):
            mse += pow(y_diff[num],2) / len(y_diff)
        print('mse = ', mse)
        writer.writerows(y)


def CNN(data,label,vali_data,vali_label):
    early_stop = EarlyStopping(monitor='val_mean_squared_error', patience=20, mode='min')
    model = Sequential()
    model.add(BatchNormalization(input_shape=data.shape[1:]))
    model.add(Conv2D(
            input_shape=data.shape[1:],
            data_format='channels_last',
            filters=32,
            kernel_size=(9,9),
            strides=(5,5),
            padding='valid',
            activation='relu',
            use_bias=True,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(0.005)
            ))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(9,9),strides=(5,5),padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
    model.add(Dropout(0.25))

    # model.add(BatchNormalization())
    # model.add(Conv2D(64, kernel_size=(5,5), strides=(3,3),padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    # # model.add(AveragePooling2D(pool_size=(2,2), strides=None, padding='valid'))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(BatchNormalization())
    # model.add(Dense(1024))
    # model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error',optimizer=optimizers.Adam(lr=1e-4,decay=5e-3),metrics=['mse'])
    model.fit(data, label, batch_size=16,epochs=1000,validation_data=(vali_data,vali_label), callbacks=[early_stop])
    model.save('regression_model.h5')
    y_pred = model.predict(vali_data)
    save_result('regress_random.csv',y_pred,vali_label)


def groupShuffle(train,label):
    x_train, x_vali, y_train, y_vali = train_test_split(train, label, test_size=0.1)
    return (x_train,y_train),(x_vali,y_vali)


def load_data():
    path = ["./Images/","./Images2"]
    pathlist = [os.listdir(path[0]),os.listdir(path[1])]
    x_train, y_train = img_collect(pathlist[0],path[0])
    x_vali, y_vali = img_collect(pathlist[1],path[1])
    x = x_train + x_vali
    y = y_train + y_vali
    idx = [index for index in range(len(y))]
    y, idx = (list(t) for t in zip(*sorted(zip(y,idx))))
    x = np.array(x)
    train = x[idx]
    label = np.array(y)
    # y = to_categorical(y)
    (x_train, y_train), (x_vali, y_vali) = groupShuffle(train,label)
    x_train = x_train.reshape(x_train.shape+(1,))
    x_vali = x_vali.reshape(x_vali.shape+(1,))
    return (x_train,y_train),(x_vali,y_vali)


def load_data2():
    path = ["./Images/","./Images2"]
    pathlist = [os.listdir(path[0]),os.listdir(path[1])]
    x_train, y_train = img_collect(pathlist[0],path[0])
    x_vali, y_vali = img_collect(pathlist[1],path[1])
    x_train = np.array(x_train)
    x_vali = np.array(x_vali)
    x_train = x_train.reshape(x_train.shape+(1,))
    x_vali = x_vali.reshape(x_vali.shape+(1,))
    return (x_train,y_train),(x_vali,y_vali)


if __name__ == "__main__":
    (train,label),(vali_train,vali_label) = load_data()
    CNN(train,label,vali_train,vali_label)
    # model = keras.models.load_model('regression_model.h5')
    # y_pred = model.predict(vali_train)
    # save_result('predict.csv',y_pred,vali_label)

