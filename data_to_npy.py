import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import string

from imageio import imread
from skimage.transform import rescale, resize, downscale_local_mean
import CNN_Regression

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


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
import copy
import time

from KFold import *
img_height = 512
img_width = 512

num_channels = 3 #1

data_batch_size = 8 #8 16 1

RATIO =0.2

def load_data(data, vali_data, label,
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

    root_path = copy.copy(data_path)
    for num_forder in range(1,10):

        data_path = root_path
        if 6>=num_forder>=2 :
            data_path+=str(num_forder)
        else:
            if 8>=num_forder >=7 :
                data_path += str(num_forder)+"_high_range"
            else:
                if num_forder==9:
                    data_path += str(num_forder)+"_200_300"

        print ("Now working with"+data_path)
        if data is not None :
            flag = False
            print(flag)
        else: print(flag)
        for lists in os.listdir(data_path):
            sub_path = os.path.join(data_path, lists)
            # print(sub_path)
            if os.path.isfile(sub_path):
                num_images += 1

    print("Total Images Number:")
    print(num_images)

    X = np.zeros((num_images, img_height, img_width, num_channels), dtype=np.float32)
    y = np.ones(num_images, dtype=np.float32)

    counter = 0
    for num_forder in range(1,10):

        data_path = root_path
        if 6>=num_forder>=2 :
            data_path+=str(num_forder)
        else:
            if 8>=num_forder >=7 :
                data_path += str(num_forder)+"_high_range"
            else:
                if num_forder==9:
                    data_path += str(num_forder)+"_200_300"

        print ("Now working with" + data_path)
        g = os.walk(data_path)
        for path,dir_list,file_list in g:
            for j, file_name in enumerate(file_list, 0):
                print(file_name)
                img = imread(os.path.join(path, file_name))
                img = resize(img, (img_height, img_width))
                X[counter] = img
                y[counter] = float(file_name[:3])
                counter+=1

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
if __name__ == '__main__':

    # X_test, X_val, y_test, y_val = None, None, None, None
    # X_test, X_val, y_test, y_val = load_data(X_test, X_val, y_test, y_val, data_path = './Images', test_size_ratio=RATIO)
    #
    # print (X_test.shape)
    # print (X_val.shape)
    #
    # np.save("./X_train_ratio="+str(RATIO)+".npy",X_test)
    # np.save("./y_train_ratio=" + str(RATIO) + ".npy",y_test)
    # #
    # np.save("./X_test_ratio=" + str(RATIO) + ".npy",X_val)
    # np.save("./y_test_ratio=" + str(RATIO) + ".npy",y_val)
    #
    print(np.load("./X_test_ratio="+str(RATIO)+".npy").shape)