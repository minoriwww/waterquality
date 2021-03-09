from __future__ import print_function
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# config.gpu_options.per_process_gpu_memory_fraction=0.5

import numpy as np
import random
import string

## from imageio import imread


# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from sklearn.externals import joblib
import joblib
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf
import cv2
import numpy as np
import keras
import csv
from keras.models import Sequential
# from keras import optimizersfac
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
from sklearn.model_selection import KFold as sklearnKF
from sklearn.utils import shuffle

from KFold import *
from bootstraping import MultiClassficationBootstraping
import CNN_Regression
from  data_loader import *

from keras.backend.tensorflow_backend import set_session
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)


img_height = 224
img_width = 224
nb_channels = 3 #1

BATCH_SIZE = 8 #8 16 1
RANDOM_STATE = 42
time_str = str(time.time())

BIN_SIZE = 20#20
MAX_LIM = 200#200
MIN_LIM = 20
nb_bins = (MAX_LIM - MIN_LIM)/BIN_SIZE

IS_SCALE = False
nb_epochs = 1000



##################################################
#   model
##################################################

def dnn_model(data=None, label=None, vali_data=None, vali_label=None, input_shape=None, **kwargs):
    # create model
    if input_shape is None and data is not None: input_shape = data.shape[1:]
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Dense(128, activation='relu',init='he_normal', kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.05)))
    model.add(Dense(256,  activation='relu',init='he_normal', kernel_regularizer=regularizers.l2(0.05), activity_regularizer=regularizers.l1(0.05)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    # model.add(Flatten())
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

def CNN(train=None,label=None,vali_train=None,vali_label=None, input_shape=None, **kwargs):
    if input_shape is None and train is not None: input_shape = train.shape[1:]

    model = Sequential()
    model.add(BatchNormalization(input_shape=train.shape[1:]))
    model.add(Conv2D(
            input_shape=train.shape[1:],
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
    # model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
    model.add(Dropout(0.25))

    # model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(5,5),strides=(1,1),padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3,3),strides=(1,1),padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    # model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
    model.add(Dropout(0.25))

    # model.add(BatchNormalization())
    # model.add(Conv2D(64, kernel_size=(5,5), strides=(3,3),padding='valid', activation='relu', use_bias=True, kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    # # model.add(AveragePooling2D(pool_size=(2,2), strides=None, padding='valid'))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(1024))
    # model.add(Dropout(0.25))
    if 'is_categorical' in kwargs and kwargs['is_categorical'] == False:
        model.add(Dense(1,kernel_initializer='he_normal'))
        model.add(Activation('linear')) #softmax
        # model = Model( inputs=inputs, outputs=result )
        model.compile(loss='mean_squared_error', #mse binary_crossentropy Dice-coefficient loss function vs cross-entropy
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=['mse'])
    elif 'is_categorical' in kwargs and kwargs['is_categorical'] == True:
        model.add(Dense(int(kwargs['nb_classes']), kernel_initializer='he_normal'))
        model.add(Activation('sigmoid')) #softmax

        model.compile(loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1e-2),
            metrics=['accuracy'])
    else:
        raise Exception('No model returned')
    print(model.summary())
    return model

#1
def pretrained_models(train=None,label=None,vali_train=None,vali_label=None, input_shape=None, model_name="densenet", **kwargs):
    """
    model_name = "inception_v3", "mobilenet", or "densenet" "resnet50" "resnet101"
    """
    if input_shape is None and train is not None:
        input_shape = train.shape[1:]
    else:
        input_shape = (img_width, img_height, nb_channels)

    model = Sequential()

    inception_v3 = keras.applications.inception_v3.InceptionV3(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None)

    mobilenet = keras.applications.mobilenet.MobileNet(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None)

    densenet = keras.applications.densenet.DenseNet121(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None)

    resnet50 = keras.applications.resnet.ResNet50(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None)
    resnet101 = keras.applications.resnet.ResNet101(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None)
    Kerasmodel = eval(model_name)


    def inner(train=train,label=label,vali_train=vali_train,vali_label=vali_label, input_shape=input_shape, model_name="densenet", **kwargs):
        nonlocal model
        model.add(Flatten())
        # model.add(Dropout(0.5))

        model.add(Dense(256, kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(0.2)))
        model.add(Dropout(0.5))


        if 'is_classfication' in kwargs and kwargs['is_classfication'] == False:
            print('categorical false')
            model.add(Dense(1,kernel_initializer='he_normal'))
            model.add(Activation('linear')) #softmax
            model = Model( inputs=Kerasmodel.input , outputs=model(Kerasmodel.output) )
            # model = Model( inputs=inputs, outputs=result )
            model.compile(loss='mean_squared_error', #mse binary_crossentropy Dice-coefficient loss function vs cross-entropy
                optimizer=optimizers.Adam(lr=1e-4),
                metrics=['mse'])
        elif 'is_classfication' in kwargs and kwargs['is_classfication'] == True:
            print('categorical true')
            model.add(Dense(int(kwargs['nb_classes']), kernel_initializer='he_normal'))
            model.add(Activation('sigmoid')) #softmax
            model = Model( inputs=Kerasmodel.input , outputs=model(Kerasmodel.output) )

            model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=1e-4),
                metrics=['accuracy'])
        else:
            raise Exception('No model returned')
        print(model.summary())
        return model
    return inner



def VGG(train=None,label=None,vali_train=None,vali_label=None, input_shape=None, **kwargs):
    """input should be 256-512 size, each channel subtracted by RGB channle mean value of all dataset
    """
    # print(train.shape[1:])

    if input_shape is None and train is not None:
        input_shape = train.shape[1:]
    else:
        input_shape = (img_width, img_height, nb_channels)

    model = Sequential()

    VGGmodel = keras.applications.vgg16.VGG16(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None
        # , classes=1000
        )
    # print(VGGmodel.output_shape[1:])


    # model.add(Flatten(input_shape=VGGmodel.output_shape[1:]))
    model.add(Flatten())
    # model.add(Dropout(0.5))

    model.add(Dense(256, kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(0.2)))
    model.add(Dropout(0.5))
    # model.add(BatchNormalization())
    # model.add(Activation('selu'))
    # model.add(Dense(64, kernel_initializer='he_normal', use_bias=True))
    # model.add(BatchNormalization())
    # model.add(Activation('selu'))


    # model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())

    if 'is_classfication' in kwargs and kwargs['is_classfication'] == False:
        print('classfication false')
        model.add(Dense(1,kernel_initializer='he_normal'))
        model.add(Activation('linear')) #softmax
        model = Model( inputs=VGGmodel.input , outputs=model(VGGmodel.output) )
        # model = Model( inputs=inputs, outputs=result )
        model.compile(loss='mean_squared_error', #mse binary_crossentropy Dice-coefficient loss function vs cross-entropy
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=['mse'])
    elif 'is_classfication' in kwargs and kwargs['is_classfication'] == True:
        print('classfication true')
        model.add(Dense(int(kwargs['nb_classes']), kernel_initializer='he_normal'))
        model.add(Activation('sigmoid')) #softmax for classes nb  larger than 2
        model = Model( inputs=VGGmodel.input , outputs=model(VGGmodel.output) )

        model.compile(loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1e-2),
            metrics=['accuracy'])
    else:
        raise Exception('No model returned')
    print(model.summary())

    return model

def classification_then_regression():
    """
    There would be 2 losses in combined network. one for classfication task and other for it subsequenced regression task. Therefore, in the input, both the bin label and ordinal scalar label should  be included.

    Then the loss should be carefully designed, include weighted classification and ordinal regression loss

    # 1. load (pre-trained) classification model
    # 2. or build a new model
    # select final layers to add classfication loss
    # Eibe Frank and Mark Hal ECML 2001 trick
    """
    if input_shape is None and train is not None:
        input_shape = train.shape[1:]
    else:
        input_shape = (img_width, img_height, nb_channels)

    model = Sequential()

    inception_v3 = keras.applications.inception_resnet_v2.InceptionResNetV2(
        include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None
        # , classes=1000
        )
    # print(VGGmodel.output_shape[1:])
    # model.add(Flatten(input_shape=VGGmodel.output_shape[1:]))
    model.add(Flatten())
    model.add(Dense(int(kwargs['nb_classes']), kernel_initializer='he_normal', activation='sigmoid'))#softmax

    model = Model( inputs=inception_v3.input , outputs=model(inception_v3.output) )

    model_regression = Sequential()
    inception_v3 = keras.applications.inception_v3.InceptionV3(include_top=False
        , weights='imagenet'
        # , input_tensor=inputs
        , input_shape=input_shape
        , pooling=None
        # , classes=1000
        )
    model_regression.add(Dense(1,kernel_initializer='he_normal', activation='linear'))

    model_regression = Model( inputs=model.input , outputs=[model.output, model_regression(VGGmodel.output)] )

    model_regression.compile(loss='mean_squared_error', #mse binary_crossentropy Dice-coefficient loss function vs cross-entropy
        optimizer=optimizers.Adam(lr=1e-4),
        metrics=['mse'])

    return model_regression





def run_model(train, label, test, test_label, vali_train=None, vali_label=None, model_fn = VGG, is_classfication=False, fold=1, nb_classes=1):
    model = model_fn(train,label,test,test_label, is_classfication=is_classfication, nb_classes=nb_classes)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', patience=30)
    # model.load_weights('./modelWights/regression_model'+model_fn.__name__+'.h5')
    model_checkpoint = ModelCheckpoint('./modelWights/weights_'+model_fn.__name__+'_fold_'+str(fold)+time_str+'.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(train, label, batch_size=BATCH_SIZE, epochs=nb_epochs, validation_data=(test,test_label), callbacks=[model_checkpoint])
    # print(history.history)
    y_pred = model.predict(test, batch_size=1, verbose=1)

    y_test = test_label
    return y_pred, y_test

def run_model_with_gen(train, label, test, test_label, vali_train=None, vali_label=None, model_fn = VGG,is_classfication=False, fold=1, nb_classes=1):
    model = model_fn(train,label,test,test_label, is_classfication=is_classfication, nb_classes=nb_classes)
    early_stop = EarlyStopping(monitor='val_mean_squared_error', patience=40)
    # model.load_weights('./modelWights/regression_model'+model_fn.__name__+'.h5')
    model_checkpoint = ModelCheckpoint('./modelWights/weights'+model_fn.__name__+'_fold_'+str(fold)+time_str+'.h5', monitor='val_loss', save_best_only=True)
    csv_logger = keras.callbacks.CSVLogger("./logs/loss_history_"+model_fn.__name__+time_str+'.csv', append=True, separator=',')

    datagen = generator_alldata(train)
    model.fit_generator(datagen.flow(train, label, batch_size=BATCH_SIZE),
                    steps_per_epoch=len(train) / 16, epochs=nb_epochs, validation_data=(vali_train,vali_label), callbacks=[model_checkpoint, csv_logger])

    y_pred = model.predict(test, batch_size=1, verbose=1)

    y_test = test_label
    return y_pred, y_test



if __name__ == '__main__':

    model_fn = pretrained_models(model_name="mobilenet") #dnn_model VGG
    data_for_training(model_fn, is_scaler=True, is_categorical=False)
    # training(model_fn,is_scaler=False, is_categorical=True, bin_method='threebins')

    # training_kfold(fold = 1, model_fn = model_fn, path=["./Images/","./Images2/","./Images3/","./Images4/"])

    # training_kfold(fold = 1, model_fn = model_fn, path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/"])
    # "./Images/","./Images2/","./Images3/","./Images4/","./Images5/", "./Images6/","./Images7_high_range/",

    #################

    training_kfold(fold = 5, model_fn = model_fn, mode = "random_empty", path=[ "./Images/","./Images2/","./Images3/","./Images4/","./Images5/", "./Images6/","./Images7_high_range/","./Images8_high_range/", "./Images9_200_300/"], is_scale=IS_SCALE)

    # training_kfold(fold = 1, model_fn = model_fn, mode = "random_empty", path=["./Images/","./Images2/","./Images3/","./Images4/"])
    # training_kfold(fold = 1, model_fn = model_fn, mode = "random_empty", path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/"])
    # training_kfold(fold = 6, model_fn = model_fn, mode = "all_empty", path=["./Images/","./Images2/","./Images3/","./Images4/","./Images5/","./Images6/"])
    # training_kfold(fold = 6, model_fn = model_fn,  path=["./Images/","./Images7_high_range/"])



    # model = model_fn(train,label,vali_train,vali_label)
    # # model.load_weights('./modelWights/weightsVGG.h5')
    # model_checkpoint = ModelCheckpoint('./modelWights/weights'+model_fn.__name__+'.h5', monitor='val_loss', save_best_only=True)
    # history = model.fit(train, label, batch_size = BATCH_SIZE, epochs=200, validation_data=(vali_train,vali_label), callbacks=[model_checkpoint])

    # model.save('./modelWights/regression_model'+model_fn.__name__+'.h5')

    # y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

    # # y_pred = np.load('imgs_mask_test.npy')
    # y_test = y_test.reshape(-1, 1)
    # y_pred = y_pred.reshape(-1, 1)

    # y_test = scaler_test.inverse_transform(y_test)
    # y_pred = scaler_test.inverse_transform(y_pred)
    # y_test = y_test.flatten()
    # y_pred = y_pred.flatten()
    # np.save('y_pred.npy', y_pred)
    # np.save('y_test_transformed.npy', y_test)


    # CNN_Regression.save_result('./result/regress_'+model_fn.__name__+str(time.time())+'.csv',y_pred,y_test)

