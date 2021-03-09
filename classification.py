import classic_regression as cr
import tensorflow.keras as keras
import tensorflow.keras.applications.resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
def loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)
import os
import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3 
session = tf.Session(config=config)

# 设置session
# KTF.set_session(session )

def get_learning_rate(cnn_type):
    if cnn_type == 'VGG16' or cnn_type == 'VGG16_DROPOUT':
        return 0.00004
    elif cnn_type == 'VGG16_KERAS':
        return 0.00005
    elif cnn_type == 'VGG19':
        return 0.00003
    elif cnn_type == 'VGG19_KERAS':
        return 0.00005
    elif cnn_type == 'RESNET50':
        return 0.00004
    elif cnn_type == 'INCEPTION_V3':
        return 0.00003
    elif cnn_type == 'SQUEEZE_NET':
        return 0.00003
    elif cnn_type == 'DENSENET_161':
        return 0.00003
    elif cnn_type == 'DENSENET_121':
        return 0.00001
    else:
        print('Error Unknown CNN type for learning rate!!')
        exit()
    return 0.00005


def get_optim(cnn_type, optim_type, learning_rate=-1):
    

    if learning_rate == -1:
        lr = get_learning_rate(cnn_type)
    else:
        lr = learning_rate
    if optim_type == 'Adam':
        optim = Adam(lr=lr)
    else:
        optim = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    print('Using {} optimizer. Learning rate: {}'.format(optim_type, lr))
    return optim

X_train, y_train, X_test, y_test = cr.load_data()

for i in range(len(y_train)):
    if y_train[i] > 300:
        y_train[i] = 30
    else:
        y_train[i] = int(y_train[i]//10)


for i in range(len(y_test)):
    if y_test[i] > 300:
        y_test[i] = 30
    else:
        y_test[i] = int(y_test[i]//10)

y_test = y_test.astype(int)
y_train = y_train.astype(int)
# print(y_test)

onehot_y_train = []
onehot_y_test = []
for i in range(len(y_test)):
    out = [0 for i in range(31)]
    out[y_test[i]]=1
    # print(out)
    onehot_y_test.append(out)

for i in range(len(y_train)):
    out = [0 for i in range(31)]
    out[y_train[i]]=1
    # print(out)
    onehot_y_train.append(out)

# print(onehot_y_train)    



dirs = "classification_result"
if not os.path.exists(dirs):
        os.makedirs(dirs)

print(X_train[0].shape)

base_model = ResNet50(include_top= False ,weights='imagenet', input_shape = X_train[0].shape)
x = base_model.output


x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = keras.layers.Dense(31, activation= 'softmax')(x)

model = keras.models.Model(inputs = base_model.input, outputs = predictions)



optim = get_optim('RESNET50', 'Adam', 1e-3)
model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

# print(X _train)

# print(onehot_y_test)
onehot_y_test = np.asarray(onehot_y_test)
onehot_y_train = np.asarray(onehot_y_train)
model.fit(X_train,onehot_y_train,epochs=50, batch_size = 8,validation_data=(X_test,onehot_y_test))
# score = model.score(X_test, y_test)
result = model.predict(X_test)

np.save(dirs+"/resnet1.npy",result)
# rmse = sqrt(mean_squared_error(y_test,result))

# plt.figure()

# plt.plot(np.arange(len(result)), y_test,'g',label='true value')
# plt.plot(np.arange(len(result)),result,'r',label='predict value')

# plt.title("ResNet50" + ' rmse: %f' % rmse)
# plt.legend()
# plt.savefig(dirs+"/"+"ResNet50"+".png",dpi=300)