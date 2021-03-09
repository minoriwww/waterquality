# import classic_regression as cr
# import os

# X_train, y_train, X_test, y_test = cr.load_data()

# for i in range(len(y_train)):
#     if y_train[i] > 300:
#         y_train[i] = 30
#     else:
#         y_train[i] = int(y_train[i]//10)


# for i in range(len(y_test)):
#     if y_test[i] > 300:
#         y_test[i] = 30
#     else:
#         y_test[i] = int(y_test[i]//10)

# y_test = y_test.astype(int)
# y_train = y_train.astype(int)
# # print(y_test)

# onehot_y_train = []
# onehot_y_test = []
# for i in range(len(y_test)):
#     out = [0 for i in range(31)]
#     out[y_test[i]]=1
#     print(out)
#     onehot_y_test.append(out)

# for i in range(len(y_train)):
#     out = [0 for i in range(31)]
#     out[y_train[i]]=1
#     print(out)
#     onehot_y_train.append(out)

# # print(onehot_y_train)    

# dirs = "classic_classification"
# if not os.path.exists(dirs):
#         os.makedirs(dirs)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

plt.switch_backend('agg')
import time
import cv2

def load_data():

    X_train = np.load("X_train_ratio=0.2.npy")
    y_train = np.load("y_train_ratio=0.2.npy")

    X_test = np.load("X_test_ratio=0.2.npy")
    y_test = np.load("y_test_ratio=0.2.npy")

    return X_train, y_train, X_test, y_test


def get_model_set():

    models = []

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()

    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(n_estimators=20)

    from sklearn import linear_model
    linear = linear_model.SGDClassifier()


    models.append(knn)
    models.append(random_forest)
    models.append(linear)

    return models
    

###########2.回归部分##########
def try_different_method(models,X_train,y_train,X_test,y_test):
    model_names=["KNN","Random Forest","Linear"]

    import os
    dirs = 'classic_classification/'+str(time.time())
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for num,model in enumerate(models):

        print("Now working on "+model_names[num])
        
        model.fit(X_train,y_train)
        
        result = model.predict(X_test)

        np.save(dirs+"/"+model_names[num]+".npy",result)
        # rmse = sqrt(mean_squared_error(y_test,result))
        score = model.score(X_test, y_test)

        plt.figure()
        
        plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
        plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
        #plt.title(model_names[num]+'score: %f'%score)
        plt.title(model_names[num] + ' score: %f' % score)
        plt.legend()
        plt.savefig(dirs+"/"+model_names[num]+".png",dpi=300)


if __name__ == "__main__":
    X_train,y_train,X_test,y_test = load_data()
    X_train_g = np.zeros(X_train.shape[:-1])
    X_test_g = np.zeros(X_test.shape[:-1])
    for i in range(X_train.shape[0]):
        X_train_g[i] = cv2.cvtColor(X_train[i], cv2.COLOR_RGB2GRAY)
    for i in range(X_test.shape[0]):
        X_test_g[i] = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2GRAY)
    X_train_g = np.stack([X_train_g] * 3, axis=-1)
    X_test_g = np.stack([X_test_g] * 3, axis=-1)

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_train_g = np.reshape(X_train, (X_train_g.shape[0], -1))
    X_test_g = np.reshape(X_test, (X_test_g.shape[0], -1))

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


    models = get_model_set()
    try_different_method(models,X_train_g,y_train,X_test_g,y_test)




