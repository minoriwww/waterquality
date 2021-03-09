import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

plt.switch_backend('agg')
import time

def load_data():

    X_train = np.load("X_train_ratio=0.2.npy")
    y_train = np.load("y_train_ratio=0.2.npy")

    X_test = np.load("X_test_ratio=0.2.npy")
    y_test = np.load("y_test_ratio=0.2.npy")
    
    X_train_ = np.zeros((X_train.shape[0], 256, 256, 3), dtype=np.float32)
    X_test_ = np.zeros((X_test.shape[0], 256, 256, 3), dtype=np.float32)
    
    counter = 0
    for  img_ in X_train:
        img = resize(img_, (256, 256))
        X_train_[counter] = img
        counter+=1
        
    counter = 0
    for img_ in X_test:
        img = resize(img_, (256, 256))
        X_test_[counter] = img
        counter+=1
    
    return X_train_, y_train, X_test_, y_test


def get_model_set():

    models = []

    from sklearn import tree
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    ####3.2线性回归####
    from sklearn import linear_model
    model_LinearRegression = linear_model.LinearRegression()
    ####3.3SVM回归####
    from sklearn import svm
    model_SVR = svm.SVR()
    ####3.4KNN回归####
    from sklearn import neighbors
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    ####3.5随机森林回归####
    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
    ####3.6Adaboost回归####
    from sklearn import ensemble
    model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
    ####3.7GBRT回归####
    from sklearn import ensemble
    model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
    ####3.8Bagging回归####
    from sklearn.ensemble import BaggingRegressor
    model_BaggingRegressor = BaggingRegressor()
    ####3.9ExtraTree极端随机树回归####
    from sklearn.tree import ExtraTreeRegressor
    model_ExtraTreeRegressor = ExtraTreeRegressor()

    models.append(model_AdaBoostRegressor)
    models.append(model_BaggingRegressor)
    models.append(model_DecisionTreeRegressor)
    models.append(model_ExtraTreeRegressor)
    models.append(model_GradientBoostingRegressor)
    models.append(model_KNeighborsRegressor)
    models.append(model_LinearRegression)
    models.append(model_RandomForestRegressor)
    models.append(model_SVR)

    return models


###########2.回归部分##########
def try_different_method(models,X_train,y_train,X_test,y_test):
    model_names=["AdaBoost Regressor","Bagging Regressor","Decision Tree Regressor","Extra Tree Regressor",
    "Gradient Boosting Regressor","KNeighbors Regressor","Linear Regression",
    "Random ForestRegressor","SVR"]

    import os

    dirs = 'classic_regression/Gray/'+str(time.time())

    # dirs = 'classic_regression/'+str(time.time())

    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    for num,model in enumerate(models):
        model.fit(X_train,y_train)
        # score = model.score(X_test, y_test)
        result = model.predict(X_test)

        np.save(dirs+"/"+model_names[num]+".npy",result)
        rmse = sqrt(mean_squared_error(y_test,result))

        plt.figure()

        plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
        plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
        #plt.title(model_names[num]+'score: %f'%score)
        plt.title(model_names[num] + ' rmse: %f' % rmse)
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
    X_train_g = np.stack([X_train_g]*3, axis=-1)
    X_test_g = np.stack([X_test_g]*3, axis=-1)

    X_train = np.reshape(X_train,(X_train.shape[0],-1))
    X_test = np.reshape(X_test,(X_test.shape[0],-1))
    X_train_g = np.reshape(X_train,(X_train_g.shape[0],-1))
    X_test_g = np.reshape(X_test,(X_test_g.shape[0],-1))

    print(X_train.shape)
    print(X_test.shape)
    models = get_model_set()
    try_different_method(models,X_train,y_train,X_test,y_test)





