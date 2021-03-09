import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.externals import joblib

def transform_y(y_pred, scaler_test):

    y_pred = y_pred.reshape(-1, 1)

    y_pred = scaler_test.inverse_transform(y_pred)

    y_pred = y_pred.flatten()
    return y_pred

plt.figure(figsize=(12,5))


model_names = ["MobileNet","K-Nearest Neighbors", "Linear","Random Forest"]

folder_names = ["1570997508.460764MobileNet","1570998378.995135KNeighborsRegressor","1571012815.106932LinearRegressor","1571012974.0192485RandomForestRegressor"]

scaler_val = joblib.load('MaxAbsScaler.pkl')

# for i,name in enumerate(model_names):

#     upper = np.load("interval_result/"+folder_names[i]+"/upper.npy")
#     lower = np.load("interval_result/"+folder_names[i]+"/lower.npy")
#     predicts = np.load("interval_result/"+folder_names[i]+"/result.npy")
#     y_test = np.load("y_test_ratio=0.2.npy")

#     interval_width = []
#     for i in range(len(upper)):
#         interval_width.append(upper[i] + lower[i])

#     bin_name = []
#     for i, item in enumerate(interval_width):
#         bin_name.append(str(i*10))
#     bin_name[-1]="300+"

    

#     if name == "MobileNet": predicts = transform_y(predicts, scaler_val)

#     rmse = sqrt(mean_squared_error(y_test,predicts))

#     print(name+" RMSE: %f" %rmse)

#     mbw = np.average(interval_width)

#     plt.plot(bin_name, interval_width,label = name+" (MBW = %f)" % mbw)

# plt.legend()
# plt.savefig("interval_width_comparison" + ".png", dpi=300)
x = np.linspace(0,500)

for i,name in enumerate(model_names):
    predicts = np.load("interval_result/"+folder_names[i]+"/result.npy")
    y_test = np.load("y_test_ratio=0.2.npy")

    if name == "MobileNet": predicts = transform_y(predicts, scaler_val)
    plt.subplot(2,2,i+1)
    
    rmse = sqrt(mean_squared_error(y_test,predicts))
    
    plt.scatter(predicts,y_test,label = model_names[i]+' (RMSE = %f)'% rmse)
    plt.plot(x,x,c='r',alpha = 0.5)
    if i+1==3 or i+1==4: plt.xlabel("Predicted Value")
    if i+1==1 or i+1==3: plt.ylabel("True Value")
    plt.legend()
plt.legend()
plt.savefig("predictions" + ".png", dpi=300)
