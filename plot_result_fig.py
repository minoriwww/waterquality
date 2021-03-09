# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("../")
import csv
import json

from multiNets import BIN_SIZE, MAX_LIM


# BIN_SIZE = 20#20
# MAX_LIM = 200#200

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

def evaluation(pred_label,true_label):
    test_pre=np.zeros(len(pred_label))
    for i in range(0,len(pred_label)):
       test_pre[i]=np.argmax(pred_label[i,])

    true_pre=np.zeros(len(pred_label))
    for i in range(0,len(true_label)):
       true_pre[i]=np.argmax(true_label[i,])

    print("mcc ="+ str(matthews_corrcoef(true_pre,test_pre)))
    print("confusion matrix="+str(confusion_matrix(true_pre,test_pre)))

def processing_y(value_float, default_method='bins', bin_size=BIN_SIZE, max_lim=MAX_LIM, min_lim=0):
    if default_method is None:
        return value_float
    if default_method == 'bins':
        nb_bins = (max_lim - min_lim)/bin_size
        if value_float>=max_lim: return nb_bins
        if value_float<min_lim: return -1
        return int(value_float/bin_size)
    if default_method == 'bins_reverse':
        return float(bin_size * value_float)

def draw_box(result_filename = [], title = ""):
    error_ls = []

    df = pd.read_csv("./result/"+result_filename[0], header=None)
    x = df.loc[1]
    y = df.loc[0]
    y = y_pred = np.array([processing_y(i, default_method=None) for i in y])
    x = y_test = np.array([processing_y(i, default_method=None) for i in x])
    # x = np.around(x)
    # y = np.around(x)
    error = y-x
    print(error)
    x = np.around(x)
    y = np.around(x)
    print(y-x)
    error_ls.append(error)
    print("RMSE:"+str(rmse(y_pred, y_test)))

    df = pd.read_csv("./result/"+result_filename[1], header=None)
    x = df.loc[1]
    y = df.loc[0]
    y = y_pred = np.array([processing_y(i, default_method='bins_reverse') for i in y])
    x = y_test = np.array([processing_y(i, default_method='bins_reverse') for i in x])
    # x = np.around(x)
    # y = np.around(x)
    error = y-x
    print(error)
    x = np.around(x)
    y = np.around(x)
    print(y-x)
    error_ls.append(error)
    print("RMSE:"+str(rmse(y_pred, y_test)))


    fig = plt.figure()
    #设置X轴标签
    plt.xlabel('bin size')
    #设置Y轴标签
    plt.ylabel('error')

    plt.boxplot(error_ls, labels=["previous", "bin 20"])
    # plt.plot(x_,y_, label='x=y')
    #设置图标
    plt.legend()
    #显示所画的图
    # plt.show()
    plt.savefig("./result/figures/"+title+"_boxplot.png")

def draw_scatter(result_filename = "", title = "", df = None, target=None, pred=None):
    figuretitle = title
    if df is None and target is None:
        df = pd.read_csv("./result/"+result_filename, header=None)
        x = df.loc[1]
        y = df.loc[0]
        y = y_pred = np.array([processing_y(i, default_method='bins_reverse') for i in y])
        x = y_test = np.array([processing_y(i, default_method='bins_reverse') for i in x])
    elif target is not None:
        y = y_pred = pred
        x = y_test = target
    
    print(mean_squared_error(y_test.reshape(-1), y_pred.reshape(-1)))
    # import bin_analyse
    # truth_list, error_list, predict_list = bin_analyse.csv_converter()
    # x = truth_list
    # y = predict_list

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #设置标题
    ax1.set_title(figuretitle)
    #设置X轴标签
    plt.xlabel('ground truth')
    #设置Y轴标签
    plt.ylabel('prediction')
    #画散点图
    plt.xlim(0, max(x))
    plt.ylim(0, max(x))
    ax1.scatter(x,y,c = 'b',marker = 'o', label='prediction')

    x_ = np.arange(1,500)
    y_ = x_
    plt.plot(x_,y_, label='x=y')
    #设置图标
    plt.legend()
    #显示所画的图
    # plt.show()
    plt.savefig("./result/figures/"+figuretitle+".png")

def add_ratio_to_csv(result_filename = "", title = ""):
    df = pd.read_csv("./result/"+result_filename, header=None)
    x = df.loc[1]
    y = df.loc[0]
    residual = df.loc[2].copy()
    ratio_residual = (residual.abs()/x.abs()).copy()
    ratio_y = (y.abs()/x.abs()).copy()

    ratio_residual = pd.DataFrame(ratio_residual.values.reshape(1, -1))
    ratio_y = pd.DataFrame(ratio_y.values.reshape(1, -1))
    print(ratio_y.values.shape)
    print(df.values.shape)
    newdf = pd.concat([df, ratio_residual, ratio_y]).reset_index(drop=True)

    metric = pd.Series(['prediction', 'groundtruth(GT)', 'residual', 'ratio_residual', 'ratio_yofGT'])
    # newdf = newdf.reset_index().reindex(metric)
    newdf.insert(0,'metric', metric)
    # newdf.set_index(metric)
    # newdf.ix['prediction'] = y
    # newdf.ix['groundtruth(GT)'] = x
    # newdf.ix['residual'] = residual
    # newdf.ix['ratio_residual'] = ratio_residual
    # newdf.ix['ratio_yofGT'] = ratio_y
    mse = ((y - x) ** 2).mean() ** .5
    newdf.to_csv("./result/"+title+"ratio_residual_"+str(mse)+"_.csv", index=True)
    return ratio_residual, ratio_y, newdf

def process_hist_res(location="result/processed_test_ep32.csv"):

    data=pd.read_csv(location, sep=';',header=None)

    readtarget = data.loc[:, 1].to_numpy()
    readpred = data.loc[:, 3].to_numpy()
#     print(readtarget)
#     print(readpred)
    target = np.array([np.array(json.loads(xi.replace(".", " "))) for xi in readtarget])
    pred = np.array([np.array(json.loads(xi)) for xi in readpred])

    pred[pred < 0] = 0
    pred[pred >300 ] = 300
    
    target[target < 0] = 0
    target[target >300 ] = 300
    
    target = target.reshape(-1)
    pred = pred.reshape(-1)
    print(pred)
    print(target)
    
    print(mean_squared_error(target, pred))
    return target, pred

if __name__ == '__main__':
    target, pred = process_hist_res()
    # draw_scatter("regress_1_fold_1VGG1540783221.82.csv", title = "image 4 random") #image 4 random
    draw_scatter("regress_1_fold_1VGG1540788567.12.csv", title = "image 5 random") #

    # draw_scatter("regress_4_fold_2VGG1540351601.56.csv", title = "image 4 fix") #
    draw_scatter("regress_5_fold_2VGG1540356929.93.csv", title = "image 5 fix") #i

#     draw_scatter("regress_VGG1539719516.41_123_4.csv", title = "123 predict image 4") #
    draw_scatter("regress_VGG1539998825.42_1234_5.csv", title = "1234 predict image 5") #

    draw_scatter("regress_VGG1540964659.24.csv", title = "12345 predict image 6") #
    
    draw_scatter(target = target, pred=pred, title = "hist block pred") 
    # draw_scatter("regress_6_fold_1VGG1540971169.83.csv", title = "image 6 fix") #
    # draw_scatter("regress_1_fold_1VGG1541094204.23.csv", title = "image 6 random") #

    # draw_scatter("regress_RegressorModelsEstimator1541535069.24.csv", title = "123456 ensemble without NN")

    # draw_scatter("regress_RegressorModelsEstimator1541744725.0.csv", title = "123456 residue ensemble with NN")
#     draw_scatter("regress_bins_4fold_VGG1550366626.18.csv", title = "fold bin size: "+str(BIN_SIZE))
    # draw_box(["regress_RegressorModelsEstimator1541744725.0.csv", "regress_VGG1547948494.19_bin20.csv"], title = "1234567 bin size: "+str(BIN_SIZE))
    # draw_scatter("regress_VGG_fold_01544619616.97.csv", title = "1234567 random empty")
#     draw_scatter("regress_VGG1543289548.43.csv", title = "123457 predict image 6")
    # add_ratio_to_csv("regress_VGG1539998825.42_1234_5.csv", title = "1234_5")
    # add_ratio_to_csv("regress_VGG1540964659.24.csv", title = "12345_6")
    # add_ratio_to_csv("regress_5_fold_2VGG1540356929.93.csv", title = "5_fix")
    # add_ratio_to_csv("regress_1_fold_1VGG1540788567.12.csv", title = "5_random")
    # add_ratio_to_csv("regress_6_fold_1VGG1540971169.83.csv", title = "6_fix")
    # add_ratio_to_csv("regress_1_fold_1VGG1541094204.23.csv", title = "5_random")
