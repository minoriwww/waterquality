import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from multiNets import *
from plot_result_fig import draw_scatter
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()

# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

def test_different_pretrain_model_with_gen():

    #regression
    is_scaler = True
    train, label, vali_train, vali_label,X_test,y_test = pre_training_data(is_scaler=is_scaler, is_categorical=False)

    model_fn = pretrained_models(model_name="inception_v3") #dnn_model VGG
    y_pred, y_test_ = run_model_with_gen(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_inception_v3_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "inception_v3")

    model_fn = pretrained_models(model_name="mobilenet") #dnn_model VGG
    y_pred, y_test_ = run_model_with_gen(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_mobilenet_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "mobilenet")

    model_fn = pretrained_models(model_name="densenet") #dnn_model VGG
    y_pred, y_test_ = run_model_with_gen(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_densenet_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "densenet")

    model_fn = pretrained_models(model_name="resnet50") #dnn_model VGG
    y_pred, y_test_ = run_model_with_gen(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_resnet50_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "resnet50")

    model_fn = pretrained_models(model_name="resnet101") #dnn_model VGG
    y_pred, y_test_ = run_model_with_gen(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_resnet101_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "resnet101")

def test_different_pretain_model():
    #regression
    is_scaler=True
    train, label, vali_train, vali_label,X_test,y_test = pre_training_data(is_scaler=is_scaler, is_categorical=False)

    model_fn = pretrained_models(model_name="inception_v3") #dnn_model VGG
    y_pred, y_test_ = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_inception_v3_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "inception_v3")

    model_fn = pretrained_models(model_name="mobilenet") #dnn_model VGG
    y_pred, y_test_ = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_mobilenet_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "mobilenet")

    model_fn = pretrained_models(model_name="densenet") #dnn_model VGG
    y_pred, y_test_ = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_densenet_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "densenet")

    model_fn = pretrained_models(model_name="resnet50") #dnn_model VGG
    y_pred, y_test_ = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_resnet50_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "resnet50")

    model_fn = pretrained_models(model_name="resnet101") #dnn_model VGG
    y_pred, y_test_ = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn,is_classfication=False, nb_classes=1)
    y_pred, y_test_ = post_training_data(y_test_, y_pred, is_scaler)
    predicted_file = 'regress_'+model_fn.__name__+"_resnet101_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test_)
    draw_scatter(predicted_file, "resnet101")



def test_different_augmentation():
    pass


def compare_traditional_methods():
    """
    run time, RMSE with label...
    """
    pass


def test_monochromatic():
    pass

def test_binning():
    """distributions with different bin size.
    """
    train, label, vali_train, vali_label,X_test,y_test = pre_training_data(is_scaler=False, is_categorical=True, is_1hot_categ=True)
    # model_fn = pretrained_models(model_name="resnet50") #dnn_model VGG
    model_fn = VGG

    print("label shape", vali_label.shape[1])
    y_pred, y_test = run_model(train, label, vali_train, vali_label,X_test,y_test, model_fn = model_fn, is_classfication=True, nb_classes=vali_label.shape[1])
    y_pred, y_test = post_training_data(y_test, y_pred)
    predicted_file = 'cardinal_'+model_fn.__name__+"_resnet50_"+time_str+'.csv'
    CNN_Regression.save_result("./result/"+predicted_file,y_pred,y_test)
    draw_scatter(predicted_file, "resnet50")

def test_classfication_then_regression():
    pass



def test_k_fold():
    pass

def test_bootstrap():
    pass


if __name__ == '__main__':
    # test_different_pretrain_model_with_gen()
    test_different_pretain_model()
    # test_binning()
