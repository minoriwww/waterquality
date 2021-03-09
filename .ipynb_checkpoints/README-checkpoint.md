# waterquality


# Required:
```
python>=3.6
pytorch>=1.2
Keras=2.2
tensorflow=1.14
```

Data available [online](http://digbio.missouri.edu/dliu/projects/waterquality/) , npy format. 

# Index:

image-image7/: 
raw image. name: XXX nb_batch a/b.jpg
XXX means content value, nb_batch means batch number of that image. (total nb of batch equals to nb of folders). a/b means testing under bottle a or b

classic_classification/, classic_regression/:
results. subfolder "Gray" have some fun images

classification_result/:
conventional methods comparsion experiment, in .py file

interval_result/:
results by interval experiments

modelWeights/:
h5 files of nn weight

result/:
csv format, comparsion table, for different model settings

result_analyse/:
measure how many predicted points in each bin(interval)

water/:
some playground for the data and model middle layer. (feat_analysis.ipynb)


website/:
old oil prediction website. 

pytorch-cifar/:
main.py: wrapper for training
models/: conventional models and new models (mainly attnResNet50.py)

/bin_analyse.py
comparsion experiments. such as test_different_pretrain_model_with_gen()

/classic_classification.py
/classic_regerssion.py
/classification.py
comparsion experiments with different conventional methods. treated as classification / regression


/CNN_regression.py
/CNN_regression2.py
/CNN.py
old models

/combined_trick_model.py
pretrained resnet18 with linear fc

/data_loader.py
load image dataset,image hight/width...

/data_to_npy.py
save the processed data as npy, 

/draw_width.py
draw predicted bin width 

/feature_test.ipynb
feature engineering, middle layer visualization, histogram analyse

/interval_process.py
calculate interval array and store

/model_eval.py
comparsion test different pretrain model

/multiNets.py
/multiNetsVgg.py
old model used in server, plain DNN/VGG

/ordinal_categorical_crossentropy.py
loss function in Keras, for ordinal loss

/plot_result_fig.py
scatter plot for prediction results

/predictioin_tocsv.py
show_roc_pr_curve, store comparsion result in csv


Please cite (for now)
```
@misc{waterquality,
  author = {minoriwww},
  title = {waterquality project and OilSS},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/minoriwww/waterquality}}
}
```
