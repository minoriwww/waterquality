from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as utils
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import time
import copy
#load pytorch pretrained models and combine classfication and regression


from data_loader import pre_training_data

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def combined_model(num_classes, feature_extract=False):

    model_ft = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def prepare_torch_model_data():
    train, label, vali_train, vali_label,X_test,y_test = pre_training_data(is_scaler=False, is_categorical=True, is_1hot_categ=False)

    train = torch.stack([torch.Tensor(i) for i in train])
    # transform to torch tensors
    label = torch.Tensor(label).to(torch.long)
    vali_train = torch.stack([torch.Tensor(i) for i in vali_train])
    vali_label = torch.Tensor(vali_label).to(torch.long)
    X_test = torch.stack([torch.Tensor(i) for i in X_test])
    y_test = torch.Tensor(y_test).to(torch.long)

    #from channle last to channle first
    train = train.permute(0,3,1,2)
    vali_train = vali_train.permute(0,3,1,2)
    X_test = X_test.permute(0,3,1,2)

    # data_dir = 'data/hymenoptera_data'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
    #                                              shuffle=True, num_workers=4)
    #               for x in ['train', 'val']}

    dataloaders = {'train':utils.DataLoader(utils.TensorDataset(train,label)),'val': utils.DataLoader(utils.TensorDataset(vali_train,vali_label))}

    dataset_sizes = {'train':len(train), 'val':len(vali_train)}
    num_classes = 2
    if len(label.size())<=1:
        t, idx = np.unique(label.numpy(), return_inverse=True)
        num_classes = len(t)+1
    if len(label.size())>1 and label.size()[1]>=2:
        num_classes = label.size()[1]

    print("tensor x size", train.size())
    print("tensor y size", label.size())
    print("num_classes", num_classes)
    return num_classes, dataloaders, dataset_sizes



if __name__ == '__main__':
    num_classes, dataloaders, dataset_sizes = prepare_torch_model_data()
    model_ft = train_model(*combined_model(num_classes), dataloaders,dataset_sizes,num_epochs=25)










