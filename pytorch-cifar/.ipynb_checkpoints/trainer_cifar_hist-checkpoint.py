'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from sklearn import preprocessing
import pickle 

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import argparse

from models import *
from utils import progress_bar
from  models import attnResNet50, resnet

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--nb_epoch',  default=50, type=int,help='resume from checkpoint')
parser.add_argument('--lr', default=2e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



import neptune
neptune.init('minoriwww/water',api_token=os.environ['NEPTUNE_API_TOKEN'])
torch.backends.cudnn.benchmark = True
run_start_time = str(time.time())

def rgb2gray(rgb):
    """
    scikit-image:
    https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_rgb_to_gray.html
    """
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #opencv
    gray = 0.2125 * r + 0.7154 * g + 0.0721  * b  #scikit-image
    return gray


class WaterImageDataset(Dataset):
    def __init__(self, x, y, stage=1, seed=42,  **kwargs):
        super(WaterImageDataset).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        
        self.set_stage_crop(stage)
        self.norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
    def __len__(self):
        return len( self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):idx = idx.tolist()

#         sample = [self.x[idx], self.y[idx]]
        return [self.crop(self.x[idx]), self.y[idx]]

    def set_stage_crop(self, stage):
        if stage == 0:
            print('Using (32, 32) crops')
#             self.crop = transforms.RandomCrop((32, 32))
            self.crop = transforms.Resize((32, 32))
        elif stage == 1:
            print('Using (512, 512) origin')
            self.crop = transforms.Resize((256, 256))
            
    def __iter__(self):
        return iter(list(zip(self.x, self.y)))


def loaddata(load_water_data=False):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
#         transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    

    if load_water_data:
        X_train = np.load('../X_train_ratio=0.2.npy')
        X_test = np.load('../X_test_ratio=0.2.npy')
        y_train = np.load('../y_train_ratio=0.2.npy')
        y_test = np.load('../y_test_ratio=0.2.npy')
        
        scaler_trn = preprocessing.MaxAbsScaler()
        y_train = scaler_trn.fit_transform(y_train.reshape(-1, 1))
        y_train = y_train.flatten()
        
        y_test = scaler_trn.transform(y_test.reshape(-1, 1))
        y_test = y_test.flatten()
#         scaler_test = preprocessing.MaxAbsScaler()
#         y_test = scaler_test.fit_transform(y_test.reshape(-1, 1))
#         y_test = y_test.flatten()
        
        pickle.dump(scaler_trn, open('../scaler_trn.pkl', 'wb'))
#         scaler = pickle.load(open('../scaler_trn.pkl', 'rb'))
        
        # print(X_test.shape)
        X_train = np.swapaxes(X_train, 1,3)
        X_test = np.swapaxes(X_test, 1,3)

        trainset = WaterImageDataset(X_train, y_train)
        trainset.transform = transform_train
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True, num_workers=4)

        testset = WaterImageDataset(X_test, y_test)
        testset.transform = transform_test
        testloader = torch.utils.data.DataLoader(testset, batch_size=3, shuffle=True, num_workers=4)
        
    else:
        trainset = torchvision.datasets.CIFAR10(root='/media/mu/NewVolume/Programs/waterquality/pytorch-cifar', train=True, download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='/media/mu/NewVolume/Programs/waterquality/pytorch-cifar', train=False, download=True, transform=transform_test)

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader

def loadmodel(nb_class=10, img_HW=8, pretrain_model='resnet18'):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    if pretrain_model == "resnet_hist18":
        net = attnResNet50.resnet18(num_classes=nb_class, image_HW=img_HW)
#         net = attnResNet50.load_resnet_imagenet(model=net, modelname="resnet18")
    elif pretrain_model == "resnet_hist50":
        net = attnResNet50.resnet50(num_classes=nb_class, image_HW=img_HW, pretrained=False)
#         net = attnResNet50.load_resnet_imagenet(model=net, modelname="resnet50")
    elif pretrain_model is None:
        net = attnResNet50.resnet50(num_classes=nb_class, image_HW=img_HW)
    elif pretrain_model == "resnet18":
        net = resnet.ResNet18(n_classes=nb_class)
#         net = attnResNet50.load_resnet_imagenet(model=net, modelname="resnet18")
    elif pretrain_model == "resnet50":
        net = resnet.ResNet50(n_classes=nb_class)
#         net = attnResNet50.load_resnet_imagenet(model=net, modelname="resnet50")



    print(net)
    with torch.no_grad():
        net = net.to(device)
        attnResNet50.initialize_weights(net)
    
#     if device == 'cuda':
#         net = torch.nn.DataParallel(net)
#         cudnn.benchmark = True
    print("Compute on device")
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt'+run_start_time+'.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

#     criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    return net, criterion, optimizer


# Training
def train(epoch, net, criterion, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    print("train mode")
    train_loss = 0
    correct = 0
    total = 0
#     MAXABSscaler = pickle.load(open('../scaler_trn.pkl', 'rb'))
    training_rec_str = ""
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        torch.cuda.empty_cache()
        
        inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
        targets.resize_(targets.shape[0], 1)
#         print(targets)
#         print(MAXABSscaler.inverse_transform(targets.cpu().data.numpy()))
        
        
        outputs = net(inputs)
#         print(outputs)
#         print(MAXABSscaler.inverse_transform(outputs.cpu().data.numpy()))
        
        eps = 1e-7
        loss = torch.sqrt(criterion(outputs, targets)+eps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            training_rec_str += "loss: "+str(loss)+" target: "+str(targets)+" outputs: "+str(outputs)+"\n"
            
            train_loss += float(loss.item())
            _, predicted = outputs.max(1)
    #         total += targets.size(0)
            neptune.log_metric('loss', (train_loss/(batch_idx+1)))
    #         neptune.log_metric('acc', 100.*correct/total)
            # correct += predicted.eq(targets).sum().item()

    #         writer.add_scalar('loss', train_loss/(batch_idx+1), batch_idx+1)
    #         writer.add_scalar('acc', 100.*correct/total, batch_idx+1)
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | '
                % (train_loss/(batch_idx+1)))
            
        torch.cuda.empty_cache()
    
    with open("../result/trainer_hist_trn_ep"+str(epoch)+"_time_"+str(time.time()), 'w+') as myWrite:
        myWrite.write(training_rec_str)
    
    return 
def test(epoch,  net, criterion, optimizer, testloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    your_file = open('predict_res.csv', 'ab')
    
    MAXABSscaler = pickle.load(open('../scaler_trn.pkl', 'rb'))
    test_rec_str = ""
    
    
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device).to(torch.float32)
            targets.resize_(targets.shape[0], 1)
            
            outputs = net(inputs)
            loss = torch.sqrt(criterion(outputs, targets))
            
            targets_trans = MAXABSscaler.inverse_transform(targets.cpu().data.numpy())
            outputs_trans = MAXABSscaler.inverse_transform(outputs.cpu().data.numpy())
            
            test_rec_str += "loss: "+str(loss)+" target: "+str(targets)+" ori_target: "+str(targets_trans)+" outputs: "+str(outputs)+" ori_output: "+str(outputs_trans)+"\n"
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            neptune.log_metric('testloss', (test_loss/(batch_idx+1)))
            neptune.log_metric('testacc', 100.*correct/total)
            # correct += predicted.eq(targets).sum().item()
                    
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    with open("../result/trainer_hist_test_ep"+str(epoch)+"_time_"+str(time.time()), 'w+') as mytestWrite:
        mytestWrite.write(test_rec_str)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt'+run_start_time+'.pth')
        best_acc = acc
    your_file.close()

if __name__ == '__main__':
#     writer = SummaryWriter(log_dir="/home/dltdc/data/projects_logs/water_logs/", filename_suffix=run_start_time)

    trainloader, testloader = loaddata(load_water_data=True)
    net, criterion, optimizer = loadmodel(nb_class=1, img_HW=256, pretrain_model="resnet50")


    # trainloader, testloader = loaddata()
    # net, criterion, optimizer = loadmodel(nb_class=10, img_HW=8, pretrain_model='resnet18')
    

    with neptune.create_experiment(name='new-model'):
        neptune.append_tag('first')
        for epoch in range(start_epoch, start_epoch+args.nb_epoch):
            train(epoch, net, criterion, optimizer, trainloader)
            test(epoch, net, criterion, optimizer, testloader)

#     writer.close()

