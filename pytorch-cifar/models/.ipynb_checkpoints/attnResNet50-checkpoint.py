import numpy as np
import math
from argparse import ArgumentParser
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import OrderedDict

import torch
import torch.nn as nn
# from torch.nn import *
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import time

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feat_map_wise_hist(inp, nb_bins=8):
    """
    https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """
    x = inp
    x = x.contiguous().to(device)
    with torch.no_grad():
        hist = x.new().to(device)
        for b in range(inp.shape[0]):
            hist_pre_ch = x.new().to(device)
            for c in range(inp.shape[1]):
                #histc

                x = inp[b,c,:,:]
                hist_pre_ch = torch.cat([hist_pre_ch,torch.histc(x, bins=nb_bins, min=x.min().item(), max=x.max().item()).cuda().view(1, nb_bins)])
#                 if x.min().item() < 0:
#                     x = x - x.min()
#                 conv_binned = torch.trunc(x * (nb_bins-1)/x.max().item()).cuda()
#                 ones = torch.ones_like(conv_binned, device=device)
#                 zeros = torch.zeros_like(conv_binned, device=device)
#                 hist_pre_ch.append(torch.tensor([torch.where(conv_binned == bin_, ones, zeros).sum()
#                                 for bin_ in range(nb_bins)], device=device))

            hist = torch.cat([hist, hist_pre_ch.view(1,inp.shape[1], nb_bins)])
#         hist = torch.stack(hist, 0).cuda()
    return hist

def feat_map_shape_hist(inp, nb_bins=8):
    """
    https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """
    x = inp
    x = x.contiguous().to(device)
    with torch.no_grad():
        hist_ = torch.histc(x, bins=inp.shape[0]*inp.shape[1]*nb_bins, min=x.min().item(), max=x.max().item()).cuda().view(inp.shape[0], inp.shape[1], nb_bins)
    
    return hist_

def rmac_hist(inp,
              L_min=7, #7 for fixed width, 1 for all
              L=7,
              nb_bins=8,
              eps=1e-7):
    '''
    https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/functional.py#L26
    '''
#     x = inp.clone().detach()
    x = torch.empty_like(inp).copy_(inp)
    with torch.no_grad():
        ovr = 0.4 # desired overlap of neighboring regions
        steps = torch.LongTensor([2, 3, 4, 5, 6, 7]) # possible regions for the long dimension

        W = x.size(3)
        H = x.size(2)

        w = min(W, H)
        w2 = math.floor(w/2.0 - 1)

        b = (max(H, W)-w)/(steps-1)
        (tmp, idx) = torch.min(torch.abs(((w**2 - w*b)/w**2)-ovr), 0) # steps(idx) regions for long dimension

        # region overplus per dimension
        Wd = 0;
        Hd = 0;
        if H < W:
            Wd = idx.item() + 1
        elif H > W:
            Hd = idx.item() + 1

        v = [feat_map_shape_hist(x)]
    #     v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

        for l in range(L_min, L+1):
            wl = math.floor(2*w/(l+1))
            wl2 = math.floor(wl/2 - 1)

            if l+Wd == 1:
                b = 0
            else:
                b = (W-wl)/(l+Wd-1)
            cenW_tmp = wl2 + torch.Tensor(range(l-1+Wd+1)).long()*b
            cenW_tmp = cenW_tmp.float()
            cenW = torch.floor(cenW_tmp) - wl2 # center coordinates
            if l+Hd == 1:
                b = 0
            else:
                b = (H-wl)/(l+Hd-1)
            cenH_tmp = wl2 + torch.Tensor(range(l-1+Wd+1)).long()*b
            cenH_tmp = cenH_tmp.float()
            cenH = torch.floor(cenH_tmp) - wl2 # center coordinates
            # print(cenH, cenW,wl, wl2)
            vt_array = []
            for i_ in cenH.tolist():
                for j_ in cenW.tolist():
                    if wl == 0:
                        continue
                    R = x[:,:,(int(i_)+torch.LongTensor(range(wl)).to(device)).tolist(),:]
                    R = R[:,:,:,(int(j_)+torch.LongTensor(range(wl)).to(device)).tolist()]

                    vt = feat_map_shape_hist(R)
#                     vt = torch.histc(R, bins=nb_bins, min=R.min().item(), max=R.max().item())

                    vt = vt / (torch.norm(vt, p=2, dim=-1, keepdim=True) + eps).expand_as(vt)
    #                 v += vt
                    vt_array+=[vt]

            vt_array = torch.stack(vt_array, -1)

            arr_along_batch = []
            for batch_id in range(x.shape[0]):
                arr_along_batch.append(F.fold(vt_array[batch_id,...], (len(cenH.tolist()),len(cenW.tolist())), (1,1)))
            arr_along_batch = torch.stack(arr_along_batch, 0).cuda()

    #         v += vt_array
#     print(arr_along_batch.shape)
    return arr_along_batch



class HistAttnBlock(nn.Module):
    """return (batch, nb_bins, H, W)
    only dim 1 have changed.
    attention reference:
    https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py#L9
    """
    def __init__(self, nb_bins=8, out_channel=32, in_img_HW = 512, hist_img_HW=8, upsample_stride=1,upsample_kernel=3):
        super(HistAttnBlock, self).__init__()
        self.nb_bins = nb_bins
        self.upsample_kernel = upsample_kernel

        self.hist_img_HW = hist_img_HW
        if hist_img_HW <= int(in_img_HW/32):
            self.hist_img_HW = int(in_img_HW/2)


        self.avgpool3d = nn.AdaptiveMaxPool3d((None,int(in_img_HW/2),int(in_img_HW/2))).cuda()
        self.avgpool2d = nn.AdaptiveMaxPool2d((in_img_HW,in_img_HW)).cuda()
#         self.conv_du = nn.Sequential(
#                         nn.Conv3d(in_channel, in_channel, self.conv3d_kernel, bias=True).cuda(),
#                         nn.ReLU(inplace=True),
#                         nn.Conv3d(2, 4, 1, bias=True),
#                         nn.Sigmoid()
#                 ).cuda()
        self.upsampleConvTrans2d = nn.ConvTranspose2d(self.nb_bins, out_channel, (self.upsample_kernel,self.upsample_kernel),
                                           stride=(upsample_stride,upsample_stride)
                                                      #np.ceil(in_img_HW//hist_img_HW).astype(int)
                                          ).cuda()
        self.upsample = nn.Upsample(size=(out_channel, in_img_HW, in_img_HW)).cuda()

        self.norm = nn.BatchNorm2d(out_channel).cuda()
        self.norm_attnonchannel = nn.BatchNorm2d(out_channel).cuda()
        self.norm3d = nn.BatchNorm3d(out_channel)
        
        self.act = nn.SELU(inplace=True).cuda()
        self.sigmoid = nn.Sigmoid().cuda()


    def forward(self, input):
        hist_pooling_res = rmac_hist(input, self.hist_img_HW, self.hist_img_HW)#CxBxhist_img_HW x hist_img_HW
#         print("hist_pooling_res", hist_pooling_res.shape)
#         redu_hist3d = self.conv_du(hist_pooling_res)
        redu_hist3d = hist_pooling_res
        res_avgpool3d = self.avgpool3d(redu_hist3d).cuda()
#         print("res_avgpool3d", res_avgpool3d.shape)
        
        attn_res = hist_pooling_res * self.norm3d(res_avgpool3d).cuda()
        attn_res_reduced = attn_res.sum(1).cuda()# reduced channel, keep hist dim
#         print("attn_res", attn_res.shape)
#         print("attn_res_reduced", attn_res_reduced.shape)
        
        attn_map_on_channel = attn_res.sum(2).cuda() # reduced hist dim, keep channel dim
#         print("attn_map_on_channel", attn_map_on_channel.shape)
        
        attn_map_on_channel = self.norm_attnonchannel(F.adaptive_max_pool2d(attn_map_on_channel, input.shape[-1]).cuda()).cuda()
#         attn_map_on_channel = self.sigmoid(attn_map_on_channel).cuda()

        # upsampled_hist
        upsampled_hist = self.upsampleConvTrans2d(attn_res_reduced, output_size=input.size()).cuda()
#         upsampled_hist = self.upsample(attn_res_reduced).cuda()

        upsampled_hist = self.norm(upsampled_hist)
#         attn_res = self.sigmoid(upsampled_hist)
        attn_res = upsampled_hist

        return attn_res, attn_map_on_channel


class HistCNN(nn.Module):
    def __init__(self, in_channel, out_channel, in_img_HW = 512, nb_bins=8, hist_img_HW=8,  upsample_stride=1, upsample_kernel=3, kernel_size=5):
        super(HistCNN, self).__init__()

#         self.conv1 = nn.Conv2d(in_channel+nb_bins, out_channel, kernel_size=kernel_size, padding=1).cuda()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=3, padding=3).cuda()
    
#         self.conv2 = nn.Conv2d(in_channel*2, in_channel*3, kernel_size=3, stride=2, padding=2).cuda()
        
#         self.conv3 = nn.Conv2d(in_channel*3, out_channel, kernel_size=3, stride=2, padding=3).cuda()

        self.norm = nn.BatchNorm2d(in_channel).cuda()
        self.norm2 = nn.BatchNorm2d(out_channel).cuda()
        self.norm3 = nn.BatchNorm2d(in_channel*3).cuda()
        self.norm4 = nn.BatchNorm2d(in_channel*4).cuda()

        
        self.histnorm = nn.BatchNorm2d(in_channel).cuda()
        
        self.act = nn.SELU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.maxpool2d = nn.AdaptiveMaxPool2d((86,86)).cuda()
#         self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).cuda()

        self.hist = HistAttnBlock(nb_bins=nb_bins, out_channel=in_channel,in_img_HW=in_img_HW, hist_img_HW=hist_img_HW, upsample_stride=upsample_stride, upsample_kernel=upsample_kernel).cuda()


    def forward(self, x, last_hist_block):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = out.view(out.size(0), -1)  # reshape
#         out = self.fc(out)
        x = self.norm(x)
        out, attn_map = self.hist(x) #out:binszie x H x W
#         print(attn_map.shape)
#         print(x.shape)
        x = x*attn_map #3 x h x w
        x = F.softplus(x)
        
#         res_last_block = self.maxpool2d(last_hist_block).max().item() #scalar

        res_last_block = self.maxpool2d(last_hist_block).cuda()

#         out = torch.cat((x,out), dim=1).cuda()
        out = out + x

#         out += res_last_block
#         out = torch.matmul(out, res_last_block)
        
#         out = self.norm(out)
#         out = F.softplus(out)
    
        out = self.conv1(out)
#         out = self.norm2(out)
        out = self.norm2(out)
        out = torch.matmul(out, res_last_block)
#         out = self.norm2(out)
#         out = F.softplus(out)
        
#         out = self.conv2(out)
#         out = self.norm3(out)
#         out = F.softplus(out)
        
#         out = self.conv3(out)
#         out = self.norm4(out)
#         out = F.softplus(out)
        
#         print(out.shape)
#         print(out.view(out.size(0), -1).shape)
#         out = self.sigmoid(out)

        return out





model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# class ResNetHist(pl.LightningModule, nn.Module):
class ResNetHist( nn.Module):

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, image_HW = 32, nb_bins=8):
        super(ResNetHist, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=3, padding=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, 1000).cuda()
        #######

        # hist Attention Part
        # input 512x512
        self.histAttnBlock0 = HistCNN(in_channel=3, out_channel=3, in_img_HW=image_HW, hist_img_HW=15,
            upsample_stride=2,
            upsample_kernel=2).cuda()
        self.histAttnBlock1 = HistCNN(in_channel=64, out_channel=64, in_img_HW=image_HW, hist_img_HW=2,
            upsample_stride=2,
            upsample_kernel=2).cuda()
#         self.histAttnBlock2 = HistCNN(in_channel=256, out_channel=256, in_img_HW=image_HW, hist_img_HW=6,
#             upsample_stride=2,
#             upsample_kernel=2).cuda()


        # input 8x8
        # self.histAttnBlock0 = HistCNN(in_channel=3, out_channel=3, in_img_HW=image_HW, hist_img_HW=15,
        #     upsample_stride=2,
        #     upsample_kernel=3).cuda()
        # self.histAttnBlock1 = HistCNN(in_channel=64, out_channel=64, in_img_HW=image_HW, hist_img_HW=6, upsample_kernel=3).cuda()
        # self.histAttnBlock2 = HistCNN(in_channel=256, out_channel=256, in_img_HW=image_HW, hist_img_HW=6, upsample_kernel=3).cuda()
        # self.histAttnBlock3 = HistCNN(in_channel=512, out_channel=512, in_img_HW=image_HW//2, hist_img_HW=4, upsample_kernel=1).cuda()

#         self.histAttnBlock4 = HistCNN(in_channel=1024, out_channel=1024, in_img_HW=image_HW//4, hist_img_HW=2, upsample_kernel=1)

        self.myfc = nn.Linear(604416#4196352# 2048,#1837056, #9216 7680
                              ,num_classes).cuda()
        
        self.lastfc =  nn.Linear(8, num_classes).cuda()

#         self.fc_dropout = nn.Dropout(p=0.5)

        self.train_loss = 0
        self.correct = 0
        self.total = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 0.5)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        #torch.save(x,'./STN_stage0.pkl')

#         x0 = self.histAttnBlock0(x, x)
#         print("After0: x0.shape: " + str(x0.shape))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
#         x = self.maxpool(x)

#         print("before1: x.shape: " + str(x.shape))
        x1 = self.histAttnBlock1(x, x)
#         print("After1: x1.shape: " + str(x1.shape))

        x = self.layer1(x)
#         print("After1: x.shape: " + str(x.shape))

#         x2 = self.histAttnBlock2(x, x1)
#         print("After2: x2.shape: " + str(x2.shape))

        x = self.layer2(x)
#         print("After2: x.shape: " + str(x.shape))

#         x3 = self.histAttnBlock3(x)
#         print("After3: x3.shape: " + str(x3.shape))

        x = self.layer3(x)
#         print("After3: x.shape: " + str(x.shape))


#         x4 = self.histAttnBlock4(x)
#         print("After4: x4.shape: " + str(x4.shape))

        x = self.layer4(x)
#         print("After4: x.shape: " + str(x.shape))


#         x = self.maxpool(x)
        x = F.avg_pool2d(x, 4)


        x = torch.cat((x.view(x.size(0), -1),
#                             x0.view(x0.size(0), -1),
                            x1.view(x1.size(0), -1),
#                             x2.view(x2.size(0), -1),
#                             x3.view(x3.size(0), -1),
#                             x4.view(x4.size(0), -1),
                      ),
                      dim=1
            ).cuda()
        print(x.view(x.size(0), -1).shape)

#         x = x + self.myfc(x1.view(x1.size(0), -1))

        x = self.myfc(x.view(x.size(0), -1))
#         x = F.softplus(x)
    
#         x = self.lastfc(x)
#         x = F.sigmoid(x)

        return x


    def training_step(self, batch, batch_nb):
        imgs, _ = batch
        self.last_imgs = imgs

        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_loss += loss.item()
        _, predicted = y_hat.max(1)
        self.total += y.size(0)
        self.correct += predicted.eq(y).sum().item()
        acc = torch.Tensor([100.*self.correct/self.total])

        tensorboard_logs = {'train_loss': loss, 'acc':acc}
        return {'loss': loss,
                'progress_bar': {'training_loss': loss, 'acc':acc},
                'log': tensorboard_logs}


    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=0.001)

#     @pl.data_loader
    def train_dataloader(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='/media/mu/NewVolume/Programs/waterquality/pytorch-cifar', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        return trainloader

#     @pl.data_loader
    def test_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='/media/mu/NewVolume/Programs/waterquality/pytorch-cifar', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
        return testloader



#     def on_epoch_end(self):
#         z = torch.randn(8, self.hparams.latent_dim)
#         # match gpu device (or keep as cpu)
#         if self.on_gpu:
#             z = z.cuda(self.last_imgs.device.index)

#         # log sampled images
#         sample_imgs = self.forward(z)
#         grid = torchvision.utils.make_grid(sample_imgs)
#         self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetHist(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetHist(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetHist(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model




####################################

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.xavier_normal(model.weight_hh_l0)
        nn.init.xavier_normal(model.weight_ih_l0)
    elif isinstance(model, nn.Conv2d):
        nn.init.xavier_normal(model.weight.data)
#         nn.init.xavier_normal(model.bias.data)


def load_resnet_imagenet(PATH=None, model=None, modelname='resnet50', **kwargs):
    """https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/40
    """
    # load part of the pre trained model
    # save
#     torch.save(pre_model.state_dict(), PATH)
    if model is None:
        model = ResNetHist(Bottleneck, [3, 4, 6, 3], **kwargs)
    if PATH is not None:
        # load
        pretrained_dict = torch.load(PATH)
    #     model = modelName(*args, **kwargs)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        pretrained_dict = model_zoo.load_url(model_urls[modelname])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model.to(device)
    return model

def initialize_weights(model):
    for m in model.modules():
        if type(m) in [nn.Linear]:
            nn.init.kaiming_normal_(m.weight.data)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.kaiming_normal_(m.weight_hh_l0)
            nn.init.kaiming_normal_(m.weight_ih_l0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
#             nn.init.kaiming_normal_(m.bias.data)

####################################

def main(hparams):

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = ResNetHist(Bottleneck, [3, 4, 6, 3], num_classes=10)
    print (model)
#     model = load_resnet_imagenet(model=model, modelname="resnet50")
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(max_nb_epochs=10, gpus=[0], default_save_path='/home/dltdc/data/projects_weights/water_weights/checkpoint/')

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#     parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)






