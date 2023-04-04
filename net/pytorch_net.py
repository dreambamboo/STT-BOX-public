# -*- coding=utf-8 -*-
"""
# pytorch network   
"""
import numpy as np 
import torch.nn as nn
from torchvision import models

class PytorchNet(nn.Module):
    def __init__(self, net_select, pretrained, output_channels):
        super().__init__()
        if net_select in ['resnet50']:
            self.mynet = ResNet(net_select, pretrained, output_channels)
        else:
            raise Exception ("Error: please choose another -net_select-.")
        print('--> net_select:{},pretrained:{}'.format(net_select, pretrained))
    def forward(self, x):
        x = self.mynet(x)
        return x



class ResNet(nn.Module):
    def __init__(self, net_select, pretrained, output_channels):
        super().__init__()
        if net_select == 'resnet50':
            self.net = models.resnet50(pretrained=pretrained)
        else:
            raise Exception ("Error: please choose another -net_select-.")
        fc_in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(fc_in_features, output_channels, bias=True)
    def forward(self, x):
        return self.net(x)

