import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *

class Discriminator(nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu', norm ='batch'),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu', norm ='batch'),
        )

        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu',
                       norm=None),
            DenseBlock(base_filter * 16, 1, activation='sigmoid', norm=None)
        )
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, netVGG, feature_layer=[9,18,27,36]):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children()))
        self.feature_layer = feature_layer

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            if ii in self.feature_layer:
                x = model(x)
                results.append(x)
        return results
        
class FeatureExtractorResnet(nn.Module):
    def __init__(self, resnet):
        super(FeatureExtractorResnet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        results = []
        results.append(self.features(x))
        return results