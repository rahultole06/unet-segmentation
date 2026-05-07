"""
Author: Rahul Tole
Modified U-NET Architecture for Image segmentation
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    Convolutional layer
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    """
    UNET model
    """
    def __init__(self, in_channels, out_channels, features):
        super(UNET, self).__init__()

        # Stores Down & Up convolutions
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down convs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up convs
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.dropout = nn.Dropout2d(p=0.6)

        self.proj = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        residuals = []

        for down in self.downs:
            x = down(x)
            residuals.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        residuals = residuals[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            residual = residuals[i//2]

            if x.shape != residual.shape:
                residual = TF.center_crop(residual, x.shape[2:])

            concat_residual = torch.concat((residual, x), dim=1)
            x = self.ups[i+1](concat_residual)
        
        return self.proj(x)



