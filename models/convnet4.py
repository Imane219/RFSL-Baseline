import torch.nn as nn
import pdb

from .models import register


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


@register('convnet4')
class ConvNet4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

#this out_dim only for cifar
@register('convnet4-512')
def ConvNet4_512():
    return ConvNet4(3,512,512,2048)

@register('convnet4-64')
def ConvNet4_64():
    return ConvNet4(3,64,64,256)