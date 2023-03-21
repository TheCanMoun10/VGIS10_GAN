import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import helper functions and network architecture:
from helper_functions import *
from network import conv_block, encoder_block, decoder_block, UNet

# input image:
x_1 = torch.rand(1,3,572,572)
x_2 = torch.rand(1, 1024, 28, 28)


# Testing conv_block
c_block = conv_block(3,64)
print(f'Conv block Tensor shape: {c_block(x_1).shape}')

# Testing encoder_block:
e_block = encoder_block()

features = e_block(x_1)
for feats in features: print(f'Encoder block Tensor shape: {feats.shape}')

# Testing decoder_block:
d_block = decoder_block()

print(f'Shape of final feature map: {d_block(x_2, features[::-1][1:]).shape}')

# Testing UNet architecture:
unet = UNet(retain_dim=True)
unet_2 = UNet()

print(f'Size of UNet tensor, retain_dim=True: {unet(x_1).shape}')
print(f'Size of UNet tensor, retain_dim=False: {unet_2(x_1).shape}')
