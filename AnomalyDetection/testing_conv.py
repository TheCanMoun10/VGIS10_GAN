#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import helper functions and network architecture:
from helper_functions import *
from network import Block, Encoder, Decoder, UNet
from utils import Avenue_DataLoader

# input image:
x_1 = torch.rand(1,3,572,572)
x_2 = torch.rand(1, 96, 128, 128)

# Testing UNet architecture:
unet = UNet(572, 572)

print(f'Size of UNet tensor: {unet(x_1).shape}')

print("Num params: ", sum(p.numel() for p in unet.parameters()))
print(unet)

#%%
transformations = transforms.Compose([
                                 #transforms.Resize((32,32)),
                                 transforms.ToTensor()
                                ])


LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 25

train_data = Avenue_DataLoader(
    video_folder= "./datasets/avenue/training/frames/",
    transform=transformations,
    resize_height=256,
    resize_width=256,
    time_step=4,
    num_pred=1,
    )

test_data = Avenue_DataLoader(
    video_folder= "./datasets/avenue/testing/frames/",
    transform=transformations,
    resize_height=256,
    resize_width=256,
    time_step=4,
    num_pred=1,
    )

train_size = len(train_data)
test_size = len(test_data)

print(train_size)
print(test_size)
#train_batch = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# %%
