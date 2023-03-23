#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import helper functions and network architecture:
from helper_functions import *
from network import UNet
from utils import Avenue_DataLoader

# input image:
# x_1 = torch.rand(1,3,572,572)
# x_2 = torch.rand(1, 96, 128, 128)

# Testing UNet architecture:
# unet = UNet(572, 572)

# print(f'Size of UNet tensor: {unet(x_1).shape}')

# print("Num params: ", sum(p.numel() for p in unet.parameters()))
# print(unet)
#%%
transformations = transforms.Compose([
                                 #transforms.Resize((32,32)),
                                 transforms.ToTensor()
                                ])


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

LEARNING_RATE = 0.0001
BATCH_SIZE = 5
IMAGE_SIZE = 64
NUM_EPOCHS = 5

train_data = Avenue_DataLoader(
    video_folder= "./datasets/avenue/training/frames/",
    transform=transformations,
    resize_height=IMAGE_SIZE,
    resize_width=IMAGE_SIZE,
    time_step=4,
    num_pred=1,
    )

test_data = Avenue_DataLoader(
    video_folder= "./datasets/avenue/testing/frames/",
    transform=transformations,
    resize_height=IMAGE_SIZE,
    resize_width=IMAGE_SIZE,
    time_step=4,
    num_pred=1,
    )

train_size = len(train_data)
test_size = len(test_data)

print(train_size)
print(test_size)

train_batch = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = UNet(IMAGE_SIZE, IMAGE_SIZE).to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
#print(model)

# Loss function and optimizer:
lossFunc = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Steps per epoch:
trainSteps = len(train_data) // BATCH_SIZE
testSteps = len(test_data) // BATCH_SIZE

print(f'Steps per epoch during training: {trainSteps}')

# Dictionary for storing training history:
H = {"train_loss": [], "test_loss": []}

for epoch in range(NUM_EPOCHS):
    model.train()
    
    total_trainloss = 0
    total_testloss = 0
    
    print("[INFO] Starting epoch [%d/%d]" % (epoch+1, NUM_EPOCHS))
    for i, data, y in enumerate(train_batch):
        data.to(device)
        y.to(device)
        
        pred = model(data)
        loss = lossFunc(pred, y)
        
        loss.backward()
        optim.step() # Gradient descent.
        
        total_trainloss += loss
        
        
        
        
        
        

# %%
