# import numpy as np
# from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import helper functions and network architecture:
from helper_functions import *
from network import LeNet5

# TODO: Add Avenue dataset loader to train.py and UNet architecture to network.py.

# Check if device is available:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

# Parameters for the network:
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 10
NUM_EPOCHS = 2

IMG_SIZE = 32
N_CLASSES = 10

# Parameters for prediction plot:
ROW_IMAGES = 10
NUM_ROWS = 5

# define transforms
transforms = transforms.Compose([
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = (0.1325,), std = (0.3105,))
                                ])

# Download and create dataset:
train_dataset = datasets.MNIST(
                               root='./datasets',
                               train = True,
                               transform = transforms,
                               download = True
                            )
        
test_dataset = datasets.MNIST(
                               root='./datasets',
                               train = False,
                               transform = transforms
                            )

# Define the Dataloader:
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LeNet5(N_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cost = nn.CrossEntropyLoss() # TODO: Change this gradient ascent loss (L1 or L2).

if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    model, optimizer, train_loss, model_accuracy  = training_loop(model, cost, optimizer, train_loader, test_loader, NUM_EPOCHS, device)
    
    plot_losses(train_loss)