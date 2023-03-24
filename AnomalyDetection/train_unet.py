import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
#from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import helper functions and network architecture:
from helper_functions import *
from network import UNet
from utils import Avenue_DataLoader

parser = argparse.ArgumentParser(description="VGIS10GAN")
#Directories
parser.add_argument("--dataset_path", type=str, default="./datasets", help="directory to data")
parser.add_argument("--dataset_type", type=str, default="avenue", help="Type of dataset, currently only avenue")
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')

#Training variables
parser.add_argument("--loss", type=str, default='L1', help="defines the loss for the model, [L1, L2]")
parser.add_argument('--batch_size', type=int, default=25, help='size of batch for training')
parser.add_argument('--epochs', type=int, default=25, help='number of epochs for training')
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate of optimizer")
parser.add_argument("--grad_ascent", type=bool, default=False, help="apply gradient ascent to generate fake anomalies")
parser.add_argument('--img_height', type=int, default=256, help='height of input images')
parser.add_argument('--img_width', type=int, default=256, help='width of input images')
parser.add_argument('--img_channels', type=int, default=3, help='channel of input images')
parser.add_argument('--time_step_length', type=int, default=5, help="length of frame sequences")

args = parser.parse_args()

transformations = transforms.Compose([
                                 #transforms.Resize((32,32)),
                                 transforms.ToTensor()
                                ])


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE)

train_folder = args.dataset_path+"/"+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+"/"+args.dataset_type+"/testing/frames"

LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
IMAGE_HEIGTH = args.img_height
IMAGE_WIDTH = args.img_width
NUM_EPOCHS = args.epochs
TIME_STEP = args.time_step_length-1

train_data = Avenue_DataLoader(
    video_folder= train_folder,
    transform=transformations,
    resize_height=IMAGE_HEIGTH,
    resize_width=IMAGE_WIDTH,
    time_step=TIME_STEP,
    num_pred=1,
    )

test_data = Avenue_DataLoader(
    video_folder= test_folder,
    transform=transformations,
    resize_height=IMAGE_HEIGTH,
    resize_width=IMAGE_WIDTH,
    time_step=TIME_STEP,
    num_pred=1,
    )

train_size = len(train_data)
test_size = len(test_data)

print(train_size)
print(test_size)

train_batch = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_batch = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = UNet(IMAGE_HEIGTH, IMAGE_WIDTH).to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)

# Dictionary for storing training history:
#H = {"train_loss": [], "test_loss": []}

log_dir = os.path.join("./experiments", args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), "w")
sys.stdout = f

# Loss function and optimizer:
if args.loss == "L2":
    lossFunc = nn.MSELoss()
elif args.loss == "L1":
    lossFunc = nn.L1Loss
    
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

# Steps per epoch:
trainSteps = len(train_data) // BATCH_SIZE
testSteps = len(test_data) // BATCH_SIZE

print(f'Steps per epoch during training: {trainSteps}')

for epoch in range(NUM_EPOCHS):
    anomaly_list = []
    model.train()
    
    total_trainloss = 0
    
    print("[INFO] Starting epoch [%d/%d]" % (epoch+1, NUM_EPOCHS))
    for i, (data) in enumerate(train_batch):
        
        imgs = torch.autograd.Variable(data).to(device)
        
        outputs = model.forward(imgs)
        loss = torch.mean(lossFunc(outputs, imgs))
        anomaly_list.append((outputs, loss))
        
        loss.backward()
        if args.grad_ascent == True:
            -optimizer.step() # Gradient ascent.
        else:
            optimizer.step() # Gradient descent.
            
        total_trainloss += loss
    
    scheduler.step()
    print('Training Loss: {:.05f}'.format(loss.item()))
    
print("Training finished")
torch.save(model, os.path.join(log_dir, 'model.pth'))
torch.save(outputs, os.path.join(log_dir, 'features.txt'))

sys.stdout = orig_stdout
f.close()

# %%
