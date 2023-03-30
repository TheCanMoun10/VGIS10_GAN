import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import DataLoader
from model.reconstruction_wo_memory import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import wandb
import argparse
import datetime

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--model', type=str, default="convAE", help='specify model architecture. default: convAE')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=64, help='height of input images')
parser.add_argument('--w', type=int, default=64, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=2, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--nega_loss', type=bool, default=False, help='Apply negative loss to model')
parser.add_argument('--nega_value', type=float, default=0.02, help='Value to degrade loss')
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--img_norm', type=str, default='mnad_norm', help='Define image normalization for dataloader: mnad_norm [-1, 1], dyn_norm [0, 1]')

args = parser.parse_args()

today = datetime.datetime.today()
timestring = f"{today.year}{today.month}{today.day}" + "{:02d}{:02d}".format(today.hour, today.minute) #format YYYYMMDDHHMM
if args.img_norm == "dyn_norm":
    norm = "Dynamic normalization [0, 1]"
else:
    norm = "MNAD normalization [-1, 1]"
    
wandb.init(project="VGIS10_MNAD",
           
           config={
               "learning_rate": args.lr,
               "timestamp" : timestring,
               "architecture" : args.model,
               "dataset": args.dataset_type,
               "epochs" : args.epochs,
               "batch size" : args.batch_size,
               "negative loss" : args.nega_loss,
               "mode" : args.mode,
               "Image normalization" : norm,
               },
            name=f'{args.mode}_{args.dataset_type}_batch{args.batch_size}_epochs{args.epochs}_lr{args.lr}_{args.img_norm}'
           
           )

torch.manual_seed(2020)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus
else:
    gpus = ""
    for i in range(len(args.gpus)):
        gpus = gpus + args.gpus[i] + ","
    os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose([
             transforms.ToTensor(),          
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0, img_norm=args.img_norm)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0, img_norm=args.img_norm)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting

model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)

params_encoder =  list(model.encoder.parameters()) 
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)
model.cuda()
wandb.watch(model)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, f'{args.exp_dir}_lr{args.lr}_{timestring}_{args.img_norm}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loss_func_mse = nn.MSELoss(reduction='none')

# Training
best_test = float("inf")
# m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items


for epoch in range(args.epochs):
    labels_list = []
    example_images = []
    input_images = []
    model.train()
    start = time.time()
    train_loss = AverageMeter()
    for j,(imgs) in enumerate(train_batch):
        
        imgs = Variable(imgs).cuda()
        outputs = model.forward(imgs, True)
        
        # Visualize images on wandb:
        pixels = outputs[0].detach().cpu().permute(1,2,0).numpy()
        np.rot90(pixels, k=0, axes=(1,0))
        
        optimizer.zero_grad()
        loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
        loss = loss_pixel
        # loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
        train_loss.update(loss.item(),imgs.size(0))
        if args.nega_loss == True:
            loss = -args.nega_value*loss 
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if(j%100 == 0):
            input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption="Input image")
            image = wandb.Image(pixels, caption=f"Generated anomaly_{j+1}_epoch{epoch+1}")
            example_images.append(image)
            input_images.append(input_image)
            print(
                f'Epoch [{epoch+1}/{args.epochs}]\t'
                f'Step [{j+1}/{len(train_batch)}]\t'
                f'Training Loss: {loss.item():.4f}'
                )
            wandb.log({'Generator Images': example_images, 'Input images': input_images})
        
        wandb.log({'Train_loss_avg':train_loss.avg ,'Loss': loss})
            
    if(epoch%5 == 0):
        torch.save(model, os.path.join(log_dir, f'{epoch}_negLoss{args.nega_loss}_model.pth'))
        #torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(loss_pixel.item()))
    print('----------------------------------------')
    print(f'Train loss avg: {train_loss.avg}')
    
torch.save(model, os.path.join(log_dir, f'{epoch}_negLoss{args.nega_loss}_model.pth'))
#torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
print('Training is finished')
print(log_dir)

# Save the model and the memory items


