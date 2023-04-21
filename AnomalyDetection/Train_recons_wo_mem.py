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
import torchvision.utils as vutils

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
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
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
parser.add_argument('--nega_loss', action='store_true', help='Apply negative loss to model')
parser.add_argument('--nega_value', type=float, default=0.02, help='Value to degrade loss')
parser.add_argument('--loss', type=str, default="MSE", help='Define the type of loss used: cross, L1, L2 (MSE)')
parser.add_argument('--img_norm', type=str, default='mnad_norm', help='Define image normalization for dataloader: mnad_norm [-1, 1], dyn_norm [0, 1]')
parser.add_argument('--wandb', action='store_true', help='Use wandb to log and visualize network training')
parser.add_argument('--sigma_noise', type=float, default=0.5, help="Sigma value for gaussian noise added to training." )

# Augmentations to dataset:
parser.add_argument("--transforms", type=str, default='none', help='Applying transforms to dataset')
parser.add_argument('--flip', type=str, default='none', help='Apply horizontal flip to training data. [ true | none ].')
parser.add_argument('--crop', type=str, default='none', help='Apply random crop on input images. [ true | none ].')
parser.add_argument('--crop_factor', type=int, default=4, help='Factor with which to crop images. height // crop_factor.')
parser.add_argument('--p_val', type=float, default=0.5, help='Probability with which to apply the transforms')
parser.add_argument('--normalize', type=bool, default=False, help='Normalize tensors')
parser.add_argument('--test_number', type=int, default=1, help='For testing porpuses, specifies the test number.')


args = parser.parse_args()

today = datetime.datetime.today()
timestring = f"{today.year}{today.month}{today.day}" + "{:02d}{:02d}".format(today.hour, today.minute) #format YYYYMMDDHHMM
if args.img_norm == "dyn_norm":
    norm = "Dynamic normalization [0, 1]"
else:
    norm = "MNAD normalization [-1, 1]"

if args.wandb:    
    wandb.init(project="VGIS10_AnomalyGeneration",
            
            config={
                "Learning_rate": args.lr,
                "Timestamp" : timestring,
                "Architecture" : args.model,
                "Dataset": args.dataset_type,
                "Epochs" : args.epochs,
                "Batch size" : args.batch_size,
                "Negative loss" : args.nega_loss,
                "Negative loss value" : args.nega_value,
                "Image normalization" : norm,
                "Loss" : args.loss,
                # "Test " : "Perfect reconstruction"
                },
                name="{0}_{1}_batch{2}_{3}_epochs{4}_lr{5}_{6}".format(args.model, args.dataset_type, args.batch_size, args.loss, args.epochs, args.lr, args.img_norm)
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
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path + args.dataset_type +"/training/frames"
test_folder = args.dataset_path + args.dataset_type +"/testing/frames"

transforms_list = []
if args.transforms == 'true':
    transforms_list.append(transforms.ToPILImage())
    if args.flip == 'true':
        transforms_list.append(transforms.RandomHorizontalFlip(args.p_val))

    if args.crop == 'true':
        transforms_list.append(transforms.RandomCrop(size=(args.h//args.crop_factor, args.w//args.crop_factor)))
        
transforms_list += [transforms.ToTensor()]

if args.normalize == True:
    transforms_list +=[transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

print("Transforms applied to tensors: ", transforms_list)

# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose(transforms_list), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, num_pred = 0, img_norm=args.img_norm)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, num_pred = 0, img_norm=args.img_norm)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting
if args.loss == "L1":
    loss_func = nn.L1Loss(reduction='none') #L1 loss
elif args.loss == "L2" or args.loss == 'MSE':
    loss_func = nn.MSELoss(reduction='none') #L2 loss
elif args.loss == 'BCE':
    loss_func =  nn.BCELoss(reduction='none')
else:
    loss_func = nn.CrossEntropyLoss(reduction='none')

# Setup generator and discriminator:
netG = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)

params_encoder =  list(netG.encoder.parameters()) 
params_decoder = list(netG.decoder.parameters())
paramsG = params_encoder + params_decoder
optimizerG = torch.optim.Adam(paramsG, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizerG,T_max =args.epochs)
netG.cuda()
if args.wandb:
    wandb.watch(netG)

# Report the training process
hourminute = '{:02d}{:02d}'.format(today.hour, today.minute)
# log_dir = os.path.join('./exp', args.dataset_type, f"Test{args.test_number}-NegaLoss{args.nega_value}") # Experiment name
log_dir = os.path.join('./exp', args.dataset_type, f"Test{args.test_number}-PerfectReconstructionMNADNorm") # Experiment name
image_folder = os.path.join(log_dir, 'images')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Training
best_test = float("inf")
m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

# Sanity check of generator and discriminator:
noise = torch.randn(args.batch_size, args.c ,args.h, args.w).cuda()
fake = netG(noise, m_items)

print(f"Sanity check of netG: \n"
      f"noise: {noise.shape} \t"
      f"netG: {fake.shape} \t" )

G_losses = [] # Generator losses

for epoch in range(args.epochs):
    labels_list = []
    example_images = []
    input_images = []
    netG.train()
    start = time.time()
    train_loss = AverageMeter()
    
    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        g_output = netG.forward(imgs, m_items, True)
        optimizerG.zero_grad()
        # print("lalala")
        if j % 200 == 0:
            if args.img_norm == "dyn_norm":
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)))
                vutils.save_image(g_output[0], os.path.join(image_folder, '%03d_%03d_fake_sample_epoch.png' % (epoch+1, j)))
            else:
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)), normalize=True)
                vutils.save_image(g_output[0], os.path.join(image_folder, '%03d_%03d_fake_sample_epoch.png' % (epoch+1, j)), normalize=True)

        ##### TRAINING GENERATOR #####:
        loss_pixels = torch.mean(loss_func(g_output, imgs))
        loss = loss_pixels
        if args.nega_loss:
            loss = -args.nega_value*loss
        train_loss.update(loss.item(), imgs.size(0))
            
        errG = loss
        
        if args.wandb:
            pixels = g_output[0].detach().cpu().permute(1,2,0).numpy()
            np.rot90(pixels, k=0, axes=(1,0))
        
        loss.backward(retain_graph=True)
        optimizerG.step()
                
        if(j%100 == 0):
            print(
                f'[{epoch+1}/{args.epochs}]\t'
                f'[{j+1}/{len(train_batch)}]\t'
                f'Loss_G: {errG.item():.6f} \t'
                f'Train loss: {train_loss.avg}'
                )
 
        if args.wandb:
            # G_losses.append(errG.item())
            wandb.log({'Loss_G' : errG.item(), 'Train_loss_average' : train_loss.avg})
            
            if (j % 200 == 0) or ((epoch == args.epochs-1) and (j == len(train_batch)-1)):
                with torch.no_grad():
                    input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption=f"Input image {j+1}_epoch{epoch+1}")
                    image = wandb.Image(pixels, caption=f"Generator Output {j+1}_epoch{epoch+1}")
                    example_images.append(image)
                    input_images.append(input_image)
                    wandb.log({'Generator Images': example_images, 'Input images': input_images})
                
    if(epoch%5 == 0):
        torch.save(netG, os.path.join(log_dir, f'netG_{epoch}_negLoss{args.nega_loss}_{args.nega_value}_model.pth'))
        # torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
        # torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(loss_pixels.item()))
    print('----------------------------------------')
    print(f'Train loss avg: {train_loss.avg}')
    
torch.save(netG, os.path.join(log_dir, f'netG_{epoch}_negLoss{args.nega_loss}_{args.nega_value}_model.pth'))
# torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
# torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
print('Training is finished')
print(log_dir)

# Save the model and the memory items


