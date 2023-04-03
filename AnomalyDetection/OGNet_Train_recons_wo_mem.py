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
from model.utils import DataLoader, gaussian
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
parser.add_argument('--train', action='store_true', help='Initialize training')
parser.add_argument('--loss', type=str, default="cross", help='Define the type of loss used: cross, L1, L2 (MSE)')
parser.add_argument('--img_norm', type=str, default='mnad_norm', help='Define image normalization for dataloader: mnad_norm [-1, 1], dyn_norm [0, 1]')
parser.add_argument('--wandb', action='store_true', help='Use wandb to log and visualize network training')
parser.add_argument('--sigma_noise', type=float, default=0.5, help="Sigma value for gaussian noise added to training." )
parser.add_argument('--training_factor', type=float, default=0.5, help="Adversarial loss training factor")
parser.add_argument('--img_dtype', type=str, default='float32', help='Image array datatype. [np.float31] | np.uint8')

# Augmentations to dataset:
parser.add_argument("--transforms", action='store_true', help='Applying transforms to dataset')
parser.add_argument('--flip', type=str, default='none', help='Apply horizontal flip to training data. [ true | none ].')
parser.add_argument('--crop', type=str, default='none', help='Apply random crop on input images. [ true | none ].')
parser.add_argument('--crop_factor', type=int, default=4, help='Factor with which to crop images. height // crop_factor.')
parser.add_argument('--p_val', type=float, default=0.5, help='Probability with which to apply the transforms')
parser.add_argument('--normalize', action='store_true', help='Normalize tensors')


args = parser.parse_args()

today = datetime.datetime.today()
timestring = f"{today.year}{today.month}{today.day}"# + "{:02d}{:02d}".format(today.hour, today.minute) #format YYYYMMDDHHMM
if args.img_norm == "dyn_norm":
    norm = "Dynamic normalization [0, 1]"
    args.normalize = False
else:
    norm = "MNAD normalization [-1, 1]"

if args.wandb:    
    wandb.init(project="VGIS10_MNAD",
            
            config={
                "learning_rate": args.lr,
                "timestamp" : timestring,
                "architecture" : args.model,
                "dataset": args.dataset_type,
                "epochs" : args.epochs,
                "batch size" : args.batch_size,
                "negative loss" : args.nega_loss,
                "Train" : args.train,
                "Image normalization" : norm,
                "Loss" : args.loss
                },
                name=f'{args.mode}_{args.dataset_type}_batch{args.batch_size}_{args.loss}_epochs{args.epochs}_lr{args.lr}_{args.img_norm}'
            
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
    
if args.img_dtype == 'uint8':
    img_dtype = np.uint8
    torch_dtyp = torch.uint8
else: 
    img_dtype = np.float32
    torch_dtyp = torch.float32
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

train_folder = args.dataset_path+args.dataset_type+"/training/frames"
test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

transforms_list = []
if args.transforms:
    transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((args.w // 2, args.h // 2)))
    if args.flip == 'true':
        transforms_list.append(transforms.ToPILImage())
        transforms_list.append(transforms.RandomHorizontalFlip(args.p_val))

    if args.crop == 'true':
        transforms_list.append(transforms.RandomCrop(size=(args.h//args.crop_factor, args.w//args.crop_factor)))
        
transforms_list += [transforms.ToTensor()]

if args.normalize:
    transforms_list +=[transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
print(transforms_list)
# Loading dataset
train_dataset = DataLoader(train_folder, transforms.Compose(transforms_list), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0, img_norm=args.img_norm, dtype=img_dtype)

test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0, img_norm=args.img_norm, dtype=img_dtype)

train_size = len(train_dataset)
test_size = len(test_dataset)

train_batch = data.DataLoader(train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)


# Model setting
beta1 = 0.5

def loss_function(tensor, comptensor, loss_func):
    if loss_func == "L1":
        loss = F.l1_loss(tensor, comptensor, reduction='none') #L1 loss
    elif loss_func == "L2" or loss_func =='MSE':
        loss = F.mse_loss(tensor, comptensor, reduction='none') #L2 loss
    elif loss_func == 'CE':
        loss =  F.cross_entropy(tensor, comptensor, reduction='none')
    else:
        loss = F.binary_cross_entropy(tensor, comptensor, reduction='none')
    
    return loss

# Setup generator and discriminator:
netG = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
netD = OpenGAN_Discriminator(ngpu=1, nc=args.c, ndf=args.fdim)
# print(netG)
# print(netD)
# netG = Generator(args.c)
# netD = Discriminator(args.c)


params_encoder =  list(netG.encoder.parameters()) 
params_decoder = list(netG.decoder.parameters())
paramsG = params_encoder + params_decoder
paramsD = netD.parameters()
netG.cuda()
netD.cuda()
if args.wandb:
    wandb.watch(netG)
    wandb.watch(netD)

# Report the training process
hourminute = '{:02d}:{:02d}'.format(today.hour, today.minute)
log_dir = os.path.join('./exp', args.dataset_type, f'{args.exp_dir}_lr{args.lr}_{timestring}', hourminute, f'{args.img_norm}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Training
best_test = float("inf")
# m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items

# Sanity check of generator and discriminator:
noise = torch.randn(args.batch_size, args.c ,args.h, args.w).cuda()
fake = netG(noise)
predLabel = netD(fake)

print(f"Sanity check of netG and netD: \n"
      f"noise: {noise.shape} \t"
      f"netG: {fake.shape} \t"
      f"netD: {predLabel.shape} \n")


img_list = []
G_losses = [] # Generator losses
D_losses = [] # Discriminator losses
iters = 0

netG.train()
netD.train()

optimizerG = torch.optim.Adam(paramsG, lr = args.lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(paramsD, lr=args.lr/1.5, betas=(beta1, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizerG,T_max =args.epochs)

fake_label = torch.ones([args.batch_size, args.h, args.w], dtype=torch_dtyp).cuda()
real_label = torch.zeros([args.batch_size, args.h, args.w], dtype=torch_dtyp).cuda()

for epoch in range(args.epochs):
    labels_list = []
    example_images = []
    input_images = []
    
    start = time.time()
    train_loss = AverageMeter()
    for j,(imgs) in enumerate(train_batch, 0):
        # Updating discriminator:
        netD.zero_grad()
        imgs = Variable(imgs).cuda()
        #b_size = imgs.size(0)
        #label = torch.full((b_size,), real_label).cuda()
        
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        sigma = args.sigma_noise ** 2
        input_w_noise = gaussian(imgs, 1, 0, sigma)
        
        # Generator inference:
        g_output_1 = netG(imgs)
        g_output = netG(input_w_noise)
        if j == 0:
            print(f"Noisy generator output: {g_output.shape}")
            print(f'Image tensor: {imgs.shape}')
        
        vutils.save_image(imgs[0], os.path.join(log_dir, '%03d_real_sample_epoch.png' % (epoch)), normalize=True)
        vutils.save_image(g_output_1[0], os.path.join(log_dir, '%03d_fake_sample_epoch.png' % (epoch)), normalize=True)
        vutils.save_image(g_output[0], os.path.join(log_dir, '%03d_noise_sample_epoch.png' % (epoch)), normalize=True)

        ##### TRAINING DISCRIMINATOR #####
        d_fake_output = netD(g_output)
        # print('d_fake_output tensor shape: {0}'.format(d_fake_output.shape))
        # print('fake_label tensor shape: {0}'.format(fake_label.shape))
        d_real_output = netD(imgs)
        # print('d_real_output tensor shape: {0}'.format(d_real_output.shape))
        # print('real_label tensor shape: {0}'.format(real_label.shape))
        d_fake_loss = loss_function(torch.squeeze(d_fake_output), fake_label, args.loss)
        print('d_fake_loss tensor shape: {0}'.format(d_fake_loss.shape))
        # print(d_fake_loss.dtype)
        d_real_loss = loss_function(torch.squeeze(d_real_output), torch.squeeze(real_label), args.loss)
        print('d_real_loss tensor shape: {0}'.format(d_real_loss.shape))
        # print(d_real_loss.dtype)
        d_sum_loss = 0.5 * (d_fake_loss + d_real_loss)
        
        output_D_real = d_real_output.detach().view(-1)
        output_D_fake = d_fake_output.detach().view(-1)
        errD = d_sum_loss
        D_x = output_D_real.mean().item()
        D_G_z1 = d_fake_output.mean().item()
        
        d_sum_loss.backward(retain_graph=True)
        
        # Updating discriminator:
        optimizerD.step()

        ##### TRAINING GENERATOR #####:
        netG.zero_grad()
        g_recon_loss = F.mse_loss(g_output, imgs)
        g_adversarial_loss = loss_function(d_fake_output, real_label, args.loss)
        g_sum_loss = (1-args.training_factor)*g_recon_loss + args.training_factor*g_adversarial_loss
        if args.nega_loss:
            g_sum_loss = -args.nega_value*g_sum_loss
            
        errG = g_sum_loss
        output_G = d_fake_output.view(-1)
        pixels = output_G[0].detach().cpu().permute(1,2,0).numpy()
        np.rot90(pixels, k=0, axes=(1,0))
        D_G_z2 = output_G.mean().item()
        
        g_sum_loss.backward()
        optimizerG.step()
                
        if(j%100 == 0):
            print(
                f'[{epoch}/{args.epochs}]\t'
                f'[{j+1}/{len(train_batch)}]\t'
                f'Loss_G: {errG.item():.4f} \t'
                f'Loss_D: {errD.item():.4f} \t'
                f'D(x): {D_x} \t'
                f'D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}'
                )
 
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        if args.wandb:
            wandb.log({'Loss_G' : G_losses ,'Loss_D': D_losses, 'D(x)': D_x, 'D(G(z))_1': D_G_z1, 'D(G(z))_2': D_G_z2,})
            
            if (iters % 200 == 0) or ((epoch == args.epochs-1) and (i == len(train_batch)-1)):
                with torch.no_grad():
                    
                    input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption="Input image")
                    image = wandb.Image(pixels, caption=f"Generated anomaly_{j+1}_epoch{epoch+1}")
                    example_images.append(image)
                    input_images.append(input_image)
                    wandb.log({'Generator Images': example_images, 'Input images': input_images})
                
    if(epoch%5 == 0):
        torch.save(netG, os.path.join(log_dir, f'netG_{epoch}_negLoss{args.nega_loss}_model.pth'))
        torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
        #torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(g_recon_loss.item()))
    print('----------------------------------------')
    print(f'Train loss avg: {train_loss.avg}')
    
torch.save(netG, os.path.join(log_dir, f'netG_{epoch}_negLoss{args.nega_loss}_model.pth'))
torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
#torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
print('Training is finished')
print(log_dir)

# Save the model and the memory items


