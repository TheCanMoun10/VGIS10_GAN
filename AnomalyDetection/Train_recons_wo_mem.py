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
# import model.reconstruction_wo_memory
# import model.DataLoader
# from model.utils import DataLoader
from modelutils import DataLoader
from reconstruction_wo_memory import *
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
parser.add_argument('--nega_value', type=float, default=0.1, help='Value to degrade loss')
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
timestring = "{0}{1}{2}".format(today.year,today.month,today.day) # + "{:02d}{:02d}".format(today.hour, today.minute) #format YYYYMMDDHHMM
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
if args.loss == "KL":
    loss_func = nn.KLDivLoss(reduction='batchmean') #KL loss
elif args.loss == "L2" or args.loss == 'MSE':
    loss_func = nn.MSELoss(reduction='none') #L2 loss
elif args.loss == 'BCE':
    loss_func =  nn.BCELoss(reduction='none')
else:
    loss_func = nn.CrossEntropyLoss(reduction='none')

# Setup generator and discriminator:
normG = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim) # normal generator.
abnormG = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim) # Abnormal generator.

params_encoder =  list(normG.encoder.parameters()) 
params_decoder = list(normG.decoder.parameters())
paramsG = params_encoder + params_decoder
norm_optimizerG = torch.optim.Adam(paramsG, lr = args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(norm_optimizerG,T_max =args.epochs)

abnorm_params_encoder = list(abnormG.encoder.parameters())
abnorm_params_decoder = list(abnormG.decoder.parameters())
abnorm_paramsG = abnorm_params_encoder + abnorm_params_decoder
abnorm_optimizerG = torch.optim.Adam(abnorm_paramsG, lr=args.lr)
abnorm_scheduler = optim.lr_scheduler.CosineAnnealingLR(abnorm_optimizerG, T_max=args.epochs)

normG.cuda()
abnormG.cuda()
if args.wandb:
    wandb.watch(normG)
    wandb.watch(abnormG)

# Report the training process
hourminute = '{:02d}{:02d}'.format(today.hour, today.minute)
if args.nega_loss:
    folder_name = "Test{0}--loss{2}-NegaLoss{1}".format(args.test_number, args.nega_value, args.loss)
else:
    folder_name = "Test{0}-loss{1}-PerfectReconstruction".format(args.test_number, args.loss)
# log_dir = os.path.join('./exp', args.dataset_type, f"Test{args.test_number}-NegaLoss{args.nega_value}") # Experiment name
log_dir = os.path.join('./exp', args.dataset_type, args.img_norm, folder_name) # Experiment name
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
fake = normG(noise, m_items)
fake_abn = abnormG(noise, m_items)

print("Sanity check of normG: \t")
print("noise: {0} \t".format(noise.shape))
print("normG: {0} \t".format(fake.shape))
print("abnormG: {0} \t".format(fake_abn.shape))

G_losses = [] # Generator losses

for epoch in range(args.epochs):
    labels_list = []
    normal_images = []
    abnormal_images = []
    input_images = []
    normG.train()
    start = time.time()
    train_loss = AverageMeter()
    abn_train_loss = AverageMeter()
    
    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()
        g_output = normG.forward(imgs, m_items, True)
        g_output_abn = abnormG.forward(imgs, m_items, True)
        norm_optimizerG.zero_grad()
        abnorm_optimizerG.zero_grad()

        if j % 500 == 0:
            if args.img_norm == "dyn_norm":
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)))
                vutils.save_image(g_output[0], os.path.join(image_folder, '%03d_%03d_normal_sample_epoch.png' % (epoch+1, j)))
                vutils.save_image(g_output_abn[0], os.path.join(image_folder, '%03d_%03d_abnormal_sample_epoch.png' % (epoch+1, j)))
            else:
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)), normalize=True)
                vutils.save_image(g_output[0], os.path.join(image_folder, '%03d_%03d_normal_sample_epoch.png' % (epoch+1, j)), normalize=True)
                vutils.save_image(g_output_abn[0], os.path.join(image_folder, '%03d_%03d_abnormal_sample_epoch.png' % (epoch+1, j)), normalize=True)

        ##### TRAINING GENERATOR #####:
        # loss_pixels = torch.mean(loss_func(g_output, imgs))
        loss_pixels = F.mse_loss(g_output, imgs) # reconstruction loss
        
        if args.nega_loss:
            abn_loss_pixels = - args.nega_value*(F.mse_loss(g_output_abn, imgs) + loss_func(g_output_abn, imgs)) # MSE-loss + KL loss
        else:
            abn_loss_pixels = F.mse_loss(g_output_abn, imgs) + loss_func(g_output_abn, imgs) # MSE-loss + KL loss
        
        loss = loss_pixels
        abnormal_loss = abn_loss_pixels
        
        train_loss.update(loss.item(), imgs.size(0))
        abn_train_loss.update(abnormal_loss.item(), imgs.size(0))
            
        errG = loss
        errG_abn = abnormal_loss
        
        if args.wandb:
            pixels = g_output[0].detach().cpu().permute(1,2,0).numpy()
            np.rot90(pixels, k=0, axes=(1,0))
            
            abn_pixels = g_output_abn[0].detach().cpu().permute(1,2,0).numpy()
            np.rot90(abn_pixels, k=0, axes=(1,0))
        
        loss.backward(retain_graph=True)
        abnormal_loss.backward(retain_graph=True)
        
        norm_optimizerG.step()
        abnorm_optimizerG.step()
                
        if(j%500 == 0):
            print("Epoch: [{0}/{1}] \t Step: [{2} / {3}] \t Loss_G: {4:.06f} \t Train loss: {5:.06f} \t Loss_Gabn: {6:.06f} \t Train loss abn: {7:.06f}".format(epoch+1, args.epochs, j+1, len(train_batch), errG.item(), train_loss.avg, errG_abn.item(), abn_train_loss.avg))
 
        if args.wandb:
            # G_losses.append(errG.item())
            # wandb.log({'Loss_G' : errG.item(), 'Train_loss_average' : train_loss.avg, ''})
            
            if (j % 500 == 0) or ((epoch == args.epochs-1) and (j == len(train_batch)-1)):
                with torch.no_grad():
                    input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption="Input image {0}_epoch{1}".format(j+1, epoch+1))
                    normal_image = wandb.Image(pixels, caption="Generator Normal Output {0}_epoch{1}".format(j+1, epoch+1))
                    abnormal_image = wandb.Image(abn_pixels, caption="Generator Abnormal Output {0}_epoch{1}".format(j+1, epoch+1))
                    normal_images.append(normal_image)
                    abnormal_images.append(abnormal_image)
                    input_images.append(input_image)
                    wandb.log({'Generator Normal Images': normal_images, 'Generator Abnormal images': abnormal_images,'Input images': input_images})
                
    if(epoch%5 == 0):
        model_name = "model_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value)    
        torch.save({
            'normG_state_dict': normG.state_dict(),
            'abnormG_state_dict': abnormG.state_dict(),
            'norm_optimizerG': norm_optimizerG.state_dict(),
            'abnorm_optimizerG': abnorm_optimizerG.state_dict(),
            }, os.path.join(log_dir, model_name))
        # torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
        # torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
    scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(loss_pixels.item()))
    print('----------------------------------------')
    print('Train loss avg: {:.06f}'.format(train_loss.avg))

model_name = "model_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value)    
torch.save({
            'normG_state_dict': normG.state_dict(),
            'abnormG_state_dict': abnormG.state_dict(),
            'norm_optimizerG': norm_optimizerG.state_dict(),
            'abnorm_optimizerG': abnorm_optimizerG.state_dict(),
            }, os.path.join(log_dir, model_name))
# torch.save(netD, os.path.join(log_dir, f'netD_{epoch}_negLoss{args.nega_loss}_model.pth'))
# torch.save(m_items, os.path.join(log_dir, f'{epoch}_m_items.pt')) 
print('Training is finished')
print(log_dir)

# Save the model and the memory items


