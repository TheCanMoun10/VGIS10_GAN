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
from losses import *

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
parser.add_argument('--model', type=str, default="DCGan", help='specify model architecture. default: DCGan, convAE')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--loss_compact', type=float, default=0.01, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.01, help='weight of the feature separateness loss')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
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
parser.add_argument('--nega_value', type=float, default=0.0, help='Value to degrade loss')
parser.add_argument('--loss', type=str, default="BCE", help='Define the type of loss used: cross, L1, L2 (MSE)')
parser.add_argument('--img_norm', type=str, default='mnad_norm', help='Define image normalization for dataloader: mnad_norm [-1, 1], dyn_norm [0, 1]')
parser.add_argument('--wandb', action='store_true', help='Use wandb to log and visualize network training')
parser.add_argument('--latent_dim', type=int, default=100, help="Dimensionality of latent space." )
parser.add_argument('--lambda_int', type=float, default=1.0, help="Weight for the intensity loss." )
parser.add_argument('--l_num', type=int, default=2, help="Exponent for the intensity loss." )
parser.add_argument('--lambda_grad', type=float, default=1.0, help="Weight for the gradient loss." )
parser.add_argument('--grad_alpha', type=float, default=1.0, help="Weight for the gradient loss." )
parser.add_argument('--lambda_adv', type=float, default=0.05, help="Weight for the adversarial loss." )

# Augmentations to dataset:
parser.add_argument("--transforms", type=str, default='none', help='Applying transforms to dataset')
parser.add_argument('--flip', type=str, default='none', help='Apply horizontal flip to training data. [ true | none ].')
parser.add_argument('--crop', type=str, default='none', help='Apply random crop on input images. [ true | none ].')
parser.add_argument('--crop_factor', type=int, default=4, help='Factor with which to crop images. height // crop_factor.')
parser.add_argument('--p_val', type=float, default=0.5, help='Probability with which to apply the transforms')
parser.add_argument('--normalize', type=bool, default=False, help='Normalize tensors')
parser.add_argument('--test_number', type=int, default=1, help='For testing porpuses, specifies the test number.')
parser.add_argument('--type', type=str, default='gan_network')


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
                "Type" : args.type,
                "Model" : args.model,
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

# Report the training process
hourminute = '{:02d}{:02d}'.format(today.hour, today.minute)
if args.nega_loss:
    folder_name = "{3}_Test{0}--loss{2}-NegaLoss{1}".format(args.test_number, args.nega_value, args.loss, args.type)
else:
    folder_name = "{2}_Test{0}-loss{1}-PerfectReconstruction".format(args.test_number, args.loss, args.type)
# log_dir = os.path.join('./exp', args.dataset_type, f"Test{args.test_number}-NegaLoss{args.nega_value}") # Experiment name
log_dir = os.path.join('./exp', args.dataset_type, args.img_norm, folder_name) # Experiment name
image_folder = os.path.join(log_dir, 'images')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

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
    loss_func = nn.KLDivLoss(reduction='batchmean').cuda() #KL loss
elif args.loss == "L2" or args.loss == 'MSE':
    loss_func = nn.MSELoss(reduction='none').cuda() #L2 loss
elif args.loss == 'BCE':
    loss_func =  nn.BCELoss(reduction='none').cuda()
else:
    loss_func = nn.CrossEntropyLoss(reduction='none').cuda()

lam_int = args.lambda_int*2
lam_grad = args.lambda_grad*2
lam_adv = args.lambda_adv
discriminator_loss = nn.BCELoss().cuda()
reconstruction_loss = nn.MSELoss().cuda()
intensity_loss = Intensity_Loss(args.l_num).cuda()
grad_loss = Gradient_Loss(args.grad_alpha, args.c).cuda()
adversarial_loss = Adversarial_Loss().cuda()


# Setup generator and discriminator:
if args.model == "convAE":
    normG = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim) # normal generator.
    params_encoder =  list(normG.encoder.parameters()) 
    params_decoder = list(normG.decoder.parameters())
    paramsG = params_encoder + params_decoder
    norm_optimizerG = torch.optim.Adam(paramsG, lr = args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(norm_optimizerG,T_max =args.epochs)
    normG.cuda()
    
    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda() # Initialize the memory items
    noise = torch.randn(args.batch_size, args.c ,args.h, args.w).cuda()
    fake = normG(noise, m_items)
    
    print("Sanity check of normG: \t")
    print("noise: {0} \t".format(noise.shape))
    print("normG: {0} \t".format(fake.shape))

netG = DCGen(num_channels=args.c, img_size=args.h, latent_dim=args.latent_dim)
netG_optimizer = torch.optim.Adam(netG.parameters(), lr=args.lr)

netD = DCDis(num_channels=args.c, img_size=args.h)
netD_optimizer = torch.optim.Adam(netD.parameters(), lr=args.lr)


netG.apply(weights_init_normal)
netD.apply(weights_init_normal)

netG.cuda()
netD.cuda()
if args.wandb:
    # wandb.watch(normG)
    wandb.watch(netG)
    wandb.watch(netD)
    
# Training
best_test = float("inf")
G_losses = [] # Generator losses

for epoch in range(args.epochs):
    labels_list = []
    normal_images = []
    abnormal_images = []
    input_images = []
    start = time.time()
    train_loss = AverageMeter()
    
    for j,(imgs) in enumerate(train_batch):
        imgs = Variable(imgs).cuda()

        # Noisy sample for generator
        if args.model == "DCGan":
            z = Variable(torch.FloatTensor(np.random.normal(0,1, (imgs.shape[0], args.latent_dim)))).cuda()
            # Generate image batch:
            g_output_abn = netG(z)
        else:    
            g_output_abn = netG(imgs)
        
        if j % 476 == 0:
            if args.img_norm == "dyn_norm":
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)))
                vutils.save_image(g_output_abn[0], os.path.join(image_folder, '%03d_%03d_recon_sample_epoch.png' % (epoch+1, j)))
            else:
                vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_%03d_real_sample_epoch.png' % (epoch+1, j)), normalize=True)
                vutils.save_image(g_output_abn[0], os.path.join(image_folder, '%03d_%03d_recon_sample_epoch.png' % (epoch+1, j)), normalize=True)

        ##### TRAINING GENERATOR #####:
        g_adv_loss = adversarial_loss(netD(g_output_abn))
        g_recon_loss = torch.mean(reconstruction_loss(g_output_abn, imgs))
        g_int_loss = intensity_loss(g_output_abn, imgs)
        g_grad_loss = grad_loss(g_output_abn, imgs)
        
        loss_g = g_recon_loss + lam_adv*g_adv_loss + lam_int*g_int_loss + lam_grad*g_grad_loss
        
        train_loss.update(loss_g.item(), imgs.size(0))
        
        if args.wandb:
            abn_pixels = g_output_abn[0].detach().cpu().permute(1,2,0).numpy()
            np.rot90(abn_pixels, k=0, axes=(1,0))
        
        netG_optimizer.zero_grad()
        
        loss_g.backward()

        netG_optimizer.step()
        
        ###### TRAINING DISCRIMINATOR #####
        netD_optimizer.zero_grad()
        
        d_loss = discriminator_loss(netD(g_output_abn.detach()), netD(imgs))
        
        d_loss.backward()
        netD_optimizer.step()
         
        if(j% 300 == 0):
            print("Epoch: [{0}/{1}] \t Step: [{2} / {3}] \t G Loss: {4:.06f} \t Train Loss avg: {5:.06f} \t D Loss: {6:.06f}".format(epoch+1, args.epochs, j+1, len(train_batch), loss_g.item(), train_loss.avg, d_loss.item()))
 
        if args.wandb:
            wandb.log({'G Loss' : loss_g.item(), 'D Loss' : d_loss.item(), 'Train Loss avg' : train_loss.avg})
            
            if (j % 300 == 0) or ((epoch == args.epochs-1) and (j == len(train_batch)-1)):
                with torch.no_grad():
                    input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption="Input image {0}_epoch{1}".format(j+1, epoch+1))
                    abnormal_image = wandb.Image(abn_pixels, caption="Generator Reconstructed Output {0}_epoch{1}".format(j+1, epoch+1))
                    abnormal_images.append(abnormal_image)
                    input_images.append(input_image)
                    wandb.log({'Reconstructed images': abnormal_images,'Input images': input_images})
                
    if(epoch%5 == 0):
        model_name = "{3}_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value, args.model)
        # model_name_dis = "netD_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value)
        # torch.save(netG.state_dict(), os.path.join(log_dir, model_name_gen))
        # torch.save(netD.state_dict(), os.path.join(log_dir, model_name_dis))
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'netG_optimizer': netG_optimizer.state_dict(),
            'netD_optimizer': netD_optimizer.state_dict(),
            }, os.path.join(log_dir, model_name))
    # scheduler.step()
    
    print('----------------------------------------')
    print('Epoch:', epoch+1)
    print('Loss: Reconstruction {:.6f}'.format(loss_g.item()))
    print('Loss: Discriminator {:.6f}'.format(d_loss.item()))
    print('----------------------------------------')
    print('Train loss avg: {:.06f}'.format(train_loss.avg))

model_name = "{3}_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value, args.model)
# model_name_dis = "netD_{0}_NegLoss{1}_{2}_model.pth".format(epoch, args.nega_loss, args.nega_value)
# torch.save(netG.state_dict(), os.path.join(log_dir, model_name_gen))
# torch.save(netD.state_dict(), os.path.join(log_dir, model_name_dis))
torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'netG_optimizer': netG_optimizer.state_dict(),
            'netD_optimizer': netD_optimizer.state_dict(),
            }, os.path.join(log_dir, model_name))
print('Training is finished')
print(log_dir)

# Save the model and the memory items


