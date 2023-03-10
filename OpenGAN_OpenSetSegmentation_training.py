from __future__ import print_function, division
import os, random, time, copy, scipy, pickle, sys, math, json, pickle

import argparse, pprint, shutil, logging, time, timeit
from pathlib import Path

from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
from scipy import ndimage, signal
import matplotlib.pyplot as plt
# import PIL.Image
from PIL import Image
from io import BytesIO
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from collections import namedtuple

from config_HRNet import models
from config_HRNet import seg_hrnet
from config_HRNet import config
from config_HRNet import update_config
from config_HRNet.modelsummary  import *
from config_HRNet.utils import *


from utils.dataset_tinyimagenet import *
from utils.dataset_cityscapes import *
from utils.eval_funcs import *


import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)

# %load_ext autoreload
# %autoreload 2

# CONFIG PARAMETERS:
# set the random seed
torch.manual_seed(0)


################## set attributes for this project/experiment ##################
# config result folder
exp_dir = './exp' # experiment directory, used for reading the init model

num_open_training_images = 1000
weight_adversarialLoss = 0.2
project_name = 'demo_step030_OpenGAN_num{}_w{:.2f}'.format(num_open_training_images, weight_adversarialLoss)




device ='cpu'
if torch.cuda.is_available(): 
    device='cuda:3'
        


ganBatchSize = 640
batch_size = 1
newsize = (-1,-1)

total_epoch_num = 50 # total number of epoch in training
insertConv = False    
embDimension = 64
#isPretrained = False
#encoder_num_layers = 18


# Number of channels in the training images. For color images this is 3
nc = 720
# Size of z latent vector (i.e. size of generator input)
nz = 64
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



save_dir = os.path.join(exp_dir, project_name)
if not os.path.exists(exp_dir): os.makedirs(exp_dir)

lr = 0.0001 # base learning rate

num_epochs = total_epoch_num
torch.cuda.device_count()
torch.cuda.empty_cache()

save_dir = os.path.join(exp_dir, project_name)
print(save_dir)    
if not os.path.exists(save_dir): os.makedirs(save_dir)

log_filename = os.path.join(save_dir, 'train.log')

# MODEL ARCHITECTURE
class CityscapesOpenPixelFeat4(Dataset):
    def __init__(self, set_name='train',
                 numImgs=500,
                 path_to_data='/scratch/dataset/Cityscapes_feat4'):        
        
        self.imgList = []
        self.current_set_len = numImgs # 2975
        if set_name=='test':  
            set_name = 'val'
            self.current_set_len = 500
        
        self.set_name = set_name
        self.path_to_data = path_to_data
        for i in range(self.current_set_len):
            self.imgList += ['{}_openpixel.pkl'.format(i)]        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):        
        filename = path.join(self.path_to_data, self.set_name, self.imgList[idx])
        with open(filename, "rb") as fn:
            openPixFeat = pickle.load(fn)
        openPixFeat = openPixFeat['feat4open_percls']
        openPixFeat = torch.cat(openPixFeat, 0).detach()
        #print(openPixFeat.shape)
        return openPixFeat
    
parser = argparse.ArgumentParser(description='Train segmentation network') 
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    default='./config_HRNet/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
                    type=str)
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)


args = parser.parse_args(r'--cfg  ./config_HRNet/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml ')
args.opts = []
update_config(config, args)    

model = eval(config.MODEL.NAME + '.get_seg_model_myModel')(config)
model_dict = model.state_dict()


model_state_file = '../openset/models/hrnet_w48_cityscapes_cls19_1024x2048_ohem_trainset.pth'
pretrained_dict = torch.load(model_state_file, map_location=lambda storage, loc: storage)


suppl_dict = {}
suppl_dict['last_1_conv.weight'] = pretrained_dict['model.last_layer.0.weight'].clone()
suppl_dict['last_1_conv.bias'] = pretrained_dict['model.last_layer.0.bias'].clone()

suppl_dict['last_2_BN.running_mean'] = pretrained_dict['model.last_layer.1.running_mean'].clone()
suppl_dict['last_2_BN.running_var'] = pretrained_dict['model.last_layer.1.running_var'].clone()
# suppl_dict['last_2_BN.num_batches_tracked'] = pretrained_dict['model.last_layer.1.num_batches_tracked']
suppl_dict['last_2_BN.weight'] = pretrained_dict['model.last_layer.1.weight'].clone()
suppl_dict['last_2_BN.bias'] = pretrained_dict['model.last_layer.1.bias'].clone()

suppl_dict['last_4_conv.weight'] = pretrained_dict['model.last_layer.3.weight'].clone()
suppl_dict['last_4_conv.bias'] = pretrained_dict['model.last_layer.3.bias'].clone()


pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                   if k[6:] in model_dict.keys()}


model_dict.update(pretrained_dict)
model_dict.update(suppl_dict)
model.load_state_dict(model_dict)


model.eval();
model.to(device);

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)     
        

class Generator(nn.Module):
    def __init__(self, ngpu=1, nz=100, ngf=64, nc=512):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d( self.nz, self.ngf * 8, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (self.ngf*8) x 4 x 4
            nn.Conv2d(self.ngf * 8, self.ngf * 4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (self.ngf*4) x 8 x 8
            nn.Conv2d( self.ngf * 4, self.ngf * 2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (self.ngf*2) x 16 x 16
            nn.Conv2d( self.ngf * 2, self.ngf*4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            # state size. (self.ngf) x 32 x 32
            nn.Conv2d( self.ngf*4, self.nc, 1, 1, 0, bias=True),
            #nn.Tanh()
            # state size. (self.nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    
class Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=512, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=True),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
netG = Generator(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
netD = Discriminator(ngpu=ngpu, nc=nc, ndf=ndf).to(device)


# Handle multi-gpu if desired
if ('cuda' in device) and (ngpu > 1): 
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)


if ('cuda' in device) and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

print(device)

noise = torch.randn(batch_size*5, nz, 1, 1, device=device)
# Generate fake image batch with G
fake = netG(noise)
predLabel = netD(fake)

print(noise.shape, fake.shape, predLabel.shape)

# SETUP DATASET
# torchvision.transforms.Normalize(mean, std, inplace=False)
imgTransformList = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

targetTransformList = transforms.Compose([
    transforms.ToTensor(),    
])

cls_datasets = {set_name: Cityscapes(root='/scratch/dataset/Cityscapes',
                                     newsize=newsize,
                                     split=set_name,
                                     mode='fine',
                                     target_type='semantic',
                                     transform=imgTransformList,
                                     target_transform=targetTransformList,
                                     transforms=None)
                for set_name in ['train', 'val']} # 'train', 

dataloaders = {set_name: DataLoader(cls_datasets[set_name],
                                    batch_size=batch_size,
                                    shuffle=set_name=='train', 
                                    num_workers=4) # num_work can be set to batch_size
               for set_name in ['train', 'val']} # 'train',


print(len(cls_datasets['train']), len(cls_datasets['val']))
classDictionary = cls_datasets['val'].classes

id2trainID = {}
id2color = {}
trainID2color = {}
id2name = {}
opensetIDlist = []
for i in range(len(classDictionary)):
    id2trainID[i] = classDictionary[i][2]
    id2color[i] = classDictionary[i][-1]
    trainID2color[classDictionary[i][2]] = classDictionary[i][-1]
    id2name[i] = classDictionary[i][0]
    if classDictionary[i][-2]:
        opensetIDlist += [i]

id2trainID_list = []
for i in range(len(id2trainID)):
    id2trainID_list.append(id2trainID[i])
id2trainID_np = np.asarray(id2trainID_list)        
        
for elm in opensetIDlist:
    print(elm, id2name[elm])
print('total# {}'.format(len(opensetIDlist)))

data_sampler = iter(dataloaders['train'])
data = next(data_sampler)
imageList, labelList = data[0], data[1]

imageList = imageList.to(device)
labelList = labelList.to(device)

imageList.shape, labelList.shape

# TRAINING SETUP
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish open and close labels
close_label = 1
open_label = 0

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr/1.5, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# SINGLE IMAGE TEST:
labelList = labelList.unsqueeze(1)
labelList = F.interpolate(labelList, scale_factor=0.25, mode='nearest')
labelList = labelList.squeeze()
H, W = labelList.squeeze().shape
trainlabelList = id2trainID_np[labelList.cpu().numpy().reshape(-1,).astype(np.int32)]
trainlabelList = trainlabelList.reshape((1,H,W))
trainlabelList = torch.from_numpy(trainlabelList)



upsampleFunc = nn.UpsamplingBilinear2d(scale_factor=4)
with torch.no_grad():
    imageList = imageList.to(device)
    logitsTensor = model(imageList).detach().cpu()
    #logitsTensor = upsampleFunc(logitsTensor)
    softmaxTensor = F.softmax(logitsTensor, dim=1)
    
    feat1Tensor = model.feat1.detach()
    feat2Tensor = model.feat2.detach()
    feat3Tensor = model.feat3.detach()
    feat4Tensor = model.feat4.detach()
    feat5Tensor = model.feat5.detach()
    
    torch.cuda.empty_cache()
    
feat4Tensor.shape, trainlabelList.shape, trainlabelList.shape[1]*trainlabelList.shape[2]

validList = trainlabelList.reshape(-1,1)
validList = ((validList>=0) & (validList<=18)).nonzero()
validList = validList[:,0]
validList = validList[torch.randperm(validList.size()[0])]
validList = validList[:ganBatchSize]

label = torch.full((ganBatchSize,), close_label, device=device)

real_cpu = feat4Tensor.squeeze()
real_cpu = real_cpu.reshape(real_cpu.shape[0], -1).permute(1,0)
real_cpu = real_cpu[validList,:].unsqueeze(-1).unsqueeze(-1).to(device)

output = netD(real_cpu).view(-1)
# Calculate loss on all-real batch
errD_real = criterion(output, label)

noise = torch.randn(ganBatchSize, nz, 1, 1, device=device)
# Generate fake image batch with G
fake = netG(noise)
label.fill_(fake_label)
# Classify all fake batch with D
output = netD(fake.detach()).view(-1)
# Calculate D's loss on the all-fake batch
errD_fake = criterion(output, label)

noise.shape, label.shape, fake.shape

# TRAINING GAN:

openPix_datasets = CityscapesOpenPixelFeat4(set_name='train', numImgs=num_open_training_images)
openPix_dataloader = DataLoader(openPix_datasets, batch_size=1, shuffle=True, num_workers=4)               

openPix_sampler = iter(openPix_dataloader)

openPixFeat = next(openPix_sampler)
openPixFeat = openPixFeat.squeeze(0)

openPixIdxList = torch.randperm(openPixFeat.size()[0])
openPixIdxList = openPixIdxList[:ganBatchSize]
openPixFeat = openPixFeat[openPixIdxList].to(device)

print(openPixFeat.shape)

# Training Loop

# Lists to keep track of progress
lossList = []
G_losses = []
D_losses = []

fake_BatchSize = int(ganBatchSize/2)
open_BatchSize = ganBatchSize



tmp_weights = torch.full((ganBatchSize+open_BatchSize+fake_BatchSize,), 1, device=device)
tmp_weights[-fake_BatchSize:] *= weight_adversarialLoss
criterionD = nn.BCELoss(weight=tmp_weights)



print("Starting Training Loop...")
# For each epoch
openPixImgCount = 0
openPix_sampler = iter(openPix_dataloader)
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, sample in enumerate(dataloaders['train'], 0):
        imageList, labelList = sample
        imageList = imageList.to(device)
        labelList = labelList.to(device)

        labelList = labelList.unsqueeze(1)
        labelList = F.interpolate(labelList, scale_factor=0.25, mode='nearest')
        labelList = labelList.squeeze()
        H, W = labelList.squeeze().shape
        trainlabelList = id2trainID_np[labelList.cpu().numpy().reshape(-1,).astype(np.int32)]
        trainlabelList = trainlabelList.reshape((1,H,W))
        trainlabelList = torch.from_numpy(trainlabelList)
        
        
        #upsampleFunc = nn.UpsamplingBilinear2d(scale_factor=4)
        with torch.no_grad():
            imageList = imageList.to(device)
            logitsTensor = model(imageList).detach().cpu()
            featTensor = model.feat4.detach()
        
        validList = trainlabelList.reshape(-1,1)
        validList = ((validList>=0) & (validList<=18)).nonzero()
        validList = validList[:,0]
        tmp = torch.randperm(validList.size()[0])        
        validList = validList[tmp[:ganBatchSize]]
                

        
        label_closeset = torch.full((ganBatchSize,), close_label, device=device)
        feat_closeset = featTensor.squeeze()
        feat_closeset = feat_closeset.reshape(feat_closeset.shape[0], -1).permute(1,0)
        feat_closeset = feat_closeset[validList,:].unsqueeze(-1).unsqueeze(-1)        
        label_open = torch.full((open_BatchSize,), open_label, device=device)
        
        openPixImgCount += 1
        feat_openset = next(openPix_sampler)
        feat_openset = feat_openset.squeeze(0)
        openPixIdxList = torch.randperm(feat_openset.size()[0])
        openPixIdxList = openPixIdxList[:open_BatchSize]
        feat_openset = feat_openset[openPixIdxList].to(device)

        if openPixImgCount==num_open_training_images:
            openPixImgCount = 0
            openPix_sampler = iter(openPix_dataloader)
        
        
        
        # generate fake images        
        noise = torch.randn(fake_BatchSize, nz, 1, 1, device=device)
        # Generate fake image batch with G
        label_fake = torch.full((fake_BatchSize,), fake_label, device=device)
        feat_fakeset = netG(noise)    
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # using close&open&fake data to update D
        netD.zero_grad()
        X = torch.cat((feat_closeset, feat_openset.to(device), feat_fakeset.detach()),0)
        label_total = torch.cat((label_closeset, label_open, label_fake),0)
                
        output = netD(X).view(-1)
        lossD = criterionD(output, label_total)
        lossD.backward()
        optimizerD.step()
        errD = lossD.mean().item()                        
            
            
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label_fakeclose = torch.full((fake_BatchSize,), close_label, device=device)        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(feat_fakeset).view(-1)
        # Calculate G's loss based on this output
        lossG = criterion(output, label_fakeclose)
        # Calculate gradients for G
        lossG.backward()
        errG = lossG.mean().item()
        # Update G
        optimizerG.step()
            
            
        # Save Losses for plotting later
        G_losses.append(errG)
        D_losses.append(errD)
        
        
        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\t\tlossG: %.4f, lossD: %.4f'
                  % (epoch, num_epochs, i, len(dataloaders['train']), 
                     errG, errD))
            
            
    cur_model_wts = copy.deepcopy(netD.state_dict())
    path_to_save_paramOnly = os.path.join(save_dir, 'epoch-{}.classifier'.format(epoch+1))
    torch.save(cur_model_wts, path_to_save_paramOnly)
    cur_model_wts = copy.deepcopy(netG.state_dict())
    path_to_save_paramOnly = os.path.join(save_dir, 'epoch-{}.GNet'.format(epoch+1))
    torch.save(cur_model_wts, path_to_save_paramOnly)
    
    
# VALIDATE RESULT
plt.figure(figsize=(10,5))
plt.title("binary cross-entropy loss in training")
plt.plot(Dopen_losses, label="Dopen_losses")
plt.plot(Dclose_losses, label="Dclose_losses")
plt.plot(Dfake_losses, label="Dfake_losses")
plt.plot(G_losses, label="G_losses")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plt.savefig('learningCurves_{}.png'.format(modelFlag), bbox_inches='tight',transparent=True)
# plt.show()