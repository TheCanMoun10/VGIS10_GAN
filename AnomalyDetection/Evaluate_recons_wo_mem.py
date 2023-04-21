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
import glob
import argparse
import torchvision.utils as vutils
import datetime
import wandb
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--model', type=str, default="convAE", help='specify model architecture. default: convAE')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=2, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
parser.add_argument('--alpha', type=float, default=1, help='weight for the anomality score')
parser.add_argument('--th', type=float, default=0.015, help='threshold for test updating')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=2, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='avenue', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='directory of data')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--wandb', action='store_true', help='Use wandb to log evaluation images')
parser.add_argument('--img_norm', type=str, default='mnad_norm', help='Define image normalization for dataloader: mnad_norm [-1, 1], dyn_norm [0, 1]')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--test_number', type=int, default=1, help='For testing porpuses, specifies the test number.')
parser.add_argument('--nega_value', type=float, default=0.1, help='Value of the negative loss applied during training.')

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
                "Image normalization": args.img_norm,
                "Timestamp" : timestring,
                "Architecture" : args.model,
                "Dataset": args.dataset_type,
                "Batch size" : args.test_batch_size,
                "Image normalization" : norm,
                },
                name="eval_{0}_{1}_batch{2}_timestep{3}".format(args.model, args.dataset_type, args.test_batch_size, args.t_length)
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

test_folder = args.dataset_path + args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1, num_pred = 0,  img_norm=args.img_norm)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
m_items = torch.load(args.m_items_dir)

labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
if args.dataset_type == 'shanghai':
    labels = np.expand_dims(labels, 0)
print("labels.npy shape: ", labels.shape)

videos = OrderedDict()
videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
for video in videos_list:
    video_name = video.split('/')[-1]
    videos[video_name] = {}
    videos[video_name]['path'] = video
    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
    videos[video_name]['frame'].sort()
    videos[video_name]['length'] = len(videos[video_name]['frame'])

labels_list = []
label_length = 0
psnr_list = {}
feature_distance_list = {}

hourminute = '{:02d}{:02d}'.format(today.hour, today.minute)
log_dir = os.path.join('./evals', args.dataset_type, args.img_norm)
# image_folder = os.path.join(log_dir, f'Test{args.test_number}-NegaLoss{args.nega_value}')
image_folder = os.path.join(log_dir, f'Test{args.test_number}-PerfectReconstructionMNADnorm')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length][:-1])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
m_items_test = m_items.clone()

model.eval()

for k,(imgs) in enumerate(test_batch):

    if k == label_length-0*(video_num+1):
        print(k,label_length)
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    imgs = Variable(imgs).cuda()

    outputs = model.forward(imgs, m_items_test, False)
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
    if args.wandb:
        wandb.log({'Image MSE' : mse_imgs})
    
    if k % 1500 == 0:
        if args.img_norm == "dyn_norm":
            vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_real_sample.png' % (k)))
            vutils.save_image(outputs[0], os.path.join(image_folder, '%03d_reconstructed_sample.png' % (k)))
        else:
            vutils.save_image(imgs[0], os.path.join(image_folder, '%03d_real_sample.png' % (k)), normalize=True)
            vutils.save_image(outputs[0], os.path.join(image_folder, '%03d_reconstructed_sample.png' % (k)), normalize=True)
        # vutils.save_image(mse_imgs[0], os.path.join(image_folder, '%03d_mse_sample.png' % (k)), normalize=True)
        
        if args.wandb:
            example_images = []
            input_images = []
            pixels = outputs[0].detach().cpu().permute(1,2,0).numpy()
            # pixels_mse = outputs[0].detach().cpu().permute(1,2,0).numpy()
            np.rot90(pixels, k=0, axes=(1,0))
            with torch.no_grad():
                input_image = wandb.Image(imgs[0].detach().cpu().permute(1,2,0).numpy(), caption=f"Input image {k}")
                image = wandb.Image(pixels, caption=f"Reconstructed Image {k}")
                example_images.append(image)
                input_images.append(input_image)
                wandb.log({'Reconstructed Images': example_images, 'Input images': input_images})
        
    # mse_feas = compactness_loss.item()
    
    # Calculating the threshold for updating at the test time
    # point_sc = point_score(outputs, imgs)

    # if  point_sc < args.th:
    #     query = F.normalize(feas, dim=1)
    #     query = query.permute(0,2,3,1) # b X h X w X d
    #     m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    # feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                    anomaly_score_list_inv(psnr_list[video_name]), args.alpha)

anomaly_score_total_list = np.asarray(anomaly_score_total_list)

accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

print('The result of ', args.dataset_type)
print('AUC: ', accuracy*100, '%')