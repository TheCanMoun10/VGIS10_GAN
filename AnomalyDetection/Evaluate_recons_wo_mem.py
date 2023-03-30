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
import wandb
import datetime

import argparse


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
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='directory of data')
#parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--model_dir', type=str, help='directory of model')
parser.add_argument('--m_items_dir', type=str, help='directory of model')
parser.add_argument('--nega_loss', type=bool, default=False, help='Apply negative loss to model')
parser.add_argument('--nega_value', type=float, default=0.02, help='Value to degrade loss')
parser.add_argument('--mode', type=str, default="eval")

args = parser.parse_args()

wandb.init(project="VGIS10_MNAD",
           
           config={
               "mode" : args.mode,
               "time_step": args.t_length,
               "architecture" : args.model,
               "dataset": args.dataset_type,
               "batch size" : args.test_batch_size,
               "negative loss" : args.nega_loss,

               },
            name=f'{args.mode}_{args.dataset_type}_batch{args.test_batch_size}_timestep{args.t_length}'
           
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

test_folder = args.dataset_path+args.dataset_type+"/testing/frames"

# Loading dataset
test_dataset = DataLoader(test_folder, transforms.Compose([
             transforms.ToTensor(),            
             ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1,num_pred = 0)

test_size = len(test_dataset)

test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size, 
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

loss_func_mse = nn.MSELoss(reduction='none')

# Loading the trained model
model = torch.load(args.model_dir)
model.cuda()
# m_items = torch.load(args.m_items_dir)


labels = np.load('./data/frame_labels_'+args.dataset_type+'.npy')
if args.dataset_type == 'shanghaitech':
    labels = np.expand_dims(labels, 0)

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
generated_images = []
input_images = []
output_images = []

print('Evaluation of', args.dataset_type)

# Setting for video anomaly detection
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
    label_length += videos[video_name]['length']
    psnr_list[video_name] = []
    feature_distance_list[video_name] = []

label_length = 0
video_num = 0
label_length += videos[videos_list[video_num].split('/')[-1]]['length']
# m_items_test = m_items.clone()

model.eval()

for k,(imgs) in enumerate(test_batch):

    if k == label_length-0*(video_num+1):
        print(k,label_length)
        video_num += 1
        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

    
    imgs = Variable(imgs).cuda()
    input_images.append(imgs[0].detach().cpu().permute(1,2,0).numpy())

    outputs = model.forward(imgs, False)
    output_images.append(outputs[0].detach().cpu().permute(1,2,0).numpy())
    pixels = outputs[0].detach().cpu().permute(1,2,0).numpy()
    np.rot90(pixels, k=0, axes=(1,0))
    
    mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (imgs[0]+1)/2)).item()
    
    if(k%10 == 0):
            print(f"Processing image [{k+1} / {len(test_batch)+1}]")
            image = wandb.Image(pixels, caption=f"Generated_anomaly_{k+1}")
            generated_images.append(image)

    psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_imgs))
    # feature_distance_list[videos_list[video_num].split('/')[-1]].append(mse_feas)
    wandb.log({'generated_images': generated_images})


# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
for video in sorted(videos_list):
    video_name = video.split('/')[-1]
    anomaly_score_total_list += score_sum(anomaly_score_list(psnr_list[video_name]), 
                                    anomaly_score_list_inv(psnr_list[video_name]), args.alpha)

# anomaly_score_total_list = np.asarray(anomaly_score_total_list)

# accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))

# print('The result of ', args.dataset_type)
# print('AUC: ', accuracy*100, '%')

from scipy.ndimage import rotate
output = output_images[50]
image = input_images[50]
output = np.rot90(output, k=0, axes=(1,0))
image = np.rot90(image, k=0, axes=(1,0))

cv2.imshow("Real image", image)

cv2.imshow("fake image", output)

horizontal = np.concatenate((image, output), axis=1)

cv2.imshow("Side by side real image and fake image", horizontal)
cv2.waitKey(0)

today = datetime.datetime.today()
timestring = f"{today.date}-{today.hour}:{today.minute}"
image_dir = os.path.join('./evals', 'figures', args.dataset_type, timestring)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

cv2.imwrite(os.path.join(image_dir, f'{args.dataset_type}_Fake_image_negloss{args.nega_loss}.jpg'), 255*output)
cv2.imwrite(os.path.join(image_dir, f'{args.dataset_type}_Real_image_negloss{args.nega_loss}.jpg'), 255*image)
cv2.imwrite(os.path.join(image_dir, f'{args.dataset_type}_Side_by_side_negloss{args.nega_loss}.jpg'), 255*horizontal)

wandb.finish()
