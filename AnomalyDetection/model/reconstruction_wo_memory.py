import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory_final_spatial_sumonly_weight_ranking_top1 import *

class Encoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel = 3):
        super(Encoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=2, stride=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=2, stride=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=2, stride=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=2, stride=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        
    def forward(self, x):

        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)
        # print(f'Generator Pool layer 1: {tensorPool1.shape} \n')

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)
        # print(f'Generator Pool layer 3: {tensorPool3.shape} \n')

        tensorConv4 = self.moduleConv4(tensorPool3)
        print(f'Generator output: {tensorConv4.shape} \n')
                
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

    
    
class Decoder(torch.nn.Module):
    def __init__(self, t_length = 5, n_channel =3):
        super(Decoder, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=1, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=1, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
                
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=1, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=1, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=1, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels = nc, out_channels=intOutput, kernel_size = 1, stride = 2, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(512, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(256, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(128, 128)
        self.moduleUpsample2 = Upsample(128, 64)
        
        # self.moduleDeconv4 = Basic(64, 64)
        # self.moduleUpsample1 = Upsample(64, 32)
        
        self.moduleDeconv1 = Gen(64,n_channel*(t_length-1),n_channel)
        
        
        
    def forward(self, x):
        
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = tensorUpsample4
        #print(f'Deconvulational layer 1: {cat4.shape}')
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = tensorUpsample3
        #print(f'Deconvulational layer 2: {cat3.shape}')
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = tensorUpsample2
        
        # tensorDeconv4 = self.moduleDeconv4(cat2)
        # tensorUpsample1 = self.moduleUpsample1(tensorDeconv4)
        # cat1 = tensorUpsample1
        
        output = self.moduleDeconv1(cat2)
        print(f'Decoder output: {output.shape}')

                
        return output
    


class convAE(torch.nn.Module):
    def __init__(self, n_channel =3,  t_length = 5, memory_size = 10, feature_dim = 512, key_dim = 512, temp_update = 0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size,feature_dim, key_dim, temp_update, temp_gather)
       

    def forward(self, x, train=True):

        fea, skip1, skip2, skip3 = self.encoder(x)
        if train:
            # updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(fea)
            
            return output #, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        
        #test
        else:
            # updated_fea, keys, softmax_score_query, softmax_score_memory,query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(fea)
            
            return output #, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss

# OpenGAN discriminator:
class OpenGAN_Discriminator(nn.Module):
    def __init__(self, ngpu=1, nc=3, ndf=512):
        super(OpenGAN_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        
        self.conv1 = nn.Conv2d(self.nc, self.ndf*8, 1, 1, 0, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.ndf*4)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.ndf*2)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(self.ndf)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()  
        
        # self.main = nn.Sequential(
        #     nn.Conv2d(self.nc, self.ndf*8, 1, 1,0, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv2d(self.ndf*8, self.ndf*4, 1, 1, 0, bias=False),
        #     # nn.BatchNorm2d(self.ndf*4),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv2d(self.ndf*4, self.ndf*2, 1, 1, 0, bias=False),
        #     # nn.BatchNorm2d(self.ndf*2),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv2d(self.ndf*2, self.ndf, 1, 1, 0, bias=False),
        #     # nn.BatchNorm2d(self.ndf),
        #     nn.LeakyReLU(0.2, inplace=True),
            
        #     nn.Conv2d(self.ndf, 1, 1, 1, 0, bias=False),
        #     nn.Sigmoid()  
        # )
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.sigmoid(x)
        
        print(f'Discriminator output: {x.shape}')
        return x