import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
# TODO: Add a UNet architecture

class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            # Architecture of LeNet-5
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride=2)
        )
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x): # The sequence which the layers will process the image.
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out
    

# Convolutional architecture:
class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64, 128)):
        super().__init__()
        self.encBlocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, x):
        blockOutputs = []
        
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        
        return blockOutputs
    
class Decoder(nn.Module):
    def __init__(self, channels=(128, 32, 16)):
        super().__init__()
        self.channels = channels
        self.upConvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        
    def forward(self, x, encFeatures):
        for i in range(len(self.channels)-1):
            x = self.upConvs[i](x)
            
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
            
        return x
    
    def crop(self, encFeatures, x):
        (_,_, H, W) = x.shape
        encFeatures = transforms.CenterCrop([H, W])(encFeatures)
        
        return encFeatures
    
class UNet(nn.Module):
    def __init__(self, image_height, image_width, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16), num_classes=1, retainDim=True):
        super().__init__()
        # initalialize encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        
        # Regression and head an class variable storing:
        self.head = nn.Conv2d(decChannels[-1], num_classes, 1)
        self.retainDim = retainDim
        self.outsize = (image_height, image_width)
        
    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        
        # pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

        # pass the decoder features through the regression head to
		# obtain the segmentation mask.
        map = self.head(decFeatures)
        
        # check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them.
        if self.retainDim:
            map = F.interpolate(map, self.outsize)
        
        # return feature map
        return map