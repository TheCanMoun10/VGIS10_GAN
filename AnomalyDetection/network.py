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
    
    
# Convolutional Architecture:
class conv_block(nn.Module):
    '''
    Convulutional block of the UNet architecture.
    Batch normalization is used to reduce internal covariance shift and stabilizes the network while training.
    Consists of two convolutional layers and two batch normalisations, used ReLU as activation function.
    Return x, the feature map of the input image.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        
        return x
        
    
# Encoder block:
class encoder_block(nn.Module):
    '''
    Consists of a conv_block followed by a 2x2 max pooling layer. Number of filters is doubled and the height and width are reduced half after every block.
    Padding is used to make sure the shape of the output feature maps remain the same as the input feature maps.
    Return the features at each specified channel.
    '''
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([conv_block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, inputs):
        feats = []
        for blocks in self.enc_blocks:
            inputs = blocks(inputs)
            feats.append(inputs)
            inputs = self.pool(inputs)
        return feats
    
        # feature_map = self.conv(inputs)
        # pool_layer = self.pool(feature_map)
        
        # return feature_map, pool_layer
    
    
    # def __init__(self, in_channels, out_channels):
    #     super().__init__()
    #     self.conv = conv_block(in_channels, out_channels)
    #     self.pool = nn.MaxPool2d((2,2))
    
    # def forward(self, inputs):
    #     feature_map = self.conv(inputs)
    #     pool_layer = self.pool(feature_map)
        
    #     return feature_map, pool_layer
    
# Decoder block:
class decoder_block(nn.Module):
    '''
    self.dec_blocks is a list of decoder blocks that perform two conv and a ReLU operations. 
    self.upConvs is a list of ConvTranspose2d operations that perform "up-convulotions". 
    The forward functions accepts the encoder_features output by the encoder and performs concatenation beofre passing to the conv_block operation.
    '''
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.upConvs = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList([conv_block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        
    def forward(self, inputs, encoder_features):
        for i in range(len(self.channels)-1):
            inputs = self.upConvs[i](inputs)
            enc_featrs = self.crop(encoder_features[i], inputs)
            inputs = torch.cat([inputs, enc_featrs], dim=1)
            inputs = self.dec_blocks[i](inputs)
            
        return inputs
    
    def crop(self, enc_featrs, inputs):
        _, _, H, W = inputs.shape # Height and width of input tensor.
        enc_featrs = transforms.CenterCrop([H, W])(enc_featrs)
        return enc_featrs
    
class UNet(nn.Module):
    def __init__(self, encoder_channels=(3,64,128,256,512,1024), decoder_channels=(1024,512,256,128,64), num_class=1, retain_dim=False, out_size=(572,572)):
        super().__init__()
        self.encoder = encoder_block(encoder_channels)
        self.decoder = decoder_block(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size
        
    def forward(self, inputs):
        encoder_features = self.encoder(inputs)
        out = self.decoder(encoder_features[::-1][0], encoder_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim: # Makes the output size the same as the input image size.
            out = F.interpolate(out, self.out_size)
        return out