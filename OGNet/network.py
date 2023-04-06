
from torch import nn


class g_net(nn.Module):
    def __init__(self, nc=1):
        super(g_net, self).__init__()
        self.nc = nc 
        self.encoder = nn.Sequential(

            nn.Conv2d(self.nc, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, stride=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(512, 256, 5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.nc, 5, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class d_net(nn.Module):
    def __init__(self, nc=1):
        super(d_net, self).__init__()
        self.nc = nc

        # self.conv1 = nn.Conv2d(self.nc, 64, 5, stride=2, padding=2)
        # self.batch1 = nn.BatchNorm2d(64)
        # self.relu1 = nn.ReLU(True)
        
        # self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        # self.batch2 = nn.BatchNorm2d(128)
        # self.relu2 = nn.ReLU(True)
            
        # self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        # self.batch3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        
        # self.conv4 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        # self.relu4 = nn.ReLU(True)
        # self.flat = Flatten(),
        # self.sigmoid = nn.Sigmoid()
        
        
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.nc, 64, 5, stride=2, padding=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(8192, 1), #Tensor shape after flattening with (64x64): [5, 512, 4, 4] --> [5, 8192] (45x45): [5, 512, 3, 3] --> [5, 4608]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        # x = self.conv1(input)
        # x = self.batch1(x)
        # x = self.relu1(x)
        
        # x = self.conv2(x)
        # x = self.batch2(x)
        # x = self.relu2(x)
        
        # x = self.conv3(x)
        # x = self.batch3(x)
        # x = self.relu3(x)        
        
        # x = self.conv4(x)
        # x = self.relu4(x)
        # x = x.view(x.size(0), -1) # flattening tensor
        # #x = Flatten(x)
        # x = nn.Linear(x.size(1), 1)
        # x = self.sigmoid(x)
        
        return x
