import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_channels, output_channel=3, dropout=0.5):
        super(UNet, self).__init__()
        self.inc = inconv(input_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.up1 = up(512, 256)
        self.up2 = up(256, 128)
        self.up3 = up(128, 64)
        self.outc = nn.Conv2d(64, output_channel, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2 = self.dropout(x2)
        x3 = self.down2(x2)
        # x3 = self.dropout(x3)
        x4 = self.down3(x3)
        # x4 = self.dropout(x4)
        x = self.up1(x4, x3)
        # x = self.dropout(x)
        x = self.up2(x, x2)
        # x = self.dropout(x)
        x = self.up3(x, x1)
        # x = self.dropout(x)
        x = self.outc(x)
 
        return torch.tanh(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class d_netclassifier(nn.Module):
    def __init__(self):
        super(d_netclassifier, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
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
            nn.Linear(32768, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x


def _test():
    rand = torch.ones([4, 12, 256, 256]).cuda()
    t = UNet(12, 3).cuda()

    r = t(rand)
    print(r.shape)
    print(r.grad_fn)
    print(r.requires_grad)
