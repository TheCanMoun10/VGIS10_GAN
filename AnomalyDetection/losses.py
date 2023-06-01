import torch
import torch.nn as nn
from torch.autograd import Variable

class Intensity_Loss(nn.Module):
    def __init__(self, l_num):
        super(Intensity_Loss, self).__init__()
        self.l_num = l_num
    def forward(self, gen_frames, img_frames):
        return torch.mean(torch.abs((gen_frames - img_frames)**self.l_num))
    
class Gradient_Loss(nn.Module):
    def __init__(self, alpha, channels):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.channels = channels
        
        filter = torch.FloatTensor([[-1, 1]]).cuda()
        
        self.filterx = filter.view(1,1,1,2).repeat(1, channels, 1, 1)
        self.filtery = filter.view(1,1,2,1).repeat(1, channels, 1, 1)
        
    def forward(self, gen_frames, img_frames):
        gen_frames_x = nn.functional.pad(gen_frames,(1,0,0,0))
        gen_frames_y = nn.functional.pad(gen_frames,(0,0,1,0))
        
        img_frames_x = nn.functional.pad(img_frames,(1,0,0,0))
        img_frames_y = nn.functional.pad(img_frames,(0,0,1,0))
        
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filterx)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filtery)
        img_dx = nn.functional.conv2d(img_frames_x, self.filterx)
        img_dy = nn.functional.conv2d(img_frames_y, self.filtery)
        
        grad_diff_x = torch.abs(img_dx - gen_dx)
        grad_diff_y = torch.abs(img_dy - gen_dy)
        
        return torch.mean(grad_diff_x**self.alpha + grad_diff_y**self.alpha)
    
class Adversarial_Loss(nn.Module):
    def __init__(self):
        super(Adversarial_Loss, self).__init__()
    def forward(self, fake_outputs):
        return torch.mean((fake_outputs-1)**2/2)
        
        