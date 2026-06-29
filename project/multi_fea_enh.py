import torch
import torch.nn as nn
import pywt
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.reduced_channels =  self.in_planes // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1) 

        self.fc1 = nn.Conv2d(self.in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) 
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  
        out = self.sigmoid(out) * x
        return out 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  
        padding = 3 if kernel_size == 7 else 1  

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x) 
        result = out * self.sa(out)
        return result

class PagModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.F_a = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.F_o = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_a, f_o):
        f_a_proj = self.F_a(f_a)
        f_o_proj = self.F_o(f_o)
        sigma = self.sigmoid((f_a_proj * f_o_proj).sum(dim=1, keepdim=True))
        output = sigma * f_o + (1 - sigma) * f_a
        return output

class WM2FE(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(WM2FE, self).__init__()
        self.dim1 = in_channels
        self.dim2 = inter_channels
        self.chunk_num = int(self.dim1 / self.dim2)
        
        self.ca_ba = CBAM(in_planes=self.dim1,)
        
        self.pagM = PagModule(self.dim2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=(1, 3), padding=(0, 1), groups=self.dim1),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=(3, 1), padding=(1, 0), groups=self.dim1),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=1),
            nn.BatchNorm2d(self.dim1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_a, F_w):
        processed_branches = None
        Fa_branchs = torch.chunk(F_a, self.chunk_num, dim=1)
        for branch in Fa_branchs:
            current_branch = branch * F_w
            current_branch = self.pagM(branch, current_branch)
            
            if processed_branches is None:
                processed_branches = current_branch
            else:
                processed_branches = torch.cat([processed_branches, current_branch], dim=1)
        
        x = self.conv(processed_branches)
        x = self.ca_ba(x)
        return x
