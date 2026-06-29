import torch
import torch.nn as nn
from pytorch_wavelets import DWT2D
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class tDWT(nn.Module):
    def __init__(self, n_level=4, wave='db1', alpha=1.5, beta=1.2,):
        super(tDWT, self).__init__()
        self.n_level = n_level
        self.wave = wave
        self.alpha = alpha
        self.beta = beta
        self.dwt = DWT2D(wave=self.wave, mode='zero', J=self.n_level).cuda()

    def forward(self, x):
        B, C, H, W = x.shape

        x_cA_feature = []
        x_cH_feature = []
        x_cV_feature = []
        x_cD_feature = []

        cA, cHVD = self.dwt(x)

        for i in range(self.n_level):
            cH, cV, cD = cHVD[i].chunk(3, dim=2)
            cH = cH.squeeze(2)
            cV = cV.squeeze(2)
            cD = cD.squeeze(2)
            x_cH_feature.append(cH)
            x_cV_feature.append(cV)
            x_cD_feature.append(cD)

        current_cA = x
        for i in range(self.n_level):
            temp_cA, _ = DWT2D(wave=self.wave, mode='zero', J=1).cuda()(current_cA)
            x_cA_feature.append(temp_cA)
            current_cA = temp_cA

        scales = []
        for i in range(1, self.n_level):
            dwt_merge = torch.cat([x_cA_feature[i], x_cH_feature[i],x_cV_feature[i],x_cD_feature[i] ], dim=1)
            scales.append(dwt_merge)

        return scales