import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self, num_imgs=4):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+1, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.Tanh()
        )


    def forward(self, adacof, phase, other, uncertainty_mask):
        x = torch.cat([adacof, phase, other, uncertainty_mask], 1)
        res = self.net(x)
        fusion_frame = adacof + res

        return fusion_frame

class FusionNet2(torch.nn.Module):

    def __init__(self, num_imgs=4):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+1, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.Sigmoid()
        )

    def forward(self, adacof, phase, other, uncertainty_mask):
        x = torch.cat([adacof, phase, other, uncertainty_mask], 1)
        alpha = self.net(x)

        fusion_frame = alpha*adacof + (1-alpha)*phase
        #print(torch.max(uncertainty_mask), torch.mean(uncertainty_mask.reshape(-1)))

        return fusion_frame, alpha