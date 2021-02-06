import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self, num_imgs=4):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
        )


    def forward(self, adacof, phase, other, uncertainty_mask):
        x = torch.cat([adacof, phase, other], 1)
        alpha = self.net(x)
        
        fusion_frame = alpha*adacof + (1-alpha)*phase
        result_frame = uncertainty_mask*fusion_frame + (1-uncertainty_mask)*adacof
  
        return result_frame