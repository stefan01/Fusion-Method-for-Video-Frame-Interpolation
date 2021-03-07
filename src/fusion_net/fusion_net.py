import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self, num_imgs=4):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+1, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
        )

class FusionNet3(torch.nn.Module):

    def __init__(self, num_imgs=4):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+1, 64, kernel_size=5, stride=1, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                # nn.Sigmoid()
                nn.Tanh()
        )

        # self.reslayer = nn.Conv2d(3*num_imgs+1, 3, kernel_size=1, stride=1)
        # self.tan = nn.Tanh()
        self.residuals = []


    def forward(self, adacof, phase, other, uncertainty_mask): 
    # save=False):
        x = torch.cat([adacof, phase, other, uncertainty_mask], 1)
        fusion_frame = self.net(x)
        
        # fusion_frame = adacof + self.net(x)
        # if save:
        #    self.residuals.append(torch.sum(self.net(x)).cpu().detach().item())

        # ResNet option
        # if resnet == True:
        #    x = self.reslayer(x)
        #    fusion_frame += x

        #fusion_frame = alpha*adacof + (1-alpha)*phase
        #print(torch.max(uncertainty_mask), torch.mean(uncertainty_mask.reshape(-1)))
        #result_frame = uncertainty_mask*fusion_frame + (1-uncertainty_mask)*adacof

        return fusion_frame