import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self, num_imgs=4, uncertainty_maps=2, kernel=3, pad=3, dil=3):
        super(FusionNet, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+uncertainty_maps, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.Tanh()
        )

        self.residuals = []


    def forward(self, adacof, phase, other, ada_uncertainty, phase_uncertainty, mode="none", save=False):
        x = torch.cat([adacof, phase, other, ada_uncertainty, phase_uncertainty], 1)
        res = self.net(x)

        if mode == "adacof":
            fusion_frame = adacof + res
        elif mode == "phase":
            fusion_frame = phase + res
        else:
            fusion_frame = res
        
        if save:
            self.residuals.append(torch.sum(res).cpu().detach().item())

        return fusion_frame

class FusionNetBoth(torch.nn.Module):

    def __init__(self, num_imgs=4, uncertainty_maps=2, kernel=3, pad=3, dil=3):
        super(FusionNetBoth, self).__init__()

        self.net = nn.Sequential(
                nn.Conv2d(3*num_imgs+uncertainty_maps, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=kernel, stride=1, padding=pad, dilation=dil),
                nn.Tanh()
        )

        self.net_alpha = nn.Sequential(
                nn.Conv2d(3*num_imgs+uncertainty_maps, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=3, dilation=3),
                nn.Sigmoid()
        )

        self.residuals = []


    def forward(self, adacof, phase, other, ada_uncertainty, phase_uncertainty, mode="alpha", save=False):
        x = torch.cat([adacof, phase, other, ada_uncertainty, phase_uncertainty], 1)
        res = self.net(x)

        alpha = self.net_alpha(x)
        fusion_frame = (alpha*adacof + (1-alpha)*phase) + res
        
        if save:
                self.residuals.append(torch.sum(res).cpu().detach().item())
                self.residuals.append(torch.sum(alpha).cpu().detach().item())

        return fusion_frame, alpha

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