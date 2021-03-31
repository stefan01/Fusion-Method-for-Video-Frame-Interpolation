import torch
from src.adacof.utility import moduleNormalize
from torch.nn import functional as F
import torch.nn as nn

class FusionNet(torch.nn.Module):

    def __init__(self, num_imgs=5, uncertainty_maps=3, kernel=3, pad=3, dil=3):
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
        
        input_channels = 3*num_imgs+uncertainty_maps
        
        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.Conv2d(32,             64, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.Conv2d(64,            128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        ])
        
        self.bottleneck_layer = nn.Conv2d(128,  128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(128,  64, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.Conv2d(64,   32, kernel_size=5, stride=1, padding=2, padding_mode='reflect'),
            nn.Conv2d(32,    3, kernel_size=1, stride=1),
        ])
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.deconvolution = nn.Upsample(scale_factor=2, mode='bilinear')

        self.residuals = []


    def forward(self, base, adacof, phase, other, maps, save=False, variant=0):
        x = torch.cat([base, adacof, phase, other, maps], 1)
        skip = []
        
        for layer in self.encoder_layers:
            x = self.relu(layer(x))
            # For skipping connections
            skip.append(x)
            x = self.max_pool(x)
            
        x = self.bottleneck_layer(x)
            
        for layer, s in zip(self.decoder_layers, skip[::-1]):
            x = self.deconvolution(self.relu(x))
            x = x + s
            x = layer(x)
            
        res = self.tanh(x)

        if variant == 1:
            fusion_frame = phase + res
        else:
            fusion_frame = base + res
        
        if save:
            self.residuals.append(torch.sum(res).cpu().detach().item())

        return fusion_frame.clamp(0, 1)
