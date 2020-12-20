import torch
from torch._C import has_openmp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

class PhaseNet(nn.Module):
    def __init__(self, pyr, device):
        super(PhaseNet, self).__init__()
        self.height = pyr.height
        self.device = device
        self.layers = self.create_architecture()

    def create_architecture(self):
        return [
            PhaseNetBlock(2, 64, 1, (1, 1), self.device),
            PhaseNetBlock(64 + 1 + 16, 64, 8, (1, 1), self.device),
            PhaseNetBlock(64 + 8 + 16, 64, 8, (1, 1), self.device),
            *[PhaseNetBlock(64 + 8 + 16, 64, 8, (3 ,3), self.device) for _ in range(8)]
        ]

    def forward(self, vals, vals2):
        # Phase Net -> 2 image Values
        # high level [256, 256, 1]
        # low level [8, 8, 1]
        # phase 10
        # 0 [256, 256, 4]
        # 1 [182, 182, 4]
        # 2 [128, 128, 4]
        # 3 [90, 90, 2, 4]
        # 4 [64, 64, 2, 4]
        # 5 [46, 46, 2, 4]
        # 6 [32, 32, 2, 4]
        # 7 [22, 22, 2, 4]
        # 8 [16, 16, 2, 4]
        # 9 [12, 12, 2, 4] -> [12, 12, 2, 4] -> [12, 12, 8]
        # amplitude
        # 0 [256, 256, 2, 4]
        # 1 [182, 182, 2, 4]
        # 2 [128, 128, 2, 4]
        # 3 [90, 90, 2, 4]
        # 4 [64, 64, 2, 4]
        # 5 [46, 46, 2, 4]
        # 6 [32, 32, 2, 4]
        # 7 [22, 22, 2, 4]
        # 8 [16, 16, 2, 4]
        # 9 [12, 12, 2, 4] -> [12, 12, 8]

        print(' START', vals.low_level.shape, vals2.low_level.shape)

        # stack low level images of input frames together
        img = torch.stack((vals.low_level[:, :, 0], vals2.low_level[:, :, 0]), -1).unsqueeze(0).permute(0, 3, 1, 2)

        # get output of first phase-net block
        feature, prediction = self.layers[0](img)

        # reverse list of steerable pyramid phase and amplitude of each frame
        phase_dim = vals.phase[::-1]
        phase_dim2 = vals2.phase[::-1]
        amplitude_dim = vals.amplitude[::-1]
        amplitude_dim2 = vals2.amplitude[::-1]

        # define low_level of output 
        low_level = prediction.clone().squeeze(0).permute(1, 2, 0)
        print('Low Level', low_level.shape)

        # Combined phase values
        # for pyramid
        phase = []
        
        # Combined amplitude values
        # for pyramid
        amplitude = []

        # Create combined phase and amplitude values
        for idx in range(1, len(self.layers)):

            # Pls refactor
            res = phase_dim[idx-1].shape[0]
            resized_feature = torch.nn.Upsample((res, res), mode='bilinear')(feature)
            resized_prediction = torch.nn.Upsample((res, res), mode='bilinear')(prediction)

            phase1 = phase_dim[idx-1][:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)
            amplitude1 = amplitude_dim[idx-1][:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)

            phase2 = phase_dim2[idx-1][:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)
            amplitude2 = amplitude_dim2[idx-1][:,:, :4].reshape(res, res, -1, 1).permute(3, 2, 0, 1)

            concat = torch.cat((resized_feature, phase1, amplitude1, phase2, amplitude2, resized_prediction), 1)

            feature, prediction = self.layers[idx](concat)
            phase_p, amplitude_p = prediction.squeeze(0)[:4], prediction.squeeze(0)[4:]

            permuted_prediction = prediction.squeeze(0).permute(1, 2, 0)
            print('TEST', torch.view_as_real(phase_p.type(torch.complex64)).shape)

            phase.append(torch.view_as_real(phase_p.type(torch.complex64)).permute(1, 2, 3, 0))
            amplitude.append(torch.view_as_real(amplitude_p.type(torch.complex64)).permute(1, 2, 3, 0))

        print(feature.shape, prediction.shape)
        # Definitely Wrong!
        high_level = amplitude[-1][:,:,0][:,:,0].unsqueeze(-1)

        # TODO
        # First no idea how to compute high level, pls fix future me
        # Second problem with phase and amplitude, they want complex values, the 2 in the dimensions e.g. [256, 256, 2, 4]
        # Third low level and high level may be swapped??
        # 4. 2 Values combining
        # 5. Use RGB Channels

        print('ENDE')
        print(vals.low_level.shape)
        print(vals.phase[0].shape)
        print(vals.high_level.shape)

        print(low_level.shape)
        print(phase[0].shape)
        print(high_level.shape)

        values = DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phase[::-1],
            amplitude=amplitude[::-1]
        )

        return values



class PhaseNetBlock(nn.Module):
    """
    PhaseNet Block
    """
    def __init__(self, c_in, c_out, pred_out, kernel_size, device):
        super(PhaseNetBlock, self).__init__()

        padding = 0
        if kernel_size == (3, 3):
            padding = 1

        self.feature_map = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ReLU()
        )
        self.prediction_map = nn.Sequential(
            nn.Conv2d(c_out, pred_out, (1, 1)),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
       f = self.feature_map(x)
       c = self.prediction_map(f)

       return f, c
