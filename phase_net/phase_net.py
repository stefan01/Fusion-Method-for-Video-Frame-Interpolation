import torch
from torch._C import has_openmp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
import math

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
        self.pyr = pyr
        self.height = pyr.height
        self.device = device
        self.layers = self.create_architecture()

    def create_architecture(self):
        return [
            PhaseNetBlock(2, 64, 1, (1, 1), self.device),
            PhaseNetBlock(64 + 1 + 16, 64, 8, (1, 1), self.device),
            PhaseNetBlock(64 + 8 + 16, 64, 8, (1, 1), self.device),
            *[PhaseNetBlock(64 + 8 + 16, 64, 8, (3 ,3), self.device) for _ in range(self.height-4)]
        ]

    def normalize_vals(self, vals):
        # Normalization (Max local amplitude1 and 2 -> 0/1) and save for later use
        # [3, 2, 26, 26] -> ampl[0] -> [2, 26, 26]
        # Normalization phase (-pi/pi -> -1/1)

        # Normalize amplitude
        amplitudes = []
        for amplitude in vals.amplitude:
            max_amplitude, _ = torch.max(amplitude, 2, True)
            #print('MAX_AMPLITUDE...')
            #print(max_amplitude)
            amplitudes.append(torch.div(amplitude, max_amplitude))
        
        phases = [torch.div(x, math.pi) for x in vals.phase]

        #print('AMPLITUDES')
        #print(amplitudes)

        #print('PHASES')
        #print(phases)

        return DecompValues(
            high_level=vals.high_level,
            low_level=vals.low_level,
            amplitude=amplitudes,
            phase=phases
            )

    def forward(self, vals):
        # Phase Net -> 2 image Values
        # high level [3, 2, 256, 256]
        # low level [3, 2, 8, 8]
        # phase 10
        # 0 [3, 8, 12, 12]
        # 1 [3, 8, 16, 16]
        # 2 [3, 8, 24, 24]
        # amplitude
        # 0 [3, 8, 12, 12]
        # 1 [3, 8, 16, 16]
        # 2 [3, 8, 24, 24]

        vals = self.normalize_vals(vals)


        print(f'Low Level: \t{vals.low_level.shape}')
        print(f'High Level: \t{vals.high_level.shape}')
        print(f'Phase Level: \t{vals.phase[0].shape}')
        print(f'Amplitude Level: \t{vals.amplitude[0].shape}')

        # get output of first phase-net block
        feature, prediction = self.layers[0](vals.low_level)

        # define low_level of output
        low_level = prediction      # TODO pyramid check value to coefficient correct shapes
        print(f'Low Level Prediction: \t{low_level.shape}')

        # Combined phase, amplitude values
        phase, amplitude = [], []

        # TODO -> check reverse list
        # Create combined phase and amplitude values
        for idx in range(len(self.layers)-1):
            # Resize
            res1, res2 = vals.phase[idx].shape[2:]
            # print((res1, res2))
            feature_r = torch.nn.Upsample((res1, res2), mode='bilinear')(feature)
            prediction_r = torch.nn.Upsample((res1, res2), mode='bilinear')(prediction)

            # print(f'Resized_feature map: \t{feature_r.shape}')
            # print(f'Resized_prediction map: \t{prediction_r.shape}')

            concat = torch.cat((feature_r, vals.phase[idx], vals.amplitude[idx], prediction_r), 1)
            # print(f'New Input to Net concat: \t{concat.shape}')
            # torch.Size([3, 81, 12, 12])

            feature, prediction = self.layers[idx+1](concat)

            # print(f'Feature map: \t{feature.shape}')        # torch.Size([3, 64, 16, 16])
            # print(f'Prediction map: \t{prediction.shape}')  # torch.Size([3, 8, 16, 16])

            res1, res2 = prediction.shape[2:]
            phase.append(prediction[:, :4, :, :].reshape(-1, 1, res1, res2))       # torch.Size([3, 1, , 16])
            amplitude.append(prediction[:, 4:, :, :].reshape(-1, 1, res1, res2))

        # Use torch.zeros with right shape, mean of both input high levels (test both) Flow levels of AdaCof
        hl_shape = vals.high_level.shape
        high_level = torch.zeros((hl_shape[0], 1, hl_shape[2], hl_shape[3]), device=self.device)

        # Prediction *pi, Amplitude *maximum value, do it on end of pyramid (before value to coeff)
        # low -> kleinstes Bild
        # high -> größtes Bild (Pyramide umgekehrt)

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
