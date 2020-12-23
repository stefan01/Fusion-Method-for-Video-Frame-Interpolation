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

    # Normalize the amplitude and phase values.
    #
    # Input amplitude values go from zero to unbound
    # Input phase values go from -pi to pi
    #
    # Amplitude values are normalized to the range 0 to 1
    # Phase values are normalized to the range -1 to 1
    def normalize_vals(self, vals):
        # Normalize amplitude
        amplitudes = []
        for amplitude in vals.amplitude:
            max_amplitude, _ = torch.max(amplitude, 2, True)
            amplitudes.append(torch.div(amplitude, max_amplitude))

        # Normalize phase
        phases = [torch.div(x, math.pi) for x in vals.phase]

        return DecompValues(
            high_level=vals.high_level,
            low_level=vals.low_level,
            amplitude=amplitudes,
            phase=phases
        )

    def forward(self, vals):
        vals = self.normalize_vals(vals)

        # Get output of first phase-net block
        feature, prediction = self.layers[0](vals.low_level)

        # define low_level of output
        low_level = prediction

        # Combined phase, amplitude values
        phase, amplitude = [], []

        for idx in range(len(self.layers)-1):
            # Resize
            res1, res2 = vals.phase[idx].shape[2:]
            # Upsample feature and prediction map to next resolution level
            feature_r = torch.nn.Upsample((res1, res2), mode='bilinear')(feature)
            prediction_r = torch.nn.Upsample((res1, res2), mode='bilinear')(prediction)

            concat = torch.cat((feature_r, vals.phase[idx], vals.amplitude[idx], prediction_r), 1)

            # pass concatenated layer through phasenet-Block
            feature, prediction = self.layers[idx+1](concat)

            # append prediction to phase and amplitude
            res1, res2 = prediction.shape[2:]
            phase.append(prediction[:, :4, :, :].reshape(-1, 1, res1, res2))
            amplitude.append(prediction[:, 4:, :, :].reshape(-1, 1, res1, res2))

        # Use torch.zeros with right shape, mean of both input high levels (test both) Flow levels of AdaCof
        hl_shape = vals.high_level.shape
        high_level = torch.zeros((hl_shape[0], 1, hl_shape[2], hl_shape[3]), device=self.device)

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
