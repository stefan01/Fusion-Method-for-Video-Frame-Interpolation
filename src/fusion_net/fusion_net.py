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

class FusionNet(nn.Module):
    def __init__(self, pyr, device, img_num=4):
        super(FusionNet, self).__init__()
        self.pyr = pyr
        self.height = pyr.height
        self.device = device
        self.img_num = img_num
        self.layers = self.create_architecture()
        self.to(self.device)
        self.eps = 1e-8

    def create_architecture(self):
        """ Create phase net architecture. """
        return nn.ModuleList([
            PhaseNetBlock(self.img_num, 64, 1, (1, 1), self.device),
            PhaseNetBlock(64 + 1 + 8 * self.img_num, 64, 8, (1, 1), self.device),
            PhaseNetBlock(64 + 8 + 8 * self.img_num, 64, 8, (1, 1), self.device),
            *[PhaseNetBlock(64 + 8 + 8 * self.img_num, 64, 8, (3 ,3), self.device) for _ in range(8)]
        ])

    def set_layers(self, start, end, freeze=True):
        """ Freeze or unfreeze layers. """
        for param in self.layers[start:end].parameters():
            param.requires_grad = freeze

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
        self.max_amplitudes = []
        batch_size = int(vals.amplitude[0].shape[0])
        for amplitude in vals.amplitude:
            max_amplitude = amplitude.reshape(batch_size, -1).max(1)[0] + self.eps

            # Save max amplitudes
            self.max_amplitudes.append(max_amplitude)

            amp_shape = amplitude.shape
            amplitudes.append((amplitude.reshape(amp_shape[0], -1).permute(1, 0) / max_amplitude).permute(1, 0).reshape(amp_shape))

        # Normalize phase
        phases = [x / math.pi for x in vals.phase]

        # Normalize low_level
        picture_num = int(batch_size/3)
        low_shape = vals.low_level.shape
        self.max_low_level = vals.low_level.reshape(batch_size, -1).max(1)[0] + self.eps
        low_level = (vals.low_level.reshape(batch_size, -1).permute(1, 0) / (self.max_low_level)).permute(1, 0).reshape(low_shape)

        return DecompValues(
            high_level=vals.high_level,
            low_level=low_level,
            amplitude=amplitudes,
            phase=phases
        )

    def reverse_normalize(self, vals, m):
        amplitudes = []
        phases = [x*math.pi for x in vals.phase]
        for i in range(m):
            max_ampl = self.max_amplitudes[i]
            amp_shape = vals.amplitude[i].shape
            batch_size = int(amp_shape[0]/self.pyr.nbands)

            amplitudes.append((vals.amplitude[i].reshape(batch_size, -1).permute(1, 0) * max_ampl).permute(1, 0).reshape(amp_shape))

        print("IN NET")
        print(len(amplitudes))
        print(amplitudes[0])
        print("#" * 10)

        for _ in range(self.height-2-m):
            phases.append(0)
            amplitudes.append(0)

        # low_level
        low_shape = vals.low_level.shape
        picture_num = int(low_shape[0]/3)
        low_level = (vals.low_level.reshape(low_shape[0], -1).permute(1, 0) * self.max_low_level).permute(1, 0).reshape(low_shape)

        return DecompValues(
            high_level=vals.high_level,
            low_level=low_level,
            amplitude=amplitudes[::-1],
            phase= phases[::-1]
            )

    def forward(self, vals, m=0):
        m = self.height-2
        vals = self.normalize_vals(vals)

        # Get output of first phase-net block
        feature, prediction = self.layers[0](vals.low_level)

        # define low_level of output
        low_level = prediction

        # Combined phase, amplitude values
        phase, amplitude = [], []

        for idx in range(m):
            # Resize
            res1, res2 = vals.phase[idx].shape[2:]
            # Upsample feature and prediction map to next resolution level
            feature_r = torch.nn.Upsample((res1, res2), mode='bilinear')(feature)
            prediction_r = torch.nn.Upsample((res1, res2), mode='bilinear')(prediction)

            concat = torch.cat((feature_r, vals.phase[idx], vals.amplitude[idx], prediction_r), 1)

            # pass concatenated layer through phasenet-Block
            i = idx+1 if idx+1 < len(self.layers)-1 else len(self.layers)-1
            feature, prediction = self.layers[i](concat)

            # append prediction to phase and amplitude
            res1, res2 = prediction.shape[2:]
            phase.append(prediction[:, :4, :, :].reshape(-1, 1, res1, res2))
            amplitude.append(prediction[:, 4:, :, :].reshape(-1, 1, res1, res2))

        print("IN NET")
        print(len(amplitude))
        print(amplitude[0].shape)
        print("#" * 10)

        # Use torch.zeros with right shape, mean of both input high levels (test both) Flow levels of AdaCof
        hl_shape = vals.high_level.shape
        #high_level = vals.high_level.mean(1).unsqueeze(1)
        high_level = torch.zeros((hl_shape[0], 1, hl_shape[2], hl_shape[3]), device=self.device)

        values = self.reverse_normalize(DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phase,
            amplitude=amplitude
        ), m)

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
            #nn.BatchNorm2d(c_out),
            nn.Conv2d(c_out, c_out, kernel_size, padding=padding, padding_mode='reflect'),
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