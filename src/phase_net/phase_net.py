import torch
import torch.nn as nn
import torch.optim as optim
from src.train.utils import DecompValues
import math

class PhaseNet(nn.Module):
    """
    Phase Net for Video Frame Interpolation
    """

    def __init__(self, pyr, device, num_img=2):
        super(PhaseNet, self).__init__()
        self.pyr = pyr
        self.height = pyr.height
        self.device = device
        self.num_img = num_img
        self.layers = self.create_architecture()
        self.to(self.device)
        self.eps = 1e-8

    def create_architecture(self):
        """ Create phase net architecture. """
        if self.num_img == 3:
            return nn.ModuleList([
                PhaseNetBlock(self.num_img, 64, self.num_img - 1, (1, 1), self.device),
                PhaseNetBlock(64 + self.num_img - 1 + 8 * self.num_img, 64, self.num_img * 4, (1, 1), self.device),
                PhaseNetBlock(64 + self.num_img * 4 + 8 * self.num_img, 64, self.num_img * 4, (1, 1), self.device),
                *[PhaseNetBlock(64 + self.num_img * 4 + 8 * self.num_img, 64, self.num_img * 4, (3 ,3), self.device) for _ in range(5)]
            ])
        return nn.ModuleList([
            PhaseNetBlock(self.num_img, 64, 1, (1, 1), self.device),
            PhaseNetBlock(64 + 1 + 8 * self.num_img, 64, 8, (1, 1), self.device),
            PhaseNetBlock(64 + 8 + 8 * self.num_img, 64, 8, (1, 1), self.device),
            *[PhaseNetBlock(64 + 8 + 8 * self.num_img, 64, 8, (3 ,3), self.device) for _ in range(5)]
        ])

    def set_layers(self, start, end, freeze=True):
        """ Freeze or unfreeze layers. """
        for param in self.layers[start:end].parameters():
            param.requires_grad = freeze

    def normalize_vals(self, vals):
        """
        Normalize the amplitude and phase values.
        Input amplitude values go from zero to unbound
        Input phase values go from -pi to pi

        Amplitude values are normalized to the range 0 to 1
        Phase values are normalized to the range -1 to 1
        """
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
        """ Reverse the normalization. """
        amplitudes = []
        phases = [x*math.pi for x in vals.phase]
        for i in range(m):
            max_ampl = self.max_amplitudes[i]
            amp_shape = vals.amplitude[i].shape
            batch_size = int(amp_shape[0]/self.pyr.nbands)

            amplitudes.append((vals.amplitude[i].reshape(batch_size, -1).permute(1, 0) * max_ampl).permute(1, 0).reshape(amp_shape))

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

    def forward(self, vals, m=None):
        """ Forward pass through network. """
        if m is None:
            m = self.height-2

        # Get output of first phase-net block for low level prediction
        feature, prediction = self.layers[0](vals.low_level)
        
        # Prediction is the linear weights between the first low level
        alpha = (prediction[:, 0]+1)/2
        low_level = alpha * vals.low_level[:, 0] + (1-alpha) * vals.low_level[:, 1]

        # Fusion Method for low level
        #if self.num_img == 4:
        #    ada_alpha = (prediction[:, 1]+1)/2
        #    fusion_alpha = (prediction[:, 2]+1)/2
        #    
        #    ada_low_level = ada_alpha * vals.low_level[:, 2] + (1-ada_alpha) * vals.low_level[:, 3]
        #    low_level = fusion_alpha * low_level + (1-fusion_alpha) * ada_low_level
        #el
        if self.num_img == 3:
            fusion_alpha = (prediction[:, 1]+1)/2
            low_level = fusion_alpha * low_level + (1-fusion_alpha) * vals.low_level[:, 2]

        # Extra dimension for low level
        low_level = low_level.unsqueeze(1)

        # Use zeros for high level prediction
        hl_shape = vals.high_level.shape
        high_level = torch.zeros((hl_shape[0], 1, hl_shape[2], hl_shape[3]), device=self.device)

        # Combined phase, amplitude values
        phases, amplitudes = [], []

        for idx in range(m):
            # Resize
            res1, res2 = vals.phase[idx].shape[2:]

            # Upsample feature and prediction map to next resolution level
            feature_r = torch.nn.Upsample((res1, res2), mode='bilinear')(feature)
            prediction_r = torch.nn.Upsample((res1, res2), mode='bilinear')(prediction)

            concat = torch.cat((feature_r, vals.phase[idx], vals.amplitude[idx], prediction_r), 1)

            del feature
            del prediction
            torch.cuda.empty_cache()

            # Pass concatenated layer through phasenet
            i = idx+1 if idx+1 < len(self.layers)-1 else len(self.layers)-1
            feature, prediction = self.layers[i](concat)

            del concat
            torch.cuda.empty_cache()
            
            # Caculate amplitude values
            beta = (prediction[:, 4:8]+1)/2
            amplitude = beta * vals.amplitude[idx][:, 4:8] + (1-beta) * vals.amplitude[idx][:, :4]

            # Fusion Method
            #if self.num_img == 4:
            #    ada_beta = (prediction[:, 8:12]+1)/2
            #    fusion_beta = (prediction[:, 12:16]+1)/2
    
            #    ada_amplitude = ada_beta * vals.amplitude[idx][:, 8:12] + (1-ada_beta) * vals.amplitude[idx][:, 12:16]
            #    amplitude = fusion_beta * amplitude + (1-fusion_beta) * ada_amplitude
            #el
            if self.num_img == 3:
                fusion_beta = (prediction[:, 8:12]+1)/2
                print(fusion_beta.mean(-1).mean(-1))
                amplitude = fusion_beta * amplitude + (1-fusion_beta) * vals.amplitude[idx][:, 8:12]

            # append prediction to phase and amplitude
            res1, res2 = prediction.shape[2:]

            phases.append(prediction[:, :4].reshape(-1, 1, res1, res2))
            amplitudes.append(amplitude.reshape(-1, 1, res1, res2))

        values = self.reverse_normalize(DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phases,
            amplitude=amplitudes
        ), m)

        return values

class PhaseNetBlock(nn.Module):
    """
    PhaseNet Block
    """
    def __init__(self, c_in, c_out, pred_out, kernel_size, device, dropout=0.5):
        super(PhaseNetBlock, self).__init__()

        padding = 0
        if kernel_size == (3, 3):
            padding = 1

        self.feature_map = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(c_out),
            nn.ELU(),
            nn.Conv2d(c_out, c_out, kernel_size, padding=padding, padding_mode='reflect'),
            nn.ELU(),
        )
        self.prediction_map = nn.Sequential(
            nn.Conv2d(c_out, pred_out, (1, 1), padding_mode='reflect'),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        f = self.feature_map(x)
        c = self.prediction_map(f)

        return f, c
