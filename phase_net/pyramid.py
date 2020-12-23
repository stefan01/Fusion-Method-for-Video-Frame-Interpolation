import numpy as np
import torch
from collections import namedtuple
from skimage import io
import copy

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import warnings
warnings.filterwarnings("ignore")

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)


class Pyramid:
    def __init__(self, height, nbands, scale_factor, device):
        self.height = height
        self.nbands = nbands
        self.scale_factor = scale_factor
        self.device = device
        self.pyr = SCFpyr_PyTorch(
            height=self.height,
            nbands=self.nbands,
            scale_factor=self.scale_factor,
            device=self.device,
        )

    def filter(self, img):
        """ Psi filter """
        coeff = self.pyr.build(img)
        # Coeff 12
        # 0 [3 256 256]
        # 1 List 4[3 256 256 2]
        # 1 List 4[3 182 182 2]
        # -1 [3 8 8]
        vals = self.coeff_to_values(coeff)
        return vals, coeff

    def inv_filter(self, vals, coeff2):
        """ Psi^{-1} filter """
        coeff = self.values_to_coeff(vals, coeff2)
        img = self.pyr.reconstruct(coeff)

        return img

    def coeff_to_values(self, coeff_n):
        coeff = copy.deepcopy(coeff_n)
        ndims = coeff[0].shape[0]
        nlevels = len(coeff)-2
        nbands = len(coeff[1])
        phase = []
        amplitude = []

        for x in range(1, nlevels+1):
            for n in range(nbands):
                coeff[x][n] = torch.view_as_complex(coeff[x][n])

        high_level = coeff[0].unsqueeze(1)
        low_level = coeff[-1].unsqueeze(1)

        for level in range(0, nlevels):
            phase_image = torch.stack([torch.imag(torch.log(coeff[level+1][band][d]))
                                      for d in range(ndims) for band in range(nbands)], -1)
            phase.append(phase_image.unsqueeze(0).permute(3, 0, 1, 2))
            amplitude_image = torch.stack([torch.abs(coeff[level+1][band][d])
                                          for d in range(ndims) for band in range(nbands)], -1)
            amplitude.append(amplitude_image.unsqueeze(0).permute(3, 0, 1, 2))

        values = DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phase,
            amplitude=amplitude
        )

        return values

    # input = [tensor, tensor, tensor]x10 ->
    # input = [[tensor]x4, [tensor]x4, [tensor]x4]x10

    def reorder(self, input, ndims):
        nbands = int(input[0].shape[0]/ndims)
        return [[input[i].reshape(nbands, ndims, input[i].shape[2], input[i].shape[3])[j]
        for j in range(int(input[0].shape[0]/ndims))] for i in range(len(input))]

    def values_to_coeff(self, values, coeff_r):
        # [3, 1, 256, 256]
        ndims, _, H, W = values.high_level.shape
        print(ndims)
        amplitude = self.reorder(values.amplitude, ndims)
        phase = self.reorder(values.phase, ndims)
        nlevels = len(phase)
        nbands = len(phase[0])
        print(nlevels, nbands)

        coeff = []
        high_level = values.high_level.squeeze(1)
        low_level = values.low_level.squeeze(1)
        coeff.append(high_level)

        print(len(phase)) # 10
        print(len(phase[0])) #4
        print(phase[0][0].shape) # [3, 256, 256]

        print(high_level.device, amplitude[0][0][0][0].device)
        #coeff.extend([[torch.stack(
            #[torch.complex(amplitude[0][level][band][d], torch.zeros(1, device=high_level.device))
            # * torch.exp(torch.complex(torch.zeros(1, device=high_level.device), phase[0][level][band][d]))
            # for d in range(0, ndims)],
            #dim=0).real
        #    for band in range(nbands)] for level in range(nlevels)])

        # Phase -> 10
        # 0 -> 4 list [3, 256, 256] angle
        # 1 -> 4 list [3, 182, 182]
        # 2 -> 4 list [3, 128, 128]

        # Amplitude -> 10
        # 0 -> 4 list [3, 256, 256] magn
        # 1 -> 4 list [3, 182, 182]
        # 2 -> 4 list [3, 128, 128]

        #1. cos(angl)*magn -> [3, 256, 256]
        #2. sin(angl)*magn -> [3, 256, 256]
        #3. stack -> [3, 256, 256, 2]

        # Result -> 10
        # 0 -> 4 list [3, 256, 256, 2]
        # 1 -> 4 list [3, 182, 182, 2]
        # 2 -> 4 list [3, 128, 128, 2]


        for level in range(nlevels): # 10
            res_in = []
            for x in range(nbands): # 4
                angle = phase[level][x]
                magnitude = amplitude[level][x]
                real = torch.cos(angle) * magnitude
                image = torch.sin(angle) * magnitude
                res_in.append(torch.stack((real, image), -1))
            coeff.append(res_in)

        coeff.append(low_level)

        print(coeff[1][0].shape, coeff_r[1][0].shape)

        print(f'coeff-div: {torch.max((coeff[1][0] - coeff_r[1][0]).abs())}')

        return coeff
