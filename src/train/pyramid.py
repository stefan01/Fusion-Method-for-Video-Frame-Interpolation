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
        coeff = self.pyr.build(img.unsqueeze(1))
        vals = self.coeff_to_values(coeff)
        return vals

    def inv_filter(self, vals):
        """ Psi^{-1} filter """
        coeff = self.values_to_coeff(vals)
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

    def reorder(self, input, ndims):
        nbands = int(input[0].shape[0]/ndims)
        return [[input[i].reshape(ndims, nbands, input[i].shape[2], input[i].shape[3])[:, j]
        for j in range(int(input[0].shape[0]/ndims))] for i in range(len(input))]

    def values_to_coeff(self, values):
        ndims, _, H, W = values.high_level.shape

        # print(values.amplitude[0])

        # reorder amplitude and phase elements to list with list of 4 tensors for each level
        amplitude = self.reorder(values.amplitude, ndims)
        phase = self.reorder(values.phase, ndims)
        nlevels = len(phase)
        nbands = len(phase[0])

        coeff = []
        high_level = values.high_level.squeeze(1)
        low_level = values.low_level.squeeze(1)
        coeff.append(high_level)

        # transform and stack phase and amplitude to real and complex dimension
        for level in range(nlevels):
            res_in = []
            for x in range(nbands):
                angle = phase[level][x]
                magnitude = amplitude[level][x]
                real = torch.cos(angle) * magnitude
                image = torch.sin(angle) * magnitude
                res_in.append(torch.stack((real, image), -1))
            coeff.append(res_in)

        coeff.append(low_level)

        return coeff
