import numpy as np
import torch
from collections import namedtuple
from skimage import io

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
        vals = self.coeff_to_values(coeff)
        return vals

    def inv_filter(self, vals):
        """ Psi^{-1} filter """
        coeff = self.values_to_coeff(vals)
        img = self.pyr.reconstruct(coeff)

        return img

    def coeff_to_values(self, coeff):
        ndims = coeff[0].shape[0]
        nlevels = len(coeff)-2
        nbands = len(coeff[1])
        phase = []
        amplitude = []

        for x in range(1, nlevels+1):
            for n in range(nbands):
                coeff[x][n] = torch.view_as_complex(coeff[x][n])

        high_level = coeff[0].permute(1, 2, 0)
        low_level = coeff[-1].permute(1, 2, 0)

        for level in range(0, nlevels):
            phase.append(torch.stack([torch.imag(torch.log(coeff[level+1][band][d]))
                                      for d in range(ndims) for band in range(nbands)], -1))
            amplitude.append(torch.stack([torch.abs(coeff[level+1][band][d])
                                          for d in range(ndims) for band in range(nbands)], -1))

        values = DecompValues(
            high_level=high_level,
            low_level=low_level,
            phase=phase,
            amplitude=amplitude
        )

        return values

    def reorder(self, input, ndims):
        H, W, N, C = input[0].shape
        nbands = int(C/ndims)
        nlevels = len(input)
        elements = [[torch.squeeze(input[j][:, :, :, i], -1)
                     for i in range(nbands*ndims)] for j in range(nlevels)]
        output = []
        for d in range(ndims):
            level = []
            for l in range(nlevels):
                level.append(elements[l][nbands*d:(d+1)*nbands])
            output.append(level)

        return output

    def values_to_coeff(self, values):
        H, W, ndims = values.high_level.shape
        amplitude = self.reorder(values.amplitude, ndims)
        phase = self.reorder(values.phase, ndims)
        nlevels = len(phase[0])
        nbands = len(phase[0][0])

        coeff = []
        high_level = values.high_level.permute(2, 0, 1)
        low_level = values.low_level.permute(2, 0, 1)
        coeff.append(torch.squeeze(high_level, -1))
        coeff.extend([[torch.stack(
            [torch.complex(amplitude[d][level][band], torch.zeros(1, device=high_level.device))
             * torch.exp(torch.complex(torch.zeros(1, device=high_level.device), phase[d][level][band]))
             for d in range(0, ndims)],
            dim=0).real
            for band in range(nbands)] for level in range(nlevels)])
        coeff.append(torch.squeeze(low_level, -1))
        return coeff
