from pyramid import Pyramid
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from phase_net import PhaseNet
import torch
from collections import namedtuple
#from trainer import get_concat_layers

def calc_loss(img1, img2, img_g, pyr, phase_net, weighting_factor=0.1):
    vals1 = pyr.filter(img1)
    vals2 = pyr.filter(img2)
    vals_g = pyr.filter(img_g)

    combined_vals = get_concat_layers(pyr, vals1, vals2)
    vals_r = phase_net(combined_vals)
    img_r = pyr.inv_filter(vals_r)

    phase_losses = []

    for (phase_r, phase_g) in zip(vals_r.phase, vals_g.phase):
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_losses.append(torch.norm(delta_psi, 1))

    phase_loss = torch.stack(phase_losses, 0).sum(0)

    l_1 = torch.norm(img_g-img_r, p=1)

    return l_1 + weighting_factor * phase_loss

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

def get_concat_layers(pyr, vals1, vals2):
    nbands = pyr.nbands

    vals1_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.amplitude]
    vals2_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.amplitude]

    vals1_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.phase]
    vals2_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.phase]

    high_level = torch.cat((vals1.high_level, vals2.high_level), 1)
    low_level = torch.cat((vals1.low_level, vals2.low_level), 1)
    phase = [torch.cat((a, b), 1) for (a, b) in zip(vals1_phase, vals2_phase)]
    amplitude = [torch.cat((a, b), 1) for (a, b) in zip(vals1_amplitude, vals2_amplitude)]

    return DecompValues(
        high_level=high_level,
        low_level=low_level,
        phase=phase[::-1],
        amplitude=amplitude[::-1]
        )