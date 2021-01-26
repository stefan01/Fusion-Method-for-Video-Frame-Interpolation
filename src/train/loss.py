from src.train.pyramid import Pyramid
import numpy as np
from skimage import io
from PIL import Image
import torch
from collections import namedtuple
import copy
from src.train.utils import *

def calc_loss(img1, img2, img_g, pyr, phase_net, weighting_factor=0.1):
    vals1 = pyr.filter(img1)
    vals2 = pyr.filter(img2)
    vals_g = pyr.filter(img_g)

    combined_vals = get_concat_layers(pyr, vals1, vals2)
    vals_r = phase_net(combined_vals)
    img_r = pyr.inv_filter(vals_r)

    low_level_loss = torch.norm(vals_r.low_level - vals_g.low_level, p=1)

    phase_losses = []

    for (phase_r, phase_g) in zip(vals_r.phase, vals_g.phase):
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_losses.append(torch.norm(delta_psi, 1))

    phase_loss = torch.stack(phase_losses, 0).sum(0)

    l_1 = torch.norm(img_g-img_r, p=1)

    return l_1 + weighting_factor * phase_loss + low_level_loss
