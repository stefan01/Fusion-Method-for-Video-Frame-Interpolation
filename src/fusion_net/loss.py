from pyramid import Pyramid
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
from collections import namedtuple
import copy

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

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

def get_concat_layers(pyr, vals1, vals2):
    """ Combines two values and transforms them so the PhaseNet can use them more easily. """
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

def get_concat_layers_inf(pyr, vals_list):
    """ Combines values and transforms them so the PhaseNet can use them more easily. 

    vals_list: List of DecompValus
    
    returns: Concatenated DecompValues
    """
    nbands = pyr.nbands

    vals1 = vals_list[0]
    vals2 = vals_list[2]

    vals_amplitude = []
    vals_phase = []

    for element in vals_list:
        # Concatenate Amplitude
        vals_amplitude.append([x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.amplitude])
        # Concatenate Phases
        vals_phase.append([x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.phase])

    # Concatenate Amplitude
    vals1_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.amplitude]
    vals2_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.amplitude]
    # Concatenate Phases
    vals1_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.phase]
    vals2_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.phase]

    print(vals1.low_level.shape)
    print(vals2.low_level.shape)

    high_level = torch.cat([ele.high_level for ele in vals_list], 1)
    low_level = torch.cat([ele.low_level for ele in vals_list], 1)
    phase = [torch.cat((a, b), 1) for (a, b) in zip(vals1_phase, vals2_phase)]
    amplitude = [torch.cat((a, b), 1) for (a, b) in zip(vals1_amplitude, vals2_amplitude)]

    phase_test = []
    for i in range(len(vals_list)):
        

    return DecompValues(
        high_level=high_level,
        low_level=low_level,
        phase=phase[::-1],
        amplitude=amplitude[::-1]
        )

def separate_vals(vals):
    """ Seperates input image batches and ground truth batches. """
    # Low level
    low_level = vals.low_level.reshape(3, -1, vals.low_level.shape[2], vals.low_level.shape[3])
    ll_1 = low_level[0].unsqueeze(1)
    ll_2 = low_level[1].unsqueeze(1)
    ll_g = low_level[2].unsqueeze(1)

    # High level
    high_level = vals.high_level.reshape(3, -1, vals.high_level.shape[2], vals.high_level.shape[3])
    hl_1 = high_level[0].unsqueeze(1)
    hl_2 = high_level[1].unsqueeze(1)
    hl_g = high_level[2].unsqueeze(1)

    # Phase
    p_1 = []
    p_2 = []
    p_g = []
    for phase in vals.phase:
        p = phase.reshape(3, -1, phase.shape[2], phase.shape[3])
        p_1.append(p[0].unsqueeze(1))
        p_2.append(p[1].unsqueeze(1))
        p_g.append(p[2].unsqueeze(1))

    # Amplitude
    a_1 = []
    a_2 = []
    a_g = []
    for ampli in vals.amplitude:
        a = ampli.reshape(3, -1, ampli.shape[2], ampli.shape[3])
        a_1.append(a[0].unsqueeze(1))
        a_2.append(a[1].unsqueeze(1))
        a_g.append(a[2].unsqueeze(1))

    # Values
    val_1 = DecompValues(
        high_level=hl_1,
        low_level=ll_1,
        phase=p_1,
        amplitude=a_1
        )
    val_2 = DecompValues(
        high_level=hl_2,
        low_level=ll_2,
        phase=p_2,
        amplitude=a_2
        )
    val_g = DecompValues(
        high_level=hl_g,
        low_level=ll_g,
        phase=p_g,
        amplitude=a_g
        )

    return val_1, val_2, val_g

def compare_vals(val1, val2, p=1):
    """ Compares two values on equality. """
    ll_err = torch.norm(val1.low_level-val2.low_level, p=p).item()
    hl_err = torch.norm(val1.high_level-val2.high_level, p=p).item()
    ph_err = 0
    for i in range(len(val1.phase)):
        ph_err += torch.norm(val1.phase[i]-val2.phase[i], p=p).item()
    amp_err = 0
    for i in range(len(val1.amplitude)):
        torch.norm(val1.amplitude[i]-val2.amplitude[i], p=p).item()

    print('Both values have a difference of:', ll_err + hl_err + ph_err + amp_err)

def exchange_vals(val_base, val_changer, start, end):
    """ Exchanges values from the changer to the base. The levels from start to end are exchanged. """
    #val_base.high_level[:] = val_changer.high_level[:]
    for level in range(start, end):
        val_base.phase[level] = val_changer.phase[level]
        val_base.amplitude[level] = val_changer.amplitude[level]

    return val_base
