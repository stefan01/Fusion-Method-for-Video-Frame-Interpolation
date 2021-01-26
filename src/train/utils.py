from src.train.pyramid import Pyramid
import numpy as np
from skimage import io
from PIL import Image
import torch
from collections import namedtuple
import copy

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
    height = pyr.height

    vals_amplitude = []
    vals_phase = []

    for element in vals_list:
        # Concatenate Amplitude
        vals_amplitude.append([x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.amplitude])
        # Concatenate Phases
        vals_phase.append([x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.phase])

    high_level = torch.cat([ele.high_level for ele in vals_list], 1)
    low_level = torch.cat([ele.low_level for ele in vals_list], 1)
    phase = [torch.cat([ele[idx] for ele in vals_phase], 1) for idx in range(height-2)]
    amplitude = [torch.cat([ele[idx] for ele in vals_amplitude], 1) for idx in range(height-2)]

    return DecompValues(
        high_level=high_level,
        low_level=low_level,
        phase=phase[::-1],
        amplitude=amplitude[::-1]
        )

def separate_vals(vals, num_input):
    """ Seperates input image batches and ground truth batches. 
    
    vals<list>: [(frame1), <frame2>, ...., <ground_truth>]

    return: List of DecompValues
    """

    low_level = vals.low_level.reshape(num_input, -1, vals.low_level.shape[2], vals.low_level.shape[3])
    high_level = vals.high_level.reshape(num_input, -1, vals.high_level.shape[2], vals.high_level.shape[3])

    out_vals = []
    
    for i in range(num_input):
        # Low Level
        ll = low_level[i].unsqueeze(1)
        
        # High Level
        hl = high_level[i].unsqueeze(1)

        # Phase
        p = [] # Phase
        for phase in vals.phase:
            v = phase.reshape(num_input, -1, phase.shape[2], phase.shape[3])
            p.append(v[i].unsqueeze(1))

        # Amplitude
        a = [] # Amplitude
        for ampli in vals.amplitude:
            amp = ampli.reshape(num_input, -1, ampli.shape[2], ampli.shape[3])
            a.append(amp[i].unsqueeze(1))

        # Values
        val = DecompValues(
            high_level=hl,
            low_level=ll,
            phase=p,
            amplitude=a
        )
        
        out_vals.append(val)
    
    return out_vals
        
    #ll_1 = low_level[0].unsqueeze(1)
    #ll_2 = low_level[1].unsqueeze(1)
    #ll_g = low_level[2].unsqueeze(1)
        
    
    #hl_1 = high_level[0].unsqueeze(1)
    #hl_2 = high_level[1].unsqueeze(1)
    #hl_g = high_level[2].unsqueeze(1)

    # Phase
    #p_1 = []
    #p_2 = []
    #p_g = []

            #p_1.append(p[0].unsqueeze(1))
            #p_2.append(p[1].unsqueeze(1))
            #p_g.append(p[2].unsqueeze(1))

    # Amplitude
    #a = []
    #a_1 = []
    #a_2 = []
    #a_g = []
    #for i in range(num_input)
    #    for ampli in vals.amplitude:
    #        a = ampli.reshape(3, -1, ampli.shape[2], ampli.shape[3])
    #        a_1.append(a[0].unsqueeze(1))
    #        a_2.append(a[1].unsqueeze(1))
    #        a_g.append(a[2].unsqueeze(1))

    # Values
    """
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

    return val_1, val_2, val_g"""

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

def pad_img(img):
    """ Pads the image to power 2 number sizes. """
    size = img.shape[:2]
    pow2_size = 2**np.ceil(np.log2(size)).astype(np.int)
    pad_size = (max(pow2_size)-size)/2

    extra_0 = int(not pad_size[0].is_integer())
    extra_1 = int(not pad_size[1].is_integer())

    pad_size = pad_size.astype(np.int)
    pad_img = np.pad(img, [(pad_size[0], pad_size[0]+extra_0), (pad_size[1], pad_size[1]+extra_1), (0, 0)], mode='constant')

    return pad_img

def calc_pyr_height(img):
    """ Calculates the height of the pyramid given an image. """
    size = img.shape[1:]
    return int(np.ceil((np.log2(min(size))-3)*2)+2)
