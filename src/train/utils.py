from src.train.pyramid import Pyramid
import numpy as np
from skimage import io
from PIL import Image
import torch
from collections import namedtuple
import copy
from src.train.transform import *

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

    vals1_amplitude = [x.reshape(
        (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.amplitude]
    vals2_amplitude = [x.reshape(
        (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.amplitude]

    vals1_phase = [x.reshape(
        (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.phase]
    vals2_phase = [x.reshape(
        (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.phase]

    high_level = torch.cat((vals1.high_level, vals2.high_level), 1)
    low_level = torch.cat((vals1.low_level, vals2.low_level), 1)
    phase = [torch.cat((a, b), 1) for (a, b) in zip(vals1_phase, vals2_phase)]
    amplitude = [torch.cat((a, b), 1)
                 for (a, b) in zip(vals1_amplitude, vals2_amplitude)]

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
        vals_amplitude.append([x.reshape(
            (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.amplitude])
        # Concatenate Phases
        vals_phase.append([x.reshape(
            (int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in element.phase])

    high_level = torch.cat([ele.high_level for ele in vals_list], 1)
    low_level = torch.cat([ele.low_level for ele in vals_list], 1)
    phase = [torch.cat([ele[idx] for ele in vals_phase], 1)
             for idx in range(height-2)]
    amplitude = [torch.cat([ele[idx] for ele in vals_amplitude], 1)
                 for idx in range(height-2)]

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

    low_level = vals.low_level.reshape(
        num_input, -1, vals.low_level.shape[2], vals.low_level.shape[3])
    high_level = vals.high_level.reshape(
        num_input, -1, vals.high_level.shape[2], vals.high_level.shape[3])

    out_vals = []

    for i in range(num_input):
        # Low Level
        ll = low_level[i].unsqueeze(1)

        # High Level
        hl = high_level[i].unsqueeze(1)

        # Phase
        p = []  # Phase
        for phase in vals.phase:
            v = phase.reshape(num_input, -1, phase.shape[2], phase.shape[3])
            p.append(v[i].unsqueeze(1))

        # Amplitude
        a = []  # Amplitude
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

    print('Both values have a difference of:',
          ll_err + hl_err + ph_err + amp_err)


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
    pow2_size = (2**(np.ceil(np.log2(size)*2)/2)).astype(np.int)
    pad_size = max(pow2_size)-size

    pad_size = pad_size.astype(np.int)
    pad_img = np.pad(
        img, [(0, pad_size[0]), (0, pad_size[1]), (0, 0)], mode='constant')

    return pad_img


def calc_pyr_height(img):
    """ Calculates the height of the pyramid given an image. """
    size = img.shape[1:]
    return int(np.ceil((np.log2(min(size))-3)*2)+2)


def preprocess(img: torch.Tensor, device, normalized=True):
    """ Preprocesses an image, so it can be directly used for the pyramid decomposition.
        Input: (B,C, H, W), Output: (B*C, H, W) normalized, lab space, float, on device. """

    if isinstance(img, torch.Tensor) or isinstance(img, np.ndarray):
        img = torch.as_tensor(img)
        dims = len(img.shape)

        if dims != 4:
            print('Image shape has to be (B, C, H, W)!')
            return img

        # Get height and width of the training image
        h, w = img.shape[2:3+1]
        hw = (h, w)

        # Normalization
        if not normalized:
            img = img/255

        # Transform into lab space
        img = rgb2lab(img).reshape((-1,) + hw)

        # Float
        img = img.float()

        # Device
        img = img.to(device)

        return img
    else:
        print('Img has a not supported type:', img.__class__)
        return img
    
def combine_values(vals_list):
    """ Combines a list of values into one values """
    # High and low values
    ll_list = [s_val.low_level for s_val in vals_list]
    hl_list = [s_val.high_level for s_val in vals_list]

    ll = torch.cat(ll_list, 0)
    hl = torch.cat(hl_list, 0)
    
    # Amplitude and phase values
    a = []
    p = []
    
    for i in range(len(vals_list[0].phase)):
      a_list = [s_val.amplitude[i] for s_val in vals_list]
      p_list = [s_val.phase[i] for s_val in vals_list]
      
      a_entry = torch.cat(a_list, 0)
      p_entry = torch.cat(p_list, 0)

      a.append(a_entry)
      p.append(p_entry)
      
    
    # Values
    val = DecompValues(
      high_level=hl,
      low_level=ll,
      phase=p,
      amplitude=a
    )
    
    return val
        
def get_last_value_levels(vals, use_levels = 1):
    # Create values with only the highest frequencies non zero (but not high level!)

    # Get important information
    ll_shape = vals.low_level.shape
    hl_shape = vals.high_level.shape
    device = vals.low_level.device
    levels = len(vals.phase)
    
    ll = torch.zeros(ll_shape).to(device)
    hl = vals.high_level.clone()#torch.zeros(hl_shape).to(device)

    # Phase
    p = []
    for i, phase in enumerate(vals.phase):
        if i >= use_levels:
            p_shape = phase.shape
            p.append(torch.zeros(p_shape).to(device))
        else:
            p.append(phase.clone())

    # Amplitude
    a = []
    for i, ampl in enumerate(vals.amplitude):
        if i >= use_levels:
            a_shape = ampl.shape
            a.append(torch.zeros(a_shape).to(device))
        else:
            a.append(ampl.clone())

    # Values
    val = DecompValues(
      high_level=hl,
      low_level=ll,
      phase=p,
      amplitude=a
    )
    
    return val
    
def get_first_value_levels(vals, use_levels = 1):
    # Create values with only the highest frequencies non zero (but not high level!)

    # Get important information
    ll_shape = vals.low_level.shape
    hl_shape = vals.high_level.shape
    device = vals.low_level.device
    levels = len(vals.phase)
    
    ll = vals.low_level.clone()##torch.zeros(ll_shape).to(device)
    hl = torch.zeros(hl_shape).to(device)

    # Phase
    p = []
    for i, phase in enumerate(vals.phase):
        if i < levels-use_levels:
            p_shape = phase.shape
            p.append(torch.zeros(p_shape).to(device))
        else:
            p.append(phase.clone())

    # Amplitude
    a = []
    for i, ampl in enumerate(vals.amplitude):
        if i < levels-use_levels:
            a_shape = ampl.shape
            a.append(torch.zeros(a_shape).to(device))
        else:
            a.append(ampl.clone())

    # Values
    val = DecompValues(
      high_level=hl,
      low_level=ll,
      phase=p,
      amplitude=a
    )
    
    return val
    
def subtract_values(vals1, vals2):
    # Subtract two values and return the absolute error values

    ll = torch.abs(vals1.low_level - vals2.low_level)
    hl = torch.abs(vals1.high_level - vals2.high_level)

    # Phase
    p = []
    for p1, p2 in zip(vals1.phase, vals2.phase):
        p.append(torch.abs(p1 - p2))

    # Amplitude
    a = []
    for a1, a2 in zip(vals1.amplitude, vals2.amplitude):
        a.append(torch.abs(a1 - a2))

    # Values
    val = DecompValues(
      high_level=hl,
      low_level=ll,
      phase=p,
      amplitude=a
    )
    
    return val
