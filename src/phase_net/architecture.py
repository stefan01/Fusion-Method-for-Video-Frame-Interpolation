import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from src.phase_net.core import PhaseNetCore
from src.train.utils import *
from src.train.transform import *
from src.train.pyramid import Pyramid
import warnings


class PhaseNet(nn.Module):
    """
    The Architecture of PhaseNet. The PhaseNetCore does not work with input images, but with pyramid decomposition.
    Therefore, the architecture converts the input images into pyramids and reconverts the predicted pyramid into an image.
    """

    def __init__(self, height: int, device: torch.device, num_img: int = 2, scale_factor: float = np.sqrt(2), nbands: int = 4):
        super(PhaseNet, self).__init__()

        # Constructor of PhaseNetCore
        self.core = PhaseNetCore(
            height, device, num_img=num_img, nbands=nbands)

        # Build pyramid
        self.pyr = Pyramid(
            height=height,
            nbands=nbands,
            scale_factor=scale_factor,
            device=device,
        )
        self.to(device)

    def load(self, path: str = './src/phase_net/phase_net.pt'):
        """ Load state dict of network from path. """
        self.core.load_state_dict(torch.load(path))

    def forward(self, img_batch: torch.tensor, high_level: bool = False, ada_pred: torch.tensor = None, m: int = None):
        # Combine images into one big batch and then create the values and separate
        vals_batch = self.pyr.filter(img_batch.float())
        vals_list = separate_vals(vals_batch, self.core.num_img)
        vals_target = None
        if m is not None:
            vals_target = vals_list[-1]
            vals_list = vals_list[:-1]
        vals_input = get_concat_layers_inf(self.pyr, vals_list)
        input = self.core.normalize_vals(vals_input)

        # Delete unnecessary vals
        del vals_batch
        del vals_list
        del vals_input
        torch.cuda.empty_cache()

        # Predicted intersected image of frame1 and frame2
        vals_pred = self.core(input, m)

        # Exchange parts for hierarchical training
        if m is not None:
            vals_pred = exchange_vals(
                vals_pred, vals_target,  0, calc_pyr_height(img_batch)-m)

        # If high_level is True, we use the high_level image of adacof
        if high_level:
            ada_pyr = self.pyr.filter(ada_pred)
            vals_pred.high_level[:] = ada_pyr.high_level

        # Transform output of the network back to an image -> inverse steerable pyramid
        prediction = self.pyr.inv_filter(vals_pred)

        return prediction, vals_pred, vals_target
