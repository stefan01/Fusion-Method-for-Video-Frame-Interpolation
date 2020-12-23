from numpy.lib.twodim_base import tri
from pyramid import Pyramid
from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader, dataloader
import numpy as np
import torch
from collections import namedtuple
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
from phase_net import PhaseNet
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")


device = utils.get_device()

pyr = Pyramid(
    height=12,
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)

# Set batch size to default value 32
batch_size = 32
#dataset = DBreader_Vimeo90k('./Trainset/vimeo/vimeo_triplet', random_crop=(256, 256))

img = Image.open('Testset/Clip1/000.png')
img2 = Image.open('Testset/Clip1/001.png')

#img.show()

img = TF.to_tensor(transforms.RandomCrop((256, 256))(img)).to(device)
img2 = TF.to_tensor(transforms.RandomCrop((256, 256))(img2)).to(device)

transforms.ToPILImage()(img.cpu()).show()

#plt.subplot(1, 2, 1)
#plt.imshow(img.cpu().squeeze(1).permute(1, 2, 0).numpy())

# Psi
vals1, coeff1 = pyr.filter(img)
vals2, _ = pyr.filter(img2)

phase_net = PhaseNet(pyr, device)
# TODO vals concatenation
# vals = (values1, value2)

# Phase Net -> 2 image Values
        # high level [3, 1, 256, 256] -> [6, 1, 256, 256]
        # low level [[3, 1, 8, 8]
        # phase 10
        # 0 [12, 1, 256, 256]
        # 1 [12, 1, 182, 182]
        # 2 [12, 1, 128, 128]
        # amplitude
        # 0 [12, 1, 256, 256]
        # 1 [12, 1, 182, 182]
        # 2 [12, 1, 128, 128]



DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)
# [12, 1, 24, 24]x2 -> [3, 4, 24, 24]x2 -> [3, 8, 24, 24]
def get_concat_layers(pyr, vals1, vals2):
    nbands = pyr.nbands
    #x.reshape(3, 4, 24, 24)
    #vals_amplitude = [torch.split(x, x.shape[1]/nbands, 0) for x in vals.amplitude]
    #vals2_amplitude = [torch.split(x, x.shape[0]/3, 0) for x in vals2.amplitude]

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


vals = get_concat_layers(pyr, vals1, vals2)
vals_r = phase_net(vals)

#    concat_layers = [torch.cat((vals.high_level, vals2.high_level))]

#    for idx in range(len(pyr.phase)-2):
#        phase = torch.cat((vals.phase[idx], vals2.phase[idx]))

#        amplitude = torch.cat((vals.amplitude[idx], vals2.ampliture[idx]))

#        concat_layers.append(torch.cat(phase, amplitude))

#    concat_layers.append(torch.cat((vals.low_level, vals2.low_level)))

#    return concat_layers

# concat_layers = get_concat_layers(pyr, vals, vals2)

# Phase Net -> 2 image Values
# high level [256, 256, 3]
# low level [8, 8, 3] -> [8, 8, 1], [8, 8, 1], [8, 8, 1]
# phase 10
    # 0 [256, 256, 2, 12]
    # 1 [182, 182, 2, 12]
    # 2 [128, 128, 2, 12]
    # 3 [90, 90, 2, 12]
    # 4 [64, 64, 2, 12]
    # 5 [46, 46, 2, 12]
    # 6 [32, 32, 2, 12]
    # 7 [22, 22, 2, 12]
    # 8 [16, 16, 2, 12]
    # 9 [12, 12, 2, 12]
# amplitude

# Psi^{-1}
img_r = pyr.inv_filter(vals_r, coeff1)

img_p = img_r.detach().cpu()
print(img_p.shape)

transforms.ToPILImage()(img_p).show()

print(img[0, 0, :20])
print(img_r[0, 0, :20])
print(img.shape, img_r.shape)
print('Endfehler', torch.max(img-img_r))

#plt.subplot(1, 2, 2)
#plt.imshow(img_r.cpu().squeeze(1).permute(1, 2, 0).numpy())
#plt.show()
