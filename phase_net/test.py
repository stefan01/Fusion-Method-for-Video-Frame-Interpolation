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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import *

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
print(img.shape)
vals1 = pyr.filter(img)
vals2 = pyr.filter(img2)

phase_net = PhaseNet(pyr, device)



DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

vals = get_concat_layers(pyr, vals1, vals1)

vals_r = phase_net(vals)
img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
transforms.ToPILImage()(img_p).show()


#print('Endfehler', torch.max(img-img_r))

optimizer = optim.Adam(phase_net.parameters(), lr=1e-3)

for epoch in range(500):
    optimizer.zero_grad()

    # Phase net image
    vals_r = phase_net(vals)
    img_r = pyr.inv_filter(vals_r)

    # Error
    #loss = torch.norm(img-img_r, p=2)
    loss = calc_loss(img, img, img, pyr, phase_net)
    loss.backward()

    optimizer.step()

    # print statistics
    running_loss = loss.item()
    print('[%d] loss: %.3f' %
            (epoch + 1, running_loss))

vals_r = phase_net(vals)
img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
transforms.ToPILImage()(img_p).show()

#plt.subplot(1, 2, 2)
#plt.imshow(img_r.cpu().squeeze(1).permute(1, 2, 0).numpy())
#plt.show()
