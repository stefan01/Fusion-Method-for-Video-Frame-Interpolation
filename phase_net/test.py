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
from phase_net import PhaseNet, PhaseNetBlock
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import *
from collections import namedtuple
import time

DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)

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
#batch_size = 32
#dataset = DBreader_Vimeo90k('./Trainset/vimeo/vimeo_triplet', random_crop=(256, 256))

# Import images
img_1 = Image.open('Testset/Clip1/000.png')
img_g = Image.open('Testset/Clip1/001.png')
img_2 = Image.open('Testset/Clip1/002.png')

img_1 = TF.to_tensor(transforms.RandomCrop((256, 256))(img_1)).to(device)
img_g = TF.to_tensor(transforms.RandomCrop((256, 256))(img_g)).to(device)
img_2 = TF.to_tensor(transforms.RandomCrop((256, 256))(img_2)).to(device)

v_test = pyr.filter(img_g)
v_test.high_level[:] = img_g.unsqueeze(1)
print(v_test.high_level)
img_test = pyr.inv_filter(v_test)
transforms.ToPILImage()(img_test.detach().cpu()).show()

#transforms.ToPILImage()(img.cpu()).show()
pyr.filter(img_1)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Filter both images together
start.record()
imgs = torch.cat((img_1, img_2, img_g), 0)
vals_all = pyr.filter(imgs)
vals_1, vals_2, vals_g = separate_vals(vals_all)
vals_1_2 = get_concat_layers(pyr, vals_1, vals_2)
end.record()
torch.cuda.synchronize()

print('Parallel filtering:', start.elapsed_time(end)/1000, 'sec')

# Filter images alone
start.record()
vals_1_old = pyr.filter(img_1)
vals_g_old = pyr.filter(img_g)
vals_2_old = pyr.filter(img_2)
vals_1_2_old = get_concat_layers(pyr, vals_1_old, vals_2_old)
end.record()
torch.cuda.synchronize()

print('Sequential filtering:', start.elapsed_time(end)/1000, 'sec')

# Create PhaseNet
phase_net = PhaseNet(pyr, device)

# Show image before training
#vals_r = phase_net(vals)
#img_r = pyr.inv_filter(vals_r)
#img_p = img_r.detach().cpu()
#transforms.ToPILImage()(img_p).show()

# Optimizer and Loss
optimizer = optim.Adam(phase_net.parameters(), lr=1e-3)
criterion = nn.L1Loss()

vals_r = phase_net(vals_1_2)

start.record()
for epoch in range(1):
    optimizer.zero_grad()

    # Phase net image
    vals_r = phase_net(vals_1_2)
    img_r = pyr.inv_filter(vals_r)

    # Error
    #loss = calc_loss(img_1, img_2, img_g, pyr, phase_net)
    #loss.backward()

    optimizer.step()
    #print(f'Epoch {epoch}  Loss {loss.item()}')
end.record()
torch.cuda.synchronize()

print('PhaseNet Time:', start.elapsed_time(end)/1000, 'sec')

vals_r = phase_net(vals)

img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
transforms.ToPILImage()(img_p).show()
