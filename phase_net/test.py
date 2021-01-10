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
img1 = Image.open('Testset/Clip1/000.png')
img_g = Image.open('Testset/Clip1/001.png')
img2 = Image.open('Testset/Clip1/002.png')

img1 = TF.to_tensor(transforms.RandomCrop((256, 256))(img1)).to(device)
img_g = TF.to_tensor(transforms.RandomCrop((256, 256))(img_g)).to(device)
img2 = TF.to_tensor(transforms.RandomCrop((256, 256))(img2)).to(device)

#transforms.ToPILImage()(img.cpu()).show()
pyr.filter(img1)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Filter both images together
start.record()
imgs = torch.cat((img1, img2, img_g), 0)
vals_test = pyr.filter(imgs)
#vals_test = transform_vals(vals_test)
end.record()
torch.cuda.synchronize()

print(start.elapsed_time(end)/1000, 'sec')

# Filter images alone
start.record()
vals1 = pyr.filter(img1)
vals_g = pyr.filter(img_g)
vals2 = pyr.filter(img2)
vals = get_concat_layers(pyr, vals1, vals2)
end.record()
torch.cuda.synchronize()

print(start.elapsed_time(end)/1000, 'sec')

seperate_vals(vals_test)
print(vals1.low_level.shape, vals1.high_level.shape)


phase_net = PhaseNet(pyr, device)
print(torch.norm(vals.low_level-vals_test.low_level, p=2))

exit()

vals_r = phase_net(vals)
img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
transforms.ToPILImage()(img_p).show()


optimizer = optim.Adam(phase_net.parameters(), lr=1e-3)
criterion = nn.L1Loss()

for epoch in range(200):
    optimizer.zero_grad()

    # Phase net image
    vals_r = phase_net(vals)
    img_r = pyr.inv_filter(vals_r)

    # Error
    loss = calc_loss(img, img, img, pyr, phase_net)
    loss.backward()

    optimizer.step()
    print(f'Epoch {epoch}  Loss {loss.item()}')


vals_r = phase_net(vals)

img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
transforms.ToPILImage()(img_p).show()
