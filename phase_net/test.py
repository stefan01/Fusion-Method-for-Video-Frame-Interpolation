from numpy.lib.twodim_base import tri
from pyramid import Pyramid
from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader, dataloader
import numpy as np
import torch
from collections import namedtuple
from skimage import io, color
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
from scipy import ndimage


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
batch_size = 32
#dataset = DBreader_Vimeo90k('./Trainset/vimeo/vimeo_triplet', random_crop=(256, 256))

img = Image.open('../Testset/Clip1/000.png')
img2 = Image.open('../Testset/Clip1/001.png')

# img = TF.to_tensor(transforms.RandomCrop((256, 256))(img)).to(device)
# img2 = TF.to_tensor(transforms.RandomCrop((256, 256))(img2)).to(device)

cropped = color.rgb2lab(transforms.RandomCrop((256, 256))(Image.open('../Testset/Clip1/000.png')))

img = TF.to_tensor(cropped).to(device).float()
img2 = TF.to_tensor(cropped).to(device).float()

print(img.reshape(-1).max(0)[0])

back = ndimage.rotate(np.flip(img.numpy().T, 0), -90, reshape=False)
plt.imshow(color.lab2rgb(back))
plt.savefig("start.png")
# transforms.ToPILImage()(img.cpu()).show()

# Psi
print(img.shape)
vals1 = pyr.filter(img)
vals2 = pyr.filter(img2)
phase_net = PhaseNet(pyr, device)


vals = get_concat_layers(pyr, vals1, vals1)
vals_r = phase_net(vals)
img_r = pyr.inv_filter(vals_r)
img_p = img_r.detach().cpu()
# transforms.ToPILImage()(img_p).show()
step = ndimage.rotate(np.flip(img_p.numpy().T, 0), -90, reshape=False)
plt.imshow(color.lab2rgb(step))
plt.savefig("step.png")

optimizer = optim.Adam(phase_net.parameters(), lr=1e-3)
criterion = nn.L1Loss()

for epoch in range(2):
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
# transforms.ToPILImage()(img_p).show()
end_img = ndimage.rotate(np.flip(img_p.numpy().T, 0), -90, reshape=False)
plt.imshow(color.lab2rgb(end_img))
plt.savefig("result.png")
print("finished")

#end_img = Image.fromarray(color.lab2rgb(transforms.ToPILImage()(img_p)), "RGB")
#end_img.show()
