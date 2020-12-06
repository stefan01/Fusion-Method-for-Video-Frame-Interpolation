from pyramid import Pyramid
import numpy as np
import torch
from collections import namedtuple
from skimage import io
import matplotlib.pyplot as plt

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import warnings
warnings.filterwarnings("ignore")


device = utils.get_device()

pyr = Pyramid(
    height=12,
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)


img = io.imread('./Lena.png')  # Format has to be (N, 1, H, W)
img = torch.from_numpy(img).to(device).reshape(
    img.shape[-1], 1, img.shape[0], img.shape[1]).float()
img /= 255


im_batch_numpy = utils.load_image_batch(
    batch_size=4, image_file='./Lena.png')
im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)


plt.subplot(1, 2, 1)
plt.imshow(img.cpu().numpy().reshape(256, 256, 3))

# Psi
vals = pyr.filter(img)

# Psi^(-1)
img_r = pyr.inv_filter(vals)

plt.subplot(1, 2, 2)
plt.imshow(img_r.reshape(
    256, 256, 3).cpu().numpy())
plt.show()
