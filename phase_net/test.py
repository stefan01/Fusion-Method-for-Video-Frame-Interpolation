from pyramid import Pyramid
from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader
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

dataset = DBreader_Vimeo90k('./Trainset/vimeo/vimeo_triplet', random_crop=(256, 256))
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

print(len(dataset[0])) #3
print(dataset[0][0].shape) # [3, 256, 256]
print(len(dataset)) # 73191

img = dataset[0][0].to(device).unsqueeze(1)

plt.subplot(1, 2, 1)
plt.imshow(img.cpu().squeeze(1).permute(1, 2, 0).numpy())

# Psi
vals = pyr.filter(img)

# Psi^(-1)
img_r = pyr.inv_filter(vals)

plt.subplot(1, 2, 2)
plt.imshow(img_r.cpu().squeeze(1).permute(1, 2, 0).numpy())
plt.show()
