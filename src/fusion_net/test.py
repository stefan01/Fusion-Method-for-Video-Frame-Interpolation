from src.train.pyramid import Pyramid
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from PIL import Image
from src.fusion_net.fusion_net import FusionNet
import steerable.utils as utils
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from src.train.loss import *
import time
from src.train.transform import *
from src.train.utils import *

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

#transforms.ToPILImage()(img.cpu()).show()
pyr.filter(img_1)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Filter both images together
start.record()
imgs = torch.cat((img_1, img_2, img_1, img_2, img_g), 0)
vals_all = pyr.filter(imgs)
vals_ = separate_vals(vals_all, 5) # TODO for new images?

vals = get_concat_layers_inf(pyr, vals_[:-1])
exit()
end.record()
torch.cuda.synchronize()

print('Parallel filtering:', start.elapsed_time(end)/1000, 'sec')
print('Sequential filtering:', start.elapsed_time(end)/1000, 'sec')

# Create FusionNet
fusion_net = FusionNet(pyr, device, 4)

# Optimizer and Loss
optimizer = optim.Adam(fusion_net.parameters(), lr=1e-3)
criterion = nn.L1Loss()

vals_result = fusion_net(vals)

start.record()
for epoch in range(1):
    optimizer.zero_grad()

    # Phase net image
    vals_result = fusion_net(vals)
    img_r = pyr.inv_filter(vals_result)

    optimizer.step()

end.record()
torch.cuda.synchronize()

print('Fusion Time:', start.elapsed_time(end)/1000, 'sec')

img_result = pyr.inv_filter(vals_result)
img_p = img_r.detach().cpu()
# transforms.ToPILImage()(img_p).show()
