import numpy as np
import torch
from collections import namedtuple
from skimage import io
from PIL import Image
from src.phase_net.phase_net import PhaseNet
from src.train.utils import *
from src.train.transform import *
from src.train.pyramid import Pyramid
import warnings
import torchvision.transforms as transforms


# Warnings and device
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')

# Import images
img_1 = np.array(Image.open('Testset/Clip11/044.png'))
img_g = np.array(Image.open('Testset/Clip11/045.png'))
img_2 = np.array(Image.open('Testset/Clip11/046.png'))

# Pad images
img_1 = pad_img(img_1)
img_g = pad_img(img_g)
img_2 = pad_img(img_2)

# To tensors
img_1 = torch.as_tensor(img_1).permute(2, 0, 1).float().to(device)
img_g = torch.as_tensor(img_g).permute(2, 0, 1).float().to(device)
img_2 = torch.as_tensor(img_2).permute(2, 0, 1).float().to(device)

print(img_1.shape)
print(calc_pyr_height(img_1))

# Build pyramid
pyr = Pyramid(
    height=calc_pyr_height(img_1),
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)

# Create PhaseNet
phase_net = PhaseNet(pyr, device).eval()
phase_net.load_state_dict(torch.load('./src/phase_net/phase_net.pt'))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
n = count_parameters(phase_net)
print(f'PhaseNet has {n} Parameters')

# Filter images and normalize
vals_1 = pyr.filter(img_1)

print(vals_1.high_level.shape)
transforms.ToPILImage()(vals_1.high_level.squeeze(1).cpu()).show()

vals_2 = pyr.filter(img_2)
vals_1_2 = get_concat_layers(pyr, vals_1, vals_2)
vals_normalized = phase_net.normalize_vals(vals_1_2)

# Delete all old values to free memory
del vals_2
del vals_1_2
torch.cuda.empty_cache()

# Predict intermediate frame
with torch.no_grad():
    vals_r = phase_net(vals_normalized)

print(f'CUDA MEMORY CONSUMPTION: {torch.cuda.memory_allocated(0)/1024**3} GB')

img_r = pyr.inv_filter(vals_r)
img_p = lab2rgb(img_r.detach().cpu())
print(img_p.shape)

# Show frame
transforms.ToPILImage()(img_p).show()
