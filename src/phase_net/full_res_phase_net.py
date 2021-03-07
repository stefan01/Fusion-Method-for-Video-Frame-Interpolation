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
device_cpu = torch.device('cpu')

# Import images
img_1 = np.array(Image.open('Testset/Clip8/000.png'))
img_g_loaded = np.array(Image.open('Testset/Clip8/001.png'))
img_2 = np.array(Image.open('Testset/Clip8/002.png'))
shape_r = img_1.shape

# Normalize and pad images
img_1 = pad_img(img_1/255)
img_g = pad_img(img_g_loaded/255)
img_2 = pad_img(img_2/255)

# To tensors
img_1 = rgb2lab_single(torch.as_tensor(img_1).permute(2, 0, 1).float()).to(device)
img_g = rgb2lab_single(torch.as_tensor(img_g).permute(2, 0, 1).float()).to(device)
img_2 = rgb2lab_single(torch.as_tensor(img_2).permute(2, 0, 1).float()).to(device)

# RGB space
#img_1 = torch.as_tensor(img_1).permute(2, 0, 1).float().to(device)
img_g_loaded = torch.as_tensor(img_g_loaded).permute(2, 0, 1).float()/255
#img_2 = torch.as_tensor(img_2).permute(2, 0, 1).float().to(device)


# Build pyramid
pyr = Pyramid(
    height=calc_pyr_height(img_1),
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)

# Create PhaseNet
phase_net = PhaseNet(pyr, device)
phase_net.load_state_dict(torch.load('./src/phase_net/phase_net.pt'))
phase_net.eval()

result = []

# Predict per channel, so we save memory
for c in range(3):
    # Filter images and normalize
    vals_1 = pyr.filter(img_1[c].unsqueeze(0).float())
    vals_2 = pyr.filter(img_2[c].unsqueeze(0).float())
    vals_1_2 = get_concat_layers(pyr, vals_1, vals_2)
    vals_normalized = phase_net.normalize_vals(vals_1_2)

    # Delete all old values to free memory
    del vals_1
    del vals_2
    del vals_1_2
    torch.cuda.empty_cache()

    # Predict intermediate frame
    with torch.no_grad():
        vals_r = phase_net(vals_normalized)

    img_r = pyr.inv_filter(vals_r).detach().cpu()
    result.append(img_r)

# Put picture together
result = torch.cat(result, 0)
img_p_pad = lab2rgb_single(result)

img_p = img_p_pad[:, :shape_r[0], :shape_r[1]]

# Show frame
transforms.ToPILImage()(img_p).show()

# Show error
img_err = torch.abs(img_p - img_g_loaded)
img_err /= img_err.max()
img_err = img_err.mean(0)
transforms.ToPILImage()(img_err).show()

# Show uncertainty
vals_1 = pyr.filter(img_1.float())
vals_2 = pyr.filter(img_2.float())
vals_g = pyr.filter(img_g.float())
vals_p = pyr.filter(img_p_pad.to(device).float())

# Show high values
hl_1 = vals_1.high_level.squeeze(1).cpu()[:, :shape_r[0], :shape_r[1]]
hl_2 = vals_2.high_level.squeeze(1).cpu()[:, :shape_r[0], :shape_r[1]]
hl_p = vals_p.high_level.squeeze(1).cpu()[:, :shape_r[0], :shape_r[1]]
unc = hl_p - (hl_1+hl_2)
unc /= unc.max()
unc = unc.mean(0)
transforms.ToPILImage()(unc).show()
