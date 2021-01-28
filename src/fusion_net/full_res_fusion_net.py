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
from types import SimpleNamespace
from src.adacof.models import Model

# Warnings and device
warnings.filterwarnings("ignore")
device = torch.device('cuda:0')
device_cpu = torch.device('cpu')

# Adacof model
adacof_args = SimpleNamespace(
    gpu_id=0,
    model='src.fusion_net.fusion_adacofnet',
    kernel_size=5,
    dilation=1,
    config='src/adacof/checkpoint/kernelsize_5/config.txt'
)
adacof_model = Model(adacof_args)
adacof_model.eval()
checkpoint = torch.load('src/adacof/checkpoint/kernelsize_5/ckpt.pth', map_location=torch.device('cpu'))
adacof_model.load(checkpoint['state_dict'])

# Import images
img_1 = np.array(Image.open('Testset/Clip1/000.png'))
img_g = np.array(Image.open('Testset/Clip1/001.png'))
img_2 = np.array(Image.open('Testset/Clip1/002.png'))
shape_r = img_1.shape

with torch.no_grad():
    frame_out1, frame_out2 = adacof_model(
        torch.as_tensor(img_1).permute(2, 0, 1).float().unsqueeze(0).to(device)/255,
        torch.as_tensor(img_2).permute(2, 0, 1).float().unsqueeze(0).to(device)/255)
    frame_out1, frame_out2 = frame_out1.squeeze(0), frame_out2.squeeze(0)
    print(img_1.shape, frame_out1.shape)
    exit()

# Normalize and pad images
img_1_pad = pad_img(img_1/255)
img_g_pad = pad_img(img_g/255)
img_2_pad = pad_img(img_2/255)

# To tensors
img_1 = rgb2lab(torch.as_tensor(img_1).permute(2, 0, 1).float()).to(device)
img_g = rgb2lab(torch.as_tensor(img_g).permute(2, 0, 1).float()).to(device)
img_2 = rgb2lab(torch.as_tensor(img_2).permute(2, 0, 1).float()).to(device)


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
    vals_1 = pyr.filter(img_1[c].unsqueeze(0))
    vals_2 = pyr.filter(img_2[c].unsqueeze(0))
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
img_p = lab2rgb(result)

img_p = img_p[:, :shape_r[0], :shape_r[1]]

# Show frame
transforms.ToPILImage()(img_p).show()
