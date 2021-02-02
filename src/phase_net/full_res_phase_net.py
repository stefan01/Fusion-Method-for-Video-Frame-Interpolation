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
img_1 = np.array(Image.open('counter_examples/basketball/00033.jpg'))
img_g = np.array(Image.open('counter_examples/basketball/00034.jpg'))
img_2 = np.array(Image.open('counter_examples/basketball/00035.jpg'))
shape_r = img_1.shape
print(shape_r)

# Normalize and pad images
img_1 = pad_img(img_1/255)
img_g = pad_img(img_g/255)
img_2 = pad_img(img_2/255)

#Image.fromarray((img_1*255).astype('uint8'), 'RGB').show()

# To tensors
img_1 = rgb2lab_single(torch.as_tensor(img_1).permute(2, 0, 1).float()).to(device)
img_g = rgb2lab_single(torch.as_tensor(img_g).permute(2, 0, 1).float()).to(device)
img_2 = rgb2lab_single(torch.as_tensor(img_2).permute(2, 0, 1).float()).to(device)


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
img_p = lab2rgb_single(result)

img_p = img_p[:, :shape_r[0], :shape_r[1]]

# Show frame
transforms.ToPILImage()(img_p).show()
