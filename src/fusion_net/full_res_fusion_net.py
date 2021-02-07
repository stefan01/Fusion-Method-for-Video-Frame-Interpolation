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
from src.fusion_net.fusion_net import FusionNet

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
img_1 = np.array(Image.open('Testset/lights/119.png'))
img_g = np.array(Image.open('Testset/lights/120.png'))
img_2 = np.array(Image.open('Testset/lights/121.png'))
shape_r = img_1.shape

with torch.no_grad():
    frame_out1, frame_out2, ada_pred, uncertainty_mask = adacof_model(
        torch.as_tensor(img_1).permute(2, 0, 1).float().unsqueeze(0).to(device)/255,
        torch.as_tensor(img_2).permute(2, 0, 1).float().unsqueeze(0).to(device)/255)
    frame_out1, frame_out2 = frame_out1.squeeze(0).permute(1, 2, 0).cpu().numpy(), frame_out2.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Normalize and pad images
img_1_pad = pad_img(img_1/255)
img_g_pad = pad_img(img_g/255)
img_2_pad = pad_img(img_2/255)
frame_out1_pad = pad_img(frame_out1)
frame_out2_pad = pad_img(frame_out2)

# To tensors
img_1 = rgb2lab_single(torch.as_tensor(img_1_pad).permute(2, 0, 1)).to(device).float()
img_g = rgb2lab_single(torch.as_tensor(img_g_pad).permute(2, 0, 1)).to(device).float()
img_2 = rgb2lab_single(torch.as_tensor(img_2_pad).permute(2, 0, 1)).to(device).float()
frame_1 = torch.as_tensor(frame_out1_pad).permute(2, 0, 1).to(device).float()
frame_2 = torch.as_tensor(frame_out2_pad).permute(2, 0, 1).to(device).float()

# Show frame
#transforms.ToPILImage()(img_1).show()
#transforms.ToPILImage()(frame_1).show()

# Build pyramid
pyr = Pyramid(
    height=calc_pyr_height(img_1),
    nbands=4,
    scale_factor=np.sqrt(2),
    device=device,
)

# Create FusionNet
fusion_net = PhaseNet(pyr, device, num_img=4)
fusion_net.load_state_dict(torch.load('./src/fusion_net/fusion_net1.pt'))
fusion_net.eval()

result = []

# Predict per channel, so we save memory
for c in range(3):
    imgs = torch.stack((img_1[c], img_2[c], frame_1[c], frame_2[c], img_g[c]), 0)

    # combine images into one big batch and then create the values and separate
    vals = pyr.filter(imgs)
    vals_list = separate_vals(vals, 5)
    vals_t = vals_list[-1]
    vals_inp = get_concat_layers_inf(pyr, vals_list[:-1])
    inp = fusion_net.normalize_vals(vals_inp)

    # Delete all old values to free memory
    del vals
    del vals_list
    del vals_t
    del vals_inp
    torch.cuda.empty_cache()

    # predicted intersected image of frame1 and frame2
    with torch.no_grad():
        vals_r = fusion_net(inp)

    img_r = pyr.inv_filter(vals_r).detach().cpu()
    result.append(img_r)

# Put picture together
result = torch.cat(result, 0)
img_p = lab2rgb_single(result)

img_p = img_p[:, :shape_r[0], :shape_r[1]]

print(shape_r)

#print(img_p.shape, ada_pred.shape, uncertainty_mask.shape)

# Fusion Net prediction
fusion_net3 = FusionNet().to(device)
fusion_net3.load_state_dict(torch.load('./src/fusion_net/fusion_net3.pt'))

phase_pred = img_p.to(device).unsqueeze(0).float()
ada_pred = ada_pred.to(device).float()
other = torch.cat([img_1[:, :shape_r[0], :shape_r[1]].reshape(-1, 3, shape_r[0], shape_r[1]), img_2[:, :shape_r[0], :shape_r[1]].reshape(-1, 3, shape_r[0], shape_r[1])], 1).float()

final_pred = fusion_net3(ada_pred, phase_pred, other, uncertainty_mask)
final_pred = final_pred.reshape(-1, final_pred.shape[2], final_pred.shape[3])


# Show frame
transforms.ToPILImage()(final_pred).show()
