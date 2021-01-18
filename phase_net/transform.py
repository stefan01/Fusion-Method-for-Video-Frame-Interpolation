from skimage import color
from PIL import Image
import torch


def rgb2lab(x):
    lab = color.rgb2lab(img_1.permute(1, 2, 0).numpy())
    lab[:,:,0] /= 100
    lab[:,:,1:] += 128
    lab[:,:,1:] /= 255
    lab = torch.tensor(lab).permute(2, 0, 1)

    return lab

def lab2rgb(x):
    rgb = x.clone().permute(1, 2, 0).numpy()
    rgb[:,:,0] *= 100
    rgb[:,:,1:] *= 255
    rgb[:,:,1:] -= 128
    rgb = color.lab2rgb(rgb)
    rgb = torch.tensor(rgb).permute(2, 0, 1)

    return rgb