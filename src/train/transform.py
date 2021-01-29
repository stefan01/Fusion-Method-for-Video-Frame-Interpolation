from skimage import color
from PIL import Image
import torch


def rgb2lab(img: torch.Tensor):
    """ Transforms a batch of rgb images into lab color space. Dimensions: [B, C, H, W] """
    lab = color.rgb2lab(img.cpu().detach().permute(0, 2, 3, 1).numpy())
    lab[:,:,:,0] /= 100
    lab[:,:,:,1:] += 128
    lab[:,:,:,1:] /= 255
    lab = torch.tensor(lab).permute(0, 3, 1, 2)

    return lab

def lab2rgb(img: torch.Tensor):
    """ Transforms a batch of lab images into rgb color space. Dimensions: [B, C, H, W] """
    rgb = img.clone().cpu().detach().permute(0, 2, 3, 1).numpy()
    rgb[:,:,:,0] *= 100
    rgb[:,:,:,1:] *= 255
    rgb[:,:,:,1:] -= 128
    rgb = color.lab2rgb(rgb)
    rgb = torch.tensor(rgb).permute(0, 3, 1, 2)

    return rgb

def lab2rgb_single(img: torch.Tensor):
    """ Transforms a single lab image into rgb color space. Dimensions: [C, H, W] """
    rgb = img.clone().cpu().detach().permute(1, 2, 0).numpy()
    rgb[:,:,0] *= 100
    rgb[:,:,1:] *= 255
    rgb[:,:,1:] -= 128
    rgb = color.lab2rgb(rgb)
    rgb = torch.tensor(rgb).permute(2, 0, 1)

    return rgb