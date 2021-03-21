from piq import ssim, LPIPS, psnr
import numpy as np
import torch


def evaluate_image(prediction, target):
    """
    Returns all measurements for the image
    """
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target)[
        0]  # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    ssd_measure = torch.sqrt(torch.sum(torch.square(prediction - target)))

    return np.array([ssim_measure.numpy(), lpips_measure.numpy(), psnr_measure.numpy(), ssd_measure.numpy()])
