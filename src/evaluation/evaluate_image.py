from piq import ssim, LPIPS, psnr
import numpy as np


def evaluate_image(prediction, target):
    """
    Returns all measurements for the image
    """
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target)[
        0]  # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    return np.array([ssim_measure.numpy(), lpips_measure.numpy(), psnr_measure.numpy()])
