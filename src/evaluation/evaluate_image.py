from piq import ssim, LPIPS, psnr
import numpy as np
import torch
import src.fusion_net.interpolate_twoframe as fusion_interp


def evaluate_image(prediction_im, target_im):
    """
    Returns all measurements for the image
    """
    prediction = fusion_interp.crop_center(
        prediction_im.permute(1, 2, 0), 512, 512).permute(2, 0, 1)
    target = fusion_interp.crop_center(
        target_im.permute(1, 2, 0), 512, 512).permute(2, 0, 1)

    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target)[
        0]  # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    ssd_measure = torch.sqrt(torch.sum(torch.square(prediction - target)))

    l1_measure = torch.sum(prediction - target)

    mse_measure = torch.mean(prediction - target)

    variance_measure = torch.var(prediction - target)

    return np.array([ssim_measure.numpy(), lpips_measure.numpy(), psnr_measure.numpy(), ssd_measure.numpy(), l1_measure.numpy(), mse_measure.numpy(), variance_measure.numpy()])
