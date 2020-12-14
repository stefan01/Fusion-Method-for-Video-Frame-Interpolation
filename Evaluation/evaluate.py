import torch
from piq import ssim, LPIPS, psnr

def evaluate(prediction, target):
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target) # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    return (ssim_measure, lpips_measure, psnr_measure)

prediction = torch.rand(4, 3, 256, 256, requires_grad=True)
target = torch.rand(4, 3, 256, 256)
print(evaluate(prediction, target))