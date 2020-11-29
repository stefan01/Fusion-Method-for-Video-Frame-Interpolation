from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import matplotlib.pyplot as plt
import torch
import cv2

device = torch.device('cuda:0')

# Load batch of images [N,1,H,W]
im_batch_numpy = cv2.imread(
    'E:/Documents/TU Darmstadt/1 M.Sc/Deep Learning in der CV/Fusion-Method-for-Video-Frame-Interpolation/Lenna.png', 0)

im_batch_torch = torch.tensor(
    im_batch_numpy, dtype=torch.float32).to(device)
im_batch_torch = im_batch_torch.reshape(
    1, 1, im_batch_numpy.shape[0], im_batch_numpy.shape[1])

print(im_batch_torch.shape)

# Initialize Complex Steerbale Pyramid
pyr = SCFpyr_PyTorch(height=5, nbands=4, scale_factor=2, device=device)

# Decompose entire batch of images
coeff = pyr.build(im_batch_torch)
print(coeff)

# Reconstruct batch of images again
im_batch_reconstructed = pyr.reconstruct(coeff)

plt.imshow(im_batch_reconstructed.reshape(512, 512).cpu().numpy(), cmap='gray')
plt.show()

# Visualization
coeff_single = utils.extract_from_batch(coeff, 0)
coeff_grid = utils.make_grid_coeff(coeff, normalize=True)
cv2.imshow('Complex Steerable Pyramid', coeff_grid)
cv2.waitKey(0)
