import numpy as np
import torch
from collections import namedtuple
import code
from skimage import io
import matplotlib.pyplot as plt

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import warnings
warnings.filterwarnings("ignore")

device = utils.get_device()

pyr = SCFpyr_PyTorch(
    height=12,
    nbands=4,
    # sqrt(2) not working correctly. Pyramid still uses 2 instead.
    scale_factor=np.sqrt(2),
    device=device,
)

img = io.imread('./Lena.png')  # Format has to be (N, 1, H, W)
img = torch.from_numpy(img).to(device).reshape(
    img.shape[-1], 1, img.shape[0], img.shape[1]).float()
img /= 255


im_batch_numpy = utils.load_image_batch(
    batch_size=4, image_file='./Lena.png')
im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)


DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)


"""def extract_from_batch(coeff_batch, example_idx=0):
    if not isinstance(coeff_batch, list):
        raise ValueError('Batch of coefficients must be a list')
    coeff = []  # coefficient for single example
    for coeff_level in coeff_batch:
        if isinstance(coeff_level, torch.Tensor):
            # Low- or High-Pass
            coeff_level_numpy = coeff_level[example_idx]
            coeff.append(coeff_level_numpy)
        elif isinstance(coeff_level, list):
            coeff_orientations_numpy = []
            for coeff_orientation in coeff_level:
                if isinstance(coeff_orientation, torch.Tensor):
                    coeff_orientation_numpy = coeff_orientation[example_idx]
                    coeff_orientation_numpy = coeff_orientation_numpy[:,
                                                                      :, 0] + 1j*coeff_orientation_numpy[:, :, 1]
                coeff_orientations_numpy.append(coeff_orientation_numpy)
            coeff.append(coeff_orientations_numpy)
    return coeff


coeff = extract_from_batch(coeff, 0)"""

# code.interact(local=locals())


def coeff_to_values(coeff):
    ndims = coeff[0].shape[0]   # number of pyramids
    nlevels = len(coeff)-2   # without first/last lvl, so 5-2 = 3 lvl
    nbands = len(coeff[1])   # 4 orientations
    phase = []
    amplitude = []

    high_level = coeff[0].permute(1, 2, 0)  # (H, W, ndims)
    low_level = coeff[-1].permute(1, 2, 0)  # (H, W, ndims)

    for level in range(0, nlevels):
        phase.append(torch.stack([torch.imag(torch.log(coeff[level+1][band][d].type(torch.complex64)))
                                  for d in range(ndims) for band in range(nbands)], -1))
        amplitude.append(torch.stack([torch.abs(coeff[level+1][band][d])
                                      for d in range(ndims) for band in range(nbands)], -1))
    print(phase[0].shape, amplitude[0].shape)

    values = DecompValues(
        high_level=high_level,
        low_level=low_level,
        phase=phase,
        amplitude=amplitude
    )

    return values


def reorder(input, ndims):
    H, W, N, C = input[0].shape
    nbands = int(C/ndims)
    nlevels = len(input)
    elements = [[torch.squeeze(input[j][:, :, :, i], -1)
                 for i in range(nbands*ndims)] for j in range(nlevels)]
    output = []
    for d in range(ndims):
        level = []
        for l in range(nlevels):
            level.append(elements[l][nbands*d:(d+1)*nbands])
        output.append(level)

    return output


def values_to_coeff(values):
    H, W, ndims = values.high_level.shape
    amplitude = reorder(values.amplitude, ndims)
    phase = reorder(values.phase, ndims)
    nlevels = len(phase[0])
    nbands = len(phase[0][0])

    coeff = []
    high_level = values.high_level.permute(2, 0, 1)
    low_level = values.low_level.permute(2, 0, 1)
    coeff.append(torch.squeeze(high_level, -1))
    coeff.extend([[torch.stack(
        [torch.complex(amplitude[d][level][band], torch.zeros(1, device=high_level.device))
         * torch.exp(torch.complex(torch.zeros(1, device=high_level.device), phase[d][level][band]))
         for d in range(0, ndims)],
        dim=0).real
        for band in range(nbands)] for level in range(nlevels)])
    coeff.append(torch.squeeze(low_level, -1))
    return coeff


plt.subplot(1, 2, 1)
plt.imshow(img.cpu().numpy().reshape(256, 256, 3))

# Psi
coeff = pyr.build(img)
vals = coeff_to_values(coeff)

# Psi^(-1)
coeff_r = values_to_coeff(vals)
im_batch_reconstructed = pyr.reconstruct(coeff_r)

print(vals.high_level.shape)

for i in range(1, 12):
    print(coeff[i][0].shape)
# print(len(coeff))    # 5
# print(coeff[0].shape)  # (N, 200, 200)
# print(coeff[1][0].shape)  # 4 List (N, 200, 200, 2)
# print(len(coeff[2]))
# print(len(coeff[3]))
# print(coeff[4].shape)  # (N, 25, 25)

#print(((coeff_r[3][1] - coeff[3][1]).abs() < 0.00001).all().item())
print(len(coeff_r))   # 5
print(len(coeff_r[1]))  # 4
print(coeff_r[1][0].shape)  # 4 List (200, 200, 2)

plt.subplot(1, 2, 2)
plt.imshow(im_batch_reconstructed.reshape(
    256, 256, 3).cpu().numpy())
plt.show()
