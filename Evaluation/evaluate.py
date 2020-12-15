import sys, os, glob, subprocess
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from piq import ssim, LPIPS, psnr

tmp_dir = 'tmp'
os.makedirs(tmp_dir, exist_ok=True)

#prediction = torch.rand(4, 3, 256, 256, requires_grad=True)
#target = torch.rand(4, 3, 256, 256)
#print(evaluate(prediction, target))

# Returns all measurements for the image
def evaluate_image(prediction, target):
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target) # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    return (ssim_measure, lpips_measure, psnr_measure)

# Interpolates image a and b (file paths)
# and saves the result at output
def interpolate(a, b, output):
    print(f'Interpolating {a} and {b} to {output}')
    subprocess.run(
        f'{sys.executable} ' \
        '../AdaCoF/interpolate_twoframe.py ' \
        '--first_frame {a} ' \
        '--second_frame {b} ' \
        '--output_frame {output} ' \
        '--checkpoint ../AdaCoF/checkpoint/kernelsize_5/ckpt.pth ' \
        '--config ../AdaCoF/checkpoint/kernelsize_5/config.txt')

# Interpolates triples of images
# from a dataset (list of filenames)
def interpolate_dataset(dataset_path):
    print(f'Interpolating Dataset {dataset_path}')
    dataset = glob.glob(f'{dataset_path}/*.png')
    it = range(0, len(dataset)-2)
    print(dataset_path)
    for i in tqdm(iterable=it, total=len(it)):
        interpolated_filename = os.path.splitext(dataset[i+1])[1]
        interpolate(dataset[i], dataset[i+2], f'{tmp_dir}/{interpolated_filename}')

# Evaluates a dataset.
# Takes interpolated images a and c
# and compares the result with b
def evaluate_dataset(dataset_path):
    prediction = datasets.ImageFolder(tmp_dir)
    target = datasets.ImageFolder(dataset_path)
    return evaluate_image(prediction, target)


testsets = ['Clip1', 'Clip2']
for testset in testsets:
    testset_path = f'../Testset/{testset}'
    interpolate_dataset(testset_path)
    result = evaluate_dataset(testset_path)
    print(f'Result: {result}')