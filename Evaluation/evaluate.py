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
    subprocess.run([
        f'{sys.executable}',
        '../AdaCoF/interpolate_twoframe.py',
        f'--first_frame', a,
        f'--second_frame', b,
        f'--output_frame', output,
        '--checkpoint', '../AdaCoF/checkpoint/kernelsize_5/ckpt.pth',
        '--config', '../AdaCoF/checkpoint/kernelsize_5/config.txt'])

# Interpolates triples of images
# from a dataset (list of filenames)
def interpolate_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path)
    print(f'Interpolating Dataset {dataset_name}')
    dataset = sorted(glob.glob(f'{dataset_path}/*.png'))
    it = range(0, len(dataset)-2)
    print(dataset_path)
    for i in tqdm(iterable=it, total=len(it)):
        interpolated_filename = os.path.basename(dataset[i+1])
        output_path = f'{tmp_dir}/{dataset_name}/{interpolated_filename}'
        os.makedirs(f'{tmp_dir}/{dataset_name}', exist_ok=True)
        interpolate(dataset[i], dataset[i+2], output_path)

# Evaluates a dataset.
# Takes interpolated images a and c
# and compares the result with b
def evaluate_datasets(dataset_path):
    #predictionFolder = datasets.ImageFolder(tmp_dir, transform=transforms.ToTensor)
    #targetFolder = datasets.ImageFolder(dataset_path, transform=transforms.ToTensor)

    #predictionsLoader = torch.utils.data.DataLoader(predictionFolder)
    #targetsLoader = torch.utils.data.DataLoader(targetFolder)

    #print(len(predictionsLoader))
    #print(len(targetsLoader))

    #predictions = iter(predictionsLoader)
    #targets = iter(targetsLoader)

    # Skip first image
    #next(targets)

    #results = []

    #it = range(1, len(targets))
    #it = zip(predictions, targets)

    #for (prediction, target) in tqdm(iterable=it, total=len(predictionsLoader)):
    #    result = evaluate_image(prediction, target)
    #    results.append(result)

    prediction_folder = glob.glob(f'{tmp_dir}/*.png')
    target_folder = glob.glob(f'{dataset_path}/*.png')

    

    return results


testsets = ['Clip1', 'Clip2']
for testset in testsets:
    if(not os.path.isdir(f'{tmp_dir}/{testset}')):
        testset_path = f'../Testset/{testset}'
        interpolate_dataset(testset_path)

result = evaluate_datasets('../Testset/')
print(f'Result: {result}')