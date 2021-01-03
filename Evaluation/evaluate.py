import sys, os, glob, subprocess
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from piq import ssim, LPIPS, psnr
from PIL import Image
import matplotlib.pyplot as plt

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
def evaluate_dataset(dataset_path):
    print(f'Evaluating Dataset {dataset_path}')
    prediction_folder = sorted(glob.glob(f'{tmp_dir}/{dataset_path}/*.png'))
    target_folder = sorted(glob.glob(f'../Testset/{dataset_path}/*.png'))

    output_path = os.path.dirname(os.path.dirname(dataset_path)) + "visual_result"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eval_results = []

    it = range(1, len(prediction_folder)) 

    for i in tqdm(iterable=it, total=len(it)):
        # Load Images
        image_prediction = Image.open(prediction_folder[i])
        image_target = Image.open(target_folder[i])

        tensor_prediction = TF.to_tensor(image_prediction)
        tensor_target = TF.to_tensor(image_target)

        # Evaluate
        eval_result = evaluate_image(tensor_prediction, tensor_target)
        eval_results.append(eval_result)

        # draw images
        #draw_difference(prediction_folder[i], target_folder[i], output_path, eval_result[0], i)

    return eval_results


def draw_difference(pred_img, target_img, out_path, error, number):
    difference = torch.abs(target_img - pred_img)

    plt.subplot(131)
    plt.imshow(target_img)
    plt.title('Target Image')

    plt.subplot(132)
    plt.imshow(pred_img)
    plt.title('Predicted Image')

    plt.subplot(133)
    plt.imshow(difference)
    plt.title('Difference Image')

    name = f"img_{number}_{error}.png"
    plt.savefig(out_path + "/" + name)



testsets = ['Clip1', 'Clip2']
for testset in testsets:
    if(not os.path.isdir(f'{tmp_dir}/{testset}')):
        testset_path = f'../Testset/{testset}'
        interpolate_dataset(testset_path)

for testset in testsets:
    result = evaluate_dataset(testset)
    print(f'Result for {testset}: {result}')
