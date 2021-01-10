import sys, os, glob, subprocess
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from piq import ssim, LPIPS, psnr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

tmp_dir = 'tmp'
os.makedirs(tmp_dir, exist_ok=True)

#prediction = torch.rand(4, 3, 256, 256, requires_grad=True)
#target = torch.rand(4, 3, 256, 256)
#print(evaluate(prediction, target))

# Returns all measurements for the image
def evaluate_image(prediction, target):
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target)[0] # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    return np.array([ssim_measure.numpy(), lpips_measure.numpy(), psnr_measure.numpy()])

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

    plt.subplot(1, 3, 1)
    plt.imshow(target_img)
    plt.title('Target Image')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_img)
    plt.title('Predicted Image')

    plt.subplot(1, 3, 3)
    plt.imshow(difference)
    plt.title('Difference Image')

    name = f"img_{number}_{error}.png"
    plt.savefig(out_path + "/" + name)


def draw_measurements(datasets, datasets_results):
    avg_data = []
    var_data = []
    for dataset_results in datasets_results:
        avg = np.average(dataset_results, axis=0)
        avg_data.append(avg)

        var = np.var(dataset_results, axis=0)
        var_data.append(var)

    avg_data = np.concatenate(avg_data)
    var_data = np.concatenate(var_data)
    
    print(avg_data.shape)
    print(avg_data)
    print(var_data.shape)
    print(var_data)
    y_pos = np.arange(avg_data.shape[0])
    
    plt.subplot(1, 3, 1)
    plt.errorbar(y_pos, avg_data[:,0], var_data[:,0])
    plt.xticks(y_pos, datasets)
    plt.title('SSIM')

    plt.subplot(1, 3, 2)
    plt.errorbar(y_pos, avg_data[:,1], var_data[:,0])
    plt.xticks(y_pos, datasets)
    plt.title('LPIPS')

    plt.subplot(1, 3, 3)
    plt.errorbar(y_pos, avg_data[:,2], var_data[:,0])
    plt.xticks(y_pos, datasets)
    plt.title('PSNR')

    plt.show()

testsets = ['Clip1', 'Clip2', 'Clip3', 'Clip4', 'Clip5', 'Clip6', 'Clip7', 'Clip8', 'Clip9', 'Clip10', 'Clip11']

# Interpolate
for testset in testsets:
    if not os.path.isdir(f'{tmp_dir}/{testset}'):
        testset_path = f'../Testset/{testset}'
        interpolate_dataset(testset_path)

# Evaluate Results
results_np = []
for testset in testsets:
    result_path = f'result_{testset}.npy'
    if os.path.exists(result_path):
        result_np = np.load(result_path)
    else:
        result = evaluate_dataset(testset)
        print(f'Result for {testset}: {result}')
        result_np = np.array(result)
        np.save(f'result_{testset}.npy', result_np)
    results_np.append(result_np)

# Show Results
draw_measurements(testsets, results_np)
