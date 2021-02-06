import sys, os, glob, subprocess
from matplotlib import image
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import cv2
from piq import ssim, LPIPS, psnr
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
import random
import argparse
from datetime import datetime

import src.adacof.interpolate_twoframe as adacof_interp
import src.phase_net.interpolate_twoframe as phasenet_interp
import src.fusion_net.interpolate_twoframe as fusion_interp

parser = argparse.ArgumentParser(description='Evaluation')

# Evaluation Parameters
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--adacof', action='store_true')
parser.add_argument('--phase', action='store_true')
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--base_dir', type=str, default=os.path.join('Evaluation', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
parser.add_argument('--img_output', type=str, default='interpolated')
parser.add_argument('--max_num', type=int, default=10)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--test_sets', type=str, nargs='+', 
    default=['airboard_1', 'airplane_landing', 'airtable_3', 'basketball_1', 'water_ski_2', 'yoyo', \
            'MODE_SH0280', 'MODE_SH0440', 'MODE_SH0450', 'MODE_SH0740', 'MODE_SH0780', 'MODE_SH1010', 'MODE_SH1270', \
            'Flashlight', 'firework', 'lights', 'sun'])

# Adacof Parameters
parser.add_argument('--adacof_model', type=str, default='src.adacof.models.adacofnet')
parser.add_argument('--adacof_checkpoint', type=str, default='./src/adacof/checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--adacof_config', type=str, default='./src/adacof/checkpoint/kernelsize_5/config.txt')
parser.add_argument('--adacof_kernel_size', type=int, default=5)
parser.add_argument('--adacof_dilation', type=int, default=1)

# Phasenet Parameters
parser.add_argument('--phasenet_checkpoint', type=str, default='./src/phase_net/phase_net.pt')
parser.add_argument('--phasenet_replace_high_level', action='store_true')

# Fusion Parameters
parser.add_argument('--fusion_checkpoint', type=str, default='./src/fusion_net/fusion_net.pt')
parser.add_argument('--fusion_adacof_model', type=str, default='src.fusion_net.fusion_adacofnet')
parser.add_argument('--fusion_model', type=int, default=1)

# Returns all measurements for the image
def evaluate_image(prediction, target):
    ssim_measure = ssim(prediction, target)
    lpips_measure = LPIPS(reduction='none')(prediction, target)[0] # Seems to be only available as loss function
    psnr_measure = psnr(prediction, target)

    return np.array([ssim_measure.numpy(), lpips_measure.numpy(), psnr_measure.numpy()])

# Interpolates image a and b (file paths)
# using adacof and saves the result at output
def interpolate_adacof(args, a, b, output):
    if not os.path.exists(output):
        print('Interpolating {} and {} to {} with adacof'.format(a, b, output))
        with torch.no_grad():
            adacof_interp.interp(SimpleNamespace(
                gpu_id=args.gpu_id,
                model=args.adacof_model,
                kernel_size=args.adacof_kernel_size,
                dilation=args.adacof_dilation,
                first_frame=a,
                second_frame=b,
                output_frame=output,
                checkpoint=args.adacof_checkpoint,
                config=args.adacof_config
            ))
        torch.cuda.empty_cache()
    

def interpolate_phasenet(args, a, b, output):
    if not os.path.exists(output):
        print('Interpolating {} and {} to {} with phasenet'.format(a, b, output))
        with torch.no_grad():
            phasenet_interp.interp(SimpleNamespace(
                gpu_id=args.gpu_id,
                first_frame=a,
                second_frame=b,
                output_frame=output,
                checkpoint=args.phasenet_checkpoint,
                high_level=args.phasenet_replace_high_level
            ))
        torch.cuda.empty_cache()

def interpolate_fusion(args, a, b, output):
    if not os.path.exists(output):
        print('Interpolating {} and {} to {} with fusion method'.format(a, b, output))
        with torch.no_grad():
            fusion_interp.interp(SimpleNamespace(
                gpu_id=args.gpu_id,
                adacof_model=args.fusion_adacof_model,
                adacof_kernel_size=args.adacof_kernel_size,
                adacof_dilation=args.adacof_dilation,
                first_frame=a,
                second_frame=b,
                output_frame=output,
                adacof_checkpoint=args.adacof_checkpoint,
                adacof_config=args.adacof_config,
                checkpoint=args.fusion_checkpoint,
                model=args.fusion_model
            ))
        torch.cuda.empty_cache()

# Interpolates triples of images
# from a dataset (list of filenames)
def interpolate_dataset(args, dataset_path, max_num=None):
    dataset_name = os.path.basename(dataset_path)
    print('Interpolating Dataset {}'.format(dataset_name))
    dataset = sorted(glob.glob('{}/*.png'.format(dataset_path)))
    if not dataset:
        dataset = sorted(glob.glob('{}/*.jpg'.format(dataset_path)))

    num = len(dataset)-2
    start = 0
    end = num
    if max_num and max_num < num:
        start = random.randint(0, num - max_num)
        end = start + max_num
    
    it = range(start, end)
    print(dataset_path)
    for i in tqdm(iterable=it, total=len(it)):
        interpolated_filename = '{}.png'.format(str(i).zfill(3))
        output_path_adacof = os.path.join(args.base_dir, args.img_output, dataset_name, 'adacof')
        output_path_phasenet = os.path.join(args.base_dir, args.img_output, dataset_name, 'phasenet')
        output_path_fusion = os.path.join(args.base_dir, args.img_output, dataset_name, 'fusion')

        output_path_adacof_image = os.path.join(output_path_adacof, interpolated_filename)
        output_path_phasenet_image = os.path.join(output_path_phasenet, interpolated_filename)
        output_path_fusion_image = os.path.join(output_path_fusion, interpolated_filename)

        # Interpolate and create output folders if they don't exist yet
        if args.adacof:
            os.makedirs(output_path_adacof, exist_ok=True)
            interpolate_adacof(args, dataset[i], dataset[i+2], output_path_adacof_image)
        if args.phase:
            os.makedirs(output_path_phasenet, exist_ok=True)
            interpolate_phasenet(args, dataset[i], dataset[i+2], output_path_phasenet_image)
        if args.fusion:
            os.makedirs(output_path_fusion, exist_ok=True)
            interpolate_fusion(args, dataset[i], dataset[i+2], output_path_fusion_image)

# Evaluates a dataset.
# Takes interpolated images a and c
# and compares the result with ground truth b
def evaluate_dataset(args, dataset_path):
    print('Evaluating Dataset ', dataset_path)
    output_path_adacof = os.path.join(args.base_dir, args.img_output, dataset_path, 'adacof')
    output_path_phasenet = os.path.join(args.base_dir, args.img_output, dataset_path, 'phasenet')
    output_path_fusion = os.path.join(args.base_dir, args.img_output, dataset_path, 'fusion')

    if args.adacof:
        prediction_folder_adacof = sorted(glob.glob(os.path.join(output_path_adacof, '*')))
        num_img = len(prediction_folder_adacof)
        first_img = prediction_folder_adacof[0]
    if args.phase:
        prediction_folder_phasenet = sorted(glob.glob(os.path.join(output_path_phasenet, '*')))
        num_img = len(prediction_folder_phasenet)
        first_img = prediction_folder_phasenet[0]
    if args.fusion:
        prediction_folder_fusion = sorted(glob.glob(os.path.join(output_path_fusion, '*')))
        num_img = len(prediction_folder_fusion)
        first_img = prediction_folder_fusion[0]
    
    target_folder = sorted(glob.glob(os.path.join('Testset', dataset_path, '*')))

    eval_results = []

    # Skip ground truth pictures if it has offset (max_num)
    start_index = int(os.path.splitext(os.path.basename(first_img))[0])-1

    it = range(1, num_img)

    for i in tqdm(iterable=it, total=len(it)):
        # Load reference images
        image_target = Image.open(target_folder[start_index + i])
        tensor_target = TF.to_tensor(image_target)

        eval_image_results = []

        if args.adacof:
            # Load Images
            image_prediction_adacof = Image.open(prediction_folder_adacof[i])
            tensor_prediction_adacof = TF.to_tensor(image_prediction_adacof)

            # Evaluate
            eval_result_adacof = evaluate_image(tensor_prediction_adacof, tensor_target)
            eval_image_results.append(eval_result_adacof)
        if args.phase:
            # Load Images
            image_prediction_phasenet = Image.open(prediction_folder_phasenet[i])
            tensor_prediction_phasenet = TF.to_tensor(image_prediction_phasenet)

            # Evaluate
            eval_result_phasenet = evaluate_image(tensor_prediction_phasenet, tensor_target)
            eval_image_results.append(eval_result_phasenet)
        if args.fusion:
            # Load Images
            image_prediction_fusion = Image.open(prediction_folder_fusion[i])
            tensor_prediction_fusion = TF.to_tensor(image_prediction_fusion)

            # Evaluate
            eval_result_fusion = evaluate_image(tensor_prediction_fusion, tensor_target)
            eval_image_results.append(eval_result_fusion)

        eval_results.append(np.stack(eval_image_results))
    
    return eval_results


def create_images(args, testset, test_path, inter_path):
    output_path = os.path.join(args.base_dir, 'visual_result')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, i in enumerate(testset):
        print('Evaluating {}'.format(i))
        out = os.path.join(output_path, i)

        if not os.path.exists(out):
            os.makedirs(out)
        
        ground_truth = [os.path.join(test_path, i, filename) for filename in sorted(os.listdir(test_path + "/" + i))][1:-1]
        inter_image_adacof = [os.path.join(inter_path, i, 'adacof', interpolate) for interpolate in sorted(os.listdir(os.path.join(inter_path, i, 'adacof')))]
        inter_image_phasenet = [os.path.join(inter_path, i, 'phasenet', interpolate) for interpolate in sorted(os.listdir(os.path.join(inter_path, i, 'phasenet')))]
        inter_image_fusion = [os.path.join(inter_path, i, 'fusion', interpolate) for interpolate in sorted(os.listdir(os.path.join(inter_path, i, 'fusion')))]
        error = np.load("Evaluation/result_" + i + ".npy")

        # Skip ground truth pictures if it has offset (max_num)
        start_index = int(os.path.splitext(os.path.basename(inter_image_adacof[0]))[0])-1

        for image_idx in range(len(inter_image_adacof) - 1): # TODO: Could be that error is missing one entry?
            if args.adacof:
                adacof_img = np.asarray(Image.open(inter_image_adacof[image_idx]))
            if args.phase:
                phase_img = np.asarray(Image.open(inter_image_phasenet[image_idx]))
            if args.fusion:
                fusion_img = np.asarray(Image.open(inter_image_fusion[image_idx]))
            
            draw_difference(adacof_img,
                            phase_img,
                            fusion_img,
                            np.asarray(Image.open(ground_truth[start_index + image_idx])),
                            out, error[image_idx], image_idx)
        
        
        input_images = sorted(glob.glob(os.path.join(out, '*.png')))
        print(input_images)
        images_to_video(input_images, os.path.join(out, 'result.avi'), framerate=10)


def draw_difference(pred_img_adacof, pred_img_phasenet, pred_img_fusion, target_img, out_path, error, number):
    name = 'img_{}_{}.png'.format(str(number).zfill(3), error[0, 0])

    if os.path.exists(out_path + "/" + name):
        return

    difference_adacof = np.average(np.abs(target_img - pred_img_adacof), axis=2)
    difference_phasenet = np.average(np.abs(target_img - pred_img_phasenet), axis=2)
    difference_fusion = np.average(np.abs(target_img - pred_img_fusion), axis=2)
    
    plt.subplot(3, 1, 1)
    plt.imshow(target_img)
    plt.axis('off')
    plt.title('Target Image')

    plt.subplot(3, 3, 4)
    plt.imshow(pred_img_adacof)
    plt.axis('off')
    plt.title('AdaCoF Image')

    plt.subplot(3, 3, 5)
    plt.imshow(pred_img_fusion)
    plt.axis('off')
    plt.title('Fusion Image')

    plt.subplot(3, 3, 6)
    plt.imshow(pred_img_phasenet)
    plt.axis('off')
    plt.title('Phasenet Image')

    plt.subplot(3, 3, 7)
    plt.imshow(difference_adacof, interpolation='none', cmap='plasma', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('AdaCoF Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(error[0, 0], error[0, 1], error[0, 2]), ha='center')

    plt.subplot(3, 3, 8)
    plt.imshow(difference_fusion, interpolation='none', cmap='plasma', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('Fusion Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(error[1, 0], error[1, 1], error[1, 2]), ha='center')


    plt.subplot(3, 3, 9)
    plt.imshow(difference_phasenet, interpolation='none', cmap='plasma', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('Phasenet Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(error[2, 0], error[2, 1], error[2, 2]), ha='center')

    plt.savefig(os.path.join(out_path, name), dpi=600)
    plt.clf()

def images_to_video(input_images, output_file, framerate=30):
    print(f'Combining images to video')
    imgs = []
    size = (1280, 720)
    for image_file in input_images:
        print(image_file)
        img = cv2.imread(image_file)
        height, width, layers = img.shape
        size = (width, height)
        imgs.append(img)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter.fourcc(*'mp4v'), framerate, size)
    for img in tqdm(iterable=imgs, total=len(imgs)):
        out.write(img)
    out.release()

def draw_measurements(args, datasets, datasets_results, title):
    avg_data = []
    std_data = []
    min_data = []
    max_data = []

    for dataset_results in datasets_results:
        avg = np.average(dataset_results, axis=0)
        avg_data.append(avg)

        std = np.std(dataset_results, axis=0)
        std_data.append(std)

        min = np.min(dataset_results, axis=0)
        min_data.append(min)
        
        max = np.max(dataset_results, axis=0)
        max_data.append(max)

    avg_data = np.stack(avg_data)
    std_data = np.stack(std_data)
    min_data = np.stack(min_data)
    max_data = np.stack(max_data)
    
    y_pos = np.arange(avg_data.shape[0])

    legend_order = [1,2,0]

    plt.figure(figsize=(30, 10))

    plt.suptitle(title)
    
    plt.subplot(1, 3, 1)
    plt.errorbar(y_pos, avg_data[:,0], std_data[:,0], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:,0], 'o', label='MIN')
    plt.plot(y_pos, max_data[:,0], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.title('SSIM')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order])

    plt.subplot(1, 3, 2)
    plt.errorbar(y_pos, avg_data[:,1], std_data[:,1], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:,1], 'o', label='MIN')
    plt.plot(y_pos, max_data[:,1], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.title('LPIPS')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order])

    plt.subplot(1, 3, 3)
    plt.errorbar(y_pos, avg_data[:,2], std_data[:,2], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:,2], 'o', label='MIN')
    plt.plot(y_pos, max_data[:,2], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.title('PSNR')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order],[labels[idx] for idx in legend_order])

    plt.savefig(os.path.join(args.base_dir, 'results_{}.png'.format(title)), dpi=600)
    plt.clf()

def main():
    eval(parser.parse_args())

def eval(args):
    random.seed(args.seed)
    img_output_dir = os.path.join(args.base_dir, args.img_output)
    os.makedirs(img_output_dir, exist_ok=True)

    # Interpolate
    for testset in args.test_sets:
        testset_path = os.path.join('Testset', testset)
        interpolate_dataset(args, testset_path, max_num=args.max_num)

    # Evaluate Results
    results_np = []
    for testset in args.test_sets:
        result_path = os.path.join(args.base_dir, 'result_{}.npy'.format(testset))
        if os.path.exists(result_path):
            result_np = np.load(result_path)
        else:
            result = evaluate_dataset(args, testset)
            print('Result for {}: {}'.format(testset, result))
            result_np = np.array(result)
            np.save(result_path, result_np)
        results_np.append(result_np)

    testset_path = 'Testset/'

    if args.adacof and args.phase and args.fusion:
        create_images(args, args.test_sets, testset_path, img_output_dir)

    # Show Results
    i = 0
    if args.adacof:
        results_adacof = [r[:,i] for r in results_np]
        draw_measurements(args, args.test_sets, results_adacof, 'AdaCoF')
        i = i + 1
    if args.phase:
        results_phasenet = [r[:,i] for r in results_np]
        draw_measurements(args, args.test_sets, results_phasenet, 'Phasenet')
        i = i + 1
    if args.fusion:
        results_fusion = [r[:,i] for r in results_np]
        draw_measurements(args, args.test_sets, results_fusion, 'Fusion')
        i = i + 1

if __name__ == "__main__":
    main()