import sys
import os
import glob
import subprocess
from matplotlib import image
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from datetime import datetime
from types import SimpleNamespace

import src.evaluation.evaluate_image as evaluate_image
import src.evaluation.interpolate as interpolate
import src.evaluation.visualizations as visualizations

from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet
from src.fusion_net.fusion_net import FusionNet
from src.train.pyramid import Pyramid

parser = argparse.ArgumentParser(description='Evaluation')

# Evaluation Parameters
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--adacof', action='store_true')
parser.add_argument('--phase', action='store_true')
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--base_dir', type=str, default=os.path.join(
    'Evaluation', datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
parser.add_argument('--img_output', type=str, default='interpolated')
parser.add_argument('--max_num', type=int, default=10)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--test_sets', type=str, nargs='+',
                    default=['airboard_1', 'airplane_landing', 'airtable_3', 'basketball_1', 'water_ski_2', 'yoyo',
                             'MODE_SH0280', 'MODE_SH0440', 'MODE_SH0450', 'MODE_SH0740', 'MODE_SH0780', 'MODE_SH1010', 'MODE_SH1270',
                             'Flashlight', 'firework', 'lights', 'sun'])
#parser.add_argument('--combine-results', type=str, nargs='+', action='append')

# Adacof Parameters
parser.add_argument('--adacof_model', type=str,
                    default='src.adacof.models.adacofnet')
parser.add_argument('--adacof_checkpoint', type=str,
                    default='./src/adacof/checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--adacof_config', type=str,
                    default='./src/adacof/checkpoint/kernelsize_5/config.txt')
parser.add_argument('--adacof_kernel_size', type=int, default=5)
parser.add_argument('--adacof_dilation', type=int, default=1)

# Phasenet Parameters
parser.add_argument('--phasenet_checkpoint', type=str,
                    default='./src/phase_net/phase_net.pt')
parser.add_argument('--phasenet_replace_high_level', action='store_true')

# Fusion Parameters
parser.add_argument('--fusion_checkpoint', type=str,
                    default='./src/fusion_net/fusion_net.pt')
parser.add_argument('--fusion_adacof_model', type=str,
                    default='src.fusion_net.fusion_adacofnet')
parser.add_argument('--fusion_model', type=int, default=1)
parser.add_argument('--fusion_replace_high_level', action='store_true')
parser.add_argument('--vimeo_testset', action='store_true')
parser.add_argument('--mode', type=str, default='alpha')


def evaluate_dataset(args, dataset_path):
    """
    Evaluates a dataset.
    Takes interpolated images a and c
    and compares the result with ground truth b
    """
    print('Evaluating Dataset ', dataset_path)

    dataset_name = os.path.basename(dataset_path)

    if args.vimeo_testset:
        output_path_adacof = os.path.join(
            args.base_dir, args.img_output, 'adacof', dataset_path, '*', 'im2.png')
        output_path_phasenet = os.path.join(
            args.base_dir, args.img_output, 'phasenet', dataset_path, '*', 'im2.png')
        output_path_fusion = os.path.join(
            args.base_dir, args.img_output, 'fusion', dataset_path, '*', 'im2.png')
    else:
        output_path_adacof = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'adacof', '*')
        output_path_phasenet = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'phasenet', '*')
        output_path_fusion = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'fusion', '*')

        print(dataset_name)
        print(output_path_adacof)

    if args.adacof:
        prediction_folder_adacof = sorted(
            glob.glob(output_path_adacof))
        num_img = len(prediction_folder_adacof)
        first_img = prediction_folder_adacof[0]
    if args.phase:
        prediction_folder_phasenet = sorted(
            glob.glob(output_path_phasenet))
        num_img = len(prediction_folder_phasenet)
        first_img = prediction_folder_phasenet[0]
    if args.fusion:
        prediction_folder_fusion = sorted(
            glob.glob(output_path_fusion))
        num_img = len(prediction_folder_fusion)
        first_img = prediction_folder_fusion[0]

    if args.vimeo_testset:
        target_folder = sorted(
            glob.glob(os.path.join('Testset', 'vimeo_interp_test',
                                   'target', '*', '*', 'im2.png')))
    else:
        target_folder = sorted(
            glob.glob(os.path.join('Testset', dataset_path, '*')))

    eval_results = []

    if args.vimeo_testset:
        start_index = 0
    else:
        # Skip ground truth pictures if it has offset (max_num)
        start_index = int(os.path.splitext(os.path.basename(first_img))[0])-1

    it = range(0, num_img)

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
            eval_result_adacof = evaluate_image.evaluate_image(
                tensor_prediction_adacof, tensor_target)
            eval_image_results.append(eval_result_adacof)
        if args.phase:
            # Load Images
            image_prediction_phasenet = Image.open(
                prediction_folder_phasenet[i])
            tensor_prediction_phasenet = TF.to_tensor(
                image_prediction_phasenet)

            # Evaluate
            eval_result_phasenet = evaluate_image.evaluate_image(
                tensor_prediction_phasenet, tensor_target)
            eval_image_results.append(eval_result_phasenet)
        if args.fusion:
            # Load Images
            image_prediction_fusion = Image.open(prediction_folder_fusion[i])
            tensor_prediction_fusion = TF.to_tensor(image_prediction_fusion)

            # Evaluate
            eval_result_fusion = evaluate_image.evaluate_image(
                tensor_prediction_fusion, tensor_target)
            eval_image_results.append(eval_result_fusion)

        eval_results.append(np.stack(eval_image_results))

    return eval_results


def main():
    eval(parser.parse_args())


def eval(args):
    device = torch.device('cuda:{}'.format(args.gpu_id))
    random.seed(args.seed)
    img_output_dir = os.path.join(args.base_dir, args.img_output)
    os.makedirs(img_output_dir, exist_ok=True)

    # Create AdaCoFNet
    adacof_args = SimpleNamespace(
        gpu_id=args.gpu_id,
        model=args.fusion_adacof_model,
        kernel_size=args.adacof_kernel_size,
        dilation=args.adacof_dilation,
        config=args.adacof_config
    )
    adacof_model = Model(adacof_args)
    adacof_model.eval()
    checkpoint = torch.load(args.adacof_checkpoint,
                            map_location=torch.device('cpu'))
    adacof_model.load(checkpoint['state_dict'])

    # Create FusionNet
    fusion_net = FusionNet().to(device)
    fusion_net.load_state_dict(torch.load(args.fusion_checkpoint))
    fusion_net.eval()

    # Interpolate
    if args.vimeo_testset:
        interpolate.interpolate_dataset(
            args, adacof_model, fusion_net)
    else:
        for testset in args.test_sets:
            testset_path = os.path.join('Testset', testset)
            interpolate.interpolate_dataset(
                args, adacof_model, fusion_net, testset_path, max_num=args.max_num)

    # Evaluate Results
    if args.vimeo_testset:
        # Override test_sets with vimeo
        testsets_path = sorted(
            glob.glob(os.path.join('Testset', 'vimeo_interp_test',
                                   'target', '*')))
        args.test_sets = [os.path.basename(x) for x in testsets_path]

    results_np = []
    for testset in args.test_sets:
        result_path = os.path.join(
            args.base_dir, 'result_{}.npy'.format(os.path.basename(testset)))
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
        visualizations.create_images(
            args, args.test_sets, testset_path, img_output_dir)

        # Show Results
    '''i = 0
    if args.adacof:
        results_adacof = [r[:, i] for r in results_np]
        visualizations.draw_measurements(
            args, args.test_sets, results_adacof, 'AdaCoF')
        i = i + 1
    if args.phase:
        results_phasenet = [r[:, i] for r in results_np]
        visualizations.draw_measurements(
            args, args.test_sets, results_phasenet, 'Phasenet')
        i = i + 1
    if args.fusion:
        results_fusion = [r[:, i] for r in results_np]
        visualizations.draw_measurements(
            args, args.test_sets, results_fusion, 'Fusion')
        i = i + 1'''

    visualizations.draw_measurements(
        args, args.test_sets, results_np)


if __name__ == "__main__":
    main()
