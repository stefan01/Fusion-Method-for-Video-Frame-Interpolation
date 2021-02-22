import sys
import os
import glob
import subprocess
from types import SimpleNamespace
import torch
from torchvision import datasets, transforms
import random
from tqdm import tqdm

import src.adacof.interpolate_twoframe as adacof_interp
import src.phase_net.interpolate_twoframe as phasenet_interp
import src.fusion_net.interpolate_twoframe as fusion_interp


def interpolate_adacof(args, a, b, output):
    """
    Interpolates image a and b (file paths)
    using adacof and saves the result at output
    """
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
    """
    Interpolates image a and b (file paths)
    using phasenet and saves the result at output
    """
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


def interpolate_fusion(args, adacof_model, fusion_net, a, b, output):
    """
    Interpolates image a and b (file paths)
    using fusion and saves the result at output
    """
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
                model=args.fusion_model,
                loaded_adacof_model=adacof_model,
                loaded_fusion_net=fusion_net,
                high_level=args.fusion_replace_high_level
            ))
        torch.cuda.empty_cache()


def interpolate_dataset(args, adacof_model, fusion_net, dataset_path, max_num=None):
    """
    Interpolates triples of images
    from a dataset (list of filenames)
    """
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

    print('Start: {}'.format(start))
    print('End: {}'.format(end))

    for i in tqdm(iterable=it, total=len(it)):
        interpolated_filename = '{}.png'.format(str(i+1).zfill(4))
        output_path_adacof = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'adacof')
        output_path_phasenet = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'phasenet')
        output_path_fusion = os.path.join(
            args.base_dir, args.img_output, dataset_name, 'fusion')

        output_path_adacof_image = os.path.join(
            output_path_adacof, interpolated_filename)
        output_path_phasenet_image = os.path.join(
            output_path_phasenet, interpolated_filename)
        output_path_fusion_image = os.path.join(
            output_path_fusion, interpolated_filename)

        # Interpolate and create output folders if they don't exist yet
        if args.adacof:
            os.makedirs(output_path_adacof, exist_ok=True)
            interpolate_adacof(
                args, dataset[i], dataset[i+2], output_path_adacof_image)
        if args.phase:
            os.makedirs(output_path_phasenet, exist_ok=True)
            interpolate_phasenet(
                args, dataset[i], dataset[i+2], output_path_phasenet_image)
        if args.fusion:
            os.makedirs(output_path_fusion, exist_ok=True)
            interpolate_fusion(args, adacof_model, fusion_net,
                               dataset[i], dataset[i+2], output_path_fusion_image)