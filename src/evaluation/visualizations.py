import sys
import os
import glob
import subprocess
from matplotlib import image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import cv2


def create_images(args, testset, test_path, inter_path):
    """
    Creates images containing the target frame,
    the interpolated frames and difference frames
    """
    output_path = os.path.join(args.base_dir, 'visual_result')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, i in enumerate(testset):
        print('Evaluating {}'.format(i))
        out = os.path.join(output_path, i)

        if not os.path.exists(out):
            os.makedirs(out)

        ground_truth = [os.path.join(test_path, i, filename) for filename in sorted(
            os.listdir(test_path + "/" + i))][1:-1]
        inter_image_adacof = [os.path.join(inter_path, i, 'adacof', interpolate) for interpolate in sorted(
            os.listdir(os.path.join(inter_path, i, 'adacof')))]
        inter_image_phasenet = [os.path.join(inter_path, i, 'phasenet', interpolate) for interpolate in sorted(
            os.listdir(os.path.join(inter_path, i, 'phasenet')))]
        inter_image_fusion = [os.path.join(inter_path, i, 'fusion', interpolate) for interpolate in sorted(
            os.listdir(os.path.join(inter_path, i, 'fusion')))]
        error = np.load(os.path.join(args.base_dir, 'result_{}.npy'.format(i)))

        # Skip ground truth pictures if it has offset (max_num)
        start_index = int(os.path.splitext(
            os.path.basename(inter_image_adacof[0]))[0]) - 1

        # TODO: Could be that error is missing one entry?
        for image_idx in range(len(inter_image_adacof)):
            if args.adacof:
                adacof_img = np.asarray(Image.open(
                    inter_image_adacof[image_idx]))
            if args.phase:
                phase_img = np.asarray(Image.open(
                    inter_image_phasenet[image_idx]))
            if args.fusion:
                fusion_img = np.asarray(Image.open(
                    inter_image_fusion[image_idx]))

            print('AdaCoF: {}'.format(inter_image_adacof[image_idx]))
            print('Phase: {}'.format(inter_image_phasenet[image_idx]))
            print('Fusion: {}'.format(inter_image_fusion[image_idx]))
            print('Ground truth: {}'.format(
                ground_truth[start_index + image_idx]))

            draw_difference(adacof_img,
                            phase_img,
                            fusion_img,
                            np.asarray(Image.open(
                                ground_truth[start_index + image_idx])),
                            out, error[image_idx], image_idx)

        input_images = sorted(glob.glob(os.path.join(out, '*.png')))
        print(input_images)
        images_to_video(input_images, os.path.join(
            out, 'result.avi'), framerate=10)


def draw_difference(pred_img_adacof, pred_img_phasenet, pred_img_fusion, target_img, out_path, error, number):
    """ 
    Draws a single frame containing the target,
    the interpolated frames and difference frames
    """
    name = 'img_{}_{}.png'.format(str(number).zfill(4), error[0, 0])

    # Only generate if not already present
    if os.path.exists(out_path + "/" + name):
        return

    # Convert type to int so the substraction doesn't result in an underflow
    target_img_int = target_img.astype(np.int)
    pred_img_adacof_int = pred_img_adacof.astype(np.int)
    pred_img_phasenet_int = pred_img_phasenet.astype(np.int)
    pred_img_fusion_int = pred_img_fusion.astype(np.int)

    # Calculate difference images
    difference_adacof = np.average(
        np.abs(target_img_int - pred_img_adacof_int), axis=2)  # axis=2
    difference_phasenet = np.average(
        np.abs(target_img_int - pred_img_phasenet_int), axis=2)
    difference_fusion = np.average(
        np.abs(target_img_int - pred_img_fusion_int), axis=2)

    # Plot target image
    plt.subplot(3, 1, 1)
    plt.imshow(target_img)
    plt.axis('off')
    plt.title('Target Image')

    # Plot adacof interpolated image
    plt.subplot(3, 3, 4)
    plt.imshow(pred_img_adacof)
    plt.axis('off')
    plt.title('AdaCoF Image')

    # Plot fusion interpolated image
    plt.subplot(3, 3, 5)
    plt.imshow(pred_img_fusion)
    plt.axis('off')
    plt.title('Fusion Image')

    # Plot phasenet interpolated image
    plt.subplot(3, 3, 6)
    plt.imshow(pred_img_phasenet)
    plt.axis('off')
    plt.title('Phasenet Image')

    # Plot adacof difference
    plt.subplot(3, 3, 7)
    plt.imshow(difference_adacof, interpolation='none',
               cmap='jet', vmin=0, vmax=100)
    plt.axis('off')
    plt.colorbar()
    plt.title('AdaCoF Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
        error[0, 0], error[0, 1], error[0, 2]), ha='center')

    # Plot fusion difference
    plt.subplot(3, 3, 8)
    plt.imshow(difference_fusion, interpolation='none',
               cmap='jet', vmin=0, vmax=100)
    plt.axis('off')
    plt.colorbar()
    plt.title('Fusion Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
        error[1, 0], error[1, 1], error[1, 2]), ha='center')

    # Plot phasenet difference
    plt.subplot(3, 3, 9)
    plt.imshow(difference_phasenet, interpolation='none',
               cmap='jet', vmin=0, vmax=100)
    plt.axis('off')
    plt.colorbar()
    plt.title('Phasenet Diff')
    plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
        error[2, 0], error[2, 1], error[2, 2]), ha='center')

    plt.savefig(os.path.join(out_path, name), dpi=600)
    plt.clf()


def images_to_video(input_images, output_file, framerate=30):
    """
    Converts a sequence of images to a video
    with the given framerate
    """
    print(f'Combining images to video')
    imgs = []
    size = (1280, 720)
    for image_file in input_images:
        print(image_file)
        img = cv2.imread(image_file)
        height, width, layers = img.shape
        size = (width, height)
        imgs.append(img)

    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter.fourcc(*'mp4v'), framerate, size)
    for img in tqdm(iterable=imgs, total=len(imgs)):
        out.write(img)
    out.release()


def draw_measurements(args, datasets, datasets_results, title):
    """
    Saves a image containing measurement results
    """
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

    legend_order = [1, 2, 0]

    plt.figure(figsize=(30, 10))

    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.errorbar(y_pos, avg_data[:, 0],
                 std_data[:, 0], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:, 0], 'o', label='MIN')
    plt.plot(y_pos, max_data[:, 0], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('SSIM')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.subplot(1, 3, 2)
    plt.errorbar(y_pos, avg_data[:, 1],
                 std_data[:, 1], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:, 1], 'o', label='MIN')
    plt.plot(y_pos, max_data[:, 1], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('LPIPS')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.subplot(1, 3, 3)
    plt.errorbar(y_pos, avg_data[:, 2],
                 std_data[:, 2], fmt='o', label='AVG + STD')
    plt.plot(y_pos, min_data[:, 2], 'o', label='MIN')
    plt.plot(y_pos, max_data[:, 2], 'o', label='MAX')
    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('PSNR')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.savefig(os.path.join(args.base_dir,
                             'results_{}.png'.format(title)), dpi=600)
    plt.clf()
