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
import src.fusion_net.interpolate_twoframe as fusion_interp


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

        if args.vimeo_testset:
            ground_truth = sorted(
                glob.glob(os.path.join('Testset', 'vimeo_interp_test',
                                       'target', i, '*', 'im2.png')))

            inter_image_adacof = sorted(
                glob.glob(os.path.join(inter_path, 'adacof', i, '*', 'im2.png')))

            inter_image_phasenet = sorted(
                glob.glob(os.path.join(inter_path, 'phasenet', i, '*', 'im2.png')))

            inter_image_fusion = sorted(
                glob.glob(os.path.join(inter_path, 'fusion', i, '*', 'im2.png')))

            start_index = 0
        else:
            ground_truth = sorted(
                glob.glob(os.path.join(test_path, i, '*')))[1:-1]
            # ground_truth = [os.path.join(test_path, i, filename) for filename in sorted(
            #    os.listdir(test_path + "/" + i))][1:-1]

            inter_image_adacof = sorted(
                glob.glob(os.path.join(inter_path, os.path.basename(i), 'adacof', '*')))
            # inter_image_adacof = [os.path.join(inter_path, i, 'adacof', interpolate) for interpolate in sorted(
            #    os.listdir(os.path.join(inter_path, i, 'adacof')))]

            inter_image_phasenet = sorted(
                glob.glob(os.path.join(inter_path, os.path.basename(i), 'phasenet', '*')))
            # inter_image_phasenet = [os.path.join(inter_path, i, 'phasenet', interpolate) for interpolate in sorted(
            #    os.listdir(os.path.join(inter_path, i, 'phasenet')))]

            inter_image_fusion = sorted(
                glob.glob(os.path.join(inter_path, os.path.basename(i), 'fusion', '*')))
            # inter_image_fusion = [os.path.join(inter_path, i, 'fusion', interpolate) for interpolate in sorted(
            #    os.listdir(os.path.join(inter_path, i, 'fusion')))]

            # Skip ground truth pictures if it has offset (max_num)
            start_index = int(os.path.splitext(
                os.path.basename(inter_image_adacof[0]))[0]) - 1

        # error = np.load(os.path.join(
        #    args.base_dir, 'result_{}.npy'.format(os.path.basename(i))))

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

            target_img = np.asarray(Image.open(
                ground_truth[start_index + image_idx]))

            # draw_difference(fusion_interp.crop_center(adacof_img, 512, 512),
            #                fusion_interp.crop_center(phase_img, 512, 512),
            #                fusion_interp.crop_center(fusion_img, 512, 512),
            #                fusion_interp.crop_center(target_img, 512, 512),
            #                out, error[image_idx], image_idx)

            draw_difference(fusion_interp.crop_center(adacof_img, 256, 256),
                            fusion_interp.crop_center(phase_img, 256, 256),
                            fusion_interp.crop_center(fusion_img, 256, 256),
                            fusion_interp.crop_center(target_img, 256, 256),
                            out, image_idx)

        input_images = sorted(glob.glob(os.path.join(out, '*.png')))
        print(input_images)

        video_output_path = os.path.join(
            out, 'result.avi')
        if not os.path.exists(video_output_path):
            images_to_video(input_images, video_output_path, framerate=10)


def draw_difference(pred_img_adacof, pred_img_phasenet, pred_img_fusion, target_img, out_path, number):
    """
    Draws a single frame containing the target,
    the interpolated frames and difference frames
    """
    #name = 'img_{}_{}.png'.format(str(number).zfill(4), error[0, 0])
    name = 'img_{}.png'.format(str(number).zfill(4))

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
    # plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
    #    error[0, 0], error[0, 1], error[0, 2]), ha='center')

    # Plot fusion difference
    plt.subplot(3, 3, 8)
    plt.imshow(difference_fusion, interpolation='none',
               cmap='jet', vmin=0, vmax=100)
    plt.axis('off')
    plt.colorbar()
    plt.title('Fusion Diff')
    # plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
    #    error[1, 0], error[1, 1], error[1, 2]), ha='center')

    # Plot phasenet difference
    plt.subplot(3, 3, 9)
    plt.imshow(difference_phasenet, interpolation='none',
               cmap='jet', vmin=0, vmax=100)
    plt.axis('off')
    plt.colorbar()
    plt.title('Phasenet Diff')
    # plt.text(500, 1300, 'SSIM: {:.2f}\nLPIPS: {:.2f}\nPSNR: {:.2f}'.format(
    #    error[2, 0], error[2, 1], error[2, 2]), ha='center')

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


def draw_measurements_old(args, datasets, datasets_results):
    """
    Saves a image containing measurement results
    """
    print(datasets)
    print(datasets_results.shape)
    avg_data = []
    std_data = []
    min_data = []
    max_data = []

    for dataset_results in datasets_results:
        avg = np.average(dataset_results, axis=1)
        avg_data.append(avg)

        std = np.std(dataset_results, axis=1)
        std_data.append(std)

        min = np.min(dataset_results, axis=1)
        min_data.append(min)

        max = np.max(dataset_results, axis=1)
        max_data.append(max)

    avg_data = np.stack(avg_data)
    std_data = np.stack(std_data)
    min_data = np.stack(min_data)
    max_data = np.stack(max_data)

    y_pos = np.arange(avg_data.shape[0])

    legend_order = [1, 2, 0]

    plt.figure(figsize=(30, 10))

    plt.suptitle('Results')

    plt.subplot(1, 3, 1)
    for i in range(datasets_results.shape[0]):
        plt.errorbar(y_pos, avg_data[i, :, 0],
                     std_data[i, :, 0], fmt='o', label='AVG + STD')
        plt.plot(y_pos, min_data[i, :, 0], 'o', label='MIN')
        plt.plot(y_pos, max_data[i, :, 0], 'o', label='MAX')

    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('SSIM')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.subplot(1, 3, 2)
    for i in range(datasets_results.shape[0]):
        plt.errorbar(y_pos, avg_data[i, :, 1],
                     std_data[i, :, 1], fmt='o', label='{} AVG + STD')
        plt.plot(y_pos, min_data[i, :, 1], 'o', label='MIN')
        plt.plot(y_pos, max_data[i, :, 1], 'o', label='MAX')

    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('LPIPS')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.subplot(1, 3, 3)
    for i in range(datasets_results.shape[0]):
        plt.errorbar(y_pos, avg_data[i, :, 2],
                     std_data[i, :, 2], fmt='o', label='AVG + STD')
        plt.plot(y_pos, min_data[i, :, 2], 'o', label='MIN')
        plt.plot(y_pos, max_data[i, :, 2], 'o', label='MAX')

    plt.xticks(y_pos, datasets)
    plt.grid()
    plt.ylim(bottom=0)
    plt.title('PSNR')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in legend_order], [labels[idx]
                                                        for idx in legend_order])

    plt.savefig(os.path.join(args.base_dir,
                             'results.png'), dpi=600)
    plt.clf()


def draw_measurements(args, datasets, datasets_results):
    """
    Saves a image containing measurement results
    """
    # datasets_results: [dataset, frame, interpolation_method, metric]
    bar_width = 0.2

    interpolation_methods = ['AdaCoF', 'Phase',
                             'Fusion', 'Baseline'][:datasets_results[0].shape[1]]

    avg_data = []
    std_data = []
    min_data = []
    max_data = []
    for dataset_results in datasets_results:
        # avg over frame (2)
        avg = np.average(dataset_results, axis=0)
        avg_data.append(avg)

        std = np.std(dataset_results, axis=0)
        std_data.append(std)

        min = np.min(dataset_results, axis=0)
        min_data.append(min)

        max = np.max(dataset_results, axis=0)
        max_data.append(max)

    x = np.arange(len(avg_data))

    legend_order = [1, 2, 0]

    #plt.figure(figsize=(len(datasets_results), 10))

    fig, axs = plt.subplots(
        4, sharex=True, figsize=(len(datasets_results)+7, 13))

    plt.title('Results')
    for interpolation_method_idx in range(datasets_results[0].shape[1]):
        avg_plot = np.array([x[interpolation_method_idx, 0] for x in avg_data])
        axs[0].set_title('SSIM')
        axs[0].grid()
        rects0 = axs[0].bar(x + interpolation_method_idx*bar_width - (datasets_results[0].shape[1]*bar_width)/2 + 0.5*bar_width, avg_plot, bar_width-0.02,
                            align='center', label=interpolation_methods[interpolation_method_idx])
        axs[0].legend(bbox_to_anchor=(1, 1))
        autolabel(axs[0], rects0)

        avg_plot = np.array([x[interpolation_method_idx, 1] for x in avg_data])
        axs[1].set_title('LPIPS')
        axs[1].grid()
        rects1 = axs[1].bar(x + interpolation_method_idx*bar_width - (datasets_results[0].shape[1]*bar_width)/2 + 0.5*bar_width, avg_plot, bar_width-0.02,
                            align='center', label=interpolation_methods[interpolation_method_idx])
        #axs[1].legend(bbox_to_anchor=(1.1, 1))
        autolabel(axs[1], rects1)

        avg_plot = np.array([x[interpolation_method_idx, 2] for x in avg_data])
        axs[2].set_title('PSNR')
        axs[2].grid()
        rects2 = axs[2].bar(x + interpolation_method_idx*bar_width - (datasets_results[0].shape[1]*bar_width)/2 + 0.5*bar_width, avg_plot, bar_width-0.02,
                            align='center', label=interpolation_methods[interpolation_method_idx])
        #axs[2].legend(bbox_to_anchor=(1, 1))
        autolabel(axs[2], rects2)

        avg_plot = np.array([x[interpolation_method_idx, 3] for x in avg_data])
        axs[3].set_title('SSD')
        axs[3].grid()
        rects3 = axs[3].bar(x + interpolation_method_idx*bar_width - (datasets_results[0].shape[1]*bar_width)/2 + 0.5*bar_width, avg_plot, bar_width-0.02,
                            align='center', label=interpolation_methods[interpolation_method_idx])
        #axs[3].legend(bbox_to_anchor=(1.08, 1))
        autolabel(axs[3], rects3)

    # axs[4].table(cellText=datasets_results)

    plt.xticks(x, datasets)
    plt.ylim(bottom=0)

    plt.savefig(os.path.join(args.base_dir,
                             'results.png'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
