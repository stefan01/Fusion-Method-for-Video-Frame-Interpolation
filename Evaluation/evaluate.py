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
# using adacof and saves the result at output
def interpolate_adacof(a, b, output):
    print('Interpolating {} and {} to {} with adacof'.format(a, b, output))
    subprocess.run([
        sys.executable,
        '../AdaCoF/interpolate_twoframe.py',
        '--first_frame', a,
        '--second_frame', b,
        '--output_frame', output,
        '--checkpoint', '../AdaCoF/checkpoint/kernelsize_5/ckpt.pth',
        '--config', '../AdaCoF/checkpoint/kernelsize_5/config.txt'])

def interpolate_phasenet(a, b, output):
    print('Interpolating {} and {} to {} with phasenet'.format(a, b, output))
    subprocess.run([
        sys.executable,
        '../AdaCoF/interpolate_twoframe.py',
        '--first_frame', a,
        '--second_frame', b,
        '--output_frame', output,
        '--checkpoint', '../AdaCoF/checkpoint/kernelsize_5/ckpt.pth',
        '--config', '../AdaCoF/checkpoint/kernelsize_5/config.txt'])

# Interpolates triples of images
# from a dataset (list of filenames)
def interpolate_dataset(dataset_path):
    dataset_name = os.path.basename(dataset_path)
    print('Interpolating Dataset {}'.format(dataset_name))
    dataset = sorted(glob.glob('{}/*.png'.format(dataset_path)))
    if not dataset:
        dataset = sorted(glob.glob('{}/*.jpg'.format(dataset_path)))
    it = range(0, len(dataset)-2)
    print(dataset_path)
    for i in tqdm(iterable=it, total=len(it)):
        interpolated_filename = os.path.basename(dataset[i+1])
        output_path_adacof = '{}/{}/adacof/{}'.format(tmp_dir, dataset_name, interpolated_filename)
        output_path_phasenet = '{}/{}/phasenet/{}'.format(tmp_dir, dataset_name, interpolated_filename)
        os.makedirs('{}/{}'.format(tmp_dir, dataset_name), exist_ok=True)
        interpolate_adacof(dataset[i], dataset[i+2], output_path_adacof)
        interpolate_phasenet(dataset[i], dataset[i+2], output_path_phasenet)

# Evaluates a dataset.
# Takes interpolated images a and c
# and compares the result with b
def evaluate_dataset(dataset_path):
    print('Evaluating Dataset ', dataset_path)
    prediction_folder_adacof = sorted(glob.glob('{}/{}/adacof/*.png'.format(tmp_dir, dataset_path)))
    prediction_folder_phasenet = sorted(glob.glob('{}/{}/phasenet/*.png'.format(tmp_dir, dataset_path)))
    target_folder = sorted(glob.glob('../Testset/{}/*.png'.format(dataset_path)))

    output_path = os.path.dirname(os.path.dirname(dataset_path)) + "visual_result"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    eval_results = []

    it = range(1, len(prediction_folder_adacof)) 

    for i in tqdm(iterable=it, total=len(it)):
        # Load Images
        image_prediction_adacof = Image.open(prediction_folder_adacof[i])
        image_prediction_phasenet = Image.open(prediction_folder_phasenet[i])
        image_target = Image.open(target_folder[i])

        tensor_prediction_adacof = TF.to_tensor(image_prediction_adacof)
        tensor_prediction_phasenet = TF.to_tensor(image_prediction_phasenet)
        tensor_target = TF.to_tensor(image_target)

        # Evaluate
        eval_result_adacof = evaluate_image(tensor_prediction_adacof, tensor_target)
        eval_result_phasenet = evaluate_image(tensor_prediction_phasenet, tensor_target)

        eval_results.append(np.stack(eval_result_adacof, eval_result_phasenet))

    return eval_results


def create_images(testset, test_path, inter_path):
    if not os.path.exists('visual_result'):
        os.makedirs('visual_result')
    for idx, i in enumerate(testset):
        print('Evaluating {}'.format(i))
        out = 'visual_result/' + i
        if not os.path.exists(out):
            os.makedirs(out)
        ground_truth = [test_path + i + "/" + filename for filename in os.listdir(test_path + "/" + i)][1:-1]
        inter_image = [inter_path + i + "/" + interpolate for interpolate in os.listdir(inter_path + "/" + i)]
        error = np.load("result_" + i + ".npy")
        for image_idx in range(len(inter_image) - 1): # TODO: Could be that error is missing one entry?
            draw_difference(np.asarray(Image.open(inter_image[image_idx])), 
                            np.asarray(Image.open(ground_truth[image_idx])),
                            out, error[image_idx], image_idx)
        
        
        input_images = sorted(glob.glob('{}/*.png'.format(out)))
        print(input_images)
        images_to_video(input_images, '{}/result.avi'.format(out), framerate=10)


def draw_difference(pred_img, target_img, out_path, error, number):
    name = 'img_{}_{}.png'.format(str(number).zfill(3), error[0])

    if os.path.exists(out_path + "/" + name):
        return

    difference = np.average(np.abs(target_img - pred_img), axis=2)
    
    plt.subplot(2, 2, 1)
    plt.imshow(target_img)
    plt.axis('off')
    plt.title('Target Image')

    plt.subplot(2, 2, 2)
    plt.imshow(pred_img)
    plt.axis('off')
    plt.title('Predicted Image')

    plt.subplot(2, 1, 2)
    plt.imshow(difference, interpolation='none', cmap='plasma', vmin=0, vmax=255)
    plt.axis('off')
    plt.colorbar()
    plt.title('Difference Image')

    plt.savefig(out_path + "/" + name, dpi=600)
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

def draw_measurements(datasets, datasets_results):
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

    plt.show()

testsets = ['Clip1', 'Clip2', 'Clip3', 'Clip4', 'Clip5', 'Clip6', 'Clip7', 'Clip8', 'Clip9', 'Clip10', 'Clip11', \
    'airboard_1', 'airplane_landing', 'airtable_3', 'basketball_1', 'water_ski_2', 'yoyo']

# Interpolate
for testset in testsets:
    if not os.path.isdir('{}/{}'.format(tmp_dir, testset)):
        testset_path = '../Testset/{}'.format(testset)
        interpolate_dataset(testset_path)

# Evaluate Results
results_np = []
for testset in testsets:
    result_path = 'result_{}.npy'.format(testset)
    if os.path.exists(result_path):
        result_np = np.load(result_path)
    else:
        result = evaluate_dataset(testset)
        print('Result for {}: {}'.format(testset, result))
        result_np = np.array(result)
        np.save('result_{}.npy'.format(testset), result_np)
    results_np.append(result_np)

testset_path = '../Testset/'
interpolate_path = 'tmp/'
create_images(testsets, testset_path, interpolate_path)

# Show Results
draw_measurements(testsets, results_np)
