import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image as imwrite
from src.train.transform import *
from src.train.utils import *
from src.train.pyramid import Pyramid
from torch.autograd import Variable
import warnings
import numpy as np
import torch
from collections import namedtuple
from PIL import Image
from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet
from types import SimpleNamespace
from src.fusion_net.fusion_net import FusionNet
from src.train.utils import *
from scipy.ndimage import maximum_filter, median_filter, sobel, convolve
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Two-frame Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--adacof_model', type=str,
                    default='src.fusion_net.fusion_adacofnet')
parser.add_argument('--adacof_checkpoint', type=str,
                    default='./src/adacof/checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--adacof_config', type=str,
                    default='./src/adacof/checkpoint/kernelsize_5/config.txt')
parser.add_argument('--adacof_kernel_size', type=int, default=5)
parser.add_argument('--adacof_dilation', type=int, default=1)


parser.add_argument('--first_frame', type=str,
                    default='./Testset/lights/119.png')
parser.add_argument('--second_frame', type=str,
                    default='./Testset/lights/121.png')
parser.add_argument('--ground_truth', type=str,
                    default='./Testset/lights/120.png')
parser.add_argument('--out_dir', type=str,
                    default='./paper/predictions/')
parser.add_argument('--output', type=str, default='lights')


transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main(loaded_adacof_model, loaded_fusion_net):
    interp(parser.parse_args(), loaded_fusion_net)


def interp(args, loaded_adacof_model=None, loaded_fusion_net=None, high_level=False):
    torch.cuda.set_device(args.gpu_id)
    # Warnings and device
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:{}'.format(args.gpu_id))

    # Adacof model
    if loaded_adacof_model:
        adacof_model = loaded_adacof_model
    else:
        adacof_args = SimpleNamespace(
            gpu_id=args.gpu_id,
            model=args.adacof_model,
            kernel_size=args.adacof_kernel_size,
            dilation=args.adacof_dilation,
            config=args.adacof_config
        )
        adacof_model = Model(adacof_args)
        adacof_model.eval()
        checkpoint = torch.load(args.adacof_checkpoint,
                                map_location=torch.device('cpu'))
        adacof_model.load(checkpoint['state_dict'])

    # Import images
    rgb_frame1 = np.array(Image.open(args.first_frame))
    rgb_frame2 = np.array(Image.open(args.second_frame))
    rgb_ground_truth = np.array(Image.open(args.ground_truth))
    shape_r = rgb_frame1.shape

    with torch.no_grad():
        frame_out1, frame_out2, ada_res, uncertainty_mask = adacof_model(
            torch.as_tensor(rgb_frame1).permute(
                2, 0, 1).float().unsqueeze(0).to(device)/255,
            torch.as_tensor(rgb_frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)/255)

        frame_out1, frame_out2 = frame_out1.squeeze(0).permute(
            1, 2, 0).cpu().numpy(), frame_out2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_ada_res = ada_res.squeeze(0).permute(1, 2, 0).cpu().numpy()

        frame_out1_pad = pad_img(frame_out1)
        frame_out2_pad = pad_img(frame_out2)
        frame_ada_res_pad = pad_img(frame_ada_res)
        
    # Save uncertainty adacof map
    u_ada = uncertainty_mask.squeeze(1).detach().cpu()
    u_ada = transforms.ToPILImage()(u_ada)
    u_ada.save(args.out_dir + f'{args.output}_uncertainty_adacof1.png')
    
    # Save adacof prediction
    p_ada = torch.as_tensor(frame_ada_res).permute(2, 0, 1)
    p_ada = transforms.ToPILImage()(p_ada)
    p_ada.save(args.out_dir + f'{args.output}_prediction_adacof.png')
    
    
    # Save difference adacof/ground truth map
    difference_adacof = np.average(
        np.abs(rgb_ground_truth/255 - frame_ada_res), axis=2)
    
    plt.imshow(difference_adacof, interpolation='none',
               cmap='jet', vmin=0, vmax=1)
    plt.savefig(args.out_dir + f'{args.output}_adacof_diff.png', dpi=600)

    # Normalize and pad images
    img_1_pad = pad_img(rgb_frame1/255)
    img_2_pad = pad_img(rgb_frame2/255)
    img_g_pad = pad_img(rgb_ground_truth/255)
    
    # RGB images
    rgb_frame1 = torch.as_tensor(img_1_pad).to(device).permute(2, 0, 1)
    rgb_frame2 = torch.as_tensor(img_2_pad).to(device).permute(2, 0, 1)
    rgb_ground_truth = torch.as_tensor(img_g_pad).to(device).permute(2, 0, 1)

    # To tensors
    img_1 = rgb2lab_single(torch.as_tensor(
        img_1_pad).permute(2, 0, 1).float()).to(device)
    img_2 = rgb2lab_single(torch.as_tensor(
        img_2_pad).permute(2, 0, 1).float()).to(device)
    frame_1 = rgb2lab_single(torch.as_tensor(
        frame_out1_pad).permute(2, 0, 1).float()).to(device)
    frame_2 = rgb2lab_single(torch.as_tensor(
        frame_out2_pad).permute(2, 0, 1).float()).to(device)
    frame_res = rgb2lab_single(torch.as_tensor(
        frame_ada_res_pad).permute(2, 0, 1).float()).to(device)

    # Build pyramid
    pyr = Pyramid(
        height=calc_pyr_height(img_1),
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )

    # High level
    #if args.high_level:
    #    ada_pyr = pyr.filter(frame_res)
    #    ada_hl = ada_pyr.high_level.clone().detach()
    #    del ada_pyr
    #    torch.cuda.empty_cache()

    # Create FusionNet
    num_img = 2
    load_path = './src/phase_net/phase_net.pt'

    phase_net = PhaseNet(pyr, device, num_img=num_img)
    phase_net.load_state_dict(torch.load(load_path))
    phase_net.eval()

  
    # Save intermediate results
    result = []
    values = []


    # Predict per channel, so we save memory
    for c in range(3):
        imgs = torch.stack((img_1[c], img_2[c]), 0)

        # combine images into one big batch and then create the values and separate
        vals = pyr.filter(imgs.float())
        vals_list = separate_vals(vals, num_img)
        vals_t = vals_list[-1]
        vals_inp = get_concat_layers_inf(pyr, vals_list)
        inp = phase_net.normalize_vals(vals_inp)

        # Delete all old values to free memory
        del vals
        del vals_list
        del vals_t
        del vals_inp
        torch.cuda.empty_cache()

        # predicted intersected image of frame1 and frame2
        with torch.no_grad():
            vals_r = phase_net(inp)

        #if high_level:
        #    vals_r.high_level[:] = ada_hl[c]

        img_r = pyr.inv_filter(vals_r).detach().cpu()
        values.append(vals_r)
        result.append(img_r)

    # Put picture together
    #lab_pred = torch.cat(result, 0)
    vals_r = combine_values(values)
    lab_pred = pyr.inv_filter(vals_r).detach().cpu()
    
    rgb_pred = lab2rgb_single(lab_pred)
    img_p = rgb_pred[:, :shape_r[0], :shape_r[1]]
    
    # Save phase net prediction
    p_phase = torch.as_tensor(img_p).detach().cpu()
    p_phase = transforms.ToPILImage()(p_phase)
    p_phase.save(args.out_dir + f'{args.output}_prediction_phase_net.png')
    
    # Phase Net uncertainty map
    # First get high level residuals of both input frames
    ada_pred = torch.as_tensor(frame_ada_res_pad).to(device).permute(2, 0, 1)

    img_batch = torch.cat((ada_pred, rgb_pred.to(device), rgb_ground_truth.to(device)), 0)
    img_batch = img_batch.reshape(-1, img_batch.shape[1], img_batch.shape[2])
    num_vals = 3

    vals_batch = pyr.filter(img_batch.float())
    vals_ada, vals_ph, vals_g = separate_vals(vals_batch, num_vals)
    
    # Since it's not easily possible to predict high level residuals we extract the highest frequencies we can extract
    vals_high = get_last_value_levels(vals_ada, use_levels=1)
    vals_high_ph = get_last_value_levels(vals_r, use_levels=1)
    h_freq = pyr.inv_filter(vals_high).detach().cpu()[:, :shape_r[0], :shape_r[1]].mean(0)
    h_freq_ph = pyr.inv_filter(vals_high_ph).detach().cpu()[:, :shape_r[0], :shape_r[1]].mean(0)
    h_freq_diff = torch.abs(h_freq - h_freq_ph)
    h_freq_diff = (h_freq_diff*100).clamp(min=0, max=1.0)
    #k = np.ones((8, 8))/64
    #h_freq_diff = convolve(h_freq_diff, k, mode='constant', cval=0.0)
    #h_freq_diff = maximum_filter(h_freq_diff.numpy(), size=10)
    h_freq_diff = gaussian_filter(h_freq_diff, 5)
    ##h_freq_diff = median_filter(h_freq_diff, size=5)

    u_phase = torch.as_tensor(h_freq_diff)
    u_phase = u_phase.detach().cpu()
    u_phase = transforms.ToPILImage()(u_phase)
    u_phase.save(args.out_dir + f'{args.output}_uncertainty_phase_net.png')
    
    # For debugging
    vals_high_real = get_last_value_levels(vals_g, use_levels=1)
    h_freq_real = pyr.inv_filter(vals_high_real).detach().cpu()[:, :shape_r[0], :shape_r[1]].mean(0)
    print(h_freq_real.max())
    h_freq_diff_real = torch.abs(h_freq_real - h_freq_ph)
    h_freq_diff_real = (h_freq_diff_real*50).clamp(min=0, max=1.0)
    
    u_real = torch.as_tensor(h_freq_diff_real)
    u_real = u_real.detach().cpu()
    u_real = transforms.ToPILImage()(u_real)
    u_real.save(args.out_dir + f'{args.output}_uncertainty_real_phase_net.png')
    
    # Second adacof uncertainty map for finding artifacts
    vals_diff = subtract_values(vals_ph, vals_ada)
    vals_diff = get_first_value_levels(vals_diff, use_levels=10)
    freq_diff = pyr.inv_filter(vals_diff).detach().cpu()[:, :shape_r[0], :shape_r[1]].mean(0)*30
    
    u_ada = torch.as_tensor(freq_diff)
    u_ada = u_ada.detach().cpu()
    u_ada = transforms.ToPILImage()(u_ada)
    u_ada.save(args.out_dir + f'{args.output}_uncertainty_adacof2a.png')

    freq_diff_median = median_filter(freq_diff, size=50)
    freq_diff = torch.abs(freq_diff - freq_diff_median)
    freq_diff = (freq_diff * 5).clamp(0, 1)
    
    u_ada = torch.as_tensor(freq_diff)
    u_ada = u_ada.detach().cpu()
    u_ada = transforms.ToPILImage()(u_ada)
    u_ada.save(args.out_dir + f'{args.output}_uncertainty_adacof2b.png')

if __name__ == "__main__":
    main(None, None)
