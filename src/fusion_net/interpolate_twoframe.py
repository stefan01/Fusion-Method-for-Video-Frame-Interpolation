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

parser.add_argument('--checkpoint', type=str,
                    default='./src/fusion_net/fusion_net.pt')

parser.add_argument('--first_frame', type=str,
                    default='counter_examples/lights/001.png')
parser.add_argument('--second_frame', type=str,
                    default='counter_examples/lights/003.png')
parser.add_argument('--output_frame', type=str, default='./output.png')

parser.add_argument('--output_phase', action='store_true')
parser.add_argument('--output_adacof', action='store_true')
parser.add_argument('--output_frame_phase', type=str,
                    default='./output_phase.png')
parser.add_argument('--output_frame_adacof', type=str,
                    default='./output_adacof.png')

parser.add_argument('--model', type=int, default=1)

parser.add_argument('--high_level', action='store_true')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():
    interp(parser.parse_args())


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]


def interp(args, high_level=False):
    torch.cuda.set_device(args.gpu_id)
    # Warnings and device
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:{}'.format(args.gpu_id))

    # Adacof model
    if args.loaded_adacof_model:
        adacof_model = args.loaded_adacof_model
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
    img1 = np.array(Image.open(args.first_frame))
    img2 = np.array(Image.open(args.second_frame))

    # Crop images
    dim = 512
    img1 = crop_center(img1, dim, dim)
    img2 = crop_center(img2, dim, dim)

    # Import images
    rgb_frame1 = torch.as_tensor(img1).permute(
        2, 0, 1).float().to(device)/255
    rgb_frame2 = torch.as_tensor(img2).permute(
        2, 0, 1).float().to(device)/255
    shape_r = rgb_frame1.shape
    # 3, 256, 256

    # Build pyramid
    pyr = Pyramid(
        height=calc_pyr_height(rgb_frame1),
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )

    # Create PhaseNet
    num_img = 2
    load_path = './src/phase_net/phase_net.pt'

    phase_net = PhaseNet(pyr, device, num_img=num_img)
    phase_net.load_state_dict(torch.load(load_path))
    phase_net.eval()

    fusion_net = args.loaded_fusion_net
    # Create Fusion Net
    #fusion_net = FusionNet().to(device)
    # fusion_net.load_state_dict(torch.load(args.checkpoint))
    # fusion_net.eval()

    # Transform into lab space
    lab_frame1 = rgb2lab_single(rgb_frame1).to(device)
    lab_frame2 = rgb2lab_single(rgb_frame2).to(device)

    # Rgb frames
    rgb_frame1 = rgb_frame1.unsqueeze(0)
    rgb_frame2 = rgb_frame2.unsqueeze(0)

    with torch.no_grad():
        ada_frame1, ada_frame2, ada_pred, flow_var_map = adacof_model(
            rgb_frame1, rgb_frame2)

        ada_frame1 = ada_frame1.reshape(
            -1, ada_frame1.shape[2], ada_frame1.shape[3]).to(device).float()
        ada_frame2 = ada_frame2.reshape(
            -1, ada_frame2.shape[2], ada_frame2.shape[3]).to(device).float()
        ada_pred = ada_pred.reshape(-1,
                                    ada_pred.shape[2], ada_pred.shape[3]).to(device).float()
        flow_var_map = flow_var_map.squeeze(1)

    # PhaseNet input preparations
    img_batch = torch.cat((lab_frame1, lab_frame2), 0)
    num_vals = 2

    # Combine images into one big batch and then create the values and separate
    vals_batch = pyr.filter(img_batch.float())
    vals_list = separate_vals(vals_batch, num_vals)
    vals_input = get_concat_layers_inf(pyr, vals_list)
    input = phase_net.normalize_vals(vals_input)

    # Delete unnecessary vals
    del vals_batch
    del vals_list
    del vals_input
    torch.cuda.empty_cache()

    # PhaseNet Vals Prediction
    with torch.no_grad():
        vals_pred = phase_net(input)

    # RGB prediction without new high levels
    lab_pred = pyr.inv_filter(vals_pred)
    lab_pred = lab_pred.reshape(-1, 3,
                                lab_pred.shape[1], lab_pred.shape[2]).float()
    r_shape = lab_pred.shape
    rgb_pred = torch.cat([lab2rgb_single(l) for l in lab_pred], 0).to(device)

    # Exchange high levels of phase net, but copy before that uncertainty of phase
    phase_pred = rgb_pred.clone()

    # Phase Net uncertainty map
    ada_pred = torch.as_tensor(ada_pred).to(device)
    img_batch = torch.cat((ada_pred, rgb_pred), 0)
    num_vals = 2

    vals_batch = pyr.filter(img_batch.float())
    vals_ada, vals_ph = separate_vals(vals_batch, num_vals)

    vals_high = get_last_value_levels(vals_ada, use_levels=1)
    vals_high_ph = get_last_value_levels(vals_ph, use_levels=1)
    h_freq = pyr.inv_filter(vals_high).detach().cpu().reshape(r_shape).mean(1)
    h_freq_ph = pyr.inv_filter(
        vals_high_ph).detach().cpu().reshape(r_shape).mean(1)
    h_freq_diff = torch.abs(h_freq - h_freq_ph)
    h_freq_diff = (h_freq_diff*100).clamp(min=0, max=1.0)
    h_freq_diff = torch.stack(
        [torch.as_tensor(gaussian_filter(h, 5)) for h in h_freq_diff])
    phase_uncertainty = h_freq_diff.to(device)

    # Adacof uncertainty map for finding artifacts
    vals_diff = subtract_values(vals_ph, vals_ada)
    vals_diff = get_first_value_levels(vals_diff, use_levels=6)
    freq_diff = pyr.inv_filter(
        vals_diff).detach().cpu().reshape(r_shape).mean(1)*30
    freq_diff_median = torch.stack(
        [torch.as_tensor(median_filter(f, size=50)) for f in freq_diff])
    freq_diff = torch.abs(freq_diff - freq_diff_median)
    freq_diff = (freq_diff * 5).clamp(0, 1)
    ada_uncertainty = freq_diff.to(device)

    # Get baseline
    with torch.no_grad():
        _, _, ada_inbetween1, _ = adacof_model(
            rgb_frame1, phase_pred.reshape(r_shape).float())
        ada_inbetween1 = ada_inbetween1.to(device).float()

        _, _, ada_inbetween2, _ = adacof_model(
            phase_pred.reshape(r_shape).float(), rgb_frame2)
        ada_inbetween2 = ada_inbetween2.to(device).float()

        _, _, base, _ = adacof_model(ada_inbetween1, ada_inbetween2)
        base = base.to(device).float()

    if False:
        u_phase = torch.as_tensor(
            rgb_frame1.reshape(r_shape)[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_1.png')

        u_phase = torch.as_tensor(
            rgb_frame2.reshape(r_shape)[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_2.png')

        u_phase = torch.as_tensor(
            phase_pred.reshape(r_shape)[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_phase_pred.png')

        u_phase = torch.as_tensor(ada_pred.reshape(r_shape)[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_ada_pred.png')

        u_phase = torch.as_tensor(h_freq_diff[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_phase_unc.png')

        u_phase = torch.as_tensor(flow_var_map[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_var_flow.png')

        u_phase = torch.as_tensor(freq_diff[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_ada_unc.png')

        u_phase = torch.as_tensor(base[0]).detach().cpu()
        u_phase = transforms.ToPILImage()(u_phase)
        u_phase.save(f'./1test_baseline.png')

    # Fusion Net prediction
    phase_pred = phase_pred.reshape(r_shape).to(device).float()
    ada_pred = ada_pred.reshape(r_shape).to(device).float()

    if(args.output_phase):
        imwrite(phase_pred.squeeze(0).clone(),
                args.output_frame_phase, range=(0, 1))
    if(args.output_adacof):
        imwrite(ada_pred.squeeze(0).clone(),
                args.output_frame_adacof, range=(0, 1))

    other = torch.cat([lab_frame1.reshape(r_shape),
                       lab_frame2.reshape(r_shape)], 1).to(device).float()
    maps = torch.stack([ada_uncertainty, phase_uncertainty,
                        flow_var_map], 1).to(device).float()

    # Predict
    final_pred = fusion_net(base, ada_pred, phase_pred, other, maps)
    final_pred = final_pred.reshape(-1,
                                    final_pred.shape[2], final_pred.shape[3])

    imwrite(final_pred.clone(), args.output_frame, range=(0, 1))


if __name__ == "__main__":
    main(None, None)
