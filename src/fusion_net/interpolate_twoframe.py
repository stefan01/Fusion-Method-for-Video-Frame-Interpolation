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
                    default='./src/fusion_net/fusion_net1.pt')

parser.add_argument('--first_frame', type=str,
                    default='./sample_twoframe/0.png')
parser.add_argument('--second_frame', type=str,
                    default='./sample_twoframe/1.png')
parser.add_argument('--output_frame', type=str, default='./output.png')

parser.add_argument('--model', type=int, default=1)

parser.add_argument('--high_level', action='store_true')

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
    img_1 = np.array(Image.open(args.first_frame))
    img_2 = np.array(Image.open(args.second_frame))
    shape_r = img_1.shape

    with torch.no_grad():
        frame_out1, frame_out2, ada_res, uncertainty_mask = adacof_model(
            torch.as_tensor(img_1).permute(
                2, 0, 1).float().unsqueeze(0).to(device)/255,
            torch.as_tensor(img_2).permute(2, 0, 1).float().unsqueeze(0).to(device)/255)

        frame_out1, frame_out2 = frame_out1.squeeze(0).permute(
            1, 2, 0).cpu().numpy(), frame_out2.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_ada_res = ada_res.squeeze(0).permute(1, 2, 0).cpu().numpy()

        frame_out1_pad = pad_img(frame_out1)
        frame_out2_pad = pad_img(frame_out2)
        frame_ada_res_pad = pad_img(frame_ada_res)

    # Normalize and pad images
    img_1_pad = pad_img(img_1/255)
    img_2_pad = pad_img(img_2/255)

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
    if args.high_level:
        ada_pyr = pyr.filter(frame_res)
        ada_hl = ada_pyr.high_level.clone().detach()
        del ada_pyr
        torch.cuda.empty_cache()

    # Create FusionNet
    if args.model == 1:
        num_img = 4
        load_path = './src/fusion_net/fusion_net1.pt'
    if args.model == 2:
        num_img = 3
        load_path = './src/fusion_net/fusion_net2.pt'
    if args.model == 3:
        num_img = 2
        load_path = './src/phase_net/phase_net.pt'

    if loaded_fusion_net:
        fusion_net = loaded_fusion_net
        fusion_net.pyr = pyr
    else:
        fusion_net = PhaseNet(pyr, device, num_img=num_img)
        fusion_net.load_state_dict(torch.load(load_path))
        fusion_net.eval()

    result = []

    # Predict per channel, so we save memory
    for c in range(3):
        if args.model == 1:
            imgs = torch.stack((img_1[c], img_2[c], frame_1[c], frame_2[c]), 0)
        elif args.model == 2:
            imgs = torch.stack((img_1[c], img_2[c], frame_res[c]), 0)
        elif args.model == 3:
            imgs = torch.stack((img_1[c], img_2[c]), 0)

        # combine images into one big batch and then create the values and separate
        vals = pyr.filter(imgs.float())
        vals_list = separate_vals(vals, num_img)
        vals_t = vals_list[-1]
        vals_inp = get_concat_layers_inf(pyr, vals_list)
        inp = fusion_net.normalize_vals(vals_inp)

        if args.model == 3:
            hl_1 = vals_list[0].high_level.squeeze(1)
            hl_2 = vals_list[1].high_level.squeeze(1)
            hl_p = vals_list[2].high_level.squeeze(1)
            phase_uncertainty = hl_p - (hl_1+hl_2)/2
            phase_uncertainty = phase_uncertainty.reshape(-1, 3, phase_uncertainty.shape[1], phase_uncertainty.shape[2])
            phase_uncertainty = phase_uncertainty.mean(1).unsqueeze(1)

            del hl_1
            del hl_2
            del hl_p

        # Delete all old values to free memory
        del vals
        del vals_list
        del vals_t
        del vals_inp

        torch.cuda.empty_cache()

        # predicted intersected image of frame1 and frame2
        with torch.no_grad():
            vals_r = fusion_net(inp)

        if high_level:
            vals_r.high_level[:] = ada_hl[c]

        img_r = pyr.inv_filter(vals_r).detach().cpu()
        result.append(img_r)

    # Put picture together
    result = torch.cat(result, 0)
    img_p = lab2rgb_single(result)

    img_p = img_p[:, :shape_r[0], :shape_r[1]]

    # Fusion Net prediction
    if args.model == 3:
        # Fusion Net prediction
        fusion_net3 = FusionNet().to(device)
        # fusion_net3.load_state_dict(torch.load(
        #     './src/fusion_net/fusion_net3.pt'))
        fusion_net3.load_state_dict(torch.load(args.fusion_checkpoint))

        phase_pred = result[:, :shape_r[0], :shape_r[1]].to(
            device).unsqueeze(0).float()
        ada_pred = rgb2lab(ada_res).to(device).float()
        other = torch.cat([img_1[:, :shape_r[0], :shape_r[1]].reshape(-1, 3, shape_r[0], shape_r[1]),
                           img_2[:, :shape_r[0], :shape_r[1]].reshape(-1, 3, shape_r[0], shape_r[1])], 1).float()

        if self.args.model == "alpha":
            final_pred, _ = fusion_net3(ada_pred, phase_pred, other, uncertainty_mask, phase_uncertainty, mode="alpha")
        else:
            final_pred = fusion_net3(ada_pred, phase_pred, other, uncertainty_mask, phase_uncertainty, mode=args.mode)
        img_p = final_pred.reshape(-1,
                                   final_pred.shape[2], final_pred.shape[3])
        img_p = lab2rgb_single(img_p)

    imwrite(img_p.clone(), args.output_frame, range=(0, 1))


if __name__ == "__main__":
    main(None, None)
