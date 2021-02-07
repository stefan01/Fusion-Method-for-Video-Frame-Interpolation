import argparse
from PIL import Image
import torch
from torchvision import transforms
import os
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable

from src.train.transform import *
from src.train.utils import *
from src.train.pyramid import Pyramid
import warnings
import numpy as np
from collections import namedtuple
from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet
from types import SimpleNamespace
from src.fusion_net.fusion_net import FusionNet

import src.fusion_net.interpolate_twoframe as fusion_interp

parser = argparse.ArgumentParser(description='Video Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--adacof_model', type=str, default='src.fusion_net.fusion_adacofnet')
parser.add_argument('--adacof_checkpoint', type=str, default='./src/adacof/checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--adacof_config', type=str, default='./src/adacof/checkpoint/kernelsize_5/config.txt')
parser.add_argument('--adacof_kernel_size', type=int, default=5)
parser.add_argument('--adacof_dilation', type=int, default=1)

parser.add_argument('--checkpoint', type=str, default='./src/fusion_net/fusion_net1.pt')

parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
parser.add_argument('--zpad', type=int, default=3, help='zero padding of frame name.')

parser.add_argument('--input_video', type=str, default='./sample_video')
parser.add_argument('--output_video', type=str, default='./interpolated_video')

parser.add_argument('--model', type=int, default=1)

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main(loaded_adacof_model=None, loaded_fusion_net=None):
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    # Warnings and device
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:{}'.format(args.gpu_id))

    print('Loading the model...')

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
        checkpoint = torch.load(args.adacof_checkpoint, map_location=torch.device('cpu'))
        adacof_model.load(checkpoint['state_dict'])

    base_dir = args.input_video

    if not os.path.exists(args.output_video):
        os.makedirs(args.output_video)

    frame_len = len([name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))])

    for idx in range(frame_len - 1):
        idx += args.index_from
        print(idx, '/', frame_len - 1, end='\r')

        frame_name1 = base_dir + '/' + str(idx).zfill(args.zpad) + '.png'
        frame_name2 = base_dir + '/' + str(idx + 1).zfill(args.zpad) + '.png'

        frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
        #frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

        #model.eval()
        #frame_out = model(frame1, frame2)

        # interpolate
        with torch.no_grad():
            fusion_interp.interp(SimpleNamespace(
                gpu_id=args.gpu_id,
                adacof_model=args.adacof_model,
                adacof_kernel_size=args.adacof_kernel_size,
                adacof_dilation=args.adacof_dilation,
                first_frame=frame_name1,
                second_frame=frame_name2,
                output_frame=args.output_video + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.png',
                adacof_checkpoint=args.adacof_checkpoint,
                adacof_config=args.adacof_config,
                checkpoint=args.checkpoint,
                model=args.model,
                loaded_adacof_model=adacof_model,
                loaded_fusion_net=loaded_fusion_net
            ))
        torch.cuda.empty_cache()

        imwrite(frame1.clone(), args.output_video + '/' + str((idx - args.index_from) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))
        #imwrite(frame_out.clone(), args.output_video + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))

    # last frame
    print(frame_len - 1, '/', frame_len - 1)
    frame_name_last = base_dir + '/' + str(frame_len + args.index_from - 1).zfill(args.zpad) + '.png'
    frame_last = to_variable(transform(Image.open(frame_name_last)).unsqueeze(0))
    imwrite(frame_last.clone(), args.output_video + '/' + str((frame_len - 1) * 2 + args.index_from).zfill(args.zpad) + '.png', range=(0, 1))


if __name__ == "__main__":
    main(None, None)
