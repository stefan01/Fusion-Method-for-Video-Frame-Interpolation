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
from src.phase_net.phase_net import PhaseNet

parser = argparse.ArgumentParser(description='Two-frame Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--adacof_model', type=str, default='src.fusion_net.fusion_adacofnet')
parser.add_argument('--adacof_checkpoint', type=str, default='./src/adacof/checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--adacof_config', type=str, default='./src/adacof/checkpoint/kernelsize_5/config.txt')
parser.add_argument('--adacof_kernel_size', type=int, default=5)
parser.add_argument('--adacof_dilation', type=int, default=1)

parser.add_argument('--checkpoint', type=str, default='./src/phase_net/phase_net.pt')

parser.add_argument('--first_frame', type=str, default='./sample_twoframe/0.png')
parser.add_argument('--second_frame', type=str, default='./sample_twoframe/1.png')
parser.add_argument('--output_frame', type=str, default='./output.png')

transform = transforms.Compose([transforms.ToTensor()])

def main():
    interp(parser.parse_args())

def interp(args):
    torch.cuda.set_device(args.gpu_id)
    # Warnings and device
    warnings.filterwarnings("ignore")
    device = torch.device('cuda:{}'.format(args.gpu_id))

    # Adacof model
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

    # Import images
    img_1 = np.array(Image.open(args.first_frame))
    img_2 = np.array(Image.open(args.second_frame))
    shape_r = img_1.shape

    with torch.no_grad():
        frame_out1, frame_out2 = adacof_model(
            torch.as_tensor(img_1).permute(2, 0, 1).float().unsqueeze(0).to(device)/255,
            torch.as_tensor(img_2).permute(2, 0, 1).float().unsqueeze(0).to(device)/255)
        frame_out1, frame_out2 = frame_out1.squeeze(0), frame_out2.squeeze(0)

    # Normalize and pad images
    img_1 = pad_img(img_1/255)
    img_2 = pad_img(img_2/255)
    
    # To tensors
    img_1 = rgb2lab(torch.as_tensor(img_1).permute(2, 0, 1).float()).to(device)
    img_2 = rgb2lab(torch.as_tensor(img_2).permute(2, 0, 1).float()).to(device)


    # Build pyramid
    pyr = Pyramid(
        height=calc_pyr_height(img_1),
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )

    # Create PhaseNet
    phase_net = PhaseNet(pyr, device)
    phase_net.load_state_dict(torch.load(args.checkpoint))
    phase_net.eval()

    result = []

    # Predict per channel, so we save memory
    for c in range(3):
        # Filter images and normalize
        vals_1 = pyr.filter(img_1[c].unsqueeze(0))
        vals_2 = pyr.filter(img_2[c].unsqueeze(0))
        vals_1_2 = get_concat_layers(pyr, vals_1, vals_2)
        vals_normalized = phase_net.normalize_vals(vals_1_2)

        # Delete all old values to free memory
        del vals_1
        del vals_2
        del vals_1_2
        torch.cuda.empty_cache()

        # Predict intermediate frame
        with torch.no_grad():
            vals_r = phase_net(vals_normalized)

        img_r = pyr.inv_filter(vals_r).detach().cpu()
        result.append(img_r)

    # Put picture together
    result = torch.cat(result, 0)
    img_p = lab2rgb(result)

    img_p = img_p[:, :shape_r[0], :shape_r[1]]

    imwrite(img_p.clone(), args.output_frame, range=(0, 1))


if __name__ == "__main__":
    main()
