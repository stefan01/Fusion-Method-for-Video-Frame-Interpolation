from torch.utils.data import DataLoader
import argparse
import torch
import random
import numpy as np
from types import SimpleNamespace
import datetime

from src.train.datareader import DBreader_Vimeo90k
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from src.fusion_net.trainer import Trainer
from src.train.pyramid import Pyramid

# import Models
from src.adacof.models import Model as AdaCofModel
from src.fusion_net.fusion_net import FusionNet, FusionNet2
from src.phase_net.phase_net import PhaseNet

parser = argparse.ArgumentParser(description='FusionNet-Pytorch')

# Parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./Trainset/vimeo/vimeo_triplet')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=1, help='max epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='seed')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=0, help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')


def main():
    # Get args and set device
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    out_dir = f"./output_fusion_net_3"

    # RNG init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    dataset = DBreader_Vimeo90k(args.train, random_crop=(256, 256))
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Pyramid and Network
    device = torch.device(f'cuda:{args.gpu_id}')
    pyr = Pyramid(
        height=12,
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )

    # Create Fusion Net
    fusion_net = FusionNet().to(device)

    # Load phase net
    phase_net = PhaseNet(pyr, device, num_img=2)
    phase_net.eval()
    phase_net.load_state_dict(torch.load('src/phase_net/phase_net.pt'))

    # Load adacof model
    adacof_args = SimpleNamespace(
        gpu_id=args.gpu_id,
        model='src.fusion_net.fusion_adacofnet',
        kernel_size=5,
        dilation=1,
        config='src/adacof/checkpoint/kernelsize_5/config.txt'
    )
    adacof_model = AdaCofModel(adacof_args)
    adacof_model.eval()
    checkpoint = torch.load('src/adacof/checkpoint/kernelsize_5/ckpt.pth', map_location=torch.device('cpu'))
    adacof_model.load(checkpoint['state_dict'])


    # Load model if given
    start_epoch = 0
    if args.load is not None:
        fusion_net.load_state_dict(torch.load(args.load))

    # Set trainer
    my_trainer = Trainer(args, train_loader, fusion_net, phase_net, adacof_model, my_pyr=pyr, lr=args.lr, weight_decay=args.weight_decay,
                                start_epoch=start_epoch, out_dir=out_dir)

    # Log training
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    # Train
    while not my_trainer.terminate():
        my_trainer.train()
        torch.save(fusion_net.state_dict(), out_dir + f'/checkpoint/model_{my_trainer.current_epoch}.pt')

    loss_hist = np.asarray(my_trainer.loss_history)
    np.savetxt(out_dir + '/loss_hist.txt', loss_hist)

    my_trainer.close()


if __name__ == "__main__":
    main()
