# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/train.py
# Train Script

from torch.utils.data import DataLoader
import argparse
import torch
import datetime
import steerable.utils as utils
import torchvision.transforms as transforms
import numpy as np
import random
from src.train.datareader import DBreader_Vimeo90k
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from src.train.trainer import Trainer
from src.train.pyramid import Pyramid
from src.phase_net.phase_net import PhaseNet

parser = argparse.ArgumentParser(description='PhaseNet-Pytorch')

# Parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./Trainset/vimeo/vimeo_triplet')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=2, help='max epochs')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--m', type=int, default=None, help='layers to train from 0 to m')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=0, help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Method Settings
parser.add_argument('--mode', type=str, default='phase', help='phase, fusion')
parser.add_argument('--high_level', type=bool, default=False, help='replace high-level with adacof output or not')

transform = transforms.Compose([transforms.ToTensor()])


def main():
    # Get args and set device
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    hl_str = '_hl' if args.high_level else ''
    out_dir = f"./output_{args.mode}_net{hl_str}"

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

    # PhaseNet
    model = PhaseNet(pyr, device, num_img= 4 if args.mode == 'fusion' else 2)
    m = 10
    #model.set_layers(0, 5, freeze=True)
    #if args.m is not None:
        #m = args.m
        #model.set_layers(0, m, freeze=True)
        #model.set_layers(m+1, 9, freeze=True)

    # Load model if given
    start_epoch = 0
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # Set trainer
    my_trainer = Trainer(args, train_loader, model, my_pyr=pyr, lr=args.lr, weight_decay=args.weight_decay,
                                start_epoch=start_epoch, out_dir=out_dir, m=m)

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
        my_trainer.test()
        torch.save(model.state_dict(), out_dir + f'/checkpoint/model_{my_trainer.current_epoch}.pt')

    loss_hist = np.asarray(my_trainer.loss_history)
    np.savetxt(out_dir + '/loss_hist.txt', loss_hist)

    my_trainer.close()


if __name__ == "__main__":
    main()
