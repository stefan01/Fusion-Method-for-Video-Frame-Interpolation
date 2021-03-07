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
from src.phase_net.architecture import PhaseNet
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='PhaseNet-Pytorch')

# Parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str,
                    default='./Trainset/vimeo/vimeo_triplet')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=1, help='max epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--m', type=int, default=None,
                    help='layers to train from 0 to m')
parser.add_argument('--m_update', type=int, default=500,
                    help='number of batches after updating m')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=0,
                    help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=float,
                    default=0, help='weight decay')

# Method Settings
parser.add_argument('--mode', type=str, default='phase', help='phase, fusion')
parser.add_argument('--high_level', type=bool, default=False,
                    help='replace high-level with adacof output or not')
parser.add_argument('--model', type=int, default=0,
                    help='Define which fusion model')

transform = transforms.Compose([transforms.ToTensor()])


def main():
    # Get args and set device
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    hl_str = '_hl' if args.high_level else ''
    fusion_model = '_' + str(args.model) if args.model != 0 else ''
    out_dir = f"./output_{args.mode}_net{hl_str}{fusion_model}"

    # RNG init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    dataset = DBreader_Vimeo90k(args.train, random_crop=(256, 256))
    train_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Pyramid and Network
    device = torch.device(f'cuda:{args.gpu_id}')
    pyr = Pyramid(
        height=12,
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )

    # PhaseNet
    if args.mode == 'fusion':
        num = 4 if args.model == 0 else 3
    else:
        num = 2
    model = PhaseNet(12, device, num_img=num)
    m = 10

    if args.m is not None:
        m = args.m

    # Load model if given
    start_epoch = 0
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))

    # Set trainer
    my_trainer = Trainer(args, train_loader, model, pyr=pyr, lr=args.lr, weight_decay=args.weight_decay,
                         start_epoch=start_epoch, out_dir=out_dir, m=m, m_update=args.m_update)

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
        torch.save(model.state_dict(), out_dir +
                   f'/checkpoint/model_{my_trainer.current_epoch}.pt')

    # Delete Testfiles
    os.remove(my_trainer.out_dir + '/log_train.txt')
    os.remove(my_trainer.out_dir + '/loss_graph_train.png')

    loss_hist = np.asarray(my_trainer.loss_history)
    np.savetxt(my_trainer.out_dir + '/log.txt', loss_hist)
    plt.plot([i for i in range(len(loss_hist))], loss_hist)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    model_idx = '_' + str(args.model) if args.mode == 'fusion' else ''
    plt.title(
        f'Loss_{args.mode}{model_idx}_{args.epochs}_{my_trainer.max_step}')
    plt.savefig(my_trainer.out_dir + '/loss_graph.png')

    my_trainer.close()


if __name__ == "__main__":
    main()
