# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/train.py

from datareader import DBreader_Vimeo90k
from torch.utils.data import DataLoader
import argparse
import torch
from trainer import Trainer
import datetime
from phase_net import PhaseNet
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
import torchvision.transforms as transforms
from pyramid import Pyramid
import numpy as np
import random

parser = argparse.ArgumentParser(description='PhaseNet-Pytorch')

# Parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./Trainset/vimeo/vimeo_triplet')
parser.add_argument('--out_dir', type=str, default='./output_phase_net_train')
parser.add_argument('--load', type=str, default=None)

# Learning Options
parser.add_argument('--epochs', type=int, default=10, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--seed', type=int, default=1, help='Seed')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=0, help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

transform = transforms.Compose([transforms.ToTensor()])


def main():
    #torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    # RNG init
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Dataset
    dataset = DBreader_Vimeo90k(args.train, random_crop=(256, 256))
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Pyramid and Network
    device = utils.get_device()
    pyr = Pyramid(
        height=12,
        nbands=4,
        scale_factor=np.sqrt(2),
        device=device,
    )
    model = PhaseNet(pyr, device)

    start_epoch = 0
    if args.load is not None:
        model.load_state_dict(torch.load(args.load + f'/model_{start_epoch}.pt'))

    my_trainer = Trainer(args, train_loader, model, my_pyr=pyr, batch_size=args.batch_size, start_epoch=start_epoch)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(args.out_dir + '/config.txt', 'a') as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()
        torch.save(model.state_dict(), args.out_dir + f'/checkpoint/model_{my_trainer.current_epoch}.pt')

    loss_hist = np.asarray(my_trainer.loss_history)
    np.savetxt(args.out_dir + '/loss_hist.txt', loss_hist)

    my_trainer.close()


if __name__ == "__main__":
    main()
