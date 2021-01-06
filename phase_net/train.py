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

parser = argparse.ArgumentParser(description='PhaseNet-Pytorch')

# Parameters
# Hardware Setting
parser.add_argument('--gpu_id', type=int, default=0)

# Directory Setting
parser.add_argument('--train', type=str, default='./Trainset/vimeo/vimeo_triplet')
parser.add_argument('--out_dir', type=str, default='./output_phase_net_train')
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--test_input', type=str, default='./test_input/middlebury_others/input')
parser.add_argument('--gt', type=str, default='./test_input/middlebury_others/gt')

# Learning Options
parser.add_argument('--epochs', type=int, default=50, help='Max Epochs')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=20, help='learning rate decay per N epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

transform = transforms.Compose([transforms.ToTensor()])


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    dataset = DBreader_Vimeo90k(args.train, random_crop=(256, 256))
    # TestDB = Middlebury_other(args.test_input, args.gt)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
        checkpoint = torch.load(args.load)
        model.load(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

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
