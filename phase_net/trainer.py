# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/trainer.py

import os
import torch
from torch import nn
# from loss import calc_loss
from collections import namedtuple
from PIL import Image
import pickle
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from loss import *


DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)


class Trainer:
    def __init__(self, args, train_loader, my_model, my_pyr, batch_size=1,
                 lR=0.001, weight_decay= 0.0001, start_epoch=0,
                 epoch=10, show_Image=False):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.model = my_model
        self.pyr = my_pyr
        self.batch_size = batch_size
        self.lR = lR
        self.wD = weight_decay
        self.epoch = epoch
        self.current_epoch = start_epoch
        self.show_Image = show_Image
        self.device = my_pyr.device

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lR,
                                          weight_decay=self.wD)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.result_dir = args.out_dir + '/result'
        self.ckpt_dir = args.out_dir + '/checkpoint'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.out_dir + '/log.txt', 'w')
        self.loss_history = []

    def train(self):
        # Train
        for batch_idx, triple in enumerate(self.train_loader):
            # define heigth and width of the training image
            heigth = triple[0].shape[2]
            width = triple[0].shape[3]

            # get input data of first and second frame and the corresponding target image
            frame1 = triple[0].to(self.device).reshape((-1, heigth, width))
            target = triple[1].to(self.device).reshape((-1, heigth, width))
            frame2 = triple[2].to(self.device).reshape((-1, heigth, width))

            # combine images into one big batch and then create the values and separate
            imgs = torch.cat((frame1, frame2, target), 0)
            vals = self.pyr.filter(imgs)
            vals_1, vals_2, vals_t = separate_vals(vals)
            inp = get_concat_layers(self.pyr, vals_1, vals_2)

            # predicted intersected image of frame1 and frame2
            vals_o = self.model(inp)

            # Exchange parts for hierarchical training
            vals_o = exchange_vals(vals_o, vals_t, 0, 10)

            # transform output of the network back to an image -> inverse steerable pyramid
            output = self.pyr.inv_filter(vals_o)

            loss, p1, p2 = get_loss(vals_o, vals_t, output, target, self.pyr)  # input, target

            # === The backpropagation
            # Reset the gradients
            self.optimizer.zero_grad()
            # Calculate new gradients with backpropagation
            loss.backward()
            # Tune weights accoring to optimizer (it has the learnrate and weight decay as defined above)
            self.optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.cpu().detach().numpy()
            # Append to loss history
            self.loss_history.append(loss)

            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f} Percentages: {:<4f}, {:<4f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item(), p1, p2))

                if self.show_Image:
                    self.show_img(output)

        self.current_epoch += 1

    def test(self):
        # Get test images
        img = Image.open('Testset/Clip1/000.png')
        img_t = Image.open('Testset/Clip1/001.png')
        img2 = Image.open('Testset/Clip1/002.png')

        # Images to tensors
        img = TF.to_tensor(transforms.CenterCrop(256)(img)).to(self.device)
        img_t = TF.to_tensor(transforms.CenterCrop(256)(img_t)).to(self.device)
        img2 = TF.to_tensor(transforms.CenterCrop(256)(img2)).to(self.device)

        # Get values
        vals1 = self.pyr.filter(img)
        vals2 = self.pyr.filter(img2)
        vals_t = self.pyr.filter(img_t)

        # Concat values
        vals = get_concat_layers(self.pyr, vals1, vals2)

        # Forward pass through phase net
        vals_o = self.model(vals)

        # Exchange parts for hierarchical training
        vals_o = exchange_vals(vals_o, vals_t, 0, 10)

        # Reconstruct image and save
        img_r = self.pyr.inv_filter(vals_o)
        img_r = img_r.detach().cpu()
        img_r = transforms.ToPILImage()(img_r)
        img_r.save(self.args.out_dir + f'/result/img_{self.current_epoch}.png')

        # Save also truth
        #img_t = transforms.ToPILImage()(img_t.detach().cpu())
        #img_t.save(self.args.out_dir + f'/result/truth_{self.current_epoch}.png')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()

    def show_img(self, img):
        img_p = img.detach().cpu()
        transforms.ToPILImage()(img_p).show()

def get_loss(vals_o, vals_t, output, target, pyr, weighting_factor=0.1):
    """ PhaseNet special loss. """
    phase_losses = []
    l1loss = nn.L1Loss()

    for (phase_r, phase_g) in zip(vals_o.phase, vals_t.phase):
        #delta_psi = phase_r - phase_g
        #phase_losses.append(torch.norm(delta_psi, 1))
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_losses.append(torch.norm(delta_psi, 1))

    phase_loss = torch.stack(phase_losses, 0).sum(0)

    l_1 = l1loss(output, target)

    total_loss = l_1 + weighting_factor * phase_loss
    l_1_p = 100*l_1.detach() / total_loss
    phase_loss_p = 100*weighting_factor * phase_loss.detach() / total_loss

    return total_loss, l_1_p, phase_loss_p