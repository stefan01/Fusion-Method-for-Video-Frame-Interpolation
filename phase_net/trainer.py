v# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/trainer.py

import os
import torch
from torch import nn
# from loss import calc_loss
from collections import namedtuple


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

            # create steerable pyramid for the input frames
            fram1_pyr = self.pyr.filter(frame1)
            fram2_pyr = self.pyr.filter(frame2)

            inp = get_concat_layers(self.pyr, fram1_pyr, fram2_pyr)

            # predicted intersected image of fram1 and frame2
            output_pyr = self.model(inp)

            # transform output of the network back to an image -> inverse steerable pyramid
            output = self.pyr.inv_filter(output_pyr)

            loss = calc_loss(output_pyr, output, target, self.pyr)  # input, target
            #(img1, img2, img_g, pyr, phase_net, weighting_factor=0.1)

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
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item()))

                if self.show_Image:
                    self.show_img(output)

        self.current_epoch += 1

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()

    def show_img(self, img):
        img_p = img.detach().cpu()
        transforms.ToPILImage()(img_p).show()

def get_concat_layers(pyr, vals1, vals2):
    nbands = pyr.nbands

    vals1_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.amplitude]
    vals2_amplitude = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.amplitude]

    vals1_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals1.phase]
    vals2_phase = [x.reshape((int(x.shape[0] / nbands), nbands, x.shape[2], x.shape[3])) for x in vals2.phase]

    high_level = torch.cat((vals1.high_level, vals2.high_level), 1)
    low_level = torch.cat((vals1.low_level, vals2.low_level), 1)
    phase = [torch.cat((a, b), 1) for (a, b) in zip(vals1_phase, vals2_phase)]
    amplitude = [torch.cat((a, b), 1) for (a, b) in zip(vals1_amplitude, vals2_amplitude)]

    return DecompValues(
        high_level=high_level,
        low_level=low_level,
        phase=phase[::-1],
        amplitude=amplitude[::-1]
    )

def calc_loss(vals_r, output, target, pyr, weighting_factor=0.1):
    vals_g = pyr.filter(target)

    phase_losses = []

    for (phase_r, phase_g) in zip(vals_r.phase, vals_g.phase):
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_losses.append(torch.norm(delta_psi, 1))

    phase_loss = torch.stack(phase_losses, 0).sum(0)

    l_1 = torch.norm(target-output, p=1)

    return l_1 + weighting_factor * phase_loss