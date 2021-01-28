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
from src.train.loss import *
from src.train.utils import *
from src.train.transform import *
from types import SimpleNamespace

# import Models
from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet


DecompValues = namedtuple(
    'values',
    'high_level, '
    'phase, '
    'amplitude, '
    'low_level'
)


class Trainer:
    def __init__(self, args, train_loader, my_model, my_pyr, batch_size=1,
                 lR=0.001, weight_decay= 0.0, start_epoch=0,
                 m=0):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.model = my_model
        self.pyr = my_pyr
        self.batch_size = batch_size
        self.lR = lR
        self.wD = weight_decay
        self.current_epoch = start_epoch
        self.device = my_pyr.device
        self.m = m

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lR,
                                          weight_decay=self.wD)

        if not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)
        self.result_dir = self.args.out_dir + '/result'
        self.ckpt_dir = self.args.out_dir + '/checkpoint'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(self.args.out_dir + '/log.txt', 'w')
        self.loss_history = []

        if self.args.mode == "fusion":
            # Load AdaCoF
            adacof_args = SimpleNamespace(
                gpu_id=0,
                model='src.fusion_net.fusion_adacofnet',
                kernel_size=5,
                dilation=1,
                config='src/adacof/checkpoint/kernelsize_5/config.txt'
            )
            self.adacof_model = Model(adacof_args).eval()
            checkpoint = torch.load('src/adacof/checkpoint/kernelsize_5/ckpt.pth', map_location=torch.device('cpu'))
            self.adacof_model.load(checkpoint['state_dict'])

    def train(self):
        # Train
        for batch_idx, triple in enumerate(self.train_loader):
            # define heigth and width of the training image
            heigth = triple[0].shape[2]
            width = triple[0].shape[3]

            frame1 = []
            target = []
            frame2 = []
            for b in range(self.batch_size):
                frame1.append(rgb2lab(triple[0][b].reshape(-1, heigth, width)))
                target.append(rgb2lab(triple[1][b].reshape(-1, heigth, width)))
                frame2.append(rgb2lab(triple[2][b].reshape(-1, heigth, width)))
            frame1 = torch.cat(frame1, 0).to(self.device)
            target = torch.cat(target, 0).to(self.device)
            frame2 = torch.cat(frame2, 0).to(self.device)

            if self.args.mode == "fusion":
                ada_frame1 = triple[0].reshape(-1, heigth, width).to(self.device)
                ada_frame2 = triple[2].reshape(-1, heigth, width).to(self.device)
                with torch.no_grad():
                    frame_out1, frame_out2 = self.adacof_model(ada_frame1.unsqueeze(0), ada_frame2.unsqueeze(0))
                imgs = torch.cat((frame1, frame2, frame_out1.squeeze(0), frame_out2.squeeze(0), target), 0)
            else:
                imgs = torch.cat((frame1, frame2, target), 0)

            # combine images into one big batch and then create the values and separate
            vals = self.pyr.filter(imgs)
            vals_list = separate_vals(vals, 3 if self.args.mode == "phasenet" else 5)
            vals_t = vals_list[-1]
            vals_inp = get_concat_layers_inf(self.pyr, vals_list[:-1])
            inp = self.model.normalize_vals(vals_inp)

            # Delete unnecessary vals
            del vals
            del vals_list
            del vals_inp
            torch.cuda.empty_cache()

            # predicted intersected image of frame1 and frame2
            vals_o = self.model(inp, self.m)

            # Exchange parts for hierarchical training
            #vals_o = exchange_vals(vals_o, vals_t,  0, 10-self.m)

            # transform output of the network back to an image -> inverse steerable pyramid
            output = self.pyr.inv_filter(vals_o)

            # Calculate the loss
            loss, p1, p2 = get_loss(vals_o, vals_t, output, target, self.pyr)

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
                self.test(int(batch_idx/100))
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f} Percentages: {:<4f}, {:<4f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item(), p1, p2))
                torch.save(self.model.state_dict(), self.args.out_dir + f'/checkpoint/model_{self.current_epoch}_{int(batch_idx/100)}.pt')

        self.current_epoch += 1

    def test(self, idx=None):
        # Get test images
        frame1 = Image.open('Testset/Clip1/000.png')
        target = Image.open('Testset/Clip1/001.png')
        frame2 = Image.open('Testset/Clip1/002.png')

        # Images to tensors
        frame1 = TF.to_tensor(transforms.CenterCrop(256)(frame1))
        target = TF.to_tensor(transforms.CenterCrop(256)(target))
        frame2 = TF.to_tensor(transforms.CenterCrop(256)(frame2))

        ada_frame1 = frame1.to(self.device)
        ada_frame2 = frame2.to(self.device)

        # Bring images into LAB color space
        frame1 = rgb2lab(frame1).to(self.device)
        target = rgb2lab(target).to(self.device)
        frame2 = rgb2lab(frame2).to(self.device)

        if self.args.mode == "fusion":
            with torch.no_grad():
                frame_out1, frame_out2 = self.adacof_model(ada_frame1.unsqueeze(0), ada_frame2.unsqueeze(0))
            imgs = torch.cat((frame1, frame2, frame_out1.squeeze(0), frame_out2.squeeze(0), target), 0)
        else:
            imgs = torch.cat((frame1, frame2, target), 0)

        # combine images into one big batch and then create the values and separate
        vals = self.pyr.filter(imgs)
        vals_list = separate_vals(vals, 3 if self.args.mode == "phasenet" else 5)
        vals_t = vals_list[-1]
        vals_inp = get_concat_layers_inf(self.pyr, vals_list[:-1])
        inp = self.model.normalize_vals(vals_inp)

        # Forward pass through phase net
        self.model.eval()
        vals_o = self.model(inp, self.m)
        self.model.train()

        # Exchange parts for hierarchical training
        #vals_o = exchange_vals(vals_o, vals_t, 0, 10-self.m)

        # Reconstruct image and save
        img_r = self.pyr.inv_filter(vals_o)
        img_r = lab2rgb(img_r)
        img_r = img_r.detach().cpu()
        img_r = transforms.ToPILImage()(img_r)
        if idx is not None:
            name = f'/result/img_{self.current_epoch}_{idx}.png'
        else:
            name = f'/result/img_{self.current_epoch}.png'
        img_r.save(self.args.out_dir + name)

        # Save also truth
        #img_t = transforms.ToPILImage()(img_t.detach().cpu())
        #img_t.save(self.args.out_dir + f'/result/truth_{self.current_epoch}.png')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()

def get_loss(vals_o, vals_t, output, target, pyr, weighting_factor=0.01):
    """ PhaseNet special loss. """
    phase_loss = 0
    l1loss = nn.L1Loss()

    for idx, (phase_r, phase_g) in enumerate(zip(vals_o.phase, vals_t.phase)):
        phase_r_2 = phase_r.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)
        phase_g_2 = phase_g.reshape(-1, pyr.nbands, phase_r.shape[2], phase_r.shape[3]).permute(1, 0, 2, 3)

        for (orientation_r, orientation_g) in zip(phase_r_2, phase_g_2):
            delta_psi = torch.atan2(torch.sin(orientation_g - orientation_r), torch.cos(orientation_g - orientation_r))
            phase_loss += l1loss(delta_psi, torch.zeros(delta_psi.shape, device=delta_psi.device))

    #low_loss = l1loss(vals_o.low_level, vals_t.low_level)
    l_1 = l1loss(output, target)

    total_loss = l_1 + weighting_factor*phase_loss
    l_1_p = 100*l_1.detach() / total_loss
    phase_loss_p = 100*weighting_factor*phase_loss.detach() / total_loss

    return total_loss, l_1_p, phase_loss_p