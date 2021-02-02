# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/trainer.py

import os
import torch
from torch import nn
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from src.train.loss import *
from src.train.utils import *
from src.train.transform import *
from src.train.loss import *
from types import SimpleNamespace
from skimage import io

# import Models
from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet

class Trainer:
    """
    Trainer for fusion and phase net
    """

    def __init__(self, args, train_loader, my_model, my_pyr, lr=0.001,
                weight_decay=0.0, start_epoch=0, out_dir='.', m=0):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.model = my_model
        self.pyr = my_pyr
        self.current_epoch = start_epoch
        self.device = my_pyr.device
        self.m = m
        self.m_update = 100
        self.out_dir = out_dir

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.result_dir = self.out_dir + '/result'
        self.ckpt_dir = self.out_dir + '/checkpoint'
        self.create_folders()

        self.logfile = open(self.out_dir + '/log.txt', 'w')
        self.loss_history = []

        # Load AdaCoF
        if self.args.mode == 'fusion' or self.args.high_level:
            adacof_args = SimpleNamespace(
                gpu_id=self.args.gpu_id,
                model='src.fusion_net.fusion_adacofnet',
                kernel_size=5,
                dilation=1,
                config='src/adacof/checkpoint/kernelsize_5/config.txt'
            )
            self.adacof_model = Model(adacof_args)
            self.adacof_model.eval()
            checkpoint = torch.load('src/adacof/checkpoint/kernelsize_5/ckpt.pth', map_location=torch.device('cpu'))
            self.adacof_model.load(checkpoint['state_dict'])

    def create_folders(self):
        """ Checks whether the folders for results and checkpoints already exists or creates them. """
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def predict(self, lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target):
        """ Given batch of input images, predict the intermediate images. """

        # If we need the images of adacof for high_level in pyramid or to concat with fames in phasenet
        if self.args.mode == "fusion" or self.args.high_level:
            with torch.no_grad():
                ada_frame1, ada_frame2, ada_pred = self.adacof_model(rgb_frame1, rgb_frame2)
                ada_frame1 = rgb2lab(ada_frame1.reshape(-1, 3, ada_frame1.shape[2], ada_frame1.shape[3]))
                ada_frame2 = rgb2lab(ada_frame2.reshape(-1, 3, ada_frame2.shape[2], ada_frame2.shape[3]))
                ada_pred = rgb2lab(ada_pred.reshape(-1, 3, ada_pred.shape[2], ada_pred.shape[3]))
                 
                ada_frame1 = ada_frame1.reshape(-1, ada_frame1.shape[2], ada_frame1.shape[3]).to(self.device).float()
                ada_frame2 = ada_frame2.reshape(-1, ada_frame2.shape[2], ada_frame2.shape[3]).to(self.device).float()
                ada_pred = ada_pred.reshape(-1, ada_pred.shape[2], ada_pred.shape[3]).to(self.device).float()

        # Depending on modus (Fusion: concat adacof and phasenet images) concat number of images.
        if self.args.mode == 'fusion' and self.args.model == 1:
            img_batch = torch.cat((lab_frame1, lab_frame2, ada_frame1, ada_frame2, target), 0)
            num_vals = 5
        elif self.args.mode == 'fusion' and self.args.model == 2:
            img_batch = torch.cat((lab_frame1, lab_frame2, ada_pred, target), 0)
            num_vals = 4
        elif self.args.mode == 'phase':
            img_batch = torch.cat((lab_frame1, lab_frame2, target), 0)
            num_vals = 3

        # Combine images into one big batch and then create the values and separate
        vals_batch = self.pyr.filter(img_batch.float())
        vals_list = separate_vals(vals_batch, num_vals)
        vals_target = vals_list[-1]
        vals_input = get_concat_layers_inf(self.pyr, vals_list[:-1])
        input = self.model.normalize_vals(vals_input)

        # Delete unnecessary vals
        del vals_batch
        del vals_list
        del vals_input
        torch.cuda.empty_cache()

        # Predicted intersected image of frame1 and frame2
        vals_pred = self.model(input, self.m)

        # Exchange parts for hierarchical training
        vals_pred = exchange_vals(vals_pred, vals_target,  0, 10-self.m)

        # If high_level is True, we use the high_level image of adacof
        if self.args.high_level:
            ada_pyr = self.pyr.filter(ada_pred.squeeze(0))
            vals_pred.high_level[:] = ada_pyr.high_level

        # Transform output of the network back to an image -> inverse steerable pyramid
        prediction = self.pyr.inv_filter(vals_pred)

        return prediction, vals_pred, vals_target

    def train(self):
        """ Train the model. """
        for batch_idx, triple in enumerate(self.train_loader):
            # Get height and width of the training image
            h = triple[0].shape[2]
            w = triple[0].shape[3]
            hw = (h, w)

            # Transform into lab space
            lab_frame1 = rgb2lab(triple[0]).reshape((-1,) + hw).to(self.device)
            target = rgb2lab(triple[1]).reshape((-1,) + hw).to(self.device)
            lab_frame2 = rgb2lab(triple[2]).reshape((-1,) + hw).to(self.device)

            # Transform rgb images for adacof
            rgb_frame1 = triple[0].reshape((1, -1,) + hw).to(self.device)
            rgb_frame2 = triple[2].reshape((1, -1,) + hw).to(self.device)

            # Predict intermediate frames
            prediction, vals_pred, vals_target = self.predict(lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target)

            # Calculate the loss
            loss, p1, p2 = get_loss(vals_pred, vals_target, prediction, target, self.pyr)

            # Update net using backprop and gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.cpu().detach().numpy()
            self.loss_history.append(loss)

            if batch_idx % 100 == 0:
                self.test(idx=int(batch_idx/100), paths=['counter_examples/basketball/pad_00033.jpg', 'counter_examples/basketball/pad_00034.jpg', 'counter_examples/basketball/pad_00035.jpg'], name='basketball')
                self.test(idx=int(batch_idx/100), paths=['counter_examples/Clip11/pad_033.png', 'counter_examples/Clip11/pad_034.png', 'counter_examples/Clip11/pad_035.png'], name='Clip11')
                self.test(idx=int(batch_idx/100), paths=['counter_examples/Clip2/pad_010.png', 'counter_examples/Clip2/pad_011.png', 'counter_examples/Clip2/pad_012.png'], name='Clip2')
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f} Percentages: {:<4f}, {:<4f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item(), p1, p2))
                torch.save(self.model.state_dict(), self.out_dir + f'/checkpoint/model_{self.current_epoch}_{int(batch_idx/100)}.pt')

            # Hierarchical training update
            if batch_idx % self.m_update == 0 and batch_idx > 0 and self.m < 10:
                self.m += 1
                #if self.m <= 7:
                #  self.model.set_layers(0, self.m-1, freeze=False)

        self.current_epoch += 1

    def test(self, idx=None, paths=None, name=''):
        # If no paths are given to test, break
        if paths is None:
            return
        
        # Get test images
        frame1 = io.imread(paths[0])
        target = io.imread(paths[1])
        frame2 = io.imread(paths[2])
        
        if frame1.shape[-1] == 4:
            frame1 = frame1[:,:,:3]
            target = target[:,:,:3]
            frame2 = frame2[:,:,:3]

        # Images to tensors
        frame1 = torch.as_tensor(frame1).permute(2, 0, 1)/255
        target = torch.as_tensor(target).permute(2, 0, 1)/255
        frame2 = torch.as_tensor(frame2).permute(2, 0, 1)/255
        rgb_frame1 = frame1.unsqueeze(0).float().to(self.device)
        rgb_frame2 = frame2.unsqueeze(0).float().to(self.device)

        # Bring images into LAB color space
        lab_frame1 = rgb2lab_single(frame1).squeeze(0).to(self.device)
        target = rgb2lab_single(target).squeeze(0).to(self.device)
        lab_frame2 = rgb2lab_single(frame2).squeeze(0).to(self.device)

        # Predict intermediate frames
        self.model.eval()
        with torch.no_grad():
            prediction, vals_pred, vals_target = self.predict(lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target)
        self.model.train()

        # Reconstruct image and save
        img = lab2rgb_single(prediction)
        img = img.detach().cpu()
        img = transforms.ToPILImage()(img)
        if idx is not None:
            name = f'/result/{name}_{self.current_epoch}_{idx}.png'
        else:
            name = f'/result/{name}_{self.current_epoch}.png'
        img.save(self.out_dir + name)

        # Save also truth
        #img_t = transforms.ToPILImage()(img_t.detach().cpu())
        #img_t.save(self.out_dir + f'/result/truth_{self.current_epoch}.png')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()

