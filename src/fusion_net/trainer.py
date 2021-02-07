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
from piq import ssim, SSIMLoss, LPIPS
from types import SimpleNamespace
from skimage import io

# import Models
from src.adacof.models import Model
from src.phase_net.phase_net import PhaseNet

class Trainer:
    """
    Trainer for fusion and phase net
    """

    def __init__(self, args, train_loader, fusion_net, phase_net, adacof_model, my_pyr, lr=0.001,
                weight_decay=0.0, start_epoch=0, out_dir='.'):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.pyr = my_pyr
        self.current_epoch = start_epoch
        self.device = my_pyr.device
        self.criterion = LPIPS()

        # Models
        self.fusion_net = fusion_net
        self.phase_net = phase_net
        self.adacof_model = adacof_model

        self.out_dir = out_dir
        self.optimizer = torch.optim.Adam(self.fusion_net.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.result_dir = self.out_dir + '/result'
        self.ckpt_dir = self.out_dir + '/checkpoint'
        self.create_folders()

        self.logfile = open(self.out_dir + '/log.txt', 'w')
        self.loss_history = []

        

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

        # Adacof prediction
        with torch.no_grad():
            ada_frame1, ada_frame2, ada_pred, uncertainty_mask = self.adacof_model(rgb_frame1, rgb_frame2)

            ada_frame1 = rgb2lab(ada_frame1)
            ada_frame2 = rgb2lab(ada_frame2)
            ada_pred = rgb2lab(ada_pred)
             
            ada_frame1 = ada_frame1.reshape(-1, ada_frame1.shape[2], ada_frame1.shape[3]).to(self.device).float()
            ada_frame2 = ada_frame2.reshape(-1, ada_frame2.shape[2], ada_frame2.shape[3]).to(self.device).float()
            ada_pred = ada_pred.reshape(-1, ada_pred.shape[2], ada_pred.shape[3]).to(self.device).float()

        # PhaseNet input preparations
        img_batch = torch.cat((lab_frame1, lab_frame2, target), 0)
        num_vals = 3

        # Combine images into one big batch and then create the values and separate
        vals_batch = self.pyr.filter(img_batch.float())
        vals_list = separate_vals(vals_batch, num_vals)
        vals_target = vals_list[-1]
        vals_input = get_concat_layers_inf(self.pyr, vals_list[:-1])
        input = self.phase_net.normalize_vals(vals_input)

        # Delete unnecessary vals
        del vals_batch
        del vals_list
        del vals_input
        torch.cuda.empty_cache()

        # PhaseNet Vals Prediction
        with torch.no_grad():
            vals_pred = self.phase_net(input)

        # Exchange high levels of phase net
        ada_pyr = self.pyr.filter(ada_pred.squeeze(0))
        vals_pred.high_level[:] = ada_pyr.high_level

        # Get phase net prediction image
        phase_pred = self.pyr.inv_filter(vals_pred)
        
        
        # Fusion Net prediction
        phase_pred = phase_pred.reshape(-1, 3, phase_pred.shape[1], phase_pred.shape[2]).float()
        ada_pred = ada_pred.reshape(-1, 3, ada_pred.shape[1], ada_pred.shape[2]).float()
        other = torch.cat([lab_frame1.reshape(-1, 3, lab_frame1.shape[1], lab_frame1.shape[2]), lab_frame2.reshape(-1, 3, lab_frame2.shape[1], lab_frame2.shape[2])], 1).float()
        final_pred = self.fusion_net(ada_pred, phase_pred, other, uncertainty_mask)
        final_pred = final_pred.reshape(-1, final_pred.shape[2], final_pred.shape[3])

        return final_pred

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
            rgb_frame1 = triple[0].to(self.device)
            rgb_frame2 = triple[2].to(self.device)

            # Predict intermediate frames
            prediction = self.predict(lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target)

            # Calculate the loss
            prediction = torch.clip(prediction, 0, 1)
            prediction = prediction.reshape(-1, 3, prediction.shape[1], prediction.shape[2]).float()
            target = target.reshape(-1, 3, target.shape[1], target.shape[2]).float()
            loss = self.criterion(prediction, target)

            # Update net using backprop and gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.cpu().detach().numpy()
            self.loss_history.append(loss)

            if batch_idx % 100 == 0:
                self.test(idx=int(batch_idx/100), paths=['counter_examples/lights/001.png', 'counter_examples/lights/002.png', 'counter_examples/lights/003.png'], name='basketball')
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item()))
                torch.save(self.fusion_net.state_dict(), self.out_dir + f'/checkpoint/model_{self.current_epoch}_{int(batch_idx/100)}.pt')

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
        self.fusion_net.eval()
        with torch.no_grad():
            prediction = self.predict(lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target)
        self.fusion_net.train()

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

