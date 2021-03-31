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
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, median_filter, sobel, convolve
from scipy.ndimage.filters import gaussian_filter

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
        self.l1 = nn.L1Loss()

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

    def predict(self, lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target, debug=False):
        """ Given batch of input images, predict the intermediate images. """
        # Adacof prediction
        with torch.no_grad():
            ada_frame1, ada_frame2, ada_pred, flow_var_map = self.adacof_model(
                rgb_frame1, rgb_frame2)

            ada_frame1 = ada_frame1.reshape(
                -1, ada_frame1.shape[2], ada_frame1.shape[3]).to(self.device).float()
            ada_frame2 = ada_frame2.reshape(
                -1, ada_frame2.shape[2], ada_frame2.shape[3]).to(self.device).float()
            ada_pred = ada_pred.reshape(
                -1, ada_pred.shape[2], ada_pred.shape[3]).to(self.device).float()
            flow_var_map = flow_var_map.squeeze(1)

        # PhaseNet input preparations
        img_batch = torch.cat((lab_frame1, lab_frame2), 0)
        num_vals = 2

        # Combine images into one big batch and then create the values and separate
        vals_batch = self.pyr.filter(img_batch.float())
        vals_list = separate_vals(vals_batch, num_vals)
        vals_input = get_concat_layers_inf(self.pyr, vals_list)
        input = self.phase_net.normalize_vals(vals_input)

        # Delete unnecessary vals
        del vals_batch
        del vals_list
        del vals_input
        torch.cuda.empty_cache()

        # PhaseNet Vals Prediction
        with torch.no_grad():
            vals_pred = self.phase_net(input)

        # RGB prediction without new high levels
        lab_pred = self.pyr.inv_filter(vals_pred)
        lab_pred = lab_pred.reshape(-1, 3,
                                    lab_pred.shape[1], lab_pred.shape[2]).float()
        r_shape = lab_pred.shape
        rgb_pred = torch.cat([lab2rgb_single(l)
                              for l in lab_pred], 0).to(self.device)

        # Exchange high levels of phase net, but copy before that uncertainty of phase
        #ada_pyr = self.pyr.filter(ada_pred.squeeze(0))
        #vals_pred.high_level[:] = ada_pyr.high_level

        # Get phase net prediction image (B*C, H, W)
        #phase_pred = self.pyr.inv_filter(vals_pred).reshape(r_shape)
        #phase_pred = torch.cat([lab2rgb_single(l) for l in phase_pred], 0)
        phase_pred = rgb_pred.clone()

        # Phase Net uncertainty map
        ada_pred = torch.as_tensor(ada_pred).to(self.device)
        img_batch = torch.cat((ada_pred, rgb_pred), 0)
        num_vals = 2

        vals_batch = self.pyr.filter(img_batch.float())
        vals_ada, vals_ph = separate_vals(vals_batch, num_vals)

        vals_high = get_last_value_levels(vals_ada, use_levels=1)
        vals_high_ph = get_last_value_levels(vals_ph, use_levels=1)
        h_freq = self.pyr.inv_filter(
            vals_high).detach().cpu().reshape(r_shape).mean(1)
        h_freq_ph = self.pyr.inv_filter(
            vals_high_ph).detach().cpu().reshape(r_shape).mean(1)
        h_freq_diff = torch.abs(h_freq - h_freq_ph)
        h_freq_diff = (h_freq_diff*100).clamp(min=0, max=1.0)
        h_freq_diff = torch.stack(
            [torch.as_tensor(gaussian_filter(h, 5)) for h in h_freq_diff])
        phase_uncertainty = h_freq_diff.to(self.device)

        # Adacof uncertainty map for finding artifacts
        vals_diff = subtract_values(vals_ph, vals_ada)
        vals_diff = get_first_value_levels(vals_diff, use_levels=6)
        freq_diff = self.pyr.inv_filter(
            vals_diff).detach().cpu().reshape(r_shape).mean(1)*30
        freq_diff_median = torch.stack(
            [torch.as_tensor(median_filter(f, size=50)) for f in freq_diff])
        freq_diff = torch.abs(freq_diff - freq_diff_median)
        freq_diff = (freq_diff * 5).clamp(0, 1)
        ada_uncertainty = freq_diff.to(self.device)

        # Get baseline
        with torch.no_grad():
            _, _, ada_inbetween1, _ = self.adacof_model(
                rgb_frame1, phase_pred.reshape(r_shape).float())
            ada_inbetween1 = ada_inbetween1.to(self.device).float()

            _, _, ada_inbetween2, _ = self.adacof_model(
                phase_pred.reshape(r_shape).float(), rgb_frame2)
            ada_inbetween2 = ada_inbetween2.to(self.device).float()

            _, _, base, _ = self.adacof_model(ada_inbetween1, ada_inbetween2)
            base = base.to(self.device).float()

        # Debugging
        if debug:
            u_phase = torch.as_tensor(
                rgb_frame1.reshape(r_shape)[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_1.png')

            u_phase = torch.as_tensor(
                rgb_frame2.reshape(r_shape)[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_2.png')

            u_phase = torch.as_tensor(
                phase_pred.reshape(r_shape)[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_phase_pred.png')

            u_phase = torch.as_tensor(
                ada_pred.reshape(r_shape)[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_ada_pred.png')

            u_phase = torch.as_tensor(h_freq_diff[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_phase_unc.png')

            u_phase = torch.as_tensor(flow_var_map[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_var_flow.png')

            u_phase = torch.as_tensor(freq_diff[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_ada_unc.png')

            rgb_ground_truth = target.reshape(r_shape)
            u_phase = torch.as_tensor(rgb_ground_truth[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_ground_truth.png')

            u_phase = torch.as_tensor(base[0]).detach().cpu()
            u_phase = transforms.ToPILImage()(u_phase)
            u_phase.save(f'./test_baseline.png')

        # Fusion Net prediction
        phase_pred = phase_pred.reshape(r_shape).to(self.device).float()
        ada_pred = ada_pred.reshape(r_shape).to(self.device).float()

        other = torch.cat([lab_frame1.reshape(r_shape), lab_frame2.reshape(
            r_shape)], 1).to(self.device).float()
        maps = torch.stack([ada_uncertainty, phase_uncertainty, flow_var_map], 1).to(
            self.device).float()
        save = self.args.save

        # Predict
        final_pred = self.fusion_net(
            base, ada_pred, phase_pred, other, maps, save=save, variant=0)
        final_pred = final_pred.reshape(-1,
                                        final_pred.shape[2], final_pred.shape[3])

        return final_pred, phase_pred

    def train(self):
        """ Train the model. """
        for batch_idx, triple in enumerate(self.train_loader):
            # Get height and width of the training image
            h = triple[0].shape[2]
            w = triple[0].shape[3]
            hw = (h, w)

            # Transform into lab space
            lab_frame1 = rgb2lab(triple[0]).reshape((-1,) + hw).to(self.device)
            lab_frame2 = rgb2lab(triple[2]).reshape((-1,) + hw).to(self.device)

            # But not target
            target = triple[1].reshape((-1,) + hw).to(self.device)

            # Transform rgb images for adacof
            rgb_frame1 = triple[0].to(self.device)
            rgb_frame2 = triple[2].to(self.device)

            # Predict intermediate frames
            prediction, phase_pred = self.predict(
                lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target)

            # Calculate the loss
            prediction = torch.clip(prediction, 0, 1)
            prediction = prediction.reshape(-1, 3,
                                            prediction.shape[1], prediction.shape[2]).float()
            target = target.reshape(-1, 3,
                                    target.shape[1], target.shape[2]).float()

            # Loss
            #l = torch.abs(target - prediction)
            loss = self.l1(target, prediction)

            # Update net using backprop and gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.cpu().detach().numpy()
            self.loss_history.append(loss)

            if batch_idx % 50 == 0:
                self.test(idx=int(batch_idx/50), paths=['counter_examples/lights/001.png',
                                                        'counter_examples/lights/002.png', 'counter_examples/lights/003.png'], name='lights')
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                                                                            '[' + str(self.current_epoch) + '/' + str(
                                                                                self.args.epochs) + ']', 'Step: ',
                                                                            '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item()))
                torch.save(self.fusion_net.state_dict(), self.out_dir +
                           f'/checkpoint/model_{self.current_epoch}_{int(batch_idx/100)}.pt')

                loss_hist = np.asarray(self.loss_history)
                np.savetxt(self.out_dir + '/log_train.txt', loss_hist)
                plt.plot([i for i in range(len(loss_hist))], loss_hist)
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title(
                    f'Loss_fusion_{self.current_epoch}_{batch_idx}/{self.max_step}')
                plt.savefig(self.out_dir + '/loss_graph_train.png')

            if self.args.save:
                res_value = str(self.fusion_net.residuals)
                #np.savetxt(self.out_dir + '/log_train.txt', loss_hist)
                with open(self.out_dir + '/residuals_train.txt', "a") as a_file:
                    a_file.write("\n")
                    a_file.write(res_value)
                self.fusion_net.residuals = []

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
            frame1 = frame1[:, :, :3]
            target = target[:, :, :3]
            frame2 = frame2[:, :, :3]

        # Images to tensors
        frame1 = torch.as_tensor(frame1).permute(2, 0, 1)/255
        target = torch.as_tensor(target).permute(2, 0, 1)/255
        frame2 = torch.as_tensor(frame2).permute(2, 0, 1)/255
        rgb_frame1 = frame1.unsqueeze(0).float().to(self.device)
        rgb_frame2 = frame2.unsqueeze(0).float().to(self.device)

        # Bring images into LAB color space
        lab_frame1 = rgb2lab_single(frame1).squeeze(0).to(self.device)
        target = target.squeeze(0).to(self.device)
        lab_frame2 = rgb2lab_single(frame2).squeeze(0).to(self.device)

        # Predict intermediate frames
        self.fusion_net.eval()
        with torch.no_grad():
            prediction, _ = self.predict(
                lab_frame1, lab_frame2, rgb_frame1, rgb_frame2, target, debug=False)
        self.fusion_net.train()

        # Reconstruct image and save
        img = prediction.detach().cpu()
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
