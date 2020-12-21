# With changes from https://github.com/HyeongminLEE/AdaCoF-pytorch/blob/master/trainer.py

import os
import torch
import torch
from torch import nn


class Trainer:
    def __init__(self, args, train_loader, test_loader, my_model, my_pyr, my_loss, batch_size,
                 lR=0.001, weight_decay= 0.0001, start_epoch=0,
                 epoch=10):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.model = my_model
        self.pyr = my_pyr
        self.loss = my_loss
        self.batch_size = batch_size
        self.lR = lR
        self.wD = weight_decay
        self.epoch = epoch
        self.current_epoch = start_epoch

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

        # Initial Test
        # self.model.eval()
        # self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')

    def train(self):
        # Train
        for batch_idx, triple in enumerate(self.train_loader):
            # define heigth and width of the training image
            heigth = triple[0].shape[2]
            width = triple[0].shape[3]

            # get input data of first and second frame and the corresponding target image
            frame1 = triple[0].reshape((self.batch_size*3, 1, heigth, width)),
            target = triple[1].reshape((self.batch_size*3, 1, heigth, width)),
            frame2 = triple[2].reshape((self.batch_size*3, 1, heigth, width))

            # create steerable pyramid for the input frames
            fram1_pyr = self.pyr.filter(frame1)
            fram2_pyr = self.pyr.filter(frame2)

            # predicted intersected image of fram1 and frame2
            output_pyr = self.model(fram1_pyr, fram2_pyr)

            # transform output of the network back to an image -> inverse steerable pyramid
            output = self.pyr.inv_filter(output_pyr)

            loss = self.loss(output, target)  # input, target

            # === The backpropagation
            # Reset the gradients
            self.optimizer.zero_grad()
            # Calculate new gradients with backpropagation
            loss.backward()
            # Tune weights accoring to optimizer (it has the learnrate and weight decay as defined above)
            self.optimizer.step()

            # To do output stuff with loss and image, we have to detach() and convert back to numpy.
            loss = loss.detach().numpy()
            # Append to loss history
            self.loss_history.append(loss)

            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ',
                      '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']', 'Step: ',
                      '[' + str(batch_idx) + '/' + str(self.max_step) + ']', 'train loss: ', loss.item()))
        self.current_epoch += 1

    def test(self):
        # Test
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.get_state_dict()}, self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')
        self.logfile.write('\n')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()
