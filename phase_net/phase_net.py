import numpy as np
import torch
from torch import nn
import tensorflow as tf
from torchvision import datasets, transforms
from project.phasnet.conv_Block import conv_block


class Phasenet(nn.Module):

    def __init__(self, pyramid, values, prediction_function, ksizes,
                 kgroups, ksizes_predictions=[], start_index=0,
                 pred_residual=False, padding='valid',
                 # normalization=tf.layers.batch_normalization, # TODO normalization layer with torch
                 is_training=True, reuse_variables=False, ctf=False):
        """
        Creates the phasenet model.

        :param pyramid: steerable pyramid
        :param values:
        :param prediction_function:
        :param ksizes:
        :param kgroups:
        :param ksizes_predictions:
        :param start_index:
        :param pred_residual:
        :param padding:
        :param normalization:
        :param is_training:
        :param reuse_variables:
        :param ctf:
        """
        super(Phasenet, self).__init__()

        # can be ignored, is always 1
        if (len(ksizes_predictions) == 0):
            ksizes_predictions = ksizes[:]

        self.act_f = torch.nn.ReLU()    # Activation function
        self.pyramid = pyramid          # steerable pyramid
        self.values = values

        # parameters
        self.scale_factor = pyramid.scale_factor    # how many images per level of steerable pyramid
        self.nlayers = pyramid.height - 2           # number of concatenated layers

        self.reuse_variables = reuse_variables
        self.ksizes = ksizes
        self.prediction_function = prediction_function
        self.ksizes_predictions = ksizes_predictions
        self.reuse_variables = reuse_variables
        self.kgroups = kgroups
        self.start_index = start_index
        self.pred_residual = pred_residual
        self.is_training = is_training
        self.ctf = ctf

        print('Building channel decoder with scale factor ' + str(self.scale_factor))
        # variable to save prediction map after each step
        self.predictions = []

        # left/right decompositions
        self.concat_layers, self.dims = get_concat_layers(self.values, self.pyramid, self.scale_factor)

    def forward(self, inp_tensor):
        conv = lambda inp_channels, nfilters, ksize, nrepetitions, normalization, reuse: conv_block(
            inp_channels,
            nfilters=nfilters,
            ksize=ksize,
            nrepetitions=nrepetitions,
            strides=1,
            dilation=1,
            # padding=padding,
            normalization_block=normalization,
            # is_training=is_training,
            reuse=reuse)

        print('layers --------- ')
        print("Scope : low_res --- Reusing : " + str(self.reuse_variables))
        conv_layer = conv(self.concat_layers[-1], nfilters=64, ksize=self.ksizes[0], nrepetitions=1, normalization=None,
                          reuse=self.reuse_variables)
        local_prediction = self.prediction_function[0](conv_layer, ksize=self.ksizes_predictions[0],
                                                  reuse=self.reuse_variables)  # alpha for low res
        self.predictions.insert(0, local_prediction)

        previous_scope = 'low_res'
        max_scope = self.kgroups[0]

        for i in range(1, len(self.ksizes)):
            current_scope = 'decoder_group_' + str(self.kgroups[i])

            reuse = self.reuse_variables
            if (self.kgroups[
                i] <= max_scope):  # At test time there could be more pyramid levels (due to images larger than 256), than phasenet has been trained for -> reuse weights of last level for the additional levels
                reuse = True
            max_scope = max(self.kgroups[i], max_scope)
            print("Scope : " + current_scope + " --- Reusing : " + str(reuse))

            # convolutions
            # resize previous prediction & layer
            print('conv')
            resized_prediction = tf.image.resize_bilinear(local_prediction, self.dims[self.nlayers - i - self.start_index])
            resized_layer = tf.image.resize_bilinear(conv_layer, self.dims[self.nlayers - i - self.start_index])
            conv_layer = conv(
                tf.concat([resized_layer, self.concat_layers[self.nlayers - i + 1 - self.start_index], resized_prediction],
                          axis=3), nfilters=64, ksize=self.ksizes[i], nrepetitions=2, normalization=self.normalization,
                reuse=reuse)

            print('prediction')
            local_prediction = self.prediction_function[1](conv_layer, ksize=self.ksizes_predictions[i], reuse=reuse)
            if (self.pred_residual):
                local_prediction += self.scale_factor * resized_prediction
                # convert them to be between [-pi,pi]
                local_prediction = tf.atan2(tf.sin(local_prediction * np.pi), tf.cos(local_prediction * np.pi))
                # convert them to be between [-1,1]
                local_prediction /= np.pi
            self.predictions.insert(0, local_prediction)

        return self.predictions


# prepare steerable pyramid for phasenet
def get_concat_layers(values, pyramid, scale_factor):
    """ Returns list of tensors of steerable pyramid at all scales"""
    concat_layers = []
    dims = []

    # add high pass residual of both images (real valued)
    # concatenate images
    concat_layers.append(torch.cat([values[0].high_level, values[1].high_level], axis=3))
    # normalize to [0,1]
    # take max value of each image channel for normalization
    # concat_layers[0] /= tf.reduce_max(tf.maximum(concat_layers[0], 0.00001), axis=[1,2,3], keep_dims=True)
    # TODO: axis check which dimension
    treshold = torch.zeros((values.shape[0], values.shape[1], values.shape[2]))
    treshold[:, :] = 0.00001
    concat_layers[0] /= torch.max(torch.max(concat_layers[0], treshold), axis=3, keep_dims=True)

    # add band pass levels (phase and amplitude values) of pyramid of both images
    for l in range(pyramid.height-2): #-2 for normal phasenet
        phase = torch.cat([values[0].phase[l], values[1].phase[l]], axis=3)
        phase /= np.pi # normalize between [-1,1]
        amplitude = torch.concat([values[0].amplitude[l], values[1].amplitude[l]], axis=3)
        # normalize to [0,1]
        tresh = torch.zeros((values.shape[0], values.shape[1], values.shape[2]))
        tresh[:, :] = 0.00001
        amplitude /= torch.max(torch.max(torch.abs(amplitude),tresh), axis=3, keep_dims=True)
        concat_layers.append(torch.cat([phase, amplitude], axis=3))
        concat_layers[l].get_shape()[1], concat_layers[l].get_shape()[2]
        # get dimensions
        height, width = phase.get_shape()[1], phase.get_shape()[2]
        dims.append([height.to(dtype=torch.int32), width.to(dtype=torch.int32)])

    # low residual
    concat_layers.append(torch.cat([values[0].low_level, values[1].low_level], axis=3))
    # normalize to [0,1]
    concat_layers[-1] /= torch.max(torch.max(tf.abs(concat_layers[-1]), treshold), axis=3, keep_dims=True)

    print('nlayers')
    print(pyramid.height-2)
    print('dim low residual')
    print(concat_layers[-1])
    print('\n')

    return concat_layers, dims