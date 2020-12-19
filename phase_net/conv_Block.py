# code snippets for phasenet from tinkerflow (internal only), need to be rewritten
import torch
import torchvision
from torch import nn


def conv_block(in_tensor, nrepetitions, inp_channels, nfilters, ksize, strides, dilation, padding='same',
               activation_fn=torch.nn.ReLU(), normalization_block=None,
               is_training=True, reuse=None):
    """ Build convolution block that is used with different options in the architecture.

        Args:
            in_tensor:              Input tensor
            nrepetitions:           Number of times the (CONV + activation_fn) layers should be repeated
            nfilters:               Number of filters/channels of the output
            ksize:                  Kernel size for the convolutions
            strides:                strides is only applied once
                                        - positive values indicate that it should be applied at the first iteration
                                        - negative values indicate that it should be applied at the last iteration

            dilation:               integer used to have conv "a trous"
            padding:                Can be 'same' or 'valid'
            activation_fn:          Activation function. Default tf.nn.relu.
                                    Can be either list of activations (len(activation_fn) == nrepetitions in that case)
                                    or just a single activation function to be used for all repetitions
            normalization_block:    Function to be used for normalization. Can be for instance:
                                        - None
                                        - tinkerflow.models.blocks.instance_normalization
                                        - tf.layers.batch_normalization

            is_training             Boolean indicating if graph is in training mode. Only meaningful with
                                    tf.layers.batch_normalization as normalization block

        Returns:
            A tensor corresponding to the output of the block

    """

    conv_block = in_tensor

    # Allow for different activation functions for each repetition.
    if not isinstance(activation_fn, list):
        activation_functions = [activation_fn] * nrepetitions
    else:
        assert (len(activation_fn) == nrepetitions)
        activation_functions = activation_fn

    # CONV Block : nrepetitions x (CONV + activation_fn )
    for i in range(0, nrepetitions):

        if ((i == 0) and (strides > 0)):
            stride_value = strides
        elif ((i == nrepetitions - 1) and (stride_value < 0)):
            stride_value = - strides
        else:
            stride_value = 1

        name = 'conv_' + str(i)
        conv_layer = nn.Conv2d(
            in_channels=inp_channels,
            out_channels=nfilters,  # Number of channels produced by the convolution
            kernel_size=ksize,      # Size of the convolving kernel
            stride=stride_value,    # Stride of the convolution
            # dilation_rate=dilation,
            # padding=padding,
            # activation=activation_functions[i],
            # name=name,
            # reuse=reuse
            )

        conv_block = conv_layer(conv_block)

    # Batch norm
    name = 'norm_' + str(i)
    if (normalization_block != None):
        conv_block = normalization_block(
            inputs=conv_block,
            training=is_training,
            name=name,
            reuse=reuse)

    return conv_block