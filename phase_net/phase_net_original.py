""" PhaseNet Model
"""

from __future__ import division
import tensorflow as tf
import numpy as np
import tinkerflow.models.blocks
import src.filterbank as filterbank
import numpy as np

# prepare steerable pyramid for phasenet
def get_concat_layers(values, pyramid, scale_factor):
    """ Returns list of tensors of steerable pyramid at all scales"""
    concat_layers = []
    dims = []

    # add high pass residual of both images (real valued)
    concat_layers.append(tf.concat([values[0].high_level, values[1].high_level], axis=3))
    # normalize to [0,1]
    concat_layers[0] /= tf.reduce_max(tf.maximum(concat_layers[0], 0.00001), axis=[1,2,3], keep_dims=True)

    # add band pass levels (phase and amplitude values) of pyramid of both images
    for l in range(pyramid.height-2): #-2 for normal phasenet
        phase = tf.concat([values[0].phase[l], values[1].phase[l]], axis=3)
        phase /= np.pi # normalize between [-1,1]
        amplitude = tf.concat([values[0].amplitude[l], values[1].amplitude[l]], axis=3)
        # normalize to [0,1]        
        amplitude /= tf.reduce_max(tf.maximum(tf.abs(amplitude),0.00001), axis=[1,2,3], keep_dims=True)
        concat_layers.append(tf.concat([phase, amplitude], axis=3))
        concat_layers[l].get_shape()[1], concat_layers[l].get_shape()[2]
        # get dimensions
        height, width = phase.get_shape()[1], phase.get_shape()[2]
        dims.append([tf.to_int32(height), tf.to_int32(width)])

    # low residual
    concat_layers.append(tf.concat([values[0].low_level, values[1].low_level], axis=3))
    # normalize to [0,1]    
    concat_layers[-1] /= tf.reduce_max(tf.maximum(tf.abs(concat_layers[-1]), 0.00001), axis=[1,2,3], keep_dims=True)

    print('nlayers')
    print(pyramid.height-2)
    print('dim low residual')
    print(concat_layers[-1])
    print('\n')

    return concat_layers, dims

# build structure of phasenet
def build_channel_decoder_only(pyramid, values, prediction_function, ksizes, kgroups, ksizes_predictions = [], start_index=0, pred_residual=False, padding = 'valid',  normalization = tf.layers.batch_normalization, is_training = True, reuse_variables = False, ctf = False):
    """ Build PhaseNet model using tinkerflow building blocks

        Args:
            in_tensor:                  Input tensor
            prediction_function:        For flexibility, the prediction function. Must be callable using a single tensor as arugment
            with_reuse:                 Boolean indicating whether to share weights
            pyramid:                    Steerable pyramid object (contains nlayers, nbands, scale factor)


        Returns:
            list of tensors with predictions at all scales

    """
    # can be ignored, is always 1
    if (len(ksizes_predictions) == 0):
        ksizes_predictions = ksizes[:]

    # parameters
    scale_factor=pyramid.scale_factor
    nlayers=pyramid.height-2

    print('Building channel decoder with scale factor '+str(scale_factor))
    predictions = []

    # left/right decompositions
    concat_layers, dims = get_concat_layers(values, pyramid, scale_factor)

    #
    var_scope = 'channel_decoder'
    with tf.variable_scope(var_scope):
    #generates the conv layer structure of a phasenet block, details in appendix of paper
        conv = lambda in_tensor, nfilters, ksize, nrepetitions, normalization, reuse : tinkerflow.models.blocks.conv_block( 
            in_tensor,
            nfilters = nfilters,
            ksize = ksize,
            nrepetitions = nrepetitions,
            strides = 1,
            dilation = 1,
            padding = padding,
            normalization_block = normalization,
            is_training = is_training,
            reuse = reuse)

        print('layers --------- ')

        # low residual level
        with tf.variable_scope('low_res', reuse_variables):
            print("Scope : low_res --- Reusing : "+str(reuse_variables))
            conv_layer = conv(concat_layers[-1], nfilters = 64, ksize = ksizes[0], nrepetitions = 1, normalization=None, reuse = reuse_variables)
            local_prediction = prediction_function[0](conv_layer, ksize = ksizes_predictions[0], reuse=reuse_variables) # alpha for low res
            predictions.insert(0, local_prediction)

        #
        previous_scope = 'low_res'
        max_scope = kgroups[0]

        for i in range(1, len(ksizes)):
            current_scope = 'decoder_group_'+str(kgroups[i])

            with tf.variable_scope(current_scope, reuse_variables):
                reuse = reuse_variables
                if ( kgroups[i] <= max_scope ): # At test time there could be more pyramid levels (due to images larger than 256), than phasenet has been trained for -> reuse weights of last level for the additional levels
                    reuse = True
                max_scope = max(kgroups[i], max_scope)
                print("Scope : " + current_scope + " --- Reusing : "+str(reuse))

                # conv
                with tf.variable_scope('conv'):
                    # resize previous prediction & layer
                    resized_prediction  = tf.image.resize_bilinear(local_prediction, dims[nlayers-i-start_index])
                    resized_layer = tf.image.resize_bilinear(conv_layer, dims[nlayers-i-start_index])
                    conv_layer = conv(tf.concat([resized_layer, concat_layers[nlayers-i+1-start_index], resized_prediction], axis=3), nfilters = 64, ksize = ksizes[i], nrepetitions = 2, normalization = normalization, reuse = reuse)

                with tf.variable_scope('prediction'):
                    local_prediction =  prediction_function[1](conv_layer, ksize = ksizes_predictions[i], reuse=reuse)
                    if(pred_residual):
                        local_prediction += scale_factor*resized_prediction
                        # convert them to be between [-pi,pi]
                        local_prediction = tf.atan2(tf.sin(local_prediction*np.pi), tf.cos(local_prediction*np.pi))
                        # convert them to be between [-1,1]
                        local_prediction /= np.pi
                    predictions.insert(0, local_prediction)

                previous_scope = current_scope

    return predictions
