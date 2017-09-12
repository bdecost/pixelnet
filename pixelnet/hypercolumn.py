"""
implementation of sparse upsampling of keras tensors
using bilinear interpolation
"""

import numpy as np
import tensorflow as tf

from keras.models import Model
from keras import backend as K
from keras.layers import Concatenate, Input, Lambda, BatchNormalization, Activation


def get_values(data, batch, x, y):
    """ construct index tensor for tf.gather_nd """
    coords = tf.stack((batch, x, y), 2)
    indices = tf.cast(coords, tf.int32)
    return tf.gather_nd(data, indices)

def offset(x1, x2):
    """ helper for bilinear upsample:
    make offset tensor the same rank as data tensor for broadcasting.
    """
    dx = x2 - x1
    return tf.expand_dims(dx, axis=-1)

def sparse_upsample_bilinear(inputs, **arguments):
    """ upsample the input tensor `data` with indices in input tensor sel
    performs sparse bilinear interpolation
    indices should explicitly contain the sample index: (b, i, j)
    """
    data, coords = inputs
    w = tf.cast(tf.shape(data)[1], tf.float32)
    h = tf.cast(tf.shape(data)[2], tf.float32)

    # transform fractional coordinates to feature map coordinates
    batch = coords[:,:,0]
    x = w * coords[:,:,1]
    y = h * coords[:,:,2]

    # clip feature map coordinates
    # results in incorrect interpolation
    # for samples in the last row and/or column
    x1, y1 = tf.floor(x), tf.floor(y)
    x1 = tf.clip_by_value(x1, 0, w-3)
    y1 = tf.clip_by_value(y1, 0, h-3)
    x2, y2 = 1 + x1, 1 + y1

    # horizontal interpolation first
    top = get_values(data, batch, x1, y2) * offset(x, x2) +  get_values(data, batch, x2, y2) * offset(x1, x)
    bottom = get_values(data, batch, x1, y1) * offset(x, x2)  + get_values(data, batch, x2, y1) * offset(x1, x)

    # vertical interpolation
    interp =  bottom * offset(y, y2) + top * offset(y1, y)
    return interp

def sparse_upsample_output_shape(input_shape):
    # sparse_upsample expects two input tensors:
    # the feature map tensor and an index tensor
    data_shape, index_shape = input_shape
    assert K.backend() == 'tensorflow'
    assert len(data_shape) == 4 # only valid for 4D tensors
    assert len(index_shape) == 3
    return (data_shape[0], index_shape[0], data_shape[3])

def random_foreground_indices(L, npix=4096, bgval=-1):
    """ Sample random foreground pixels from 3D label array L: return indices (including batch index)
    same number of pixels from each image so that the samples fit into the coordinate/output tensors
    npix=4096 yields ~2% of foreground pixels in 1024x1024 particle micrograph inputs
    """
    n, h, w = L.shape
    
    # sample coordinates should include the batch index for tf.gather_nd
    ind = []

    for idx in range(n):
        # get coordinates for all foreground pixels
        y, x = np.where(L[idx] > bgval)
        minibatch = np.ones(x.size) * idx
        pixels = np.stack((minibatch, x/w, y/h), axis=1)
        
        # get a random subset
        idx = np.random.choice(range(pixels.shape[0]), npix)
        ind.append(pixels[idx])
    
    return np.stack(ind, axis=0)

def build_model(base_model, input_layers, mode='dense', batchnorm=False):
    inputdata = base_model.input
    
    batchsize, h, w = tf.shape(inputdata)[0], tf.shape(inputdata)[1], tf.shape(inputdata)[2]
    
    X = [base_model.get_layer(layername).output for layername in input_layers]

    if batchnorm:
        X = [
            BatchNormalization(scale=False, name='{}_bn'.format(name))(x)
            for x, name in zip(X, input_layers)
            ]
    
    X = [Activation('relu')(x) for x in X]

    if mode == 'dense':
        inputs = inputdata
        upsample = Lambda(
            lambda t: tf.image.resize_images(t, (h,w)),
            output_shape=lambda s: (s[0], h, w, s[-1]),
            name='tf_upsample'
        )

        hc_layers = [upsample(x) for x in X]


    elif mode == 'sparse':
        inputcoords = Input(shape=(None, 3,), dtype='float32')
        inputs = [inputdata, inputcoords]
        upsample = Lambda(
            sparse_upsample_bilinear,
            output_shape=sparse_upsample_output_shape,
            name='sparse_upsample'
        )
    
        hc_layers = [upsample([x, inputcoords]) for x in X]
    
    hc = Concatenate(name='hypercolumn')(hc_layers)
    
    return Model(inputs, hc)
