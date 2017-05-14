""" keras implementation of PixelNet architecture. """
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Layer, Concatenate, Dropout, Dense
from keras import backend as K

from keras.applications.inception_v3 import conv2d_bn

from .upsample import sparse_upsample, sparse_upsample_output_shape

def pixelnet_model(nclasses=4, inference=False):
    """ Use sparse upsample implementations to define a PixelNet model

    @article{pixelnet,
      title={PixelNet: {R}epresentation of the pixels, by the pixels, and for the pixels},
      author={Bansal, Aayush
              and Chen, Xinlei,
              and  Russell, Bryan
              and Gupta, Abhinav
              and Ramanan, Deva},
      Journal={arXiv preprint arXiv:1702.06506},
      year={2017}
    }

    TODO: add batch normalization to conv layers (for training from scratch)
    TODO: consider removing dropout from conv layers

    From the paper and their notes on github, it seems like the semantic segmentation
    task should work either with linear classifier + BatchNorm, or with MLP without BatchNorm.
    """
    
    # a single input channel for grayscale micrographs...
    inputdata = Input(shape=(None,None,1))

    # coordinate tensor: (batch, index)
    # the index axis should contain (b, i, j)
    # i.e. batch index, row index, column index
    inputcoord = Input(shape=(None, 3,), dtype='float32')

    x = conv2d_bn(inputdata, 16, 3, 3, name='block1_conv1')
    x = conv2d_bn(x, 16, 3, 3, name='block1_conv2')
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = conv2d_bn(x1, 32, 3, 3, name='block2_conv1')
    x = conv2d_bn(x, 32, 3, 3, name='block2_conv2')
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = conv2d_bn(x2, 64, 3, 3, name='block3_conv1')
    x = conv2d_bn(x, 64, 3, 3, name='block3_conv2')
    x = conv2d_bn(x, 64, 3, 3, name='block3_conv3')
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = conv2d_bn(x3, 128, 3, 3, name='block4_conv1')
    x = conv2d_bn(x, 128, 3, 3, name='block4_conv2')
    x = conv2d_bn(x, 128, 3, 3, name='block4_conv3')
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = conv2d_bn(x4, 128, 3, 3, name='block5_conv1')
    x = conv2d_bn(x, 128, 3, 3, name='block5_conv2')
    x = conv2d_bn(x, 128, 3, 3, name='block5_conv3')
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    batchsize, h, w = tf.shape(inputdata)[0], tf.shape(inputdata)[1], tf.shape(inputdata)[2]

    if inference:
        print('inference phase.')
        upsample = Lambda(
            lambda t: tf.image.resize_images(t, (h,w)),
            output_shape=lambda s: (s[0], h, w, s[-1]),
            name='tf_upsample'
        )
        h1 = upsample(x1)
        h2 = upsample(x2)
        h3 = upsample(x3)
        h4 = upsample(x4)
        h5 = upsample(x5)

    else:
        print('training phase.')
        upsample = Lambda(
            sparse_upsample,
            output_shape=sparse_upsample_output_shape,
            name='sparse_upsample'
        )
        h1 = upsample([x1, inputcoord])
        h2 = upsample([x2, inputcoord])
        h3 = upsample([x3, inputcoord])
        h4 = upsample([x4, inputcoord])
        h5 = upsample([x5, inputcoord])

    # now we have shape (batch, sample, channel)
    x = Concatenate()([h1, h2, h3, h4, h5])

    # flatten into pixel features
    nchannels = tf.shape(x)[-1]

    flatten_pixels = Lambda(
        lambda t: K.reshape(t, (-1, nchannels)),
        output_shape=lambda s: (-1, s[-1]),
        name='flatten_pixel_features'
    )
    x = flatten_pixels(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(nclasses, activation='softmax', name='predictions')(x)

    if inference:
        # note: output_shape wants the static shape (None, None, None, nclasses)
        # but reshape needs the dynamic shape (int, int, int, nclasses)
        b_dyn, h_dyn, w_dyn, _ = inputdata.shape
        unflatten = Lambda(
            lambda t: K.reshape(t, (batchsize, h, w, nclasses)),
            output_shape=lambda s: (b_dyn, h_dyn, w_dyn, nclasses),
            name='unflatten_pixel_features'
        )
    else:
        # note: output_shape wants the static shape (None, None, nclasses)
        # but reshape needs the dynamic shape (int, int, nclasses)
        # actually, batchsize can be either, but npix needs to be the dynamic value.
        npix = K.shape(inputcoord)[1]
        _, npix_dyn, _ = inputcoord.shape
        unflatten = Lambda(
            lambda t: K.reshape(t, (batchsize, npix, nclasses)),
            # output_shape=lambda s: (4, 2048, nclasses),
            output_shape=lambda s: (batchsize, npix_dyn, nclasses),
            name='unflatten_pixel_features'
        )
    x = unflatten(x)

    if inference:
        model = Model(inputs=inputdata, outputs=x)
    else:
        model = Model(inputs=[inputdata, inputcoord], outputs=x)
    return model
