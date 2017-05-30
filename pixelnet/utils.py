#!/usr/bin/env python
import os
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.utils.np_utils import to_categorical

def random_pixel_samples(images, labels, batchsize=4, npix=2048, nclasses=4, replace_samples=True, categorical=True):
    """ generate random samples of pixels in batches of training images """
    n_images = images.shape[0]
    
    pixel_labels = tf.placeholder(tf.float32, shape=(batchsize, npix))
    while True:
        # choose random batch of training images, with replacement
        im_idx = np.random.choice(range(n_images), batchsize, replace=replace_samples)
        sample_images = images[im_idx]
        target_labels = labels[im_idx]
        
        # sample coordinates should include the batch index for tf.gather_nd
        coords = np.ones((batchsize, npix, 3))
        coords = coords * np.arange(batchsize)[:,np.newaxis,np.newaxis]

        # choose random pixel coordinates on (0, 1) interval
        p = np.random.random((batchsize,npix,2))
        coords[:,:,1:] = p

        # get sample pixel labels
        ind = coords * np.array([1, images.shape[1], images.shape[2]])
        ind = ind.astype(np.int32)
        bb, xx, yy = ind[:,:,0], ind[:,:,1], ind[:,:,2]
        pixel_labels = target_labels[bb,xx,yy]

        if categorical:
            # convert labels to categorical indicators for cross-entropy loss
            s = pixel_labels.shape
            pixel_labels = to_categorical(pixel_labels.flat, num_classes=nclasses)
            pixel_labels = pixel_labels.reshape((s[0], s[1], nclasses))

        yield ([sample_images, coords], pixel_labels)
        
def stratified_training_samples():
    while True:
        coords = np.random.random((1,2048,3))
        coords *= np.array([0, 1, 1])
