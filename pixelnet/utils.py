#!/usr/bin/env python
import os
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.utils.np_utils import to_categorical

def random_training_samples(images, labels, batchsize=4, ntrain=20, npix=2048, nclasses=4):
    """ generate random samples of pixels in batches of four training images """
    pixel_labels = tf.placeholder(tf.float32, shape=(batchsize, npix))
    while True:
        # sample coordinates should include the batch index for tf.gather_nd
        coords = np.ones((batchsize, npix, 3))
        coords = coords * np.arange(batchsize)[:,np.newaxis,np.newaxis]

        # choose random pixel coordinates on (0, 1) interval
        p = np.random.random((batchsize,npix,2))
        coords[:,:,1:] = p

        # choose random batch of training images, with replacement
        im_idx = np.random.choice(range(ntrain), batchsize, replace=True)
        t_ims = images[im_idx]

        # select the corresponding label images
        target_labels = labels[im_idx]
        ind = coords * np.array([1, images.shape[1], images.shape[2]])
        ind = ind.astype(np.int32)

        # get sample pixel labels
        bb, xx, yy = ind[:,:,0], ind[:,:,1], ind[:,:,2]
        pixel_labels = target_labels[bb,xx,yy]
        
        # convert labels to categorical indicators for cross-entropy loss
        s = pixel_labels.shape
        pixel_labels = to_categorical(pixel_labels.flat, num_classes=NCLASSES)
        pixel_labels = pixel_labels.reshape((s[0], s[1], NCLASSES))

        yield ([t_ims, coords], pixel_labels)

def random_validation_samples(images, labels, batchsize=4, ntrain=20, npix=2048, nclasses=4):
    """ generate random samples of pixels in batches of four validation images """
    pixel_labels = tf.placeholder(tf.float32, shape=(batchsize, npix))
    while True:
        # sample coordinates should include the batch index for tf.gather_nd
        coords = np.ones((batchsize, npix, 3))
        coords = coords * np.arange(batchsize)[:,np.newaxis,np.newaxis]

        # choose random pixel coordinates on (0, 1) interval
        p = np.random.random((batchsize,npix,2))
        coords[:,:,1:] = p

        # choose random batch of training images, with replacement
        im_idx = np.random.choice(range(ntrain,images.shape[0]), batchsize, replace=True)
        t_ims = images[im_idx]

        # select the corresponding label images
        target_labels = labels[im_idx]
        ind = coords * np.array([1, images.shape[1], images.shape[2]])
        ind = ind.astype(np.int32)

        # get sample pixel labels
        bb, xx, yy = ind[:,:,0], ind[:,:,1], ind[:,:,2]
        pixel_labels = target_labels[bb,xx,yy]

        # convert labels to categorical indicators for cross-entropy loss
        s = pixel_labels.shape
        pixel_labels = to_categorical(pixel_labels.flat, num_classes=nclasses)
        pixel_labels = pixel_labels.reshape((s[0], s[1], nclasses))

        yield ([t_ims, coords], pixel_labels)
        
def stratified_training_samples():
    while True:
        coords = np.random.random((1,2048,3))
        coords *= np.array([0, 1, 1])
