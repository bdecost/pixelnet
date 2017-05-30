#!/usr/bin/env python
import os
import numpy as np

import tensorflow as tf

from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import sys
sys.path.append(os.getcwd())
sys.path.append('../uhcs-segment')

from pixelnet.pixelnet import pixelnet_model
from pixelnet.utils import random_pixel_samples
from uhcsseg import data

# suppress some of the noisier tensorflow log messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    datafile = '../uhcs-segment/data/uhcs.h5'
    images, labels, names = data.load_dataset(datafile, cropbar=38)
    print(images.shape)
    images = data.preprocess_images(images)

    # add channel axis
    images = images[:,:,:,np.newaxis]
    N, h, w, _ = images.shape

    batchsize = 4
    ntrain = 20
    npix = 2048
    nclasses = 4
    
    steps_per_epoch = ntrain * h * w / (batchsize*npix)
    # steps_per_epoch = 100
    print('steps_per_epoch:')
    print(steps_per_epoch)
    
    opt = optimizers.Adam()
    model = pixelnet_model()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    csv_logger = CSVLogger('run/training-1.log')
    checkpoint = ModelCheckpoint('run/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # note: keras/engine/training.py:L132 --> is not None
    f = model.fit_generator(
        random_pixel_samples(images[:ntrain], labels[:ntrain]),
        steps_per_epoch,
        epochs=10,
        callbacks=[csv_logger, checkpoint, early_stopping],
        validation_data=random_pixel_samples(images[ntrain:], labels[ntrain:]),
        validation_steps=10
    )
