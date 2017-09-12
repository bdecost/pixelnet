import os

import keras.backend as K
from keras.layers import Flatten, Reshape
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Conv1D, Conv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.models import Model
from keras import applications

WEIGHTS_DIR='/mnt/data/users/holmlab/.keras/models'

def load_imagenet_weights(model, weights_dir=WEIGHTS_DIR):
    # load convolution layers
    weights_path = os.path.join(weights_dir, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights(weights_path, by_name=True)

    vgg_model = applications.vgg16.VGG16(include_top=True, weights=None)
    vgg_model.load_weights(weights_path)
    # load classifier layer weights, reshape, and insert
    for layername in ['fc1', 'fc2', 'predictions']:        
        layer = model.get_layer('{}-conv'.format(layername))

        l = vgg_model.get_layer(layername)
        w, b = l.get_weights()

        # get target shapes
        # only need to reshape kernel weights
        w_fc, b_fc = layer.weights        
        w = w.reshape(w_fc.get_shape())    
        layer.set_weights([w, b])

    return model

def fully_conv_model(include_top=True, weights='imagenet', 
                     input_tensor=None, input_shape=(None,None,3),
                     classes=1000):
        
    print(input_shape)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    # use explicit ReLU layer to allow pixelnet model to apply batchnorm
    # before concatenation into hypercolumns
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = Activation('relu', name='block1_conv1_relu')(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = Activation('relu', name='block1_conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = Activation('relu', name='block2_conv1_relu')(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = Activation('relu', name='block2_conv2_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = Activation('relu', name='block3_conv1_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = Activation('relu', name='block3_conv2_relu')(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = Activation('relu', name='block3_conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = Activation('relu', name='block4_conv1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = Activation('relu', name='block4_conv2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = Activation('relu', name='block4_conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = Activation('relu', name='block5_conv1_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = Activation('relu', name='block5_conv2_relu')(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = Activation('relu', name='block5_conv3_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Conv2D(4096, (7, 7), padding='valid', name='fc1-conv')(x)
        x = Activation('relu', name='fc1_relu')(x)
        x = Conv2D(4096, (1,1), padding='valid', name='fc2-conv')(x)
        x = Activation('relu', name='fc2_relu')(x)
        x = Conv2D(classes, (1,1), activation='softmax', name='predictions-conv')(x)
        
    model = Model(img_input, x)

    if weights == 'imagenet':
        model = load_imagenet_weights(model)

    return model


