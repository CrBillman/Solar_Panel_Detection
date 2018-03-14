from __future__ import print_function

import os
#from skimage.transform import resize
#from skimage.io import imsave
import numpy as np
import keras.metrics as kmetrics
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend

class Unet():
    """
    Unet neural network as defined by Ronneberger et al 2015.
    """
    
    def __init__(self, rowImg, colImg, nChannels):
        """
        Initialize Unet object.
        
        Parameters
        ----------
        rowImg: int
            number of rows in input images
        colImag: int
            number of cols in input images
        nChannels: int
            number of channels (e.g. 3 for RGB)
        """
        
        self.rowImg = rowImg
        self.colImg = colImg
        self.nChannels = nChannels
    
    def create_unet(self, filterSize=(3,3), strideConv=(1,1),
                    poolSize=(2,2), stridePool=(2,2),
                    filterSizeUp=(2,2), strideConvUp=(2,2),
                    actType='relu', padType='same',
                    nF1=64, nF2=128, nF3=256, nF4=512, nF5=1024):
        """
        Create unet model using Keras layers.
        Defaults match that used in Ronneberger et al 2015.
        Defaults can be changed, but must ensure that parameters
        allow convolution layers to match for concatenation.
        
        Parameters
        ----------
        nF(1-5): int
            number of filters for that layer
            Note: different parameter for each UNET step
        filterSize: tuple of ints
            filter size for that layer
            Note: same parameter for each UNET layer
        poolSize: tuple of ints
            pool size for max pooling steps
            Note: same parameter for each pooling step
        filterSizeUp: tuple of ints
            filter size of upsampling using Conv2dTranspose
        actType: string
            activation type (e.g. relu)
            Note: same parameter for each activation
        padType: string
            type of padding to use on edges
        
        Returns
        -------
        model: keras model object
        
        """
        
        # Create initial input using image row/column/channel sizes
        inputs = Input(shape=(self.rowImg, self.colImg, self.nChannels))
        
        # --Downsampling--
        
        # First downsampling set
        conv1a = Conv2D(nF1, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(inputs)
        conv1b = Conv2D(nF1, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv1a)
        pool1 = MaxPooling2D(pool_size=poolSize, strides=stridePool,
                             padding=padType)(conv1b)
        
        # Second downsampling set
        conv2a = Conv2D(nF2, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(pool1)
        conv2b = Conv2D(nF2, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv2a)
        pool2 = MaxPooling2D(pool_size=poolSize, strides=stridePool,
                             padding=padType)(conv2b)
        
        # Third downsampling set
        conv3a = Conv2D(nF3, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(pool2)
        conv3b = Conv2D(nF3, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv3a)
        pool3 = MaxPooling2D(pool_size=poolSize, strides=stridePool,
                             padding=padType)(conv3b)
        
        # Fourth downsampling set
        conv4a = Conv2D(nF4, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(pool3)
        conv4b = Conv2D(nF4, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv4a)
        pool4 = MaxPooling2D(pool_size=poolSize, strides=stridePool,
                             padding=padType)(conv4b)
        
        # Fifth downsampling set
        conv5a = Conv2D(nF5, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(pool4)
        conv5b = Conv2D(nF5, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv5a)
        
        #--Upsampling and Concatenation--
        # **CROPPING?!?!? TO DO
        
        # First upsampling set
        convUp6 = Conv2DTranspose(nF4, filterSizeUp,
                                  strides=strideConvUp, padding=padType)(conv5b)
        concat6 = concatenate([conv4b, convUp6], axis=3)
        conv6a = Conv2D(nF4, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(concat6)
        conv6b = Conv2D(nF4, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv6a)
        
        # Second upsampling set
        convUp7 = Conv2DTranspose(nF3, filterSizeUp,
                                  strides=strideConvUp, padding=padType)(conv6b)
        concat7 = concatenate([conv3b, convUp7], axis=3)
        conv7a = Conv2D(nF3, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(concat7)
        conv7b = Conv2D(nF3, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv7a)

        # Third upsampling set
        convUp8 = Conv2DTranspose(nF2, filterSizeUp,
                                  strides=strideConvUp, padding=padType)(conv7b)
        concat8 = concatenate([conv2b, convUp8], axis=3)
        conv8a = Conv2D(nF2, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(concat8)
        conv8b = Conv2D(nF2, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv8a)

        # Fourth upsampling set
        convUp9 = Conv2DTranspose(nF1, filterSizeUp,
                                  strides=strideConvUp, padding=padType)(conv8b)
        concat9 = concatenate([conv1b, convUp9], axis=3)
        conv9a = Conv2D(nF1, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(concat9)
        conv9b = Conv2D(nF1, filterSize, activation=actType,
                        strides=strideConv, padding=padType)(conv9a)
        
        # Final convolution with sigmoid activation for target prediction
        # **Two filter layers for 0,1 binary target?
        convFinal = Conv2D(2, (1, 1), activation='sigmoid')(conv9b)
                                
        # Create and compile model
        model = Model(inputs=inputs, outputs=convFinal)
        model.compile(optimizer=Adam(lr=0.001), loss=kmetrics.binary_crossentropy,
                      metrics=[kmetrics.categorical_accuracy])
        
        return model

