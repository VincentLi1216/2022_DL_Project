import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import cv2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from keras.layers import Input, DepthwiseConv2D
from keras.layers import BatchNormalization
from keras.layers import ReLU, AvgPool2D
from keras import Model

def pretrained_model_crop_part1_load_model(which_model):
    try:
        model = tf.keras.models.load_model('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_'+which_model+'_model.h5')
    except:
        print("Can't find model. Use fine-tuned pretrained model.")
        mobile = tf.keras.applications.mobilenet.MobileNet()
        mobile_layers = mobile.layers[-5].output

        output = Dense(units=1, activation='relu')(mobile_layers)
        model = Model(inputs=mobile.input, outputs=output)
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0001))
        for layer in model.layers[:-8]:
            layer.trainable = False

    checkpoint_path = 'C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_'+which_model+'_model_checkpoint.ckpt'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

    if os.path.exists(checkpoint_path) == True:
        model.load_weights(checkpoint_path)

    return model, checkpoint_callback
