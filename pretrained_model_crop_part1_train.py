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

from utk_data_preprocessing import *
from pretrained_model_crop_part1_load_model import *
from pretrained_model_crop_part1_prepare_data import *


#--


# Data Preprocessing
df = utk_data_preprocessing('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//utk_data//crop_part1//')
print('\nData Preprocessing Finished.')


#--


# 等等會分成3個model，分別算age, gender, race
# 0:age, 1:gender, 2:race
agetrain, agevalid, agetest = pretrained_model_crop_part1_prepare_data(0, df)
gendertrain, gendervalid, gendertest = pretrained_model_crop_part1_prepare_data(1, df)
racetrain, racevalid, racetest = pretrained_model_crop_part1_prepare_data(2, df)
print('\nData Prepared.')


#--


# load model and train

def train_age():
    age_model, age_checkpoint_callback = pretrained_model_crop_part1_load_model('age')
    age_history = age_model.fit(agetrain, epochs=epochs_set, shuffle=True, validation_data=agevalid, callbacks=age_checkpoint_callback)
    age_model.save('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_age_model.h5')

def train_gender():
    gender_model, gender_checkpoint_callback = pretrained_model_crop_part1_load_model('gender')
    gender_history = gender_model.fit(gendertrain, epochs=epochs_set, shuffle=True, validation_data=gendervalid, callbacks=gender_checkpoint_callback)
    gender_model.save('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_gender_model.h5')

def train_race():
    race_model, race_checkpoint_callback = pretrained_model_crop_part1_load_model('race')
    race_history = race_model.fit(racetrain, epochs=epochs_set, shuffle=True, validation_data=racevalid, callbacks=race_checkpoint_callback)
    race_model.save('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_race_model.h5')

print('\nModel Loaded.')

epochs_set = 1
train_age()
train_gender()
train_race()