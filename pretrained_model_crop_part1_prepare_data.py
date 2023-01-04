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


def pretrained_model_crop_part1_prepare_data(which_model, df):
    # 準備輸入與標籤資料

    x = []
    y = []
    for i in range(len(df)):
        df['Images'].iloc[i] = cv2.resize(df['Images'].iloc[i], (224, 224), interpolation=cv2.INTER_AREA)
        ar = np.asarray(df['Images'].iloc[i])
        x.append(ar)
        agegenrace = [int(df['Ages'].iloc[i]), int(df['Genders'].iloc[i]), int(df['Races'].iloc[i])]
        y.append(agegenrace)
    x = np.array(x)
    y_age = df['Ages']
    y_gender = df['Genders']
    y_race = df['Races']


    # generate data

    x_train_age, x_valid_test_age, y_train_age, y_valid_test_age = train_test_split(x, y_age, test_size=0.3, random_state = 42)
    x_train_gender, x_valid_test_gender, y_train_gender, y_valid_test_gender = train_test_split(x, y_gender, test_size=0.3, random_state = 42)
    x_train_race, x_valid_test_race, y_train_race, y_valid_test_race = train_test_split(x, y_race, test_size=0.3, random_state = 42)

    x_valid_age, x_test_age, y_valid_age, y_test_age = train_test_split(x_valid_test_age, y_valid_test_age, test_size=0.33, random_state = 42)
    x_valid_gender, x_test_gender, y_valid_gender, y_test_gender = train_test_split(x_valid_test_gender, y_valid_test_gender, test_size=0.33, random_state = 42)
    x_valid_race, x_test_race, y_valid_race, y_test_race = train_test_split(x_valid_test_race, y_valid_test_race, test_size=0.33, random_state = 42)

    datagen = ImageDataGenerator(rescale=1./255., width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    if which_model == 0:
        train_data = datagen.flow(x_train_age, y_train_age, batch_size=100)
        valid_data = valid_test_datagen.flow(x_valid_age, y_valid_age, batch_size=100)
        test_data = valid_test_datagen.flow(x_test_age, y_test_age, batch_size=100)

    elif which_model == 1:
        train_data = datagen.flow(x_train_gender, y_train_gender, batch_size=100)
        valid_data = valid_test_datagen.flow(x_valid_gender, y_valid_gender, batch_size=100)
        test_data = valid_test_datagen.flow(x_test_gender, y_test_gender, batch_size=100)

    elif which_model == 2:
        train_data = datagen.flow(x_train_race, y_train_race, batch_size=100)
        valid_data = valid_test_datagen.flow(x_valid_race, y_valid_race, batch_size=100)
        test_data = valid_test_datagen.flow(x_test_race, y_test_race, batch_size=100)

    else:
        print("pretrained_model_crop_part1_prepare_data() error!")


    return train_data, valid_data, test_data