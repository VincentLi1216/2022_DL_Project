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


def process_and_predict(image, age_model, gender_model, race_model):
    print("Image original shape:",image.shape)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 224, 224, 3)

    age = age_model.predict(image).astype(int)
    gender = np.round(gender_model.predict(image))
    race = np.round(race_model.predict(image))

    if gender == 0:
        gender = 'Male'
    elif gender == 1:
        gender = 'Female'

    if race == 0:
        race = 'Denoting White'
    elif race == 1:
        race = 'Black'
    elif race == 2:
        race = 'Asian'
    elif race == 3:
        race = 'Indian'
    elif race == 4:
        race = 'Others'

    # print('Age:', int(age), '\n Gender:', gender, '\n Race:', race)

    return age[0][0][0][0], gender, race


def pretrained_model_crop_part1_predict(img):
    age_model = tf.keras.models.load_model('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_' + 'age' + '_model.h5')
    gender_model = tf.keras.models.load_model('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_' + 'gender' + '_model.h5')
    race_model = tf.keras.models.load_model('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_' + 'race' + '_model.h5')

    age, gender, race = process_and_predict(img, age_model, gender_model, race_model)

    return age, gender, race


if __name__ == '__main__':
    path = 'C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//utk_data//UTKFace//31_0_0_20170120134310511.jpg.chip.jpg'
    age, gender, race = pretrained_model_crop_part1_predict(cv2.imread(path, 1))

    print('Age:', age, '\nGender:', gender, '\nRace:', race)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', cv2.imread(path, 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
