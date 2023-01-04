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

def utk_data_preprocessing(filepath):
    images = []
    ages = []
    genders = []
    races = []

    length = len(os.listdir(filepath))
    for i in os.listdir(filepath)[0:length]:
        if i.count('_') != 3:
            continue
        split = i.split('_')
        ages.append(int(split[0]))
        genders.append(int(split[1]))
        races.append(int(split[2]))
        images.append(
            cv2.imread(filepath + i))

    # ---

    images_series = pd.Series(list(images), name='Images')
    ages_series = pd.Series(list(ages), name='Ages')
    genders_series = pd.Series(list(genders), name='Genders')
    races_series = pd.Series(list(races), name='Races')

    df = pd.concat([images_series, ages_series, genders_series, races_series], axis=1)

    # ---

    # seaborn初始設定
    sns.set_theme()
    # sns.distplot(df['Ages'], kde=True, bins=50)

    # ---

    # 計算最多的那個要抽樣多少，才會跟第二多的一樣多
    bar_values = []
    for h in sns.distplot(df['Ages'], kde=True, bins=50).patches:
        bar_values.append(round(h.get_height(), 3))
    bar_values.sort(reverse=True)
    # print('bar_values:', bar_values)

    # ---

    # 讓最多的那個直接跟第二多的一樣數量
    under4s = []
    for i in range(len(df)):
        if df['Ages'].iloc[i] <= 4:
            under4s.append(df.iloc[i])
    under4s = pd.DataFrame(under4s)
    under4s = under4s.sample(frac=bar_values[1] / bar_values[0])

    df = df[df['Ages'] > 4]

    df = pd.concat([df, under4s], ignore_index=True)

    # ---

    # 把大於80歲的人拿掉
    df.drop(df[df['Ages'] > 80].index, inplace=True)

    return df