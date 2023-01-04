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


#---


# Data Preprocessing
df = utk_data_preprocessing('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//utk_data//crop_part1//')


#---


# 準備輸入與標籤資料
x = []
y = []
for i in range(len(df)):
    df['Images'].iloc[i] = cv2.resize(df['Images'].iloc[i], (200,200), interpolation=cv2.INTER_AREA)
    ar = np.asarray(df['Images'].iloc[i])
    x.append(ar)
    agegenrace = [int(df['Ages'].iloc[i]), int(df['Genders'].iloc[i]), int(df['Races'].iloc[i])]
    y.append(agegenrace)
x = np.array(x)


#---


# 等等會分成3個model，分別算age, gender, race
# y是標籤
y_age = df['Ages']
y_gender = df['Genders']
y_race = df['Races']

x_train_age, x_valid_test_age, y_train_age, y_valid_test_age = train_test_split(x, y_age, test_size=0.3, random_state = 42)
x_train_gender, x_valid_test_gender, y_train_gender, y_valid_test_gender = train_test_split(x, y_gender, test_size=0.3, random_state = 42)
x_train_race, x_valid_test_race, y_train_race, y_valid_test_race = train_test_split(x, y_race, test_size=0.3, random_state = 42)

x_valid_age, x_test_age, y_valid_age, y_test_age = train_test_split(x_valid_test_age, y_valid_test_age, test_size=0.33, random_state = 42)
x_valid_gender, x_test_gender, y_valid_gender, y_test_gender = train_test_split(x_valid_test_gender, y_valid_test_gender, test_size=0.33, random_state = 42)
x_valid_race, x_test_race, y_valid_race, y_test_race = train_test_split(x_valid_test_race, y_valid_test_race, test_size=0.33, random_state = 42)


#---


mobile = tf.keras.applications.mobilenet.MobileNet()
mobile_layers = mobile.layers[-5].output


# age model
output = Dense (units = 1, activation = 'relu')(mobile_layers)
age_model = Model(inputs=mobile.input, outputs=output)
age_model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0001))
for layer in age_model.layers[:-8]:
    layer.trainable = False
age_model.summary()


# gender model
output = Dense (units = 1, activation = 'sigmoid')(mobile_layers)
gender_model = Model(inputs=mobile.input, outputs=output)
gender_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
for layer in gender_model.layers[:-8]:
    layer.trainable = False
gender_model.summary()


# race model
output = Dense (units = 5, activation = 'softmax')(mobile_layers)
race_model = Model(inputs=mobile.input, outputs=output)
race_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
for layer in race_model.layers[:-8]:
    layer.trainable = False
race_model.summary()


#---


# Training
datagen = ImageDataGenerator(rescale=1./255., width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

agetrain = datagen.flow(x_train_age, y_train_age, batch_size=100)
agevalid = valid_test_datagen.flow(x_valid_age, y_valid_age, batch_size=100)
agetest = valid_test_datagen.flow(x_test_age, y_test_age, batch_size=100)

checkpoint_path = 'C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_age_model_checkpoint.ckpt'
# checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

if os.path.exists(checkpoint_path) == True:
    age_model.load_weights(checkpoint_path)

age_history = age_model.fit(agetrain, epochs=1, shuffle=True, validation_data=agevalid, callbacks=checkpoint_callback)

age_model.save('C://Users//allen//Documents//PyCharm//Projects//2022_DL_Project//age_gen_race_model//pretrained_model_crop_part1_age_model.h5')

age_test_loss, age_test_acc = age_model.evaluate(agetest)

print( "\nAge Test Loss:", round(age_test_loss, 2) )
print( "Age Test Accuracy:", round(age_test_acc, 2) )