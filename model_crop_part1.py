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


#---


images = []
ages = []
genders = []
races = []

length = len(os.listdir('C://Users//allen//Documents//PyCharm//Projects//age_gender_race_detection//utk_data//crop_part1//'))
for i in os.listdir('C://Users//allen//Documents//PyCharm//Projects//age_gender_race_detection//utk_data//crop_part1//')[0:length]:
    if i.count('_') != 3:
        continue
    split = i.split('_')
    ages.append(int(split[0]))
    genders.append(int(split[1]))
    races.append(int(split[2]))
    images.append(cv2.imread('C://Users//allen//Documents//PyCharm//Projects//age_gender_race_detection//utk_data//crop_part1//' + i))


#---


images_series = pd.Series(list(images), name = 'Images')
ages_series = pd.Series(list(ages), name = 'Ages')
genders_series = pd.Series(list(genders), name = 'Genders')
races_series = pd.Series(list(races), name = 'Races')

df = pd.concat([images_series, ages_series, genders_series, races_series], axis=1)


#---


# seaborn初始設定
sns.set_theme()
# sns.distplot(df['Ages'], kde=True, bins=50)


#---


# 計算最多的那個要抽樣多少，才會跟第二多的一樣多
bar_values = []
for h in sns.distplot(df['Ages'], kde=True, bins=50).patches:
    bar_values.append(round(h.get_height(),3))
bar_values.sort(reverse = True)
# print('bar_values:', bar_values)


#---


# 讓最多的那個直接跟第二多的一樣數量
under4s = []
for i in range(len(df)):
    if df['Ages'].iloc[i] <= 4:
        under4s.append(df.iloc[i])
under4s = pd.DataFrame(under4s)
under4s = under4s.sample(frac=bar_values[1]/bar_values[0])

df = df[df['Ages'] > 4]

df = pd.concat([df, under4s], ignore_index = True)


#---


# 把大於80歲的人拿掉
df.drop(df[df['Ages']>80].index, inplace=True)


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


def mobilnet_block (x_, filters, strides):
    x_ = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x_)
    x_ = BatchNormalization()(x_)
    x_ = ReLU()(x_)
    x_ = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x_)
    x_ = BatchNormalization()(x_)
    x_ = ReLU()(x_)
    return x_


# age model
input = Input(shape = (200,200,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
    x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
x = AvgPool2D (pool_size = 7, strides = 1)(x)
# x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
output = Dense (units = 1, activation = 'relu')(x)
age_model = Model(inputs=input, outputs=output)
age_model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=0.0001))
age_model.summary()


# gender model
input = Input(shape = (200,200,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
    x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
output = Dense (units = 1, activation = 'sigmoid')(x)
gender_model = Model(inputs=input, outputs=output)
gender_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
gender_model.summary()


# race model
input = Input(shape = (200,200,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
    x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
output = Dense (units = 5, activation = 'softmax')(x)
race_model = Model(inputs=input, outputs=output)
race_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
race_model.summary()


#---


# Training
datagen = ImageDataGenerator(rescale=1./255., width_shift_range = 0.1, height_shift_range = 0.1, horizontal_flip = True)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

agetrain = datagen.flow(x_train_age, y_train_age, batch_size=100)
agevalid = valid_test_datagen.flow(x_valid_age, y_valid_age, batch_size=100)
agetest = valid_test_datagen.flow(x_test_age, y_test_age, batch_size=100)

checkpoint_path = 'C://Users//allen//Documents//PyCharm//Projects//age_gender_race_detection//model_crop_part1_checkpoint.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

if os.path.exists(checkpoint_path) == True:
    age_model.load_weights(checkpoint_path)
age_history = age_model.fit(agetrain, epochs=50, shuffle=True, validation_data=agevalid, callbacks=checkpoint_callback)

age_test_loss, age_test_acc = age_model.evaluate(agetest)

print( "\nAge Test Loss:", round(age_test_loss, 2) )
print( "Age Test Accuracy:", round(age_test_acc, 2) )