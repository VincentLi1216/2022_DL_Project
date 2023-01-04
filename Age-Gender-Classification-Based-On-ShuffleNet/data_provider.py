import tensorflow.keras as tk
import config as cf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Datasets(object):
    def __init__(self, trainable=True):
        self.all_data = []
        self.convert_data_format(trainable)
        self.trainable = trainable

    def gen(self):
        # data_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=30,
        #                               width_shift_range=0.4, height_shift_range=0.4, horizontal_flip=True, vertical_flip=True)
        images = []
        age_labels = []
        gender_labels = []

        while True:
            np.random.shuffle(self.all_data)
            for i in range(len(self.all_data)):
                image, age_label, gender_label = self.all_data[i]
                age_label = tk.utils.to_categorical(age_label, num_classes=cf.NUM_AGE_CLASSES)
                gender_label = tk.utils.to_categorical(gender_label, num_classes=cf.NUM_GENDER_CLASSES)
                images.append(image)
                age_labels.append(age_label)
                gender_labels.append(gender_label)

                if len(images) == cf.BATCH_SIZE:
                    images = np.array(images) / 255.0
                    # if self.trainable:
                    #     data_gen.fit(images, augment=True)
                    age_labels = np.array(age_labels)
                    gender_labels = np.array(gender_labels)
                    yield images, {"age_output": age_labels, "gender_output": gender_labels}
                    images = []
                    age_labels = []
                    gender_labels = []

            if len(images):
                images = np.array(images) / 255.0
                # if self.trainable:
                #     data_gen.fit(images, augment=True)
                age_labels = np.array(age_labels)
                gender_labels = np.array(gender_labels)
                yield images, {"age_output": age_labels, "gender_output": gender_labels}
                images = []
                age_labels = []
                gender_labels = []

    def convert_data_format(self, trainable):
        if trainable:
            data = np.load(os.path.join(os.getcwd(), 'data/train_224x224.npy'), allow_pickle=True)
        else:
            data = np.load(os.path.join(os.getcwd(), 'data/test_224x224.npy'), allow_pickle=True)

        self.all_data = data
        if trainable:
            print('Number of train data data:', str(len(self.all_data)))
        else:
            print('Number of the test data', str(len(self.all_data)))