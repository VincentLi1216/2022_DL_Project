import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
import config as cf
from data_provider import Datasets
from model_layers import _block


class ShuffleNet(object):
    def __init__(self, trainable=True, scale_factor=1, pooling='avg', input_shape=(224, 224, 3),
                 groups=3, load_model=None, botteleneck_ratio=0.25):
        self.input_shape = input_shape
        self.groups = groups
        self.scale_factor = scale_factor
        self.num_shuffle_units = [3, 7, 3]
        self.out_channels_in_stage = None
        self.bottleneck_ratio = botteleneck_ratio
        self.pooling = pooling
        self.load_model = load_model

        # define input
        self.img_input = self._define_input()

        # build model
        self.model = self.build_model(self.img_input)

        # Compile the model
        losses = {
            "age_output": "categorical_crossentropy",
            "gender_output": "categorical_crossentropy"
        }

        self.model.compile(loss=losses, optimizer=Adam(1e-3), metrics=['acc'])
        
        if trainable:
            self.model.summary()
            self.train_data = Datasets(trainable=True)
            self.test_data = Datasets(trainable=False)

    def _define_input(self):
        out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

        if self.pooling not in ['max', 'avg']:
            raise ValueError('Invalid value for pooling')

        # calculate output channels for each stage
        exp = np.insert(np.arange(0, len(self.num_shuffle_units)), 0, 0)
        self.out_channels_in_stage = 2 ** exp
        self.out_channels_in_stage *= out_dim_stage_two[self.groups]
        self.out_channels_in_stage[0] = 24  # fist stage has always 24 output channels
        self.out_channels_in_stage *= self.scale_factor
        self.out_channels_in_stage = self.out_channels_in_stage.astype(int)

        img_input = layers.Input(shape=self.input_shape)
        return img_input

    @staticmethod
    def build_age_branch(x):
        # Output age branch
        predictions_age = layers.Dense(cf.NUM_AGE_CLASSES, activation="softmax", name='age_output')(x)

        return predictions_age

    @staticmethod
    def build_gender_branch(x):
        # Output gender branch
        predictions_gender = layers.Dense(cf.NUM_GENDER_CLASSES, activation="softmax", name='gender_output')(x)

        return predictions_gender

    def build_model(self, input_img):
        x = layers.Conv2D(filters=self.out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False,
                          strides=(2, 2), activation='relu', name='conv1')(input_img)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool1')(x)

        # create stages containing shuffle-net units beginning at stage 2
        for stage in range(len(self.num_shuffle_units)):
            repeat = self.num_shuffle_units[stage]
            x = _block(x, self.out_channels_in_stage, repeat=repeat, bottleneck_ratio=self.bottleneck_ratio,
                       groups=self.groups, stage=stage + 2)

        if self.pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        elif self.pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='global_pool')(x)

        # Output layer
        predictions_age = self.build_age_branch(x)
        predictions_gender = self.build_gender_branch(x)

        model = Model(inputs=input_img, outputs=[predictions_age, predictions_gender], name="ShuffleNet")

        if self.load_model is not None:
            model.load_weights(self.load_model, by_name=True)

        return model

    def train(self):
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_age_output_acc', factor=0.9, patience=5, verbose=1, )
        # Model Checkpoint
        cpt_save = ModelCheckpoint('./weight.h5', save_best_only=True, monitor='val_age_output_acc', mode='max', save_weights_only=True)

        learn_rates = [0.02, 0.005, 0.001, 0.0005]
        lr_scheduler = LearningRateScheduler(lambda epoch: learn_rates[epoch // 30])

        print("Training......")
        step_val = len(self.test_data.all_data) // cf.BATCH_SIZE
        step_train = len(self.train_data.all_data) // cf.BATCH_SIZE // 2
          
        self.model.fit(self.train_data.gen(), batch_size=cf.BATCH_SIZE, steps_per_epoch=step_train,
                       callbacks=[cpt_save, reduce_lr, lr_scheduler], validation_data=self.test_data.gen(), validation_steps=step_val,
                       verbose=1, epochs=cf.NUM_EPOCHS, shuffle=True)
