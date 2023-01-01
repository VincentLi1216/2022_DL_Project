import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from os import *
from matplotlib import pyplot as plt

model = load_model('models/imageclassifier2.h5')

def prediction(dir):
    img = cv2.imread(dir)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    print(yhat)
    if yhat > 0.5:
        print(f'Predicted class is Straight')
    else:
        print(f'Predicted class is Gay')
    plt.imshow(img)
    plt.show()

folder_path = "Gay_faces/cropped"
for path in os.listdir(folder_path):
    if path.endswith(".DS_Store"):
        continue
    print(os.path.join(folder_path, path))
    prediction(os.path.join(folder_path, path))
    print("----------------------")
    print()
    print()