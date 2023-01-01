import os
import time
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import *
from util_progress_bar import progress_bar



def get_jpgs(folder_name):
    import os
    file_paths = []
    file_names = []
    # r=root, d=directories, f = files
    print("Getting all the file in", folder_name, ":")
    for r, d, f in os.walk(folder_name):
        for file in f:
            if file.endswith(".jpg"):
                file_paths.append(os.path.join(r, file))
                file_names.append(file)
    return file_paths, file_names

file_paths, file_names = get_jpgs(os.path.join("Gay_faces", "cropped"))



img = load_img("Gay_faces/cropped/0a051ec0-f2b9-41c9-8bdf-f4b060b35323.jpg")
img_array = img_to_array(img)
imgs_array = img_array
print(imgs_array.shape)

imgs = load_img("Gay_faces/cropped")

# for file_path in file_paths:
#
#
#     # load the image
#     img = load_img(file_path)
#     # print("Orignal:", type(img))
#
#     # convert to numpy array
#     img_array = img_to_array(img)
#     # print(img_array.shape)
#     # print("NumPy array info:")
#     # print(type(img_array))
#     np.append([imgs_array], [img_array], axis=0)

print(imgs_array.shape)



#nparray saving method
np.save("imgs.npy",imgs_array)
# b = np.load("filename.npy")