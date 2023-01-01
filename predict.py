import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import time
from util_progress_bar import progress_bar

# init the mp_face_detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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


# use mp_face_detection to find the faces in the image than crop it to size 100*100 pixels in the format of .jpg
def crop_face(folder_name, to_show = False):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9) as face_detection:
        time.sleep(1)  # in order to get the progress bar display normally

        # print out all the file it gets
        print("Working process:")
        for idx, path in enumerate(os.listdir(folder_name)):
            if path.endswith(".DS_Store"):
                continue
            image = cv2.imread(os.path.join(folder_name, path))  # read the img
            progress_bar(idx, len(os.listdir(folder_name)))  # update the progress bar
            orig_shape = [image.shape[1], image.shape[0]]  # get the shape of the img
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert the BGR image to RGB and process it with MediaPipe Face Detection
            orig_image = image.copy()  # get the orig copy

            # no faces detected
            if not results.detections:
                continue

            # face detected
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    plt.show()
                    # crop_img(orig_image, detection, orig_shape, path=folder_name + "/cropped", idx=idx, to_show=to_show)  # crop the img




crop_face("test1/rename")