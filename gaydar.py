# import lib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from util_warning import *

# init the mp_face_detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# load the gaydar model
model = load_model('models/imageclassifier2.h5')

# predict how gay are you
def prediction(img, face_D_to_show, crop_to_show, result_to_show, to_print_result=True):
    import cv2


    img = crop_face(image=img, face_D_to_show=face_D_to_show, crop_to_show=crop_to_show)

    # if there's nothing pass in then return
    if img is None:
        return None


    # convert to the correct color format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize the image again just for sure
    resize = tf.image.resize(img, (256, 256))

    # predict the model
    yhat = model.predict(np.expand_dims(resize / 255, 0))

    # print out the result
    if to_print_result:
        print(round(100 - yhat[0, 0] * 100, 2), "% is Gay")

    # show the result
    # todo: change the title of the window
    if result_to_show:
        plt.imshow(img)
        fig = plt.gcf()
        fig.canvas.manager.set_window_title('Prediction')
        plt.title("Why You Gay Test - " +  str(round(100 - yhat[0, 0] * 100, 2)) + "%")
        plt.show()
    return round(100 - yhat[0, 0] * 100, 2)

# use mp_face_detection to find the faces in the image than crop it to size 100*100 pixels in the format of .jpg
def crop_face(image, face_D_to_show, crop_to_show):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9) as face_detection:

        # print out all the file it gets
        print("Working process:")


        # progress_bar(idx, len(os.listdir(folder_name)))  # update the progress bar
        orig_shape = [image.shape[1], image.shape[0]]  # get the shape of the img
        results = face_detection.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))  # convert the BGR image to RGB and process it with MediaPipe Face Detection
        orig_image = image.copy()  # get the orig copy

        # no faces detected
        if not results.detections:
            # print(os.path.join(img_path, path) + " has no faces detected!")
            warning("NO Faces Detected!")
            return None

        # face detected
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                if face_D_to_show:
                    cv2.imshow("Face Detected", image)
                    cv2.waitKey(0)
                    cv2.destroyWindow("Face Detected")

                return crop_img(orig_image, detection, orig_shape, crop_to_show=crop_to_show)  # crop the img


# to crop img
def crop_img(img, detection, orig_shape, crop_to_show):
    # get all the info to crop
    orig_width = orig_shape[0]
    orig_height = orig_shape[1]

    xmin = int(detection.location_data.relative_bounding_box.xmin * orig_width)
    ymin = int(detection.location_data.relative_bounding_box.ymin * orig_height)
    width = int(detection.location_data.relative_bounding_box.width * orig_width)
    height = int(detection.location_data.relative_bounding_box.height * orig_height)

    img = img[ymin:ymin + height, xmin:xmin + width]  # got to reverse the order of x and y

    # show the img if necessary
    if crop_to_show:
        cv2.imshow("cropped", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    try:
        crp_img = cv2.resize(img, (256, 256))
        # cv2.imwrite(path+"/"+IMAGE_NAMES[idx],img)
        return crp_img
    except:
        # print out failed img's filename
        print()
        print("crop error")
        print("can Not crop the image")


# the main func
def main(folder_name="test1/orig"):
    # get all the file in the folder,
    for idx, path in enumerate(os.listdir(folder_name)):

        # only read the .jpg files
        if not path.endswith(".jpg"):
            continue

        # print out img's name
        print(path)

        # show the orig img
        image = cv2.imread(os.path.join(folder_name, path))
        cv2.imshow(path, image)
        cv2.waitKey(0)
        cv2.destroyWindow(path)

        # firstly crop the img then predict it
        prediction(image, result_to_show=True, face_D_to_show=True, crop_to_show=True)


if __name__ == "__main__":
    main("test2/orig")
