import numpy as np
import cv2
import time
import data_utils as utils
from shuffle_net import ShuffleNet
import config as cf
from mtcnn import MTCNN

def main_age_gender_detection(image):
    # Set up arguments
    arguments = utils.get_arguments()
    margin = arguments.margin
    weight_path = arguments.weight_file
    boxes = []

    # Detector use by MTCNN
    detector = MTCNN()

    # Load image
    # image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]

    # Load weights
    model = ShuffleNet(trainable=False).model
    model.load_weights(weight_path)

    # Run detector through a image(frame)
    start_time = time.time()
    result = detector.detect_faces(image)
    end_time = time.time()

    num_detected_face = len(result)
    if num_detected_face == 0:
        print('No detected face')
        exit()


    faces = np.empty((num_detected_face, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
    print("Detected {} faces on {}s".format(num_detected_face, end_time - start_time))


    # crop faces
    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']

        # coordinates of boxes
        left, top = bounding_box[0], bounding_box[1]
        right, bottom = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

        # coordinates of cropped image
        x1_crop = max(int(left), 0)
        y1_crop = max(int(top), 0)
        x2_crop = int(right)
        y2_crop = int(bottom)

        cropped_face = image[y1_crop:y2_crop, x1_crop:x2_crop, :]
        face = cv2.resize(cropped_face, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
        faces[i, :, :, :] = face
        box = (x1_crop, y1_crop, x2_crop, y2_crop)
        boxes.append(box)

    # predict
    result = model.predict(faces / 255.0)

    # Draw bounding boxes and labels on image
    image, age, gender = utils.draw_labels_and_boxes(image, boxes, result, margin)


    if image is None:
        exit()

    image = cv2.resize(image, (img_w, img_h), cv2.INTER_AREA)
    # cv2.imwrite('hoailinh_result.png', image)
    # cv2.imshow('img', image)

    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     exit()


    return age, gender




if __name__ == '__main__':
    file_name = input('Photo you want to predict\nFilename: ')
    img_path = '../utk_data/UTKFace/' + file_name + '.jpg'
    img = cv2.imread(img_path)

    main_age_gender_detection(img)

    actual_age = file_name.split('_')[0]
    actual_gender = file_name.split('_')[1]

    if actual_gender == '0':
        actual_gender = 'Male'
    elif actual_gender == '1':
        actual_gender = 'Female'
    else:
        actual_gender = '?'
    print("\nActual Age: {}\nActual Gender: {}".format(actual_age, actual_gender))

    origin_image = cv2.imread(img_path)
    # cv2.namedWindow('Origin_image', cv2.WINDOW_NORMAL)
    cv2.imshow('Origin_image', origin_image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()