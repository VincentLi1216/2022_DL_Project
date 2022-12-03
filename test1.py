import os
import time
import cv2
import mediapipe as mp
from util_progress_bar import progress_bar
from os import walk
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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
                print(file)
                # print(os.path.join(r, file))
    return file_paths, file_names

# For static images:


def find_face(folder_name, to_show = False):
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.9) as face_detection:
        time.sleep(1)

        print("Working process:")
        for idx, file in enumerate(IMAGE_FILES):
            progress_bar(idx, len(IMAGE_NAMES))
            image = cv2.imread(file)
            orig_shape = [image.shape[1], image.shape[0]]
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            orig_image = image.copy()
            # Draw face detections of each face.
            if not results.detections:
                try:
                    os.mkdir(folder_name + "/without_face/")
                except:
                    a = 1
                cv2.imwrite(folder_name + '/without_face/' + IMAGE_NAMES[idx], orig_image)
                continue

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    # print(detection)
                    crop_img(orig_image, detection, orig_shape,path="",file_name=file)
            try:
                os.mkdir(folder_name + "/with_face")
                # print(IMAGE_NAMES[idx] + " is added to " + folder_name + "/with_face")
            except:
                a = 1
                # print(IMAGE_NAMES[idx] + " is added to " + folder_name + "/with_face")
            cv2.imwrite(folder_name + '/with_face/' + IMAGE_NAMES[idx], orig_image)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

            if to_show:
                cv2.imshow('MediaPipe Face Detection', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    progress_bar(1,1)

def crop_img(img, detection, orig_shape,path,file_name):
    orig_width = orig_shape[0]
    orig_height = orig_shape[1]

    xmin = int(detection.location_data.relative_bounding_box.xmin * orig_width)
    ymin = int(detection.location_data.relative_bounding_box.ymin * orig_height)
    width = int(detection.location_data.relative_bounding_box.width * orig_width)
    height = int(detection.location_data.relative_bounding_box.height * orig_height)
    # print(xmin, ymin, width, height)

    img = img[ymin:ymin+height, xmin:xmin+width]
    # print(orig_width, orig_height)
    # img = img[846:846+500, 2040:2040+1000]
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main_folder = "test1"
IMAGE_FILES, IMAGE_NAMES = get_jpgs(main_folder + "/orig")
find_face(main_folder, to_show=False)
