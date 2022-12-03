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


def crop_face(folder_name, to_show = False):
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
                continue

            if results.detections:
                for detection in results.detections:
                    crop_img(orig_image, detection, orig_shape,path=folder_name + "/cropped",idx=idx, to_show=to_show)


    progress_bar(1,1)

def crop_img(img, detection, orig_shape,path,idx, to_show = False):
    orig_width = orig_shape[0]
    orig_height = orig_shape[1]

    xmin = int(detection.location_data.relative_bounding_box.xmin * orig_width)
    ymin = int(detection.location_data.relative_bounding_box.ymin * orig_height)
    width = int(detection.location_data.relative_bounding_box.width * orig_width)
    height = int(detection.location_data.relative_bounding_box.height * orig_height)

    img = img[ymin:ymin+height, xmin:xmin+width]

    if to_show == True:
        cv2.imshow("cropped", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    try:
        os.mkdir(path)
    except:
        a = 0

    try:
        cv2.imwrite(path+"/"+IMAGE_NAMES[idx],img)
    except:
        print()
        print(path+"/"+IMAGE_NAMES[idx])


main_folder = "Gay_faces"
IMAGE_FILES, IMAGE_NAMES = get_jpgs(main_folder + "/rename")
crop_face(main_folder, to_show=False)
