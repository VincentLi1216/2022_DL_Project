import os
import time
import cv2
import mediapipe as mp
from util_progress_bar import progress_bar

# init the mp_face_detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# use os.walk to get all the jpgs in the folder
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
    return file_paths, file_names

# use mp_face_detection to find the faces in the image than crop it to size 100*100 pixels in the format of .jpg
def crop_face(folder_name, to_show = False):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.9) as face_detection:
        time.sleep(1)  # in order to get the progress bar display normally

        # print out all the file it gets
        print("Working process:")
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)  # read the img
            progress_bar(idx, len(IMAGE_NAMES))  # update the progress bar
            orig_shape = [image.shape[1], image.shape[0]]  # get the shape of the img
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert the BGR image to RGB and process it with MediaPipe Face Detection
            orig_image = image.copy()  # get the orig copy

            # no faces detected
            if not results.detections:
                continue

            # face detected
            if results.detections:
                for detection in results.detections:
                    crop_img(orig_image, detection, orig_shape, path=folder_name + "/cropped", idx=idx, to_show=to_show)  # crop the img


    progress_bar(1,1)  # make the progress bar display 100% at the end


# to resize img
def resize(img, size=100):
    import cv2

    resized = cv2.resize(img, (size, size))
    return resized

#to crop img
def crop_img(img, detection, orig_shape,path,idx, to_show = False):
    # get all the info to crop
    orig_width = orig_shape[0]
    orig_height = orig_shape[1]

    xmin = int(detection.location_data.relative_bounding_box.xmin * orig_width)
    ymin = int(detection.location_data.relative_bounding_box.ymin * orig_height)
    width = int(detection.location_data.relative_bounding_box.width * orig_width)
    height = int(detection.location_data.relative_bounding_box.height * orig_height)

    img = img[ymin:ymin+height, xmin:xmin+width]  # got to reverse the order of x and y

    # show the img if necessary
    if to_show == True:
        cv2.imshow("cropped", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # make dir
    try:
        os.mkdir(path)
    except:
        a = 0  # do nothing

    try:
        img = resize(img, 256)
        cv2.imwrite(path+"/"+IMAGE_NAMES[idx],img)
    except:
        # print out failed img's filename
        print()
        print(path+"/"+IMAGE_NAMES[idx])

# set the main_folder(working folder)'s name
main_folder = "Straight_faces"

# save the files path and name in the variables
IMAGE_FILES, IMAGE_NAMES = get_jpgs(main_folder + "/rename")

# main call
crop_face(main_folder, to_show=False)
