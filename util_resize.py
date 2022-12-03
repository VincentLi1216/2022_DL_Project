import os
import cv2

def resize(folder_name, size=100):
    import os
    file_paths = []
    file_names = []
    # r=root, d=directories, f = files
    print("Getting all the file in", folder_name, ":")
    for r, d, f in os.walk(folder_name):
        for file in f:
            if file.endswith(".jpg"):
                # print(folder_name+"/"+file)
                img = cv2.imread(folder_name+"/"+file)
                img = cv2.resize(img, (size,size))
                cv2.imwrite(folder_name+"/"+file, img)

                file_paths.append(os.path.join(r, file))
                file_names.append(file)

                print(file)
                # print(os.path.join(r, file))
    # return file_paths, file_names

get_jpgs("Gay_faces/cropped")