import uuid
import shutil
import cv2

def rename_jpgs(folder_name):
    import os

    # r=root, d=directories, f = files
    print("Getting all the file in", folder_name, ":")
    for r, d, f in os.walk(folder_name):
        for file in f:
            if file.endswith(".jpg"):

                try:
                    os.mkdir(main_folder+"/rename/")
                except:
                    a=1
                img = cv2.imread(os.path.join(r, file))
                cv2.imwrite(main_folder+"/rename/"+str(uuid.uuid4())+".jpg", img)
                # os.rename(os.path.join(r, file), main_folder+"/rename/"+str(uuid.uuid4())+".jpg")
                print(file)
                # print(os.path.join(r, file))



main_folder = "Gay_faces"
rename_jpgs(main_folder + "/orig")