# Import required Libraries
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from gaydar import *
from util_fake_prediction_generator import *
from tkinter import filedialog as fd
from tkinter import ttk
from main_age_gender_detection import *

# craete a detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# file_image_variables
input_source = "webcam"
input_img = None
orig_input_img = None

#webcam_image_variables
screenshots = []
predict_nums= []
orig_screenshots = []
webcam_predict_finished = False

# sex_lbl_variables
sex_lbl_txt = "sexuality"

# age_lbl_variables
age_lbl_txt = "age"

# gender_lbl_variables
gender_lbl_txt = "gender"

# init the winsow
win = Tk() # Create an instance of TKinter Window or frame
win.geometry("2000x1300") # Set the size of the window


# functions
def shot_button_event(result_to_show=True):
   global input_source
   global orig_input_img
   global webcam_predict_finished
   global orig_screenshots
   global screenshots
   global predict_nums
   input_source = "webcam"

   def take_a_shot():
      new_img = cap.read()[1]
      screenshots.append(new_img)
      orig_screenshots.append(new_img.copy())
      shot_num_lbl.config(text="Screen Shot " + str(len(screenshots)) + "/5")
      print(len(screenshots))

   if len(screenshots) < 5-1:
      print("case 1")
      take_a_shot()
   elif webcam_predict_finished == True:
      print("case 2")
      webcam_predict_finished = False
      screenshots = []
      orig_screenshots = []
      predict_nums = []
      take_a_shot()
   elif len(screenshots) == 5-1 and webcam_predict_finished == False:
      print("case 3")
      take_a_shot()

      try:
         combine_predict(mode="webcam", result_to_show=result_to_show)
         shot_num_lbl.config(text="Finished")

      except:
         shot_num_lbl.config(text="Detect Error!")

      webcam_predict_finished = True

def exit():
   plt.close('all')
   win.destroy()

def open_file(result_to_show=True):
   global input_source
   global input_img
   global orig_input_img
   # file type
   filetypes = (
      ('Image Files', '*.jpg *.png'),
      ('All files', '*.*')
   )
   # show the open file dialog
   file = fd.askopenfile(filetypes=filetypes)
   print(file.name)
   img = cv2.imread(file.name)
   input_img = img
   input_source = "file"
   combine_predict(mode="file", img=img, result_to_show=result_to_show)


def combine_predict(mode, img=None, result_to_show=True):
   global predict_nums
   global orig_screenshots
   if mode == "file":
      orig_input_img = img.copy()
      predict = prediction(img, result_to_show=False, crop_to_show=False, face_D_to_show=False, to_print_result=False)
      crop_img = crop_face(orig_input_img, face_D_to_show=False, crop_to_show=False)
      predict_age, predict_gender = main_age_gender_detection(img)

   elif mode == "webcam":
      for i in range(5):
         predict_nums.append(prediction(screenshots[i], result_to_show=False, crop_to_show=False, face_D_to_show=False, to_print_result=False))
      #print(predict_nums)
      predict = round(sum(predict_nums) / len(predict_nums),2)
      crop_img = crop_face(orig_screenshots[0], face_D_to_show=False, crop_to_show=False)
      predict_age, predict_gender = main_age_gender_detection(screenshots[0])


   sex_lbl.config(text="Gay Possibility: " + str(predict) + "%")
   age_lbl.config(text="Age: " + predict_age + " Years Old")
   gender_lbl.config(text="Gender: " + predict_gender)


   if result_to_show:
      plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
      fig = plt.gcf()
      fig.canvas.manager.set_window_title('Prediction')
      plt.title("Why You Gay Test - " + str(predict) + "%")
      plt.show()

def show_frames():
   with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9) as face_detection:

      def image_config():
         label.configure(image=imgtk, width=1280, height=720)
         # Repeat after an interval to capture continiously
         label.after(20, show_frames)

      if input_source == "webcam":
         if len(screenshots) == 5:
            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(screenshots[4], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk, width=1280, height=720)
            # Repeat after an interval to capture continiously
            label.after(20, show_frames)


         else:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            success, image = cap.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
               for detection in results.detections:
                  mp_drawing.draw_detection(image, detection)

            # Get the latest frame and convert into Image
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            image_config()
      elif input_source == "file":
         # Get the latest frame and convert into Image
         cv2image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
         cv2image = cv2.resize(cv2image, (int(cv2image.shape[1]*720/cv2image.shape[0]),720))
         img = Image.fromarray(cv2image)
         # Convert image to PhotoImage
         imgtk = ImageTk.PhotoImage(image=img)
         label.imgtk = imgtk
         image_config()

# Init objects
# Create an image Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0, columnspan=3, rowspan=4)

# init shot_num_lbl
shot_num_lbl = Label(win, text="Take a Screen Shot or Select a File",font=("Arial", 15), borderwidth=3, relief="solid")
shot_num_lbl.grid(column=3,row=0)

# init shot_bttn
shot_icon = Image.open("icons/shutter.png")
shot_icon = shot_icon.resize((150,150))
shot_icon = ImageTk.PhotoImage(shot_icon)
shot_bttn = Button(win,image=shot_icon, command=lambda:shot_button_event())
shot_bttn.grid(column=3, row=1, padx=40)



# init shot_bttn_lbl
shot_bttn_lbl = Label(text="Take a Screenshot")
shot_bttn_lbl.place(x=1346, y=268)

#init file_bttn_lbl
file_bttn_lbl = Label(text="Open a File")
file_bttn_lbl.place(x=1368, y=481)

#init exit_bttn_lbl
exit_bttn_lbl = Label(text="Exit The App")
exit_bttn_lbl.place(x=1366, y=694)

# delta_x = 0
# delta_y = 0
# def move(obj, move):
#    global delta_x
#    global delta_y
#    if move == "left":
#       delta_x -= x
#    if move == "right":
#       delta_x += x
#    if move == "up":
#       delta_y -= x
#    if move == "down":
#       delta_y += x
#    obj.place(x=1368+delta_x, y=694+delta_y)
#    print("delta_x", delta_x)
#    print("delta_y", delta_y)
#
# target = exit_bttn_lbl
# # intit left_bttn
# left_bttn = Button(win, text="Left", command=lambda:move(target, "left"))
# left_bttn.grid(column=3, row=4)
#
# # intit right_bttn
# right_bttn = Button(win, text="Right", command=lambda:move(target, "right"))
# right_bttn.grid(column=3, row=5)
#
# # intit up_bttn
# up_bttn = Button(win, text="up", command=lambda:move(target, "up"))
# up_bttn.grid(column=3, row=6)
#
# # intit down_bttn
# down_bttn = Button(win, text="down", command=lambda:move(target, "down"))
# down_bttn.grid(column=3, row=7)
#
# x = 1
# def toggle_x():
#    global x
#    if x == 1:
#       x = 10
#    elif x == 10:
#       x = 100
#    elif x == 100:
#       x = 1
#    x_bttn.configure(text="x" + str(x))
# # init x_bttn
# x_bttn = Button(win, text="X" + str(x), command=toggle_x)
# x_bttn.grid(column=3, row=8)

# init open file button
open_icon = Image.open("icons/folder.png")
open_icon = open_icon.resize((150,150))
open_icon = ImageTk.PhotoImage(open_icon)
open_button = Button(win,image=open_icon, relief="solid", borderwidth=0, highlightthickness=0, background="red", command=open_file)
open_button.grid(column=3, row=2)

#init exit_bttn
exit_icon = Image.open("icons/scalable.png")
exit_icon = exit_icon.resize((150,150))
exit_icon = ImageTk.PhotoImage(exit_icon)
exit_bttn = Button(win, image=exit_icon, command=exit)
exit_bttn.grid(column=3, row=3)

# init sex_lbl
sex_lbl = Label(win, text=sex_lbl_txt, font=("Arial", 25))
sex_lbl.grid(column=0, row=4)

# init age_lbl
age_lbl = Label(win, text=age_lbl_txt, font=("Arial", 25))
age_lbl.grid(column=1, row=4)

# init gender_lbl
gender_lbl = Label(win, text=gender_lbl_txt, font=("Arial", 25))
gender_lbl.grid(column=2, row=4)

# main
cap = cv2.VideoCapture(0)
show_frames()
win.mainloop()