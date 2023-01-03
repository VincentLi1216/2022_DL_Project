# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
from gaydar import *

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("2000x1300")

# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=0)


screenshots = []
predict_nums= []

def shot_button_event(result_to_show=True):
   if len(screenshots) < 5:
      new_cap = cv2.VideoCapture(0)
      new_img = cap.read()[1]
      screenshots.append(new_img)
      shot_bttn.config(text="Screen Shot " + str(len(screenshots)) + "/5")
   if len(screenshots) == 5:
      for i in range(5):
         predict_nums.append(prediction(screenshots[i], result_to_show=False, crop_to_show=False, face_D_to_show=False, to_print_result=False))
      #print(predict_nums)
      avg_predict = round(sum(predict_nums) / len(predict_nums),2)
      crop_img = crop_face(screenshots[0], face_D_to_show=False, crop_to_show=False)
      if result_to_show:
         plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
         fig = plt.gcf()
         fig.canvas.manager.set_window_title('Prediction')
         plt.title("Why You Gay Test - " + str(avg_predict) + "%")
         plt.show()


   #print("Screen Shot Numbers:" + str(len(screenshots)))


# Create a buttons
shot_bttn = Button(win, text="Screen Shot", command=lambda:shot_button_event())
shot_bttn.grid(row=1, column=0)

cap = cv2.VideoCapture(0)

exit_bttn = Button(win, text="Exit", command=win.destroy)
exit_bttn.grid(row=2, column=0)

# Define function to show frame
def show_frames():
   with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.9) as face_detection:
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
         label.configure(image=imgtk, width=1280, height=720)
         # Repeat after an interval to capture continiously
         label.after(20, show_frames)


show_frames()
win.mainloop()