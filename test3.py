import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import cv2

# Root window
root = tk.Tk()
root.title('Display a Text File')
root.resizable(False, False)
root.geometry('550x250')

# Text editor
text = tk.Text(root, height=12)
text.grid(column=0, row=0, sticky='nsew')


def open_text_file():
    # file type
    filetypes = (
        ('Image Files', '*.jpg *.png'),
        ('All files', '*.*')
    )
    # show the open file dialog
    file = fd.askopenfile(filetypes=filetypes)
    print(file.name)
    img = cv2.imread(file.name)
    cv2.imshow("Image raed", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# open file button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=open_text_file
)

open_button.grid(column=0, row=1, sticky='w', padx=10, pady=10)


root.mainloop()