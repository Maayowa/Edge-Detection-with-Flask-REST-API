import numpy as np
import tkinter as tk
from PIL import ImageTk, Image


def array_display(img):
    root = tk.Tk()

    img = Image.fromarray(img)
    shape = img.size
    if  max( img.size ) <= 300: 
        pct_ratio = 1.0 * 300/img.size[0]
        h = int( img.size[1] * pct_ratio)
        img = img.resize( (300, h), Image.ANTIALIAS )
        shape = img.size

    image = ImageTk.PhotoImage(image = img)

    # Create Display canvas
    canvas = tk.Canvas(root, width = shape[0], height = shape[1])
    canvas.pack()
    canvas.create_image(shape[0] // 2, shape[1] // 2, image = image)

    root.mainloop()


    