import tkinter as tk
from PIL import Image, ImageTk


def showxy(event):
    '''
    show x, y coordinates of mouse click position
    event.x, event.y relative to ulc of widget (here root) 
    '''
    # xy relative to ulc of root
    #xy = 'root x=%s  y=%s' % (event.x, event.y)
    # optional xy relative to blue rectangle
    xy = 'rectangle x=%s  y=%s' % (event.x-x1, event.y-y1)
    root.title(xy)


root = tk.Tk()
root.title("Mouse click within blue rectangle ...")
# create a canvas for drawing
w = 2032
h = 1024
cv = tk.Canvas(root, width=w, height=h, bg='white')
cv.pack()
# draw a blue rectangle shape with
# upper left corner coordinates x1, y1
# lower right corner coordinates x2, y2
x1 = 0
y1 = 0
x2 = 1008
y2 = 756

image = Image.open("../data/Fairies_pieces_small.png")
photo = ImageTk.PhotoImage(image)
image_id = cv.create_image(0,0,image=photo, anchor=tk.NW)
cv.tag_bind(image_id, "<1>", showxy)
# bind left mouse click within shape rectangle
root.mainloop()
