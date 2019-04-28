import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
    

def circle_click(event):
    for circlePair in circles:
        cv.itemconfig(circlePair[0], fill="blue")
        cv.itemconfig(circlePair[1], fill="blue")

    print("event triggered")
    event_cord = (event.x-x1, event.y-y1)
    closest_dist = 20
    best_idx = 0
    for i, match in enumerate(piece_matches):
        dist1 = np.linalg.norm(match[0]-event_cord)
        dist2 = np.linalg.norm(match[1] - event_cord)
        if dist1 < closest_dist or dist2 < closest_dist:
            closest_dist = min(dist1, dist2)
            best_idx = i
            
    last_circle_1 = cv.itemconfig(circles[best_idx][0], fill="green")
    last_circle_2 = cv.itemconfig(circles[best_idx][1], fill="green")
    

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x - r, y - r, x +
    r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle


root = tk.Tk()
root.title("Mouse click within blue rectangle ...")
# create a canvas for drawing
w = 1014
h = 378
cv = tk.Canvas(root, width=w, height=h, bg='white')
cv.pack()
x1 = 0
y1 = 0
x2 = 1008
y2 = 756

# Draw images on canvas
image = Image.open("../data/Parks_Pieces.png")
image = image.resize((504, 378), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
image_id = cv.create_image(0,0,image=photo, anchor=tk.NW)

image2 = Image.open("../data/Parks_Complete_Puzzle.jpeg")
image2 = image2.resize((504, 378), Image.ANTIALIAS)
photo2 = ImageTk.PhotoImage(image2)
image_id2 = cv.create_image(510,0,image=photo2, anchor=tk.NW)

piece_matches = np.load('piece_matches.npy')
piece_matches = piece_matches / 8
piece_matches[:, 1, 0] = piece_matches[:, 1, 0] + 510

# Draw circles and bind event listener
circles = []
for match in piece_matches:
    circle1 = cv.create_circle(match[0][0], match[0][1], 5, fill="blue", outline="#DDD", width=0)
    circle2 = cv.create_circle(match[1][0], match[1][1], 5, fill="blue", outline="#DDD", width=0)
    circles.append([circle1, circle2])
    cv.tag_bind(circle1, "<1>", circle_click)
    cv.tag_bind(circle2, "<1>", circle_click)






root.mainloop()
