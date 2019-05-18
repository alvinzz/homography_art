import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl
import PIL
from PIL import ImageTk
from tkinter import *

def compute_H(pts_s, pts_t):
  assert len(pts_s) == len(pts_t)
  A = np.zeros((2 * len(pts_s) + 1,9))
  for i, (p_s, p_t) in enumerate(zip(pts_s, pts_t)):
    p_s = np.asarray((p_s[0], p_s[1], 1))
    A[2*i, 0:3] = -p_s
    A[2*i + 1, 3:6] = -p_s
    A[2*i, 6:9] = p_s * p_t[0]
    A[2*i + 1, 6:9] = p_s * p_t[1]
  A[-1,-1] = 1
  b = np.zeros(2 * len(pts_s) + 1)
  b[-1] = 1
  x = np.linalg.lstsq(A, b)[0]
  T_inv = x.reshape(3,3)
  return T_inv

def warp(target, source, H):
  h, w, d = target.shape
  x = np.arange(target.shape[0])
  y = np.arange(target.shape[1])
  x, y = np.meshgrid(x,y)
  x = x.flatten()
  y = y.flatten()
  targets = np.dstack((x, y, np.ones(len(x)))).squeeze()
  sources = targets.dot(H.T)
  sources = np.round(((sources[:,:2]) / sources[:,-1:])).astype(np.int)
  for a, b, (i, j) in zip(x, y, sources):
    if 0 <= i < source.shape[0] and 0 <= j < source.shape[1]:
      target[a,b] = source[i,j]

def annotate_image(imname, num_points=20):
  real_imname = imname.split(".")[0]
  if os.path.isfile("input/%s.pkl" % real_imname) and not reannotate: 
    print("Annotation for %s already exists" % real_imname)
    return
  image = plt.imread("input/%s" % imname)
  plt.imshow(image)
  points = plt.ginput(num_points, timeout=0)
  plt.close()
  file = open("input/%s.pkl" % real_imname, 'wb')
  pkl.dump(points, file)

def align_image(imname, points):

  def up(event):
    w.move(target, 0, -5)

  def down(event):
    w.move(target, 0, 5)

  def left(event):
    w.move(target, -5, 0)

  def right(event):
    w.move(target, 5, 0)

  def quit(event):
    nonlocal points
    points = w.coords(target)
    master.destroy()

  def pgup(event):
    points = w.coords(target)
    for i in range(len(points)):
      points[i] = np.round(points[i] * 1.1)
    w.coords(target, *points)

  def pgdown(event):
    points = w.coords(target)
    for i in range(len(points)):
      points[i] = np.round(points[i] * 0.9)
    w.coords(target, *points)

  real_imname = imname.split(".")[0]
  if os.path.isfile("input/%s.pkl" % real_imname) and not reannotate: 
    print("Annotation for %s already exists" % real_imname)
    return
  image = PIL.Image.open("input/%s" % imname)
  master = Tk()
  canvas_width, canvas_height = image.size
  w = Canvas(master, 
             width=canvas_width,
             height=canvas_height)
  w.pack()
  img = ImageTk.PhotoImage(image)
  w.create_image(0,0, anchor=NW, image=img)
  points = np.concatenate(np.round(np.flip(points, axis=1)).astype(np.int)).tolist()
  target = w.create_polygon(points, outline='blue', 
                            fill='yellow', width=3)

  master.bind_all('<Up>', up)
  master.bind_all('<Down>', down)
  master.bind_all('<Left>', left)
  master.bind_all('<Right>', right)
  master.bind_all('<q>', quit)
  master.bind_all('<Prior>', pgup)
  master.bind_all('<Next>', pgdown)
  master.mainloop()
  points = np.vstack((points[0::2], points[1::2])).T
  file = open("input/%s.pkl" % real_imname, 'wb')
  pkl.dump(points, file)

def load_image_annotation(imname):
  image = plt.imread("input/%s" % imname)
  real_imname = imname.split(".")[0]
  file = open("input/%s.pkl" % real_imname, 'rb')
  points = pkl.load(file)
  return image, np.flip(np.asarray(points), axis=1), real_imname

if __name__ == '__main__':
  f1 = 'mountain.jpg' # 
  f2 = 'paper.jpg' # image to project on
  reannotate = False
  width = 110 * 5
  height = 85 * 5
  im3 = np.zeros((height, width, 3), dtype=np.uint8)
  points3 = ((0, 0), (0, width-1), (height-1, width-1), (height-1, 0)) # Actual shape of target object, default rectangle
  annotate_image(f2, num_points=4)
  im2, points2, real_imname2 = load_image_annotation(f2) # TL, TR, BR, BL
  align_image(f1, points2)
  im1, points1, real_imname1 = load_image_annotation(f1) 
  H1 = compute_H(points3, points1)
  H2 = compute_H(points2, points3)
  hom_result = np.copy(im2)
  warp(im3, im1, H1)
  plt.imshow(im3) # display the rectified image (if you were to print or paint it)
  plt.show() 
  warp(hom_result, im3, H2)
  plt.imshow(hom_result)
  plt.show()
  filename = "output/%s_%s.jpg" % (real_imname1, real_imname2)
  plt.imsave(filename, im3, format='jpg')
  filename = "output/%s_%s_preview.jpg" % (real_imname1, real_imname2)
  plt.imsave(filename, hom_result, format='jpg')
