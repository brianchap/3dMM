''' 
Generate 2d maps representing different attributes(colors, depth, pncc, etc)
: render attributes to image space.
'''
import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt
from MorphabelModel import MorphabelModel

sys.path.append('..')
import face3d
from face3d import mesh
from face3d import mesh_numpy

###############################################
### My additions:
import cv2
import dlib
name = 'Sample2'
im = cv2.imread(name+'.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
h, w, c = im.shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

rects = detector(gray, 1)
shape = predictor(gray, rects[0])
tl_corner_x = rects[0].center().x - rects[0].width()/2 # added
tl_corner_y = rects[0].center().y - rects[0].height()/2 # added
br_corner_x = rects[0].center().x + rects[0].width()/2 # added
br_corner_y = rects[0].center().y + rects[0].height()/2 # added
rects = [(tl_corner_x, tl_corner_y), (br_corner_x, br_corner_y)]
landmarks = np.zeros((68, 2))

for i, p in enumerate(shape.parts()):
    landmarks[i] = [p.x, p.y]
    im = cv2.circle(im, (p.x, p.y), radius=3, color=(0, 0, 255), thickness=5)

bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
x = mesh_numpy.transform.from_image(landmarks, h, w)
X_ind = bfm.kpt_ind


fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter=200, isShow=False)
colors = bfm.generate_colors(np.random.rand(bfm.n_tex_para, 1))
colors = np.minimum(np.maximum(colors, 0), 1)

fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
image_vertices = mesh_numpy.transform.to_image(transformed_vertices, h, w)

###############################################

vertices = image_vertices
triangles = bfm.triangles
colors = colors/np.max(colors)

# ------------------------------ load mesh data
#C = sio.loadmat('Data/example1.mat')
#vertices = C['vertices']; colors = C['colors']; triangles = C['triangles']
#colors = colors/np.max(colors)

# ------------------------------ modify vertices(transformation. change position of obj)
# scale. target size=200 for example
#s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
# rotate 30 degree for example
#R = mesh.transform.angle2matrix([0, 30, 0]) 
# no translation. center of obj:[0,0]
#t = [0, 0, 0]
#transformed_vertices = mesh.transform.similarity_transform(vertices, s, R, t)

# ------------------------------ render settings(to 2d image)
# set h, w of rendering
#h = w = 256
# change to image coords for rendering
#image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

## --- start
save_folder = 'results/image_map'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

## 0. color map
attribute = colors
color_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
io.imsave('{}/color.jpg'.format(save_folder), np.squeeze(color_image))

## 1. depth map
z = image_vertices[:,2:]
z = z - np.min(z)
z = z/np.max(z)
attribute = z
depth_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=1)
io.imsave('{}/depth.jpg'.format(save_folder), np.squeeze(depth_image))

## 2. pncc in 'Face Alignment Across Large Poses: A 3D Solution'. for dense correspondences 
pncc = face3d.morphable_model.load.load_pncc_code('Data/BFM/Out/pncc_code.mat')
attribute = pncc
pncc_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
io.imsave('{}/pncc.jpg'.format(save_folder), np.squeeze(pncc_image))

## 3. uv coordinates in 'DenseReg: Fully convolutional dense shape regression in-the-wild'. for dense correspondences
uv_coords = face3d.morphable_model.load.load_uv_coords('Data/BFM/Out/BFM_UV.mat') #
attribute = uv_coords # note that: original paper used quantized coords, here not
uv_coords_image = mesh.render.render_colors(image_vertices, triangles, attribute, h, w, c=2) # two channels: u, v
# add one channel for show
uv_coords_image = np.concatenate((np.zeros((h, w, 1)), uv_coords_image), 2)
io.imsave('{}/uv_coords.jpg'.format(save_folder), np.squeeze(uv_coords_image))

