#use all the triangles containing a certain vertex to get the normal vector at that point
#vertices: list of coordinates for all vertices
#triangles: set of indices that form a facial triangle
#colors: same dimensions as the vertices, containing per-vertex color
import os, sys
import math
from skimage import io
from time import time
import subprocess
import matplotlib.pyplot as plt
from MorphabelModel import MorphabelModel
from IntensityPreprocessing import sRGB_generation

sys.path.append('..')
import face3d
from face3d import mesh

import cv2
import dlib
import numpy as np
import scipy.io as sio

# width of beckmann microfacet distribution
beck_width = 0.000001

# Refractive index of the material the light is coming from
eta_i = 0.5

# Refractive index of the material the light is hitting/entering
eta_t = 0.5

# Microfacet distribution function
def distr(w_h):
    if w_h[2] == 0:
        return 0
    
    sin_theta_sqr = max(0, 1 - w_h[2]*w_h[2])
    cos_theta_sqr = w_h[2] * w_h[2]

    cos_theta_4 = pow(cos_theta_sqr, 2)
    tan_sqr = sin_theta_sqr/cos_theta_sqr

    width_sqr = pow(beck_width, 2)
    return np.exp(-tan_sqr/width_sqr)/(math.pi * width_sqr * cos_theta_4)

# Monodirectional shadowing helper function
def monodir_shadowing(v, w_h):
    cos_sqr = w_h[2] * w_h[2]
    sin_sqr = max(0, 1 - cos_sqr)
    a = 1 / (beck_width * abs(np.sqrt(sin_sqr / cos_sqr)))
    if a < 1.6:
        a_sqr = a * a
        return (3.535 * a + 2.181 * a_sqr)/(1 + 2.276 * a-2.577 * a_sqr)
    else:
        return 1

# Geomtry function (using monodirectional shadowing function)
def geom(w_i, w_o, w_h):
    ret = 1
    for w in [w_i, w_o]:
        ret *= monodir_shadowing(w, w_h)
    return ret
        
# Dielectric fresnel coefficients helper function
def dielectric(ci, ct, ei, et):
    r_par = (et * ci - ei * ct)/(et * ci + ei * ct)
    r_perp = (ei * ci - et * ct)/(ei * ci + et * ct)
    return [0.5 * (r_par * r_par + r_perp * r_perp) for i in range(4)]

    
# Fresnel reflectance function for dielectrics (skin?)
def fres(cos_i):
    cos_i = abs(cos_i)
    g = np.sqrt((eta_t**2)/(eta_i**2) - 1 + cos_i**2)
    first_term = 0.5 * ((g - c)**2)/((g + c)**2)
    second_term = 1 + (((c * (g + c) - 1)**2)/((c * (g - c) + 1)**2))

    return first_term * second_term

# Implementation of torrance-sparrow microfacet BRDF
# Inputs:
# p_s ... This value represents the oiliness of the skin- [0,1]
# w_i, w_o ... These are 2 3d vectors that represent the incoming and outgoing lighting directions

# Outputs:
# Returns the torrance-sparrow brdf value

def brdf(p_s, w_i, w_o):
    cos_to = abs(w_o[2])
    cos_ti = abs(w_i[2])

    # TODO: implement edge cases for inputs
    
    w_h = w_i + w_o

    # Normalize w_h
    norm_wh = w_h / np.linalg.norm(w_h)
    f = fres(np.dot(w_i, w_h))
    g = geom(w_i, w_o, w_h)
    d = distr(w_h)
    return (f*p_s*g*d)/(4*cos_to*cos_ti)

# Prevent memory errors from large dimensions
def adder(a, b):
	c = []
	if len(a) != len(b):
		print("Dimensions invalid.")
	else:
		for i in range(len(a)):
			c.append(a[i] + b[i])
	return c

def subter(a, b):
	c = []
	if len(a) != len(b):
		print("Dimensions invalid.")
	else:
		for i in range(len(a)):
			c.append(a[i] - b[i])
	return c

def multer(a, b):
	c = []
	if len(a) != len(b):
		print("Dimensions invalid.")
	else:
		for i in range(len(a)):
			c.append(a[i]*b[i])
	return c

def divider(a, b):
	c = []
	if len(a) != len(b):
		print("Dimensions invalid.")
	else:
		for i in range(len(a)):
			c.append(a[i]/b[i])
	return c

molarExtinctionCoeffOxy = {}
molarExtinctionCoeffDoxy = {}
with open('extinction_coeff.txt') as f:
    for line in f:
       (lmbda, muoxy,mudoxy) = line.split()
       molarExtinctionCoeffOxy[int(lmbda)] = float(muoxy)
       molarExtinctionCoeffDoxy[int(lmbda)] = float(mudoxy)

def transform_test(vertices, obj, camera, source, h = 256, w = 256):
	'''
	Args:
		obj: dict contains obj transform paras
		camera: dict contains camera paras
	'''
	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
	
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

	brdf_val = []
	for index, value in enumerate(vertices):
		brdf_val.append(brdf(0.1, vertices[index]-source, camera['eye']-vertices[index]))
	midpoint = 0.5/np.percentile(brdf_val, 50)
	for index, value in enumerate(brdf_val):
		brdf_val[index] = brdf_val[index]*midpoint
		if brdf_val[index] > 1:
			brdf_val[index] = 0.999999999
		strength_val[index] = strength_val[index] * brdf_val[index]
	print('\n' + "BRDF Values")
	print(np.percentile(brdf_val, 10))
	print(np.percentile(brdf_val, 30))
	print(np.percentile(brdf_val, 50))
	print(np.percentile(brdf_val, 70))
	print(np.percentile(brdf_val, 90))

	#CHANGE THE LINE BELOW THIS FROM NORMALS -> ANGLES OR STRENGTH_VAL AS APPROPRIATE
	rendering = mesh.render.render_colors(image_vertices, triangles, strength_val, h, w, c=1)
	#print(image_vertices.shape)
	#print(triangles.shape)
	#print(normals.shape)
	rendering = np.concatenate((np.zeros((h, w, 1)), rendering), 2)
	rendering = np.concatenate((np.zeros((h, w, 1)), rendering), 2)
	#rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering

#geometric functions for normal and angle calculation
def normals_compute(vertices,triangles):
    eps = 1e-6
    no_v = vertices.shape[0] #-----number of vertices
    #no_v = triangles.shape[0]
    normals = np.zeros([no_v,3])

    for count, i in enumerate(triangles):
         v1 = vertices[i[1]] - vertices[i[0]]
         v2 = vertices[i[2]] - vertices[i[0]]
         product = np.cross(v1, v2)
         normals[i[0]] = normals[i[0]] + product
         normals[i[1]] = normals[i[1]] + product
         normals[i[2]] = normals[i[2]] + product
    normals = normals/(np.linalg.norm(normals, ord=2, axis=-1, keepdims=True)+eps)
    return normals

    #for i in np.arange(no_v):
    #      for j in np.arange(3):
    #         sub_array = triangles[:,j]
    #         #print(sub_array.shape)
    #         ixs = np.where(sub_array==i)[0]
    #         #print(ixs)
    #         in1 = (j+1)%3
    #         in2 = (j+2)%3

    #         v1 = vertices[triangles[ixs,in1],:]-vertices[triangles[ixs,j],:]
    #         v2 = vertices[triangles[ixs,in2],:]-vertices[triangles[ixs,j],:]
    #         #print(v1.shape,v2.shape)
    #         n_temp = np.cross(v2,v1)
    #         #n_temp = n_temp/(np.linalg.norm(n_temp, ord=2, axis=-1, keepdims=True)+eps)
    #         #print(n_temp.shape)
    #         n = np.sum(n_temp,axis=0)
    #         #print(n.shape)
    #         n = n/(np.linalg.norm(n,ord=2)+eps)
    #         normals[i,:] = normals[i,:]+n
    #normals = normals/(np.linalg.norm(normals, ord=2, axis=-1, keepdims=True)+eps)
    #return normals

def angles_compute(v1,v2):
    #v1 is an array of unit vectors
    #v2 is an array of corresponding unit vectors
    dot_product = np.sum(np.multiply(v1, v2),axis=1,keepdims=True)
    #for index, value in enumerate(dot_product):
    #	if dot_product[index] < 0:
    #		dot_product[index] = dot_product[index] * -1
    angles = np.arccos(dot_product)
    return angles





##########-----------------computation functions---------------------##########
def computebackScatteringCoefficient(lmbda):
  mieScattering = 2 * pow(10,-5) * pow(lmbda,-1.5)
  rayleighScattering = 2 * pow(10,12) * pow(lmbda,-4)
  return mieScattering + rayleighScattering

# mu_skin computation
def computeAbsorptionCoefficientSkin(lmbda):
  a = (lmbda - 154) / 66.2
  return 0.244 + 85.3 * np.exp(-1 * a)

# mu_blood computation
def computeAbsorptionCoefficientBlood(lmbda):
  muOxy = (2.303 * molarExtinctionCoeffOxy[lmbda] * 150)/64500
  muDoxy = (2.303 * molarExtinctionCoeffDoxy[lmbda] * 150)/64500
  return 0.75 * muOxy + 0.25 * muDoxy

# mu_dermis computation
def computeAbsorptionCoefficientDermis(lmbda,fblood):
  mublood = computeAbsorptionCoefficientBlood(lmbda)
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  # print(mublood,muskin,lmbda,fblood)
  mudermis = [x * mublood + (1-x) * muskin for x in fblood]
  # print(mudermis)
  return mudermis

def computeK(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return [np.sqrt(y*(y + 2*s)) for y in k]

def computeBeta(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return [np.sqrt(z/( z + 2*s )) for z in k]

def computeDk_Dfblood(lmbda):
  mublood = computeAbsorptionCoefficientBlood(lmbda)
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return mublood - muskin

def computeDbeta2_Dk(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return [(2 * s) / pow(a + 2 * s, 2) for a in k]

def computeDK_Dk(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return [( b + s) / np.sqrt(b*(b + 2 *s )) for b in k]

def computeDR_DK(lmbda,fblood,d):
  b = computeBeta(lmbda,fblood)
  K = computeK(lmbda,fblood)
  nr = multer([(1 - c * c) * 8 * c for c in b], d)
  dr = [pow(j,2) for j in subter(multer([pow((1 + g),2) for g in b], [np.exp(f) for f in multer(K, d)]), multer([pow((1 - h),2) for h in b], [np.exp(-1*e) for e in multer(K, d)]))]
  return divider(nr, dr)

def computeDR_Dbeta2(lmbda,fblood,d):
  b = computeBeta(lmbda,fblood)
  K = computeK(lmbda,fblood)
  expnt = multer(K, d)
  dr = subter(multer([pow((1 + g),2) for g in b], [np.exp(f) for f in expnt]), multer([pow((1 - h),2) for h in b], [np.exp(-1*e) for e in expnt]))
  trm2 = [1/m for m in dr]
  nr = subter(multer([(pow(n,2) -1)*(1 + 1/n) for n in b], [np.exp(o) for o in expnt]), multer([(1 - 1/p) for p in b], [np.exp(-1*q) for q in expnt]))
  trm1 = divider(nr, [pow(r,2) for r in dr])
  return multer((subter(trm1, trm2)), [subter(np.exp(s), np.exp(-1*s)) for s in expnt])

def computeRdermis(lmbda,fblood,d):
  DR_Dbeta2 = computeDR_Dbeta2(lmbda,fblood,d)
  Dbeta2_Dk = computeDbeta2_Dk(lmbda,fblood)
  DR_DK = computeDR_DK(lmbda,fblood,d)
  DK_Dk = computeDK_Dk(lmbda,fblood)
  Dk_Dfblood = computeDk_Dfblood(lmbda)
  #https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w16/Alotaibi_A_Biophysical_3D_ICCV_2017_paper.pdf table1 of this paper deltaFblood mentioned
  deltaFblood = 0.05
  # print("DR_Dbeta2 =",DR_Dbeta2, " Dbeta2_Dk =",Dbeta2_Dk," DR_DK = ",DR_DK, " DK_Dk = ",DK_Dk," Dk_Dfblood = ",Dk_Dfblood)
  Rdermis = [ar * Dk_Dfblood * deltaFblood for ar in adder(multer(DR_DK, DK_Dk), multer(DR_Dbeta2, Dbeta2_Dk))]
  return Rdermis

def computeAbsorptionCoefficientEpidermis(lmbda,fmel):
  mumel = 6.6 * pow(10,11) * pow(lmbda,-3.33)
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return [w * mumel + (1-w) * muskin for w in fmel]

def computeTepidermis(lmbda,fmel):
  muepidermis = computeAbsorptionCoefficientEpidermis(lmbda,fmel)
  return [np.exp(-1 * 0.004466 * v) for v in muepidermis]

def computeTotalReflectance(lmbda,fblood,fmel,d):
  Rdermis = computeRdermis(lmbda,fblood,d)
  # print("Tepidermis = ",Tepidermis)
  # print("Rdermis = ",Rdermis)
  dRdermis =  Rdermis
  return dRdermis

# def computeR(lmbda,fblood,fmel,d):
#   Tepidermis = computeTepidermis(lmbda,fmel)
#   b = computeBeta(lmbda,fblood)
#   K = computeK(lmbda,fblood)
#   R = pow((1 - b),2) * (np.exp(K*d) - np.exp(-1*K*d))
#
#   R = R/(pow((1 + b),2) * np.exp(K*d) - pow((1 - b),2) * np.exp(-1*K*d))
#
#   dRdermis = computeTotalReflectance(lmbda,fblood,fmel,d)
#
#   return pow(dRdermis,2)/pow(R,2)

def computeIntensity(lmbda,fblood,fmel,d):
  Tepidermis = computeTepidermis(lmbda,fmel)
  dRdermis = computeTotalReflectance(lmbda,fblood,fmel,d)
  return np.abs(multer([aq**2 for aq in Tepidermis], dRdermis))

#consolidated function to return strength
#consolidated function to return strength
def compute_strength(lmbda,fblood,fmel,vertices,triangles,s,d=0.129126):
    eps = 1e-6
    #- s is light source coordinates
    #- d is the avg. skin depth
    #lmbda- wavelength of incident radiation, between 400 and 730 nm

    Tepidermis = computeTepidermis(lmbda,fmel)
    b = computeBeta(lmbda,fblood)
    K = computeK(lmbda,fblood)

    direction = vertices-np.reshape(s,(1,-1))
    direction = direction/(np.linalg.norm(direction, ord=2, axis=-1, keepdims=True)+eps)

    normals = normals_compute(vertices,triangles)
    angles = angles_compute(normals,direction)
    print("Angles Values")
    print(np.min(angles))
    print(np.max(angles))
    print(angles)
    d_real = d/(np.cos(angles)+eps)
    for index, value in enumerate(d_real):
    	if d_real[index] < 0:
    		d_real[index] = d_real[index] * 0
    		#d_real[index] = d_real[index] * -1
    print('\n' + "d_real Values")
    print(np.percentile(d_real, 10))
    print(np.percentile(d_real, 30))
    print(np.percentile(d_real, 50))
    print(np.percentile(d_real, 70))
    print(np.percentile(d_real, 90))
    print(d_real)

    dRdermis = computeTotalReflectance(lmbda,fblood,fmel,d_real)

    signal = np.abs(multer([aq**2 for aq in Tepidermis], dRdermis))
    signal = np.nan_to_num(signal)
    #reshaping signal to nVert,1
    signal = np.reshape(signal,(-1,1))
    signal = signal/np.max(signal) #normalize to 0=1: if linear normalization doesnt enable good visualization, can also use dB scale
    #signal = np.tile(signal,(1,3)) #covert to RGB so that signal strength is shown as shades of grey
    print('\n' + "Strength Values")
    print(np.percentile(signal, 10))
    print(np.percentile(signal, 30))
    print(np.percentile(signal, 50))
    print(np.percentile(signal, 70))
    print(np.percentile(signal, 90))

    return signal, angles, normals

# --------- load mesh data

# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')

# --- 2. load fitted face
mat_filename = 'Sample6.mat'
mat_data = sio.loadmat(mat_filename)
image_filename = 'Sample6.jpg'
# with open(image_filename, 'rb') as file:
#    img = Image.open(file)
#    print('image size: {0}'.format(img.size))
sp = mat_data['Shape_Para']
ep = mat_data['Exp_Para']
tp = mat_data['Tex_Para']
vertices = bfm.generate_vertices(sp, ep)

# tp = bfm.get_tex_para('random')
colors = bfm.generate_colors(tp)
# colors = np.minimum(np.maximum(colors, 0), 1)

# --- 3. transform vertices to proper position
pp = mat_data['Pose_Para']
s = pp[0, 6]
# angles = [np.rad2deg(pp[0, 0]), np.rad2deg(pp[0, 1]), np.rad2deg(pp[0, 2])]
angles2 = pp[0, 0:3]
t = pp[0, 3:6]

# set prop of rendering
h = w = 450
c = 3

#s = 8e-04
#angles2 = [10, 30, 20]
#t = [0, 0, 0]
transformed_vertices = bfm.transform(vertices, s, angles2, t)
projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection
# projected_vertices[:,1] = h - projected_vertices[:,1] - 1
# --- 4. render(3d obj --> 2d image)
image_vertices = mesh.transform.to_image(projected_vertices, h, w)

vertices = image_vertices
triangles = bfm.triangles

#C = sio.loadmat('Data/example1.mat')
#vertices = C['vertices']; 
#global colors
#global triangles
#colors = C['colors']; triangles = C['triangles']
colors = colors/np.max(colors)

sRGBim = sRGB_generation()
fblood = []
fmel = []
timer = 1
for color in colors:
	k = 1
	diff = 0
	pos_of_min = [0, 0]
	min_diff = 10e30
	for r in range(len(sRGBim)):
		for c in range(len(sRGBim[0])):
			diff = np.sqrt((sRGBim[r][c][0] - color[0])**2 + (sRGBim[r][c][1] - color[1])**2 + (sRGBim[r][c][2] - color[2])**2)
			if diff < min_diff:
				pos_of_min = [r, c]
				min_diff = diff
	fblood.append(0.0007 + 0.0693*(pos_of_min[0]/len(sRGBim)))
	fmel.append(0.0043 + 0.4257*(pos_of_min[1]/len(sRGBim[0])))
	print(str(100*timer/len(colors)) + "% completed processing")
	timer = timer + 1

nm_wavelength = 550
source = (400, 100, 50)
strength_val, angles, normals = compute_strength(nm_wavelength, fblood, fmel, vertices, triangles, source)
print('\n' + "Bounds Check")
print(np.percentile(strength_val, 1))
print(np.percentile(strength_val, 100))
#print(np.mean(strength_val))
#print(np.std(strength_val))
#strength_val = strength_val - (np.mean(strength_val) - 2*np.std(strength_val))
#strength_val = ((1/4) / np.std(strength_val)) * strength_val
#strength_val = (strength_val - np.percentile(strength_val, 1))*1/(np.percentile(strength_val, 100) - np.percentile(strength_val, 1))
#for index, sing_val in enumerate(strength_val):
#	if strength_val[index] < 0:
#		strength_val[index] = 0
#	elif strength_val[index] > 1:
#		strength_val[index] = 0.9999999999
print("-----")
#for index, sing_val in enumerate(strength_val):
#	if strength_val[index] < np.percentile(strength_val, 20):
#		strength_val[index] = 0
#	elif strength_val[index] < np.percentile(strength_val, 40):
#		strength_val[index] = 0.25
#	elif strength_val[index] < np.percentile(strength_val, 60):
#		strength_val[index] = 0.5
#	elif strength_val[index] < np.percentile(strength_val, 80):
#		strength_val[index] = 0.75
#	else:
#		strength_val[index] = 1
#print(strength_val)
#strength_val = np.log(strength_val) + 1
#for index, sing_val in enumerate(strength_val):
#	if strength_val[index] < 0:
#		strength_val[index] = 0
vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

# save folder
save_folder = 'results/intensity_distro'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
options = '-delay 10 -loop 0 -layers optimize' # gif options. need ImageMagick installed.

# ---- start
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
#print(np.max(vertices[:,1]) - np.min(vertices[:,1]))

## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'
camera['eye'] = [0, 0, 250]
# scale
for factor in np.arange(0.5, 1.2, 0.1):
	obj['s'] = scale_init*factor
	obj['angles'] = [0, 0, 0]
	obj['t'] = [0, 0, 0]
	image = transform_test(vertices, obj, camera, source) 
	io.imsave('{}/1_1_{:.2f}.jpg'.format(save_folder, factor), image, quality=100)

# angles
for i in range(3):
	for angle in np.arange(-50, 51, 10):
		obj['s'] = scale_init
		obj['angles'] = [0, 0, 0]
		obj['angles'][i] = angle
		obj['t'] = [0, 0, 0]
		image = transform_test(vertices, obj, camera, source) 
		io.imsave('{}/1_2_{}_{}.jpg'.format(save_folder, i, angle), image, quality=100)

## 2. fix obj position(center=[0,0,0], front pose). change camera position&direction, using perspective projection(fovy fixed)
obj['s'] = scale_init
obj['angles'] = [0, 0, 0]
obj['t'] = [0, 0, 0]
# obj: center at [0,0,0]. size:200

camera['proj_type'] = 'perspective'
camera['at'] = [0, 0, 0]
camera['near'] = 1000
camera['far'] = -100
# eye position
camera['fovy'] = 30
camera['up'] = [0, 1, 0] #
# z-axis: eye from far to near, looking at the center of face
for p in np.arange(500, 250-1, -40): # 0.5m->0.25m
	camera['eye'] = [0, 0, p]  # stay in front of face
	image = transform_test(vertices, obj, camera, source) 
	io.imsave('{}/2_eye_1_{}.jpg'.format(save_folder, 1000-p), image, quality=100)

# y-axis: eye from down to up, looking at the center of face
for p in np.arange(-300, 301, 60): # up 0.3m -> down 0.3m
	camera['eye'] = [0, p, 250] # stay 0.25m far
	image = transform_test(vertices, obj, camera, source) 
	io.imsave('{}/2_eye_2_{}.jpg'.format(save_folder, p/6), image, quality=100)

# x-axis: eye from left to right, looking at the center of face
for p in np.arange(-300, 301, 60): # left 0.3m -> right 0.3m
	camera['eye'] = [p, 0, 250] # stay 0.25m far
	image = transform_test(vertices, obj, camera, source) 
	io.imsave('{}/2_eye_3_{}.jpg'.format(save_folder, -p/6), image, quality=100)

# up direction
camera['eye'] = [0, 0, 250] # stay in front
for p in np.arange(-50, 51, 10):
	world_up = np.array([0, 1, 0]) # default direction
	z = np.deg2rad(p)
	Rz=np.array([[math.cos(z), -math.sin(z), 0],
                 [math.sin(z),  math.cos(z), 0],
                 [     0,       0, 1]])
	up = Rz.dot(world_up[:, np.newaxis]) # rotate up direction
	# note that: rotating up direction is opposite to rotating obj
	# just imagine: rotating camera 20 degree clockwise, is equal to keeping camera fixed and rotating obj 20 degree anticlockwise.
	camera['up'] = np.squeeze(up)
	image = transform_test(vertices, obj, camera, source) 
	io.imsave('{}/2_eye_4_{}.jpg'.format(save_folder, -p), image, quality=100)

# -- delete jpg files
print('gifs have been generated, now delete jpgs')
# subprocess.call('rm {}/*.jpg'.format(save_folder), shell=True)
