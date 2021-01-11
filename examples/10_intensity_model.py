#use  all the triangles containing a certain vertex to get the normal vector at that point
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

sys.path.append('..')
import face3d
from face3d import mesh

import cv2 as cv
import dlib
import numpy as np
import scipy.io as sio
import imutils
# width of beckmann microfacet distribution
beck_width = 0.35

# Refractive index of the material the light is coming from
eta_i = 1

# Refractive index of the material the light is hitting/entering
eta_t = 1.45

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

##########-----------------skin thickness per vertex functions---------------------##########

# Helper func
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

# Helper func
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

# Maps each landmark from 1 to 60, with a corresponding epidermis thickness
def contruct_mapping():
    # Doesn't include landmarks 61 - 68 (inside of lips)
    epi_depth = {1:1.64, 2:1.37, 3:1.37, 4:1.7, 5:1.7, 6:1.5, 7:1.3, 8:1.3, 9:1.54, 10:1.3, 11:1.3, 12:1.5, 13:1.7, 14:1.7, 15:1.37, 16:1.37, 17:1.64, 18:1.64, 19:1.54, 20:1.54, 21:1.54, 22:1.77, 23:1.77, 24:1.54, 25:1.54, 26:1.54, 27:1.64, 28:1.94, 29:1.94, 30:1.58, 31:1.70, 32:1.89, 33:1.58, 34:1.58, 35:1.58, 36:1.89, 37:1.62, 38:1.43, 39:1.0, 40:1.11, 41:1.55, 42:1.14, 43:1.11, 44:1.0, 45:1.43, 46:1.62, 47:1.14, 48:1.14, 49:1.69, 50:1.89, 51:1.58, 52:1.58, 53:1.58, 54:1.89, 55:1.69, 56:1.3, 57:1.4, 58:1.54, 59:1.4, 60:1.3}
    epi_scale = 29.57
    for key in epi_depth:
        epi_depth[key] *= epi_scale/10000
    derm_depth = {1:1.47, 2:1.55, 3:1.53, 4:1.51, 5:1.51, 6:1.45, 7:1.38, 8:1.46, 9:1.53, 10:1.46, 11:1.38, 12:1.45, 13:1.51, 14:1.51, 15:1.53, 16:1.55, 17:1.47, 18:1.43, 19:1.39, 20:1.35, 21:1.46, 22:1.58, 23:1.58, 24:1.46, 25:1.35, 26:1.39, 27:1.43, 28:1.67, 29:1.77, 30:2.08, 31:1.68, 32:1.74, 33:2.12, 34:1.63, 35:2.12, 36:1.74, 37:1.3, 38:1.43, 39:1.36, 40:1.45, 41:1.59, 42:1.62, 43:1.45, 44:1.36, 45:1.43, 46:1.3, 47:1.45, 48:1.62, 49:1.78, 50:2.12, 51:1.88, 52:1.63, 53:1.88, 54:2.12, 55:1.78, 56:1.38, 57:1.46, 58:1.53, 59:1.46, 60:1.38}
    derm_scale = 758.85
    for key in derm_depth:
        derm_depth[key] *= derm_scale/10000
    return epi_depth, derm_depth

# Given two points already on the map, this function returns the coordinates
# and the thickness at the point colinear with the other two points and frac of the way up from the first point
def find_mid(shape, i1, i2, frac):
    x1, y1 = shape[i1-1]
    x2, y2 = shape[i2-1]
    xo = x1 + (x2-x1)*frac
    yo = y1 + (y2-y1)*frac
    return [int(xo), int(yo)]

# Takes in the 3DMM vertices (properly aligned to the input image), as well as the path to the face landmark pre-traines detector, and...
# Outputs: a thickness value for every single vertex in the vertices array
def get_thicknesses(vertices, path_model):
    epi_depth, derm_depth = contruct_mapping()

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_model)
    
    # load the input image, resize it, and convert it to grayscale
    image = cv.imread(image_filename)
    #image = imutils.resize(image, width=500)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)
    shape = shape[:60]
    
    # Add new points to dict and face landmarks: epidermis
    for i1, i2, frac in [[30, 2, 1/2], [4, 31, 1/4], [49, 6, 1/2], [58, 9, 1/2], [55, 12, 1/2], [14, 31, 1/4], [30, 16, 1/2], [31, 4, 1/4], [30, 3, 1/5], [29, 2, 1/6], [31, 14, 1/4], [30, 15, 1/5], [29, 16, 1/6]]:
        x1, y1 = find_mid(shape, i1, i2, frac)
        shape = np.append(shape, [[x1, y1]], axis=0)
        for e_depth in [1.37, 1.7, 1.3, 1.54, 1.3, 1.7, 1.37, 2.56, 2.59, 2.3, 2.56, 2.59, 2.3]:
            epi_depth[len(shape)] = e_depth*29.57/10000
        for d_depth in [1.55, 1.51, 1.38, 1.53, 1.38, 1.51, 1.55, 1.74, 1.58, 1.64, 1.74, 1.58, 1.64]:
            derm_depth[len(shape)] = d_depth*758.85/10000

    print(f"len shape:{len(shape)}, len dict:{len(derm_depth)}")

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = rect_to_bb(rect)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the face number
    cv.putText(image, "Face #1", (x - 10, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for j, (x, y) in enumerate(shape):
        cv.circle(image, (x, y), 2, (0, 0, 255), -1)
        
    # show the output image with the face detections + facial landmarks
    cv.imshow("Output", image)
    cv.waitKey(0)

    epi_skin_depth = np.zeros(len(vertices))
    derm_skin_depth = np.zeros(len(vertices))
    
    for i, vert in enumerate(vertices):
        vert = vert.astype(int)
        cv.circle(image, (vert[0], vert[1]), 5, (0, 255), -1)
        closest = [-1, float('inf')]
        sec_closest = [-1, float('inf')]
        for j, (x, y) in enumerate(shape):
            euc_dist = math.dist([x,y], vert[:2])
            min_closest = min(closest[1], euc_dist)

            if min_closest != closest[1]:
                sec_closest = closest.copy()
                closest = [j+1, min_closest]
            else:
                min_closest = min(sec_closest[1], euc_dist)
                sec_closest = [j+1, min_closest]
        tot_clos = closest[1]+sec_closest[1]

        # NOTE: Arbitrary weightage of closest and second closest thickness, based on distance of other.
        # Eg. if closest is 1 away and second closest is 4 away, the weight of the closer thickness is 4/5 (and the other weight is 1/5).
        epi_weight_avg = (epi_depth[closest[0]]*sec_closest[1]/tot_clos) + (epi_depth[sec_closest[0]]*closest[1]/tot_clos)
        derm_weight_avg = (derm_depth[closest[0]]*sec_closest[1]/tot_clos) + (derm_depth[sec_closest[0]]*closest[1]/tot_clos)

        epi_skin_depth[i] = epi_weight_avg #, closest[0], sec_closest[0], vert])
        derm_skin_depth[i] = derm_weight_avg #, closest[0], sec_closest[0], vert])

    sum_d = 0
    for key in derm_depth:
        sum_d += derm_depth[key]
    sum_d /= len(derm_depth)
    print(f"average dermdepth:{sum_d}")

    return np.array(epi_skin_depth), np.array(derm_skin_depth)

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
  mudermis = fblood * mublood + (1-fblood) * muskin
  # print(mudermis)
  return mudermis

def computeK(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return np.sqrt(k*(k + 2*s))

def computeBeta(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return np.sqrt(k/( k + 2*s ))

def computeDk_Dfblood(lmbda):
  mublood = computeAbsorptionCoefficientBlood(lmbda)
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return mublood - muskin

def computeDbeta2_Dk(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return (2 * s) / pow(k + 2 * s, 2)

def computeDK_Dk(lmbda,fblood):
  k = computeAbsorptionCoefficientDermis(lmbda,fblood)
  s = computebackScatteringCoefficient(lmbda)
  return ( k + s) / np.sqrt(k*(k + 2 *s ))

def computeDR_DK(lmbda,fblood,d):
  b = computeBeta(lmbda,fblood)
  K = computeK(lmbda,fblood)
  nr = (1 - b * b) * 8 * b* d
  dr = pow((pow((1 + b),2) * np.exp(K*d) - pow((1 - b),2) * np.exp(-1*K*d)),2 )
  return nr/dr

def computeDR_Dbeta2(lmbda,fblood,d):
  b = computeBeta(lmbda,fblood)
  K = computeK(lmbda,fblood)
  dr = (pow((1 + b),2) * np.exp(K*d)) - (pow((1 - b),2) * np.exp(-1*K*d))
  trm2 = 1/dr
  nr = (pow(b,2) -1) * ((np.exp(K*d)*(1 + 1/b)) - ((1 - 1/b)*np.exp(-1*K*d)))
  trm1 = nr / pow(dr,2)
  return  (trm1 - trm2) * (np.exp(K*d) - np.exp(-1*K*d))

def computeRdermis(lmbda,fblood,d):
  DR_Dbeta2 = computeDR_Dbeta2(lmbda,fblood,d)
  Dbeta2_Dk = computeDbeta2_Dk(lmbda,fblood)
  DR_DK = computeDR_DK(lmbda,fblood,d)
  DK_Dk = computeDK_Dk(lmbda,fblood)
  Dk_Dfblood = computeDk_Dfblood(lmbda)
  #https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w16/Alotaibi_A_Biophysical_3D_ICCV_2017_paper.pdf table1 of this paper deltaFblood mentioned
  deltaFblood = 0.05
  # print("DR_Dbeta2 =",DR_Dbeta2, " Dbeta2_Dk =",Dbeta2_Dk," DR_DK = ",DR_DK, " DK_Dk = ",DK_Dk," Dk_Dfblood = ",Dk_Dfblood)
  Rdermis = ( DR_DK * DK_Dk + DR_Dbeta2 * Dbeta2_Dk ) * Dk_Dfblood * deltaFblood
  return Rdermis

def computeAbsorptionCoefficientEpidermis(lmbda,fmel):
  mumel = 6.6 * pow(10,11) * pow(lmbda,-3.33)
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return fmel * mumel + (1-fmel) * muskin

def computeTepidermis(lmbda,fmel):
  muepidermis = computeAbsorptionCoefficientEpidermis(lmbda,fmel)
  return np.exp(-1 * 0.004466 * muepidermis)

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
  return np.abs(Tepidermis*Tepidermis*dRdermis)

#consolidated function to return strength
#consolidated function to return strength
def compute_strength(lmbda,fblood,fmel,vertices,triangles,s,d_epi):
    d_epi = 0.129126
    eps = 1e-6
    #- s is light source coordinates
    #- d_epi is the avg. skin depth
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
    print(f"len angles:{len(angles)}")
    print(angles)
    print(f"angles[0]:{angles[0]}")
    print(f"angles[:,0]:{angles[:,0]}")
    angles = angles[:, 0]
    cos_angles = np.cos(angles)+eps
    print(f"Finished making cos_angles, len:{len(cos_angles)}, print f10:{cos_angles[:10]}")
    d_real = d_epi/cos_angles
    print(f"Finished making d_real, len:{len(d_real)}, print f10:{d_real[:10]}")
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

    signal = np.abs(Tepidermis*Tepidermis*dRdermis)
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
orig_img = cv.imread(image_filename)
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

# NOTE: If left, as is, the vertices are inverted!
d_s = -1*pp[0, 6]
s = pp[0, 6]

# NOTE: The angles in the .mat were in radians instead of degrees
d_angles2 = [np.rad2deg(pp[0, 0]), np.rad2deg(pp[0, 1]), np.rad2deg(pp[0, 2])]
angles2 = pp[0, 0:3]
t = pp[0, 3:6]
d_t = t.copy()
# NOTE: This is necessary because the translations are relative to the bottom left corner of the image, instead of the opencv standard of top left.
d_t[1] = len(orig_img[0]) - t[1]

# set prop of rendering
h = w = 450
c = 3

#s = 8e-04
#angles2 = [-10, -100, 0]
# t = [0, 0, 0]
blank_image = np.zeros((500, 500, 3), np.uint8)
print(f"angles2:{angles2}")
depth_var_vertices = bfm.transform(vertices, d_s, d_angles2, d_t)
transformed_vertices = bfm.transform(vertices, s, angles2, t)
projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection


#orig_img = imutils.resize(orig_img, width=500)
print(f"projected_vertices len:{len(projected_vertices)}, max:{np.max(projected_vertices)}, min:{np.min(projected_vertices)}, 0th:{projected_vertices[0]}, all:{projected_vertices}")
for vert in depth_var_vertices:
    cv.circle(blank_image, (int(vert[0]), int(vert[1])), 1, (0, 255, 0), 1)
    cv.circle(orig_img, (int(vert[0]), int(vert[1])), 1, (0, 255, 0), 1)
    
cv.imshow("depth_var_vertices", blank_image)
cv.imshow("orig_img", orig_img)
cv.waitKey(0)
cv.destroyAllWindows()
    
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
orig_source = [400, 100, 50]
print(f"Num vertices:{len(vertices)}, triangles:{len(triangles)}")
epi_depth, derm_depth = get_thicknesses(depth_var_vertices, "../brdf_thickness/landmark_model/shape_predictor_68_face_landmarks.dat")
for cam_ang in [-25, 25]:
    for i in range(3):
        source = orig_source.copy()
        source[i] += cam_ang
        source = tuple(source)
        # DELETE THIS if you want to loop through different light source angles
        source = orig_source.copy()
        colors = colors/np.max(colors)

        strength_val, angles, normals = compute_strength(550, 0.045, 0.22, vertices, triangles, source, derm_depth)
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
        save_folder = f"results/intensity_distro/light_source/x{source[0]}y{source[1]}z{source[2]}"
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
                print(f"image printed:{image}")
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
        # subprocess.call('rm {}/*.jpg'.format(save_folder), shell=True)B
