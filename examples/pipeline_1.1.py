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

# Geometry function (using monodirectional shadowing function)
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
    g = np.sqrt((eta_t**2)/(eta_i**2) - 1 + cos_i**2)
    cos_i = abs(cos_i)
    first_term = 0.5 * ((g - cos_i)**2)/((g + cos_i)**2)
    second_term = 1 + (((cos_i * (g + cos_i) - 1)**2)/((cos_i * (g - cos_i) + 1)**2))
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
    f = fres(np.dot(w_i, norm_wh))
    g = geom(w_i, w_o, norm_wh)
    d = distr(norm_wh)
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

def source_channel_distr(T, channel):
    #T: temperature in Kelvin
    #channel: color channel (0/1/2) - (R/G/B)
    #l: wavelengths: in nm
    wvs = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720]
    l = wvs
    h = 6.626e-34
    k = 1.381e-23
    c = 3.0e8 
    l = [l2*1e-9 for l2 in l]
    spd = [(8*np.pi*h*c*pow(l2,-5))/(np.exp((h*c)/(k*l2*T))-1) for l2 in l]
    spd = [spd2/(np.sum(spd)) for spd2 in spd]
    spd = np.array(spd)
    spd_ch = spd*sensor_data[channel]
    wavelength_ch = 10*round(sum(multer((spd_ch/sum(spd_ch)), wvs))/10, 0)
    return wavelength_ch

##########-----------------skin thickness per vertex functions---------------------##########

# Helper func
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with Opencv2
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
    image = cv2.imread(image_filename)
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    #print(f"len shape:{len(shape)}, len dict:{len(derm_depth)}")

    # convert dlib's rectangle to a Opencv2-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the face number
    cv2.putText(image, "Face #1", (x - 10, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for j, (x, y) in enumerate(shape):
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)

    epi_skin_depth = np.zeros(len(vertices))
    derm_skin_depth = np.zeros(len(vertices))

    for i, vert in enumerate(vertices):
        vert = vert.astype(int)
        cv2.circle(image, (vert[0], vert[1]), 5, (0, 255), -1)
        closest = [-1, float('inf')]
        vec_closest = [-1, float('inf')]
        for j, (x, y) in enumerate(shape):
             euc_dist = np.sqrt((x - vert[:2][0])**2 + (y - vert[:2][1])**2)
             #euc_dist = math.dist([x,y], vert[:2])
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
    #print(f"average dermdepth:{sum_d}")

    sum_e = 0
    for key in epi_depth:
        sum_e += epi_depth[key]
    sum_e /= len(epi_depth)
    #print(f"average epidepth:{sum_e}")
    return np.array(epi_skin_depth), np.array(derm_skin_depth)

molarExtinctionCoeffOxy = {}
molarExtinctionCoeffDoxy = {}

with open('extinction_coeff.txt') as f:
    for line in f:
       (lmbda, muoxy,mudoxy) = line.split()
       molarExtinctionCoeffOxy[int(lmbda)] = float(muoxy)
       molarExtinctionCoeffDoxy[int(lmbda)] = float(mudoxy)

def transform_test(vertices, obj, camera, source, temperature, channel, h = 256, w = 256):
    '''
    Args:
        obj: dict contains obj transform paras
        camera: dict contains camera paras
    '''
    R = mesh.transform.angle2matrix(obj['angles'])
    transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
    vertices = transformed_vertices

    wavelength = source_channel_distr(temperature, channel)
    
    #wavelength = 10*round(sum(multer(spd, nm_wavelength))/10, 0)
    #epi_depth, derm_depth = get_thicknesses(depth_var_vertices, "shape_predictor_68_face_landmarks.dat")		
    strength_val, angles, normals = compute_strength(wavelength, fblood, fmel, vertices, triangles, source, derm_depth, epi_depth)
    brdf_val = []
    for index, value in enumerate(vertices):
        brdf_val.append(brdf(0.1, vertices[index]-source, camera['eye']-vertices[index]))
    midpoint = 0.5/np.percentile(brdf_val, 50)
    brdf_val = np.array(brdf_val).reshape(-1,1)
    brdf_val = brdf_val*midpoint
    brdf_val = np.minimum((brdf_val - 1),-0.0000001) + 1
    strength_val = strength_val*brdf_val

    print('\n' + "BRDF Values")
    print(np.percentile(brdf_val, 10))
    print(np.percentile(brdf_val, 30))
    print(np.percentile(brdf_val, 50))
    print(np.percentile(brdf_val, 70))
    print(np.percentile(brdf_val, 90))
    
    
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
    
    #stval = np.concatenate((strength_val, np.zeros_like(strength_val),np.zeros_like(strength_val)),1)
    #stval = np.concatenate((np.zeros_like(strength_val),stval),1)
    #CHANGE THE LINE BELOW THIS FROM NORMALS -> ANGLES OR STRENGTH_VAL AS APPROPRIATE
    #rendering = mesh.render.render_colors(image_vertices, triangles, stval, h, w, c=3)
    #rendering = np.concatenate((np.zeros((h, w, 1)), rendering), 2)
    #rendering = np.concatenate((np.zeros((h, w, 1)), rendering), 2)
    #rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return strength_val, image_vertices

#geometric functions for normal and angle calculation
def normals_compute(vertices,triangles):
    eps = 1e-6
    no_v = vertices.shape[0] #-----number of vertices
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

def angles_compute(v1,v2):
    #v1 is an array of unit vectors
    #v2 is an array of corresponding unit vectors
    dot_product = np.sum(np.multiply(v1, v2),axis=1,keepdims=True)
    angles = np.arccos(dot_product)
    return angles

##########-----------------computation functions---------------------##########
def computebackScatteringCoefficient(lmbda):
  mieScattering = 2 * pow(10,5) * pow(lmbda,-1.5)
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
  mudermis = [x * mublood + (1-x) * muskin for x in fblood]
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

def computeAbsorptionCoefficientEpidermis(lmbda,fmel,d_epi):
  mumel = [6.6 * d_epis * pow(10,11) * pow(lmbda,-3.33) for d_epis in d_epi]
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return adder(multer(fmel, mumel), [(1-w) * muskin for w in fmel])

def computeTepidermis(lmbda,fmel,d_epi):
  muepidermis = computeAbsorptionCoefficientEpidermis(lmbda,fmel,d_epi)
  return [np.exp(-1 * v) for v in muepidermis]

def computeTotalReflectance(lmbda,fblood,fmel,d):
  Rdermis = computeRdermis(lmbda,fblood,d)
  # print("Tepidermis = ",Tepidermis)
  # print("Rdermis = ",Rdermis)
  dRdermis =  Rdermis
  return dRdermis

def computeIntensity(lmbda,fblood,fmel,d,d_epi):
  Tepidermis = computeTepidermis(lmbda,fmel,d_epi)
  dRdermis = computeTotalReflectance(lmbda,fblood,fmel,d)
  return np.abs(multer([aq**2 for aq in Tepidermis], dRdermis))

#consolidated function to return strength
#consolidated function to return strength
def compute_strength(lmbda,fblood,fmel,vertices,triangles,s,d_derm,d_epi):
    eps = 1e-6
    #- s is light source coordinates
    #- d_epi + d_derm is the avg. skin depth
    #lmbda- wavelength of incident radiation, between 400 and 730 nm
    Tepidermis = computeTepidermis(lmbda,fmel, d_epi)
    b = computeBeta(lmbda,fblood)
    K = computeK(lmbda,fblood)
    
    direction = vertices-np.reshape(s,(1,-1))
    direction = direction/(np.linalg.norm(direction, ord=2, axis=-1, keepdims=True)+eps)
    
    normals = normals_compute(vertices,triangles)
    angles = angles_compute(normals,direction)
    print("Angles Values")
    print(np.min(angles))
    print(np.max(angles))
    
    d_real = divider(d_derm, np.cos(angles)+eps)
    d_real = np.maximum(d_real,0)
    # for index, value in enumerate(d_real):
    # 	if d_real[index] < 0:
    # 		d_real[index] = d_real[index] * 0
    # 		#d_real[index] = d_real[index] * -1
    print('\n' + "d_real Values")
    print(np.percentile(d_real, 10))
    print(np.percentile(d_real, 30))
    print(np.percentile(d_real, 50))
    print(np.percentile(d_real, 70))
    print(np.percentile(d_real, 90))
 
    dRdermis = computeTotalReflectance(lmbda,fblood,fmel,d_real)
    signal = np.abs(multer([aq**2 for aq in Tepidermis], dRdermis))
    signal = np.nan_to_num(signal)
    #reshaping signal to nVert,1
    signal = np.reshape(signal,(-1,1))
    #signal = signal/np.max(signal) #normalize to 0=1: if linear normalization doesnt enable good visualization, can also use dB scale
    print('\n' + "Strength Values")
    print(np.percentile(signal, 10))
    print(np.percentile(signal, 30))
    print(np.percentile(signal, 50))
    print(np.percentile(signal, 70))
    print(np.percentile(signal, 90))
    return signal, angles, normals

def calc_mel_haem(colors):
    fblood = []
    fmel = []
    timer = 1
    for color in colors:
        k = 1
        diff = 0
        pos_of_min = [0, 0]
        min_diff = 10e30
        i1 = np.square(sRGBim[:,:,0] - color[0])
        i2 = np.square(sRGBim[:,:,1] - color[1])
        i3 = np.square(sRGBim[:,:,2] - color[2])
        diff = np.sqrt(i1+i2+i3)
        mini = np.min(diff)
        k = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
        if mini < min_diff:
            pos_of_min = [k[0],k[1]]
        fblood.append(0.0007 + 0.0693*(pos_of_min[0]/len(sRGBim)))
        fmel.append(0.0043 + 0.4257*(pos_of_min[1]/len(sRGBim[0])))
        if timer % (len(colors)//3) == 0:
            print(str(100*timer/len(colors)) + "% completed processing")
        timer = timer + 1
    
    return fblood, fmel


# --------- load mesh data
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')
# --- 2. load fitted face
mat_filename = 'Sample6.mat'
mat_data = sio.loadmat(mat_filename)
image_filename = 'Sample6.jpg'
orig_img = cv2.imread(image_filename)
sp = mat_data['Shape_Para']
ep = mat_data['Exp_Para']
tp = mat_data['Tex_Para']
vertices = bfm.generate_vertices(sp, ep)
colors = bfm.generate_colors(tp)
colors = colors/np.max(colors)
triangles = bfm.triangles
pp = mat_data['Pose_Para']
s = pp[0, 6]
d_s = -1*pp[0, 6]
angles2 = pp[0, 0:3]
d_angles2 = [np.rad2deg(pp[0, 0]), np.rad2deg(pp[0, 1]), np.rad2deg(pp[0, 2])]
t = pp[0, 3:6]
d_t = t.copy()
d_t[1] = len(orig_img[0]) - t[1]

global epi_depth, derm_depth
depth_var_vertices = bfm.transform(vertices, d_s, d_angles2, d_t)
epi_depth, derm_depth = get_thicknesses(depth_var_vertices, "shape_predictor_68_face_landmarks.dat")

transformed_vertices = bfm.transform(vertices, s, angles2, t)
transformed_vertices = transformed_vertices - np.mean(transformed_vertices, 0)[np.newaxis, :]
vertices = transformed_vertices
c = 3

cam_filename = 'cam_sens.mat'
cam_data = sio.loadmat(cam_filename)
global sensor_data
sensor_data = cam_data['S']

# set prop of rendering
#h = w = 450
#c = 3

#projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection
# projected_vertices[:,1] = h - projected_vertices[:,1] - 1
# --- 4. render(3d obj --> 2d image)
#image_vertices = mesh.transform.to_image(projected_vertices, h, w)
#vertices = image_vertices
#vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]

# print(f"projected_vertices len:{len(projected_vertices)}, max:{np.max(projected_vertices)}, min:{np.min(projected_vertices)}, 0th:{projected_vertices[0]}, all:{projected_vertices}")	
# for vert in depth_var_vertices:	
# 	cv2.circle(blank_image, (int(vert[0]), int(vert[1])), 1, (0, 255, 0), 1)	
# 	cv2.circle(orig_img, (int(vert[0]), int(vert[1])), 1, (0, 255, 0), 1)	
# 	#cv2.imshow("depth_var_vertices", blank_image)	
# 	#cv2.imshow("orig_img", orig_img)	
# 	#cv2.waitKey(0)	
# 	#cv2.destroyAllWindows()

# orig_source = [400, 100, 50]	
# print(f"Num vertices:{len(vertices)}, triangles:{len(triangles)}")	
# epi_depth, derm_depth = get_thicknesses(depth_var_vertices, "shape_predictor_68_face_landmarks.dat")	
# for cam_ang in [-25, 25]:	
# 	for i in range(3):	
# 		source = orig_source.copy()	
# 		source[i] += cam_ang	
# 		source = tuple(source)	
# 		# DELETE THIS if you want to loop through different light source angles	
# 		source = orig_source.copy()

#C = sio.loadmat('Data/example1.mat')
#vertices = C['vertices']; 
#global colors
#global triangles
#colors = C['colors']; triangles = C['triangles']

global sRGBim, fblood, fmel
sRGBim = sRGB_generation() 
fblood, fmel = calc_mel_haem(colors)

# ---- start
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'
camera['eye'] = [0, 0, 250]
obj['s'] = scale_init
obj['angles'] = [0, 0, 0]
obj['t'] = [0, 0, 0] 
source = (0,0,200)
temperature = 6000 # temperature is also a variable parameter (range to be determined)
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt

#------------------------------------------------------------------------
#Testing for camera spatial positions
cam_pos_str = []
cam_verts = []
render_names = []

source = (0,0,200)
## 2. fix obj position(center=[0,0,0], front pose). change camera position&direction, using perspective projection(fovy fixed)
obj['s'] = scale_init
obj['angles'] = [0, 0, 0]
obj['t'] = [0, 0, 0]
# obj: center at [0,0,0]. size:200
#camera['proj_type'] = 'perspective'
camera['proj_type'] = 'orthographic'
camera['at'] = [0, 0, 0]
camera['near'] = 1000
camera['far'] = -100
# eye position
camera['fovy'] = 30
camera['up'] = [0, 1, 0] #
# z-axis: eye from far to near, looking at the center of face
i = 0
for p in np.arange(500, 250-1, -20): # 0.5m->0.25m
    camera['eye'] = [0, 0, p]  # stay in front of face
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    cam_pos_str.append(stv)
    cam_verts.append(vs) 
    i = i+1
    lab = 'cam_pos_x_{:>2d}'.format(p)
    render_names.append(lab)
    print(f"i:{i}")
    
print("Finished computing strengths")
# Normalizing all strengths
h = w = 256
cam_strs = np.array(cam_pos_str)
cam_strs = np.nan_to_num(cam_strs)
cam_strs = cam_strs/np.max(cam_strs)
cam_sum = np.sum(cam_strs, axis = 1)

sf = 'results/final_results/camera_positions'
fig_sf = 'results/final_results/figs'
if not os.path.exists(fig_sf):
    os.mkdir(fig_sf)

if not os.path.exists(sf):
    os.mkdir(sf)
# Rendering the strength results
for k in range(len(cam_strs)):
    img = mesh.render.render_colors(cam_verts[k], triangles, cam_strs[k], h, w, c=1)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img *= 255 # or any coefficient
    img = img.astype(np.uint8)
    io.imsave('{}/ortho_{:>2d}_'.format(sf, k) + render_names[k] + '.jpg', img, quality=100)
print("Finished rendering")
#Generating and plotting strength sum values for each of the axes
plot_z_x = np.arange(500,250-1,-20)
plot_z_y = cam_sum
plt.figure()
plt.plot(plot_z_x, plot_z_y) #(z = 50)
plt.grid(True)
plt.xlabel("Camera position z-axis values")
plt.ylabel("Normalised strength sums")
plt.title("RPPG strengths when varying the Z-axis camera position")
plt.savefig(f'{fig_sf}/ortho_cam_z.png')
sys.exit("Finished generating perspective cam z")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Testing for camera eye positions (Orthographic projection only)
cam_pos_str = []
cam_verts = []
render_names = []
for i,p in enumerate(range(-400, 401, 80)):
    camera['eye'] = [p,0,250]
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    cam_pos_str.append(stv)
    cam_verts.append(vs)
    lab = 'cam_pos_x_{:>2d}'.format(p)
    render_names.append(lab)
for i,p in enumerate(range(-400, 401, 80)):
    camera['eye'] = [0,p,250]
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    cam_pos_str.append(stv)
    cam_verts.append(vs)
    lab = 'cam_pos_y_{:>2d}'.format(p)
    render_names.append(lab)
for i,p in enumerate(range(-100, 701, 80)):
    camera['eye'] = [0,0,p]
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    cam_pos_str.append(stv)
    cam_verts.append(vs)
    lab = 'cam_pos_z_{:>2d}'.format(p)
    render_names.append(lab)

# Normalizing all strengths
h = w = 256
cam_strs = np.array(cam_pos_str)
cam_strs = np.nan_to_num(cam_strs)
cam_strs = cam_strs/np.max(cam_strs)
cam_sum = np.sum(cam_strs, axis = 1)

sf = 'results/final_results/camera_positions'
if not os.path.exists(sf):
    os.mkdir(sf)
# Rendering the strength results
for k in range(len(cam_strs)):
    img = mesh.render.render_colors(cam_verts[k], triangles, cam_strs[k], h, w, c=1)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img *= 255 # or any coefficient
    img = img.astype(np.uint8)
    io.imsave('{}/{:>2d}_'.format(sf, k) + render_names[k] + '.jpg', img, quality=100)

#Generating and plotting strength sum values for each of the axes
plot_x_x = np.array(range(-400,401,80))
plot_x_y = cam_sum[0:11,0]
plot_y_x = np.array(range(-400,401,80))
plot_y_y = cam_sum[11:22,0]
plot_z_x = np.array(range(-100,701,80))
plot_z_y = cam_sum[22:33,0]
plt.plot(plot_x_x, plot_x_y) #(x = 200)
plt.grid(True)
plt.plot(plot_y_x, plot_y_y) #(y = -325)
plt.grid(True)
plt.plot(plot_z_x, plot_z_y) #(z = 50)
plt.grid(True)

camera['eye'] = [200,-325,50]
stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
cam_pos_str[32] = stv
cam_verts[32] = vs

img = mesh.render.render_colors(cam_verts[32], triangles, cam_strs[32], h, w, c=1)
img = np.concatenate((np.zeros((h, w, 1)), img), 2)
img = np.concatenate((np.zeros((h, w, 1)), img), 2)
img *= 255 # or any coefficient
img = img.astype(np.uint8)
plt.imshow(img)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Testing for source positions (Orthographic projection only)
camera['eye'] = [200,-325,50] # Best camera eye location determined
light_pos_str = []
light_verts = []
render_names_light = []
for i,p in enumerate(range(-400, 401, 80)):
    source = (p,0,200)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'light_x_{:>2d}'.format(p)
    render_names_light.append(lab)
for i,p in enumerate(range(-400, 401, 80)):
    source = (0,p,200)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'light_y_{:>2d}'.format(p)
    render_names_light.append(lab)
for i,p in enumerate(range(-100, 701, 80)):
    source = (0,0,p)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'light_z_{:>2d}'.format(p)
    render_names_light.append(lab)

#Normalizing all strengths
h = w = 256
light_strs = np.array(light_pos_str)
light_strs = np.nan_to_num(light_strs)
light_strs = light_strs/np.max(light_strs)
light_sum = np.sum(light_strs, axis = 1)

sf = 'results/final_results/source_positions'
if not os.path.exists(sf):
    os.mkdir(sf)
# Rendering the strength results
for k in range(len(light_strs)):
    img = mesh.render.render_colors(light_verts[k], triangles, light_strs[k], h, w, c=1)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img *= 255 # or any coefficient
    img = img.astype(np.uint8)
    io.imsave('{}/{:>2d}_'.format(sf, k) + render_names_light[k] + '.jpg', img, quality=100)

#Generating and plotting strength sum values for each of the axes
plot_x_x = np.array(range(-400,401,80))
plot_x_y = light_sum[0:11,0]
plot_y_x = np.array(range(-400,401,80))
plot_y_y = light_sum[11:22,0]
plot_z_x = np.array(range(-100,701,80))
plot_z_y = light_sum[22:33,0]
plt.plot(plot_x_x, plot_x_y) #(x = 0)
plt.grid(True)
plt.plot(plot_y_x, plot_y_y) #(y = -100)
plt.grid(True)
plt.plot(plot_z_x, plot_z_y) #(z = 200 to 700)
plt.grid(True)

source = (0,-100,700)
stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
light_pos_str.append(stv)
light_verts.append(vs)
light_strs = np.array(light_pos_str)
light_strs = np.nan_to_num(light_strs)
light_strs = light_strs/np.max(light_strs)
light_sum = np.sum(light_strs, axis = 1)
img = mesh.render.render_colors(light_verts[32], triangles, light_strs[32], h, w, c=1)
img = np.concatenate((np.zeros((h, w, 1)), img), 2)
img = np.concatenate((np.zeros((h, w, 1)), img), 2)
img *= 255 # or any coefficient
img = img.astype(np.uint8)
print("")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Testing for source temperature values (Orthographic projection only)
source = (0,-100,700) # Best source position determined
temp_str = []
temp_verts = []
render_names_temp = []
for i,p in enumerate(range(250, 10250, 1000)):
    temperature = p
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    temp_str.append(stv)
    temp_verts.append(vs)
    lab = 'temp_{:>2d}'.format(p)
    render_names_temp.append(lab)

#Normalizing all strengths
h = w = 256
temp_strs = np.array(temp_str)
temp_strs = np.nan_to_num(temp_strs)
temp_strs = temp_strs/np.max(temp_strs)
temp_sum = np.sum(temp_strs, axis = 1)

sf = 'results/final_results/temperature_values'
if not os.path.exists(sf):
    os.mkdir(sf)
# Rendering the strength results
for k in range(len(temp_strs)):
    img = mesh.render.render_colors(temp_verts[k], triangles, temp_strs[k], h, w, c=1)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img = np.concatenate((np.zeros((h, w, 1)), img), 2)
    img *= 255 # or any coefficient
    img = img.astype(np.uint8)
    io.imsave('{}/{:>2d}_'.format(sf, k) + render_names_temp[k] + '.jpg', img, quality=100)

#Generating and plotting strength sum values for each of the axes
plot_x_x = np.array(range(250,10250,1000))
plot_x_y = temp_sum[0:10,0]
plt.plot(plot_x_x, plot_x_y) #(x = 0)
plt.grid(True)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
