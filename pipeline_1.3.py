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
import time
# width of beckmann microfacet distribution
beck_width = 0.35
# Refractive index of the material the light is coming from
eta_i = 1
# Refractive index of the material the light is hitting/entering
eta_t = 1.45

def prt_some(arr):
    print(f"f10:{arr[:10]},L10:{arr[-10:]}, max, min:{max(arr)}, {min(arr)}")
    return "printing"

def normalize(arr):
    return arr/np.linalg.norm(arr)

# Microfacet distribution function
# NOTE: theta_m is a debugging parameter, and can be removed (along with its corresponding if-else)
def distr(w_h, n):

    cos_wh = np.dot(w_h, n)
    #return 1, 0, 0
    
    #cos_wh = np.cos(theta_m)
    if cos_wh <= 0:
        return 0
    cos_theta_sqr = cos_wh ** 2
    sin_theta_sqr = 1 - cos_theta_sqr

    cos_theta_4 = pow(cos_theta_sqr, 2)
    tan_sqr = sin_theta_sqr/cos_theta_sqr

    width_sqr = pow(beck_width, 2)
    return np.exp(-tan_sqr/width_sqr)/(math.pi * width_sqr * cos_theta_4)


# Monodirectional shadowing helper function
def monodir_shadowing(v, w_h, n):
    #return 1
    cos_sqr = np.dot(v, n)
    sin_sqr = max(0, 1 - cos_sqr)
    if np.dot(v, w_h) <= 0 or np.dot(v, n) <= 0:
        return 0
    else:
        a = 1 / (beck_width * abs(np.sqrt(sin_sqr / cos_sqr)))
    if a < 1.6:
        a_sqr = a * a
        return (3.535 * a + 2.181 * a_sqr)/(1 + 2.276 * a + 2.577 * a_sqr)
    else:
        return 1

# Geometry function (using monodirectional shadowing function)
def geom(w_i, w_o, w_h, n):
    ret = 1
    for w in [w_i, w_o]:
        ret *= monodir_shadowing(w, w_h, n)
    return ret
        
# Dielectric fresnel coefficients helper function
def dielectric(ci, ct, ei, et):
    r_par = (et * ci - ei * ct)/(et * ci + ei * ct)
    r_perp = (ei * ci - et * ct)/(ei * ci + et * ct)
    return 0.5 * (r_par * r_par + r_perp * r_perp)

# Fresnel reflectance function for dielectrics (skin)
def fres(cos_i):
    #return 1
    g = (eta_t**2)/(eta_i**2) - 1 + cos_i**2
    if g <= 0:
        return 1
    
    g = np.sqrt(g)
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
def brdf(p_s, w_i, w_o, n):
    n = n/np.linalg.norm(n)
    w_i = w_i/np.linalg.norm(w_i)
    w_o = w_o/np.linalg.norm(w_o)
    cos_to = np.abs(np.dot(w_o, n))
    cos_ti = np.abs(np.dot(w_i, n))
    if cos_to == 0 or cos_ti == 0:
        return 1
    
    w_h = w_i + w_o
    # Normalize w_h
    if np.linalg.norm(w_h) == 0:
        return 1
    norm_wh = w_h / np.linalg.norm(w_h)

    #f = fres_orig(np.dot(w_i, norm_wh))
    f = fres(np.dot(w_i, norm_wh))

    g = geom(w_i, w_o, norm_wh, n)
    d = distr(norm_wh, n)
    numer = p_s * g * d
    denom = 4*cos_to*cos_ti
    return (f*p_s*g*d)/(4*cos_to*cos_ti)

# This function takes in an outgoing vector (camera vector) and returns a list of 10 vectors that are representative of its corresponding hemisphere.
def get_wos(wo):
    wo = normalize(wo)
    if wo[2] != 1:
        cross = [0,0,1]
    else:
        cross = [0,1,0]
    perp1 = normalize(np.cross(cross, wo))
    perp2 = normalize(np.cross(perp1, wo))
    return [wo, normalize(wo + perp1), normalize(wo - perp1), normalize(wo + perp2), normalize(wo - perp2),perp1,perp2,-perp1,-perp2]

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

# def source_channel_distr(T, channel):
# 	#T: temperature in Kelvin
# 	#channel: color channel (0/1/2) - (R/G/B)
# 	#l: wavelengths: in nm
# 	wvs = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720]
# 	l = wvs
# 	h = 6.626e-34
# 	k = 1.381e-23
# 	c = 3.0e8 
# 	l = [l2*1e-9 for l2 in l]
# 	spd = [(8*np.pi*h*c*pow(l2,-5))/(np.exp((h*c)/(k*l2*T))-1) for l2 in l]
# 	spd = [spd2/(np.sum(spd)) for spd2 in spd]
# 	spd = np.array(spd)
# 	spd_ch = spd*sensor_data[channel]
# 	wavelength_ch = 10*round(sum(multer((spd_ch/sum(spd_ch)), wvs))/10, 0)
# 	return wavelength_ch

##########-----------------skin thickness per vertex functions---------------------##########


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
    # NOTE THAT THIS WAS PREVIOUSLY CAST TO INT IN THE DLIB VERSION
    return [xo, yo]

# Takes in the 3DMM vertices (properly aligned to the input image), as well as the path to the face landmark pre-traines detector, and...
# Outputs: a thickness value for every single vertex in the vertices array
def get_thicknesses(vertices, path_model, image_file):
    mat_fname = "Data/BFM/Out/BFM_info.mat"
    model_info = sio.loadmat(mat_fname)
    shape = model_info['model_info']['kpt_ind'][0][0][0]
    uv_translate = model_info['model_info']['uv_coords'][0][0]
    xyshape = np.zeros((len(shape), 2))
    for i, elem in enumerate(shape):
        xyshape[i] = uv_translate[int(elem)]
    shape = xyshape
    epi_depth, derm_depth = contruct_mapping()

    shape = shape[:60]

    # Add new points to dict and face landmarks: epidermis
    edepths = [1.37, 1.7, 1.3, 1.54, 1.3, 1.7, 1.37, 2.56, 2.59, 2.3, 2.56, 2.59, 2.3, 1.37, 1.37]
    ddepths = [1.55, 1.51, 1.38, 1.53, 1.38, 1.51, 1.55, 1.74, 1.58, 1.64, 1.74, 1.58, 1.64, 1.55, 1.55]
    for index, [i1, i2, frac] in enumerate([[30, 2, 1/2], [4, 31, 1/2], [49, 6, 1/2], [58, 9, 1/2], [55, 12, 1/2], [14, 31, 1/2], [30, 16, 1/2], [31, 4, 1/13], [30, 3, 1/14], [29, 2, 1/14], [31, 14, 1/14], [30, 15, 1/14], [29, 16, 1/15], [30, 15, 1/3], [30, 3, 1/3]]):
        x1, y1 = find_mid(shape, i1, i2, frac)
        shape = np.append(shape, [[x1, y1]], axis=0)
        e_depth = edepths[index]
        epi_depth[len(shape)] = e_depth*29.57/10000
        d_depth = ddepths[index]
        derm_depth[len(shape)] = d_depth*758.85/10000
    
    epi_skin_depth = np.zeros(len(vertices))
    derm_skin_depth = np.zeros(len(vertices))

    for i, vert in enumerate(vertices):

        vert = uv_translate[i]

        closest = [-1, float('inf')]
        sec_closest = [-1, float('inf')]
        for j, (x, y) in enumerate(shape):
            euc_dist = np.sqrt((x - vert[0])**2 + (y - vert[1])**2)

            min_closest = min(closest[1], euc_dist)

            if min_closest == euc_dist:
                sec_closest = closest.copy()
                closest = [j+1, min_closest]
            else:
                min_closest = min(sec_closest[1], euc_dist)
                if min_closest == euc_dist:
                    sec_closest = [j+1, min_closest]
        tot_clos = closest[1]+sec_closest[1]

        epi_weight_avg = (epi_depth[closest[0]]*sec_closest[1]/tot_clos) + (epi_depth[sec_closest[0]]*closest[1]/tot_clos)
        derm_weight_avg = (derm_depth[closest[0]]*sec_closest[1]/tot_clos) + (derm_depth[sec_closest[0]]*closest[1]/tot_clos)
        epi_skin_depth[i] = epi_weight_avg 
        derm_skin_depth[i] = derm_weight_avg

    sum_d = 0
    for key in derm_depth:
        sum_d += derm_depth[key]
    sum_d /= len(derm_depth)

    sum_e = 0
    for key in epi_depth:
        sum_e += epi_depth[key]
    sum_e /= len(epi_depth)
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

    
    strength_val, angles, normals = compute_strength(temperature, channel, fblood, fmel, vertices, triangles, source, inp_derm_depth, inp_epi_depth)
    strength_val = np.array(strength_val)

    # NOTE: that all these other arrays (other than brdf_val) can be removed from the code. They are just there solely for debugging purposes.
    brdf_val = []

    for index, value in enumerate(vertices):
        norm_i = normals[index]
        norm_i /= np.linalg.norm(norm_i)
        wo = normalize(camera['eye'])
        tot_brdf = float('-inf')
        #wos = get_wos(camera['eye'] - vertices[index])
        wos = get_wos(source - vertices[index])
        for wo in wos:
            brdf_out = brdf(0.1, source - vertices[index], wo, -1*np.array(norm_i))
            brdf_out *= np.dot(source - vertices[index], -1*np.array(norm_i))
            if tot_brdf == float('-inf'):
                tot_brdf = brdf_out/len(wos)
            else:
                tot_brdf = tot_brdf + brdf_out/len(wos)
        curr_brdf = tot_brdf
        brdf_val.append(curr_brdf)

    brdf_val = np.array(brdf_val)
    brdf_val = np.array(brdf_val).reshape(-1,1)
    ##midpoint = 0.99/np.percentile(brdf_val, 99)
    ##brdf_val = brdf_val*midpoint
    ##brdf_val = np.minimum((brdf_val - 1),-0.0000001) + 1
    ##brdf_val = brdf_val/np.percentile(brdf_val, 100)
    ##brdf_val = brdf_val/np.amax(brdf_val)
    strength_val = strength_val*brdf_val
    #brdf_val = np.array(brdf_val).reshape(-1,1)

    print('\n' + "BRDF Values computed")
    strength_val = strength_val * (np.divide(np.square(np.max(transformed_vertices[:,1]) - np.min(transformed_vertices[:,1])),np.square(source[2] - transformed_vertices[:,2]))).reshape(-1,1)

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
  nr = subter(multer([(pow(n,2) -1)*(1 + 1/n) for n in b], [np.exp(o) for o in expnt]), multer([(pow(p,2) -1)*(1 - 1/p) for p in b], [np.exp(-1*q) for q in expnt]))
  #nr = subter(multer([(pow(n,2) -1)*(1 + 1/n) for n in b], [np.exp(o) for o in expnt]), multer([(1 - 1/p) for p in b], [np.exp(-1*q) for q in expnt]))
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
  mumel = [6.6 * pow(10,11) * pow(lmbda,-3.33) for d_epis in d_epi]
  muskin = computeAbsorptionCoefficientSkin(lmbda)
  return adder(multer(fmel, mumel), [(1-w) * muskin for w in fmel])

def computeTepidermis(lmbda,fmel,d_epi):
  muepidermis = computeAbsorptionCoefficientEpidermis(lmbda,fmel,d_epi)
  prod = multer(d_epi,muepidermis)
  return [np.exp(-1 * v) for v in prod]

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

def computeR(lmbda,fblood,fmel,d,d_epi):
    Tepidermis = computeTepidermis(lmbda,fmel,d_epi)
    b = computeBeta(lmbda,fblood)
    K = computeK(lmbda,fblood)
    expnt = multer(K, d)
    R = multer([pow((1 - h),2) for h in b], subter([np.exp(f) for f in expnt],[np.exp(-1*e) for e in expnt]))  
    R = divider(R,subter(multer([pow((1 + g),2) for g in b], [np.exp(f) for f in expnt]), multer([pow((1 - h),2) for h in b], [np.exp(-1*e) for e in expnt])))
    
    return np.abs(multer([aq**2 for aq in Tepidermis], R))	
	
	

#consolidated function to return strength
def compute_strength(T,ch,fblood,fmel,vertices,triangles,s,d_derm,d_epi):
    eps = 1e-6
    #- s is light source coordinates
    #- d_epi + d_derm is the avg. skin depth
    #lmbda- wavelength of incident radiation, between 400 and 730 nm
    direction = vertices-np.reshape(s,(1,-1))
    direction = direction/(np.linalg.norm(direction, ord=2, axis=-1, keepdims=True)+eps)
    
    normals = normals_compute(vertices,triangles)
    angles = angles_compute(normals,direction)
    print('\n'+"Angles Values computed")
    # print(np.min(angles))
    # print(np.max(angles))
    
    d_real = divider(d_derm, np.cos(angles)+eps)
    d_real = np.maximum(d_real,0)
    print('\n' + "d_real Values computed")
    # print(np.percentile(d_real, 10))
    # print(np.percentile(d_real, 30))
    # print(np.percentile(d_real, 50))
    # print(np.percentile(d_real, 70))
    # print(np.percentile(d_real, 90))

    wvs = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720]
    l = wvs
    h = 6.626e-34
    k = 1.381e-23
    c = 3.0e8 
    l = [l2*1e-9 for l2 in l]
    spd = [(8*np.pi*h*c*pow(l2,-5))/(np.exp((h*c)/(k*l2*T))-1) for l2 in l]
    spd = [spd2/(np.sum(spd)) for spd2 in spd]
    spd = np.array(spd)
    chan = spd*sensor_data[ch]
    total_sig = np.zeros((len(vertices),1))
    fbla1 = []
    [fbla1.append(0.01) for i in range(53215)]
    fbla2 = []
    [fbla2.append(0.07) for i in range(53215)]

    #alls = []
    for i in range(len(wvs)):
    	lmbda = wvs[i]
    	fblood = fbla1
    	a1 = computeR(lmbda,fblood,fmel,d_real,d_epi)
    	fblood = fbla2
    	a2 = computeR(lmbda,fblood,fmel,d_real,d_epi)
    	signal = np.abs(a2 - a1)
    	signal = np.nan_to_num(signal)
		#reshaping signal to nVert,1
    	signal = np.reshape(signal,(-1,1))
    	total_sig = total_sig + signal*chan[i]
    	#alls.append(signal)
	    #signal = signal/np.max(signal) #normalize to 0=1: if linear normalization doesnt enable good visualization, can also use dB scale

    signal = total_sig
    print('\n' + "Strength Values computed")
    # print(np.percentile(signal, 10))
    # print(np.percentile(signal, 30))
    # print(np.percentile(signal, 50))
    # print(np.percentile(signal, 70))
    # print(np.percentile(signal, 90))
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
        fblood.append(0.02 + 0.05*(pos_of_min[0]/len(sRGBim)))
        fmel.append(0.013 + 0.417*(pos_of_min[1]/len(sRGBim[0])))
        if timer%(len(colors)//3) == 0:
            print(str(100*timer/len(colors)) + "% completed processing")
        timer = timer + 1
    
    return fblood, fmel

def calc_mel_haem2(colors):
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
        fblood.append(0.07*(pos_of_min[0]/len(sRGBim[:,0]))**3)
        fmel.append(0.43*(pos_of_min[1]/len(sRGBim[0]))**3)
        if timer%(len(colors)//3) == 0:
            print(str(100*timer/len(colors)) + "% completed processing")
        timer = timer + 1
    
    return fblood, fmel

def tri_area(x1, y1, x2, y2, x3, y3): 
  
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)  
                + x3 * (y1 - y2)) / 2.0) 
  
def inTri(tri, point): 
	  
    # Calculate area of triangle ABC 
    A = tri_area(tri[0,0],tri[0,1],tri[1,0],tri[1,1],tri[2,0],tri[2,1]) 
  
    # Calculate area of triangle PBC  
    A1 = tri_area(tri[1,0],tri[1,1],tri[2,0],tri[2,1],point[0],point[1]) 
      
    # Calculate area of triangle PAC  
    A2 = tri_area(tri[0,0],tri[0,1],tri[2,0],tri[2,1],point[0],point[1]) 
      
    # Calculate area of triangle PAB  
    A3 = tri_area(tri[0,0],tri[0,1],tri[1,0],tri[1,1],point[0],point[1]) 
      
    # Check if sum of A1, A2 and A3  
    # is same as A 
    if(abs(A - (A1 + A2 + A3))<0.01): 
        return True
    else: 
        return False

def mask_eye(vertices,kpts):
	ans = []
	for i in range(len(kpts)):
		ans.append(vertices[kpts[i],:])
	
	ans2 = np.array(ans)
	x_mi = np.min(ans2[:,0])
	x_ma = np.max(ans2[:,0])
	y_mi = np.min(ans2[:,1])
	y_ma = np.max(ans2[:,1])
	z_mi = np.min(ans2[:,2])
	z_ma = np.max(ans2[:,2])
	k = []
	for i in range(len(vertices)):	
		if vertices[i,0] >= x_mi and vertices[i,0] <= x_ma:
			if vertices[i,1] >= y_mi and vertices[i,1] <= y_ma:
				if vertices[i,2] >= z_mi-4000 and vertices[i,2] <= z_ma+4000:
					k.append(i)
	
	t1 = np.concatenate((ans2[0,0:2].reshape(1,-1),ans2[1,0:2].reshape(1,-1),ans2[5,0:2].reshape(1,-1)),axis = 0)
	t2 = np.concatenate((ans2[1,0:2].reshape(1,-1),ans2[4,0:2].reshape(1,-1),ans2[5,0:2].reshape(1,-1)),axis = 0)
	t3 = np.concatenate((ans2[1,0:2].reshape(1,-1),ans2[2,0:2].reshape(1,-1),ans2[4,0:2].reshape(1,-1)),axis = 0)
	t4 = np.concatenate((ans2[2,0:2].reshape(1,-1),ans2[3,0:2].reshape(1,-1),ans2[4,0:2].reshape(1,-1)),axis = 0)
	
	eye_points = []
	for i in range(len(k)):
		
		if inTri(t1,vertices[k[i],0:2]) or inTri(t2,vertices[k[i],0:2]) or inTri(t3,vertices[k[i],0:2]) or inTri(t4,vertices[k[i],0:2]) :
			eye_points.append(k[i])

	return eye_points

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --------- load mesh data
# --- 1. load model
bfm = MorphabelModel('Data/BFM/Out/BFM.mat')
print('init bfm model success')

cam_filename = 'cam_sens.mat'
cam_data = sio.loadmat(cam_filename)
global sensor_data
sensor_data = cam_data['S']

global sRGBim, fblood, fmel
#sRGBim = sRGB_generation() 
new_mel2 = plt.imread('mel_map_2.jpg') # melanin map from Jiminez et.al.
new_mel2 = new_mel2[0:157,0:500]
new_mel2 = new_mel2/256
sRGBim = new_mel2

# --- 2. load fitted face
mat_filename00 = 'Sample6.mat'
# mat_filename01 = 'Sample7.mat'
# mat_filename02 = 'image00002.mat'
# mat_filename03 = 'image02720.mat'
# mat_filename04 = 'image00006.mat'
# mat_filename05 = 'image00013.mat'
# mat_filename06 = 'image02732.mat'
# mat_filename07 = 'image00039.mat'
# mat_filename08 = 'image00076.mat'
# mat_filename09 = 'image00134.mat'
# mat_filename10 = 'image02752.mat'
# mat_filename11 = 'image02796.mat'
# mat_filename12 = 'image00294.mat'
# mat_filename13 = 'image00441.mat'
# mat_filename14 = 'image01016.mat'
#mat_filename15 = 'image02673.mat'

mata_data = []
mata_data.append(sio.loadmat(mat_filename00))
# mata_data.append(sio.loadmat(mat_filename01))
# mata_data.append(sio.loadmat(mat_filename02))
# mata_data.append(sio.loadmat(mat_filename03))
# mata_data.append(sio.loadmat(mat_filename04))
# mata_data.append(sio.loadmat(mat_filename05))
# mata_data.append(sio.loadmat(mat_filename06))
# mata_data.append(sio.loadmat(mat_filename07))
# mata_data.append(sio.loadmat(mat_filename08))
# mata_data.append(sio.loadmat(mat_filename09))
# mata_data.append(sio.loadmat(mat_filename10))
# mata_data.append(sio.loadmat(mat_filename11))
# mata_data.append(sio.loadmat(mat_filename12))
# mata_data.append(sio.loadmat(mat_filename13))
# mata_data.append(sio.loadmat(mat_filename14))
# mata_data.append(sio.loadmat(mat_filename15))



image_filename = {}
image_filename[0] = 'Sample6.jpg'
# image_filename[1] = 'Sample7.jpg'
# image_filename[2] = 'image00002.jpg'
# image_filename[3] = 'image02720.jpg'
# image_filename[4] = 'image00006.jpg'
# image_filename[5] = 'image00013.jpg'
# image_filename[6] = 'image02732.jpg'
# image_filename[7] = 'image00039.jpg'
# image_filename[8] = 'image00076.jpg'
# image_filename[9] = 'image00134.jpg'
# image_filename[10] = 'image02752.jpg'
# image_filename[11] = 'image02796.jpg'
# image_filename[12] = 'image00294.jpg'
# image_filename[13] = 'image00441.jpg'
# image_filename[14] = 'image01016.jpg'
# image_filename[15] = 'image02673.jpg'

mat_data = mata_data[0]
image_file = image_filename[0]
orig_img = cv2.imread(image_file)

sp = mat_data['Shape_Para']
ep = mat_data['Exp_Para']
tp = mat_data['Tex_Para']
vertices = bfm.generate_vertices(sp, ep)
colors = bfm.generate_colors(tp)
#colors = np.minimum(np.maximum(colors, 0), 1)
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

#code to compute skin depth values
global inp_epi_depth, inp_derm_depth
depth_var_vertices = bfm.transform(vertices, d_s, d_angles2, d_t)
inp_epi_depth, inp_derm_depth = get_thicknesses(depth_var_vertices, "shape_predictor_68_face_landmarks.dat",image_file)
buff_derm = inp_derm_depth #buffer variable to store actual values computed
buff_epi = inp_epi_depth #buffer variable to store actual values computed
#code to scale derm_depth and epi_depth such that avg derm_depth = 0.2 and avg epi_depth = 0.01
inp_epi2 = inp_epi_depth * (0.01/np.mean(inp_epi_depth))
inp_epi_depth = inp_epi2
inp_derm2 = inp_derm_depth * (0.2/np.mean(inp_derm_depth))
inp_derm_depth = inp_derm2

#Masking out the eyes from colors variable (based on which fmel and fblood are calculated)
keys = bfm.kpt_ind[36:48]
left_eye_points = mask_eye(vertices,keys[0:6])
right_eye_points = mask_eye(vertices,keys[6:12])
for i in range(len(left_eye_points)):
	colors[left_eye_points[i],:] = float('-inf')
	inp_derm_depth[left_eye_points[i]] = 0
	inp_epi_depth[left_eye_points[i]] = 0
for i in range(len(right_eye_points)):
	colors[right_eye_points[i],:] = float('-inf')
	inp_derm_depth[right_eye_points[i]] = 0
	inp_epi_depth[right_eye_points[i]] = 0	

fblood, fmel = calc_mel_haem2(colors)

transformed_vertices = bfm.transform(vertices, s, angles2, t)
transformed_vertices = transformed_vertices - np.mean(transformed_vertices, 0)[np.newaxis, :]
vertices = transformed_vertices
c = 3

# channel = 2
# camera['eye'] = [0, 0, 250]
# source = (0,0,400)  # stay in front of face
# beg = time.time()
# stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
# end = time.time()
# print(end-beg)
# #stv = stv/np.max(stv)
# img = mesh.render.render_colors(vs,triangles,stv, 256, 256, c=1)
# plt.imshow(img)
# plt.set_cmap('inferno')
# plt.clim(0,0.005)

#mouth = bfm.model['tri_mouth']
#tri2 = bfm.full_triangles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#1 - Testing for camera eye positions (Perspective projection only)
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
temperature = 5500 # temperature is a variable parameter 
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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

cam_pos_str = []
cam_verts = []
render_names = []
# z-axis: eye from far to near, looking at the center of face
begi = time.time()
for p in np.arange(500, 250-1, -40): # 0.5m->0.25m
	beg = time.time()
	camera['eye'] = [0, 0, p]  # stay in front of face
	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
	cam_pos_str.append(stv)
	cam_verts.append(vs)
	lab = 'cp_z_{:>2d}'.format(p)
	render_names.append(lab)
	end = time.time()
	print('\n\n',end-beg,'secs time taken for this image\n')

print("\n"+ str(700/40) + "% complete \n")
# y-axis: eye from down to up, looking at the center of face

for p in np.arange(-300, 301, 60): # up 0.3m -> down 0.3m
	beg = time.time()
	camera['eye'] = [0, p, 250] # stay 0.25m far
	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
	cam_pos_str.append(stv)
	cam_verts.append(vs)
	lab = 'cp_y_{:.2f}'.format(p)
	render_names.append(lab)
	end = time.time()
	print('\n\n',end-beg,'secs time taken for this image\n')
print("\n"+ str(1800/40) + "% complete \n")
# x-axis: eye from left to right, looking at the center of face

for p in np.arange(-300, 301, 60): # left 0.3m -> right 0.3m
	beg = time.time()
	camera['eye'] = [p, 0, 250] # stay 0.25m far
	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
	cam_pos_str.append(stv)
	cam_verts.append(vs)
	lab = 'cp_x_{:.2f}'.format(p)
	render_names.append(lab)
	end = time.time()
	print('\n\n',end-beg,'secs time taken for this image\n')

print("\n"+ str(2900/40) + "% complete \n")
# up direction

camera['eye'] = [0, 0, 250] # stay in front
for p in np.arange(-50, 51, 10):
	beg = time.time()
	world_up = np.array([0, 1, 0]) # default direction
	z = np.deg2rad(p)
	Rz=np.array([[math.cos(z), -math.sin(z), 0],
                 [math.sin(z),  math.cos(z), 0],
                 [     0,       0, 1]])
	up = Rz.dot(world_up[:, np.newaxis]) # rotate up direction
	# note that: rotating up direction is opposite to rotating obj
	# just imagine: rotating camera 20 degree clockwise, is equal to keeping camera fixed and rotating obj 20 degree anticlockwise.
	camera['up'] = np.squeeze(up)
	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
	cam_pos_str.append(stv)
	cam_verts.append(vs)
	lab = 'cp_up_{:>2d}'.format(p)
	render_names.append(lab)
	end = time.time()
	print('\n\n',end-beg,'secs time taken for this image\n')

cam_strs = np.array(cam_pos_str)
cam_strs = np.nan_to_num(cam_strs)
endi = time.time()
print('\n\n',(endi-begi)/60,'mins time taken for the parameter\n')
print("\n"+ str(4000/40) + "% complete \n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2 - Testing for source positions (Perspective projection only)
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
temperature = 5500 # temperature is a variable parameter 
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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

#camera['eye'] = [200,-325,50] # Best camera eye location determined
light_pos_str = []
light_verts = []
render_names_light = []
begi = time.time()
for i,p in enumerate(range(-400, 401, 100)):
    beg = time.time()
    source = (p,0,200)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'lit_x_{:>2d}'.format(p)
    render_names_light.append(lab)
    end = time.time()
    print('\n\n',(end-beg)/60,'mins time taken for this image\n')

print("\n"+ str(900/26) + "% complete \n")

for i,p in enumerate(range(-400, 401, 100)):
    beg = time.time()
    source = (0,p,200)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'lit_y_{:>2d}'.format(p)
    render_names_light.append(lab)
    end = time.time()
    print('\n\n',(end-beg)/60,'mins time taken for this image\n')

print("\n"+ str(1800/26) + "% complete \n")

for i,p in enumerate(range(300, 1001, 100)):
    beg = time.time()
    source = (0,0,p)
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    light_pos_str.append(stv)
    light_verts.append(vs)
    lab = 'lit_z_{:>2d}'.format(p)
    render_names_light.append(lab)
    end = time.time()
    print('\n\n',(end-beg)/60,'mins time taken for this image\n')

light_strs = np.array(light_pos_str)
light_strs = np.nan_to_num(light_strs)
print("\n"+ str(2600/26) + "% complete \n")
endi = time.time()
print('\n',(endi-begi)/60,'mins time taken for the parameter\n')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3 - Testing for source temperature values 
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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

temp_str = []
temp_verts = []
render_names_temp = []
for i,p in enumerate(range(3000, 9000, 250)):
    beg = time.time()
    temperature = p
    stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
    temp_str.append(stv)
    temp_verts.append(vs)
    lab = 'temp_{:>2d}'.format(p)
    render_names_temp.append(lab)
    end = time.time()
    print('\n\n',(end-beg)/60,'mins time taken for this image\n')

temp_strs = np.array(temp_str)
temp_strs = np.nan_to_num(temp_strs)
print("\n100% complete\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#4 - Testing for object angles 
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
temperature = 5500
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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

objang_str = []
objang_verts = []
render_names_objang = []

for i in range(3):
	for angle in np.arange(-50, 51, 10):
		obj['s'] = scale_init
		obj['angles'] = [0, 0, 0]
		obj['angles'][i] = angle
		obj['t'] = [0, 0, 0]
		stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
		objang_str.append(stv)
		objang_verts.append(vs)    
		lab = 'ang_{:>2d}_{:>2d}'.format(i,angle)
		render_names_objang.append(lab)

obj_strs = np.array(objang_str)
obj_strs = np.nan_to_num(obj_strs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#5 - Testing for melanin variation 

obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
temperature = 5500
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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
fmel2 = fmel
baseline = np.array(fmel)
max_val = np.percentile(baseline,96)
low = 0.01/max_val
high = 0.43/max_val
inc = (high-low)/30

art_mel_str = []
art_mel_ver = []
render_names_mel = []

for factor in np.arange(low,(high+inc),inc):
	k1 = baseline * factor
	k1 = np.minimum(k1,0.43)
	fmel = k1
	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
	art_mel_str.append(stv)
	art_mel_ver.append(vs)	
	lab = 'mel_sc_{:.2f}'.format(factor)
	render_names_mel.append(lab)
	
	

amel_strs = np.array(art_mel_str)
amel_strs = np.nan_to_num(amel_strs)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#6 - Testing for different channels 
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
camera['eye'] = [0, 0, 250]
source = (0,0,200)
temperature = 5500
channel = 1 #channel can be 0 or 1 or 2 corresponding to (R/G/B) or (B/G/R) - doubt
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
fmel = fmel2
baseline = np.array(fmel)
max_val = np.percentile(baseline,96)
low = 0.01/max_val
high = 0.43/max_val
inc = (high-low)/10

chan_str = []
chan_ver = []
render_names_chan = []

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# all_strs_fmel = []
# chan_ver = []
# for i in np.arange(0,0.44,0.01):
# 	fmela = []
# 	for j in range(53215):
# 		fmela.append(i)
# 	fmel = fmela
# 	#print('\n' + str(fmel[0]) + '\n')
# 	stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
# 	all_strs_fmel.append(stv)
# 	chan_ver.append(vs)
# 	print(str((i*100)/0.43) + "% completed")	
# T = 5500
# wvs = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720]
# l = wvs
# h = 6.626e-34
# k = 1.381e-23
# c = 3.0e8 
# l = [l2*1e-9 for l2 in l]
# spd = [(8*np.pi*h*c*pow(l2,-5))/(np.exp((h*c)/(k*l2*T))-1) for l2 in l]
# spd = [spd2/(np.sum(spd)) for spd2 in spd]
# spd = np.array(spd)
# all_chan_mels = []
# metrics = np.zeros([len(all_strs_fmel),3])
# for k in range(3):
# 	for i in range(len(all_strs_fmel)):
# 		tot = np.zeros((len(vs),1))
# 		for j in range(33):
# 			tot = tot + all_strs_fmel[i][j,:,:]*spd[j]*sensor_data[k,j]
# 		
# 		#tot = tot*comm_brdf
# 		all_chan_mels.append(tot)
# 		metrics[i,k] = np.abs(np.mean(tot))
# 		lab = 'chan_{:>2d}_mel_{:.2f}'.format(k,(i/100))
# 		#render_names_chan.append(lab)
# all_chan_strs = np.array(all_chan_mels)
# all_chan_strs = np.nan_to_num(all_chan_strs)	
# mela = np.arange(0.0,0.44,0.01,dtype = float)
# plt.figure()
# plt.grid(True)
# plt.xlim([-0.01,0.45])
# # plt.ylim([0,0.005])
# plt.plot(mela,(metrics[:,0]),'r',mela,(metrics[:,1]),'g',mela,(metrics[:,2]),'b',linewidth=3)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@	
	

for i in range(3):
	channel = i
	for factor in np.arange(low,(high+inc),inc):
		beg = time.time()
		k1 = baseline * factor
		k1 = np.minimum(k1,0.43)
		fmel = k1
		stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
		chan_str.append(stv)
		chan_ver.append(vs)	
		lab = 'chan_{:>2d}_mel_{:.2f}'.format(i,factor)
		render_names_chan.append(lab)
		end = time.time()
		print('\n\n',end-beg,'secs time taken for this image\n')

chan_strs = np.array(chan_str)
chan_strs = np.nan_to_num(chan_strs)

stv, vs = transform_test(vertices, obj, camera, source, temperature, channel)
global comm_brdf
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL NORMALISATION CODE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#cam = 40
#light = 26
#temp = 24
#obj = 33
#amel = 31
#chan = 33

all_strs = np.concatenate((cam_strs,light_strs,temp_strs,obj_strs,amel_strs,chan_strs))
all_strs = all_strs/np.max(all_strs)

cam_strs = all_strs[0:40,:,:]
light_strs = all_strs[40:66,:,:]
temp_strs = all_strs[66:90,:,:]
obj_strs = all_strs[90:123,:,:]
amel_strs = all_strs[123:154,:,:]
chan_strs = all_strs[154:187,:,:]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#***************************************
# Rendering the camera position results
#cam_strs = cam_strs/np.max(cam_strs)
h = w = 256
cam_render_str = []
sf = 'results/final_results/camera_positions'
if not os.path.exists(sf):
    os.mkdir(sf)
maxi = 0
for k in range(len(cam_strs)):
	img = mesh.render.render_colors(cam_verts[k], triangles, cam_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	cam_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf, k) + render_names[k] + '.jpg')
print(maxi)
#0.0744
#***************************************
# Rendering the source position results
h = w = 256
light_render_str = []
sf2 = 'results/final_results/source_positions'
if not os.path.exists(sf2):
    os.mkdir(sf2)
maxi = 0
for k in range(len(light_strs)):
	img = mesh.render.render_colors(light_verts[k], triangles, light_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	light_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names_light[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf2, k) + render_names_light[k] + '.jpg')
print(maxi)
#0.3393
#***************************************
# Rendering the temperature results
h = w = 256
sf3 = 'results/final_results/temperature_values'
temp_render_str = []
if not os.path.exists(sf3):
    os.mkdir(sf3)
maxi = 0
for k in range(len(temp_strs)):
	img = mesh.render.render_colors(temp_verts[k], triangles, temp_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	temp_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names_temp[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf3, k) + render_names_temp[k] + '.jpg')
print(maxi)
#0.0566
#***************************************
# Rendering the object rotation results
h = w = 256
obj_render_str = []
sf4 = 'results/final_results/object_angles'
if not os.path.exists(sf4):
    os.mkdir(sf4)
maxi = 0
for k in range(len(obj_strs)):
	img = mesh.render.render_colors(objang_verts[k], triangles, obj_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	obj_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names_objang[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf4, k) + render_names_objang[k] + '.jpg')
print(maxi)
#0.0872
#***************************************
# Rendering the melanin variation results
h = w = 256
mel_render_str = []
sf5 = 'results/final_results/artificial_melanin_variation'
if not os.path.exists(sf5):
    os.mkdir(sf5)
maxi = 0
for k in range(len(amel_strs)):
	img = mesh.render.render_colors(art_mel_ver[k], triangles, amel_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	mel_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names_mel[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf5, k) + render_names_mel[k] + '.jpg')
print(maxi)
#maxi = 0.0585
#****************************************************

h = w = 256
chan_render_str = []
sf6 = 'results/final_results/channel_variation'
if not os.path.exists(sf6):
    os.mkdir(sf6)
maxi = 0
for k in range(len(chan_strs)):
	img = mesh.render.render_colors(chan_ver[0], triangles, chan_strs[k], h, w, c=1)
	buff = np.ones((h,w,1))
	anding = (np.logical_and(img,buff)).astype(int)
	avgstr = np.sum(img)/np.sum(anding)
	chan_render_str.append(avgstr)
	maxi = np.maximum(maxi,np.max(img))
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	#plt.colorbar()
	plt.set_cmap('inferno')
	plt.clim(0,0.1)
	plt.title('Average Strength = {:.5f}'.format(avgstr),fontdict={'fontsize': 25,'fontweight': 'bold'}) 
	plt.text(128, 245, render_names_chan[k], verticalalignment='bottom', horizontalalignment='center', color='white', fontweight='bold', fontsize=25)
	plt.savefig('{}/{:>2d}_'.format(sf6, k) + render_names_chan[k] + '.jpg')
print(maxi)
#0.0585
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#TODO : continue the code from here

#1 - Camera positions
#Generating and plotting strength sum values for each of the axes
plot_z_x = np.array(range(500,220,-40))
plot_z_y = cam_render_str[0:7]
plot_y_x = np.array(range(-50,60,10))
plot_y_y = cam_render_str[7:18]
plot_x_x = np.array(range(50,-60,-10))
plot_x_y = cam_render_str[18:29]
plot_up_x = np.array(range(50,-60,-10))
plot_up_y = cam_render_str[29:40]
plt.plot(plot_x_x, plot_x_y)
plt.grid(True)
plt.plot(plot_y_x, plot_y_y) 
plt.grid(True)
plt.plot(plot_z_x, plot_z_y) 
plt.grid(True)
plt.plot(plot_up_x, plot_up_y)
plt.grid(True)

#2 - Source positions
plot_x_x = np.array(range(-400,401,100))
plot_x_y = light_render_str[0:9]
plot_y_x = np.array(range(-400,401,100))
plot_y_y = light_render_str[9:18]
plot_z_x = np.array(range(300,1001,100))
plot_z_y = light_render_str[18:26]
plt.plot(plot_x_x, plot_x_y)
plt.grid(True)
plt.plot(plot_y_x, plot_y_y) 
plt.grid(True)
plt.plot(plot_z_x, plot_z_y) 
plt.grid(True)

#3 - Temperature values
plot_x_x = np.array(range(3000,9000,250))
plot_x_y = temp_render_str[0:24]
plt.plot(plot_x_x, plot_x_y)
plt.grid(True)

#4 - Object rotation
plot_x_x = np.array(range(-50,60,10))
plot_x_y = obj_render_str[0:11]
plot_y_x = np.array(range(-50,60,10))
plot_y_y = obj_render_str[11:22]
plot_z_x = np.array(range(-50,60,10))
plot_z_y = obj_render_str[22:33]
plt.plot(plot_x_x, plot_x_y)
plt.grid(True)
plt.plot(plot_y_x, plot_y_y) 
plt.grid(True)
plt.plot(plot_z_x, plot_z_y) 
plt.grid(True)

#5 - Melanin variation
plot_x_x = np.array(np.arange(low,(high+inc),inc))*max_val
plot_x_y = mel_render_str[0:31]
plt.plot(plot_x_x, plot_x_y)
plt.grid(True)

#6 - Color channel variation
plot_x_x = np.array(np.arange(low,(high+inc),inc))*max_val
plot_x_y = chan_render_str[0:11]
plot_y_y = chan_render_str[11:22]
plot_z_y = chan_render_str[22:33]

plt.plot(plot_x_x, plot_x_y, 'r')
plt.plot(plot_x_x, plot_y_y, 'g')
plt.plot(plot_x_x, plot_z_y, 'b')  
plt.grid(True)

plt.yscale("log")
plt.plot(plot_x_x, plot_x_y,color = 'red') 
plt.plot(plot_x_x, plot_y_y,color = 'green') 
plt.plot(plot_x_x, plot_z_y,color = 'blue') 
plt.grid(True)