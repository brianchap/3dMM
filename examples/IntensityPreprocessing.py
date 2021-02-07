from skimage import io
import csv
import matplotlib.pyplot as plt
import numpy as np
import re

# 1. DATA INITIALIZATION
deoxy_hemo_ext_coeff = [223296, 303956, 407560, 528600, 413280, 103292, 23388.8, 16156.4, 14550, 16684, 20862, 25773.6, 31589.6, 39036.4, 46592, 53412, 53788, 45072, 37020, 28324.4, 14677.2, 9443.6, 6509.6, 5148.8, 4345.2, 3750.12, 3226.56, 2795.12, 2407.92, 2051.96, 1794.28, 1540.48, 1325.88]
oxy_hemo_ext_coeff = [266232, 466840, 480360, 246072, 102580, 62816, 44480, 33209.2, 26629.2, 23684.4, 20932.8, 20035.2, 24202.4, 39956.8, 53236, 43016, 32613.2, 44496, 50104, 14400.8, 3200, 1506, 942, 610, 442, 368, 319.6, 294, 277.6, 276, 290, 314, 348]
eumelanin_ext = [14.6366, 13.6476, 12.8612, 12.0748, 11.3966, 10.7412, 10.1064, 9.6058, 9.1052, 8.6178, 8.1371, 7.6564, 7.1757, 6.7085, 6.2493, 5.7901, 5.4564, 5.132, 4.8077, 4.4977, 4.3003, 4.1029, 3.8782, 3.6155, 3.3528, 3.1789, 3.0391, 2.8993, 2.7594, 2.5931, 2.4239, 2.2548, 2.1532]
pheomelanin_ext = [12.1065, 11.2765, 10.3872, 9.5972, 8.6767, 7.9894, 7.4196, 6.7876, 6.1557, 5.6117, 5.0845, 4.5997, 4.1243, 3.7138, 3.3524, 2.9935, 2.6584, 2.3233, 1.9881, 1.653, 1.738, 1.6214, 1.4722, 1.3547, 1.2817, 1.2087, 1.1268, 1.0410, 0.9583, 0.8795, 0.8007, 0.7070, 0.6126]
illumA = [14.708, 17.6753, 20.995, 24.6709, 28.7027, 33.0859, 37.8121, 42.8693, 48.2423, 53.9132, 59.8611, 66.0635, 72.4959, 79.1326, 85.947, 92.912, 100, 107.184, 114.436, 121.731, 129.043, 136.346, 143.618, 150.836, 157.979, 165.028, 171.963, 178.769, 185.429, 191.931, 198.261, 204.409, 210.365]
XYZspace = [[0.0143, 0.000396, 0.0679], [0.0435, 0.0012, 0.2074], [0.1344, 0.0040, 0.6456], [0.2839, 0.0116, 1.3856], [0.3483, 0.0230, 1.7471], [0.3362, 0.0380, 1.7721], [0.2908, 0.06, 1.6692], [0.1954, 0.091, 1.2876], [0.0956, 0.139, 0.813], [0.032, 0.208, 0.4652], [0.0049, 0.323, 0.272], [0.0093, 0.503, 0.1582], [0.0633, 0.71, 0.0782], [0.1655, 0.862, 0.0422], [0.2904, 0.954, 0.0203], [0.4334, 0.995, 0.0087], [0.5945, 0.995, 0.0039], [0.7621, 0.9520, 0.0021], [0.9163, 0.87, 0.0017], [1.0263, 0.757, 0.0011], [1.0622, 0.613, 0.0008], [1.0026, 0.503, 0.00034], [0.8544, 0.381, 0.00019], [0.6424, 0.265, 0.00005], [0.4479, 0.175, 0.00002], [0.2835, 0.107, 0], [0.1649, 0.061, 0], [0.0874, 0.032, 0], [0.0468, 0.017, 0], [0.0227, 0.0082, 0], [0.0114, 0.0041, 0], [0.0058, 0.0021, 0], [0.0029, 0.001, 0]]
rgbCMF = []
rgbCMF1 = []
rgbCMF2 = []
rgbCMF3 = []
datafile1 = open('rgbCMF1.csv', 'r')
datafile2 = open('rgbCMF2.csv', 'r')
datafile3 = open('rgbCMF3.csv', 'r')
datareader1 = list(csv.reader(datafile1, delimiter=','))
datareader2 = list(csv.reader(datafile2, delimiter=','))
datareader3 = list(csv.reader(datafile3, delimiter=','))
for row in datareader1:
  rgbCMF1.append(row)
for row in datareader2:
  rgbCMF2.append(row)
for row in datareader3:
  rgbCMF3.append(row)
rgbCMF.append(rgbCMF1)
rgbCMF.append(rgbCMF2)
rgbCMF.append(rgbCMF3)
rgbCMF = [[[float(re.sub("[^0-9.E-]", "", i)) for i in j] for j in k] for k in rgbCMF]

# 2. HELPER FUNCTIONS
def cameraSensitivity(rgbCMF):
  w, h = 28, 99
  y = [[0 for x in range(w)] for y in range(h)] 
  redS = rgbCMF[0]
  greenS = rgbCMF[1]
  blueS = rgbCMF[2]
  for i in range(0, 28):
    for j in range(0, 33):
      y[j][i] = redS[j][i]/sum(list(map(list, zip(*redS)))[i])
    for j in range(33, 66):
      y[j][i] = greenS[j - 33][i]/sum(list(map(list, zip(*greenS)))[i])
    for j in range(66, 99):
      y[j][i] = blueS[j - 66][i]/sum(list(map(list, zip(*blueS)))[i])
  return y

def computeLightColor(e, Sr, Sg, Sb):
  return [sum(np.multiply(Sr, e)), sum(np.multiply(Sg, e)), sum(np.multiply(Sb, e))]

def computeSingleLayer(mu_a, mu_s, d):
  K = np.sqrt(mu_a * (mu_a + 2*mu_s))
  beta = np.sqrt(mu_a / (mu_a + 2*mu_s))
  R = (1 - beta**2)*(np.exp(K*d)-np.exp(-1*K*d))/(((1 + beta)**2)*np.exp(K*d)-((1 - beta)**2)*np.exp(-1*K*d))
  T = 4*beta/(((1 + beta)**2)*np.exp(K*d)-((1 - beta)**2)*np.exp(-1*K*d))
  return R, T

def fromRawTosRGB(imWB, T_RAW2XYZ):
  Ix = T_RAW2XYZ[0][0][0]*imWB[0] + T_RAW2XYZ[0][0][1]*imWB[1] + T_RAW2XYZ[0][0][2]*imWB[2]
  Iy = T_RAW2XYZ[0][0][3]*imWB[0] + T_RAW2XYZ[0][0][4]*imWB[1] + T_RAW2XYZ[0][0][5]*imWB[2]
  Iz = T_RAW2XYZ[0][0][6]*imWB[0] + T_RAW2XYZ[0][0][7]*imWB[1] + T_RAW2XYZ[0][0][8]*imWB[2]
  Ixyz = [Ix, Iy, Iz]
  
  Txyzrgb = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.057]
  R = Txyzrgb[0]*Ixyz[0] + Txyzrgb[1]*Ixyz[1] + Txyzrgb[2]*Ixyz[2]
  G = Txyzrgb[3]*Ixyz[0] + Txyzrgb[4]*Ixyz[1] + Txyzrgb[5]*Ixyz[2]
  B = Txyzrgb[6]*Ixyz[0] + Txyzrgb[7]*Ixyz[1] + Txyzrgb[8]*Ixyz[2]
  return [R, G, B]

def preparedModel(pheomelanin_ext, eumelanin_ext, deoxy_hemo_ext_coeff, oxy_hemo_ext_coeff):
  pheomelanin_concentration = 12
  melanin_concentration = 80
  thickness_epidermis = 0.02
  wavelength = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720]
  inv_wavelength = [1/w for w in wavelength]
  eumelanin_proportion = 0.61
  thickness_papillary_dermis = 0.3
  f_oxy = 0.75
  g = 66500
  c_hemo = 150
  ma_eumelanin = [melanin_concentration*thickness_epidermis*x for x in eumelanin_ext]
  ma_pheomelanin = [pheomelanin_concentration*thickness_epidermis*y for y in pheomelanin_ext]
  mu_a_melanin = [sum(aa) for aa in zip([eumelanin_proportion*z for z in ma_eumelanin], [(1 - eumelanin_proportion)*a for a in ma_pheomelanin])]
  m_oxy = [2.303*f_oxy*c_hemo/g*b for b in oxy_hemo_ext_coeff]
  m_deoxy = [2.303*(1 - f_oxy)*c_hemo/g*c for c in deoxy_hemo_ext_coeff]
  mu_a_blood = [sum(x) for x in zip(m_oxy, m_deoxy)]
  musp_Rayleigh = 2*(10**12)*(np.power(inv_wavelength, 4))
  musp_Mie = 2*(10**5)*(np.power(inv_wavelength, 1.5))
  musp_total = musp_Mie + musp_Rayleigh
  return wavelength, mu_a_melanin, m_oxy, m_deoxy, musp_total, thickness_papillary_dermis, ma_eumelanin, ma_pheomelanin, mu_a_blood

def skinModel(melanosomes_melanin, c_hemoglobin, wavelength, mu_a_melanin, m_oxy, m_deoxy, musp_total, d, mu_a_blood):
  C_he = c_hemoglobin/4
  Ye = 3/4
  abs_baseline = 0.0244 + (8.53*np.exp(([(154 - g)/66.2 for g in wavelength])))
  intermediate_step = [sum(z) for z in zip([Ye*h for h in m_oxy], [(1 - Ye)*j for j in m_deoxy])]
  m_a_epidermis = [sum(w) for w in zip([f*melanosomes_melanin for f in mu_a_melanin], [C_he*e for e in intermediate_step], [k*(1 - melanosomes_melanin - C_he) for k in abs_baseline])]
  transmittance = [np.exp(-1*m) for m in m_a_epidermis]

  mu_a_dermis = [sum(y) for y in zip([c_hemoglobin*k for k in mu_a_blood], [(1 - c_hemoglobin)*l for l in abs_baseline])]
  R_pderm, T_pderm = computeSingleLayer(mu_a_dermis, musp_total, d)
  return ([n**2 for n in transmittance])*R_pderm

def whiteBalance(rawAppearance, lightcolor):
  WBrCh = rawAppearance[0]/lightcolor[0]
  WBgCh = rawAppearance[1]/lightcolor[1]
  WBbCh = rawAppearance[2]/lightcolor[2]
  return [WBrCh, WBgCh, WBbCh]

# 3. IMPLEMENTATION
def sRGB_generation():
  wavelength_proper, mu_a_melanin, m_oxy, m_deoxy, musp_total, thickness_papillary_dermis, ma_eumelanin, ma_pheomelanin, mu_a_blood = preparedModel(pheomelanin_ext, eumelanin_ext, deoxy_hemo_ext_coeff, oxy_hemo_ext_coeff)
  Y = cameraSensitivity(rgbCMF)
  wavelength = 33
  dim = 256
  maxmelanin = 0.43
  maxhemoglobin = 0.07
  u = np.linspace(0.01, 1, dim)
  melaninvalues = 0.43 * u
  hemoglobinvalues = 0.07 * u

  Sr = np.reshape(list(map(list, zip(*Y[0:wavelength])))[0], (33, 1))
  Sg = np.reshape(list(map(list, zip(*Y[wavelength:wavelength*2])))[0], (33, 1))
  Sb = np.reshape(list(map(list, zip(*Y[wavelength*2:wavelength*3])))[0], (33, 1))
  e = np.reshape(illumA, (33, 1))
  e = e/sum(e)

  melanin, hemoglobin = np.meshgrid(melaninvalues, hemoglobinvalues)
  ar_row = len(melanin)
  ar_cl = len(melanin[0])
  a = np.zeros((1, dim))
  SpectralReflectance = np.zeros((ar_row, ar_cl, 33))
  rgbim = np.zeros((ar_row, ar_cl, 3))
  timer = 0
  for row in range(0, ar_row):
    timer = timer + 1
    for col in range(0, ar_cl):
      SpectralReflectance[row][col] = skinModel(melanin[row][col], hemoglobin[row][col], wavelength_proper, mu_a_melanin, m_oxy, m_deoxy, musp_total, thickness_papillary_dermis, mu_a_blood)
    if timer % (dim//3) == 0:
      print(str(100*timer/dim) + "% completed preprocessing")
  skincolor = SpectralReflectance*np.reshape([3*q for q in e], (1, 1, 33))
  rCh = np.zeros((dim, dim, 1))
  gCh = np.zeros((dim, dim, 1))
  bCh = np.zeros((dim, dim, 1))
  for jj in range(wavelength):
    rCh[:, :, 0] = rCh[:, :, 0] + skincolor[:, :, jj] * np.reshape(Sr, (1, 1, 33))[:, :, jj]
    gCh[:, :, 0] = gCh[:, :, 0] + skincolor[:, :, jj] * np.reshape(Sg, (1, 1, 33))[:, :, jj]
    bCh[:, :, 0] = bCh[:, :, 0] + skincolor[:, :, jj] * np.reshape(Sb, (1, 1, 33))[:, :, jj]
  Iraw = [rCh, gCh, bCh]
  Iraw = np.array(Iraw).squeeze()

  lightColor = computeLightColor(e, Sr, Sg, Sb)
  ImwhiteBalanced = whiteBalance(Iraw, lightColor)
  S = [Sr, Sg, Sb]
  S = np.array(S).squeeze()
  T = np.linalg.lstsq(list(map(list, zip(*S))), XYZspace)[0]
  T = list(map(list, zip(*T)))
  T[0] = [0.3253*xx/sum(T[0]) for xx in T[0]]
  T[1] = [0.3425*yy/sum(T[1]) for yy in T[1]]
  T[2] = [0.3723*zz/sum(T[2]) for zz in T[2]]
  T_RAW2XYZ = np.reshape(T, (1, 1, 9))
  RGBim = fromRawTosRGB(ImwhiteBalanced, T_RAW2XYZ)
  scaleSRGBim = [[[max(0, min(1, u)) for u in v] for v in w] for w in RGBim]
  sRGBim = [[[p**(1/2.2) for p in q] for q in r] for r in scaleSRGBim]
  sRGBim = np.swapaxes(np.transpose(np.array(sRGBim)), 0, 1)
  plt.imshow(sRGBim).get_figure().savefig('sanity_check.jpg')
  return sRGBim
