#https://github.com/Twinklebear/tray_rust/blob/master/src/bxdf/torrance_sparrow.rs

import numpy as np

# TODO: Additional parameters that need fixing

# width of beckmann microfacet distribution (roughness param from Donner 2006 paper)
beck_width = 0.35

# Refractive index of the material the light is coming from (air most likely)
eta_i = 1

# Refractive index of the material the light is hitting/entering (epidermis most likely)
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
    return exp(-tan_sqr/width_sqr)/(math.pi * width_sqr * cos_theta_4)

# Monodirectional shadowing helper function
def monodir_shadowing(v, w_h):
    cos_sqr = w_h[2] * w_h[2]
    sin_sqr = max(0, 1 - cos_sqr)
    a = 1 / (beck_width * abs(sqrt(sin_sqr / cos_sqr)))
    if a < 1.6:
        a_sqr = a * a
        return (3.535 * a + 2.181 * a_sqr)/(1 + 2.276 * a_2.577 * a_sqr)
    else:
        return 1

# Geomtry function (using monodirectional shadowing function)
def geom(w_i, w_o, w_h):
    ret = 1
    for w in [w_i, w_o}:
        ret *= monodir_shadowing(w, w_h)
    return ret
        
# Dielectric fresnel coefficients helper function
def dielectric(ci, ct, ei, et):
    r_par = (et * ci - ei * ct)/(et * ci + ei * ct)
    r_perp = (ei * ci - et * ct)/(ei * ci + et * ct)
    return [0.5 * (r_par * r_par + r_perp * r_perp) for i in range(4)]

# Fresnel reflectance function. My own implementation using the fresnel equations (https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf; eq. 22)
def fres(cos_i):
    cos_i = abs(cos_i)
    g = sqrt((eta_t**2)/(eta_i**2) - 1 + cos_i**2)
    first_term = 0.5 * ((g - c)**2)/((g + c)**2)
    second_term = 1 + (((c * (g + c) - 1)**2)/((c * (g - c) + 1)**2))

    return first_term * second_term
    
# Fresnel reflectance function for dielectrics (skin?) Implemented using the rust code from github
def fres_rust(cos_i):
    if cos_i < -1:
        ci = -1
    elif cos_i > 1:
        ci = 1
    else:
        ci = cos_i

    if ci > 0:
        ei = eta_i
        et = eta_t
    else:
        ei = eta_t
        et = eta_i

    sin_t = ei / et * sqrt(max(0, 1 - ci*ci))

    if sin_t >= 1:
        return [0]*4
    else:
        ct = sqrt(max(0, 1 - sin_t * sin_t))
        return dielectric(abs(ci), ct, ei, et)

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
    return (p_s * f * g * d)/(4 * cos_to * cos_ti)


def main():
    pass
