# Source: Microwave Radar and Radiometric Remote Sensing, http://mrs.eecs.umich.edu
# These MATLAB-based computer codes are made available to the remote
# sensing community with no restrictions. Users may download them and
# use them as they see fit. The codes are intended as educational tools
# with limited ranges of applicability, so no guarantees are attached to
# any of the codes. 
# Original code : RelDielConst_PureIce.m 
# (C) Jack Landy, 2018 (adapted from code for saline water dielectrics of
# Ulaby et al., 2014)
# Translated to Python by Marion Bocquet

import numpy as np


def RelDielConst_PureIce(T, f):
    """
    Code 4.3: Relative Dielectric Constant of Pure Ice
    Description: Code computes the real and imaginary parts of the relative
    dielectric constant of pure ice at any temperature -40C<T<0C and 
    frequency 1<f<1000GHz
    Input Variables:
        %T: Temperature in degree C
        %f: frequency in GHz
    Output Products:
        %epsr: real part of relative dielectric constant
        %epsi: imaginary part of relative dielectric constant
    Book Reference: Section 4-3
    MATLAB Code: RelDielConst_PureIce.m

    Example call: [A B] = RelDielConst_PureIce(T,f)
    Computes the real and imaginary components of the permitivity of pure ice
        based on the temperature value (T) in degrees C and frequency
        vector (f) and assigns them to vectors A and B respectively
    """
    T = T + 273 # represent temperature in Kelvin

    theta = (300/T) - 1

    B1 = 0.0207
    B2 = 1.16e-11
    b = 335

    alpha = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)

    betaM = (B1/T) * np.exp(b/T) / (np.exp(b/T)-1)**2  + B2 * f**2
    delBeta = np.exp(-9.963 + 0.0372 * (T - 273.16))

    beta = betaM + delBeta

    # epsr = 3.1884 + 9.1e-4 * (T - 273) * f / f

    epsr = 3.1884 + 9.1e-4 * (T-273)

    epsi = alpha / f + beta * f
    


    return epsr, epsi