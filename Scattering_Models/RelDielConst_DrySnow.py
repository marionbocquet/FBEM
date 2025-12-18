# Source: Microwave Radar and Radiometric Remote Sensing, http://mrs.eecs.umich.edu
# These MATLAB-based computer codes are made available to the remote
# sensing community with no restrictions. Users may download them and
# use them as they see fit. The codes are intended as educational tools
# with limited ranges of applicability, so no guarantees are attached to
# any of the codes. 
# MATLAB Code: RelDielConst_DrySnow.m
# Translated to python by Marion Bocquet
    
import numpy as np
from Scattering_Models.RelDielConst_PureIce import RelDielConst_PureIce 

def RelDielConst_DrySnow(T, Ps, f ):

    """
    Code 4.5: Relative Dielectric Constant of Dry Snow
    Description: Code computes the real and imaginary parts of the relative
    dielectric constant of Dry Snow
    Input Variables:
        T: Temperature in C
        Ps: Dry Snow Density in g/cm^3
        f: frequency in GHz
    Output Products:
        epsr: real part of relative dielectric constant
        epsi: imaginary part of relative dielectric constant
    Book Reference: Section 4-6
    MATLAB Code: RelDielConst_DrySnow.m

    Example call: [A B] = RelDielConst_DrySnow(T,Ps,f)
    Computes the real and imaginary components of the permitivity of Dry Snow
        %based on the temperature value (T) in degrees C, dry snow density (Ps), and frequency
        %vector (f) and assigns them to vectors A and B respectively
    """


    vi=  Ps /0.9167 

    epsr_ice, epsi_ice = RelDielConst_PureIce(T,f)

    if vi <= 0.45:
        epsr_ds = 1 + 1.4667 * vi + 1.435 * vi**3 # 0<vi<0.45
    
    if vi > 0.45:
        epsr_ds = (1+ 0.4759 * vi)**3  # vi>0.45

    epsi_ds = 0.34 * vi * epsi_ice / (1 - 0.42 * vi)**2

    return epsr_ds, epsi_ds