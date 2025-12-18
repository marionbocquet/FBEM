# Source: Microwave Radar and Radiometric Remote Sensing, http://mrs.eecs.umich.edu
# These MATLAB-based computer codes are made available to the remote
# sensing community with no restrictions. Users may download them and
# use them as they see fit. The codes are intended as educational tools
# with limited ranges of applicability, so no guarantees are attached to
# any of the codes. 
#
# Original FBEM code : I2EM_Backscatter_model.m
# Converted to Python by: Marion Bocquet

import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad
from scipy.constants import pi
from scipy.stats import norm
from numpy import sin, cos, tan, sqrt, exp, log10
from math import factorial
from scipy import special



def I2EM_Backscatter_model(fr, sig, L, thi, er, sp, xx):

    """
    
    Description:
    Code 10.1: I2EM Backscattering from Single-Scale Random Surface
    Code computes sigma_0_vv, sigma_0_hh, and sigma_0_hv for 
    single-scale random surface with a specified correlation function

    Input variables:
    fr : frequency (GHz)
    sig : rms height (cm)
    L : correlation length (cm)
    thi : incidence angle (degrees)
    er : complex dielectric constant of the scattering medium
    sp : type of correlation function: 1- exponential, 2- Gaussian, 3- x-power
    xx : coefficient for the x-power correlation function

    Output products:
    sigma_0_vv : backscattering coefficient for VV polarization (dB)
    sigma_0_hh : backscattering coefficient for HH polarization (dB)
    sigma_0_hv : backscattering coefficient for HV polarization (dB)

    Book reference: Section 10-3.9

    """
    
    #--the code calls two other functions. The first is the I2EM_Bistatic_model
    #code which is used to calculate the co-polarized responses in backscatter.
    #The second is the IEMX_model used to calculate the cross-polarized
    #response. The later code is slow as it involved double integrations.


    #--- The co-pol components:
    ths = np.copy(thi)
    phs = 180
    sigma_0_vv, sigma_0_hh = I2EM_Bistat_model(fr, sig, L, thi, phs, ths, er, sp, xx)


    #--- The cross-pol component:
    auto = 1 
    # auto = 1 allows for automatic selection of the number of spectral components 
    # auto = 0 forces the number of spectral components to be equal to 15 always.
    # selection of auto = 1 results in a slower code execution

    sigma_0_hv = IEMX_model(fr, sig, L, thi, er, sp, xx, auto)

    return sigma_0_vv, sigma_0_hh, sigma_0_hv



def I2EM_Bistat_model(fr, sig, L, thi, phs, ths, er, sp, xx):
    
    """
    Description:
    Code 10.3: I2EM Bistatic Scattering from Single-Scale Random Surface
    Code computes sigma_0_vv(thi, ths,phs) and sigma_0_hh(thi, ths,phs) 
    for single-scale random surface

    Input variables:
    er : complex dielectric constant of the scattering medium
    thi: incidence angle (deg)
    ths: scattering angle (deg)
    phs: Relative azimuth angle (deg)
    sp: type of correlation function: 1- exponential, 2- Gaussian, 
        3- x-power, 4- x-exponential
    xx: coefficient (>1) needed for the x-power and x-exponential correl. fnc.
    sig: rms height (m)
    L: correlation length (m)
    fr: frequency (GHz)

    Output products:
    sigma_0_vv : bistatic backscattering coefficient for VV polarization (dB)
    sigma_0_hh : bistatic backscattering coefficient for HH polarization (dB)
    sigma_0_hv : bistatic backscattering coefficient for HV polarization (dB)

    Book reference: Section 10-3.9
    """
    error = 1.0e8
    sig = np.copy(sig) / 100
    L = np.copy(L) / 100
    mu_r = 1.0 # Relativ permeability

    k = 2 * pi * fr / 30 # wavenumber in free space. Speed of light is in cm/s
    theta = thi * pi / 180 # transform to radian
    phi = 0
    thetas = ths * pi / 180 # transform to radian
    phis = phs * pi / 180 # transform to radian

    ks = k * sig # roughness parameter
    kl = k * L 

    ks2 = ks**2

    cs = cos(theta + 0.01)
    s = sin(theta + 0.01)

    sf = sin(phi)
    cf = cos(phi)

    ss = sin(thetas)
    css = cos(thetas)

    cfs = cos(phis)
    sfs = sin(phis)

    s2 = s**2
    #sq = sqrt(er-s2)

    kx = k * s * cf
    ky = k * s * sf
    kz = k * cs

    ksx = k * ss * cfs
    ksy = k * ss * sfs
    ksz = k * css

    # Reflection coefficients
    rt = sqrt(er - s2)
    Rvi = (er * cs - rt) / (er * cs + rt)
    Rhi = (cs - rt) / (cs + rt)

    wvnb = k * sqrt((ss * cfs - s * cf )**2 + (ss * sfs - s * sf)**2)

    Ts = 1 

    while error > 1.0e-8:
        Ts = Ts + 1
        error = ( ks2 * (cs + css)**2 )**Ts / factorial(Ts)

    # ------- calculating roughness spectrum --------

    wn, rss = roughness_spectrum(sp, xx, wvnb, sig, L, Ts)

    # ------- compute R- transition --------

    Rv0 = (sqrt(er) - 1) / (sqrt(er) + 1)
    Rh0 = -Rv0

    Ft = 8 * Rv0**2 * ss *(cs + sqrt(er - s2))/(cs * sqrt(er - s2))
    a1 = 0
    b1 = 0

    for n in range(1, Ts+1):
        a0 = (ks * cs)**(2 * n) / factorial(n)
        a1 = a1 + a0 * wn[n-1]
        b1 = b1 + a0 * (abs(Ft/2 + 2**(n+1) * Rh0/cs * exp(-(ks*cs)**2)))**2 * wn[n-1]

    St = 0.25 * abs(Ft)**2 * a1 / b1
    St0 = 1 / (abs(1 + 8 * Rv0/(cs * Ft)))**2

    Tf = 1 - St / St0

    #----------- compute average reflection coefficients ------------
    #-- these coefficients account for slope effects, especially near the
    #brewster angle. They are not important if the slope is small.

    sigx = 1.1 * sig / L
    sigy = sigx
    xxx = 3 * sigx

    """Rav, err = dblquad(lambda y, x, cs, s, er, s2, sigx, sigy: Rav_integration(x, y, cs, s, er, s2, sigx, sigy),
                       -xxx, xxx, 
                       lambda x: -xxx, 
                       lambda x: xxx,    
                       args=(cs, s, er, s2, sigx, sigy))
    """
    
    Rav, err = dblquad(Rav_integration_wrapped, 
                       -xxx, xxx, # bound for Zx
                       lambda x: -xxx,  # lower bound for Zy
                       lambda x: xxx,   # upper bound for Zy
                       args=(cs, s, er, s2, sigx, sigy))
    
    Rah, err = dblquad(Rah_integration_wrapped, 
                       -xxx, xxx, # bound for Zx
                       lambda x: -xxx,  # lower bound for Zy
                       lambda x: xxx,   # upper bound for Zy
                       args=(cs, s, er, s2, sigx, sigy))

    Rav = np.copy(Rav) /(2*pi * sigx * sigy)
    Rah = np.copy(Rah) /(2*pi * sigx * sigy)

    #-- select proper reflection coefficients

    if (thi == ths) and (phs==180): #i.e. operating in backscatter mode
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

    else:        
        # in this case, it is the bistatic configuration and average R is used
        #     Rvt = Rav + (Rv0 - Rav) .* Tf
        #     Rht = Rah + (Rh0 - Rah) .* Tf
        Rvt = np.copy(Rav)
        Rht = np.copy(Rah)

    fvv = 2 * Rvt *(s * ss - (1 + cs * css) * cfs) / (cs + css)
    fhh = -2 * Rht *(s * ss - (1 + cs * css) * cfs) /(cs + css)

    #------- Calculate the Fppup(dn) i(s) coefficients ----
    Fvvupi, Fhhupi = Fppupdn_is_calculations(+1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    Fvvups, Fhhups = Fppupdn_is_calculations(+1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    Fvvdni, Fhhdni = Fppupdn_is_calculations(-1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    Fvvdns, Fhhdns = Fppupdn_is_calculations(-1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)

    qi = k * cs
    qs = k * css

    #----- calculating  Ivv and Ihh ----

    Ivv = np.zeros((Ts, 1))
    Ihh = np.copy(Ivv)
        
    for n in range(1, Ts+1):
        Ivv[n-1] = ((kz + ksz)**n * fvv * exp(-sig**2 * kz * ksz) + 
                    0.25 * (Fvvupi * (ksz - qi)**(n-1) * exp(-sig**2 * (qi**2 - qi * (ksz - kz))) 
                            + Fvvdni * (ksz + qi)**(n-1) * exp(-sig**2 *(qi**2 + qi * (ksz - kz)))
                            + Fvvups *(kz + qs)**(n-1) * exp(-sig**2 *(qs**2 - qs * (ksz - kz)))
                            + Fvvdns *(kz - qs)**(n-1) * exp(-sig**2 *(qs**2 + qs * (ksz - kz)))))

        
        Ihh[n-1] = ((kz + ksz)**n * fhh * exp(-sig**2 * kz * ksz) + 
                    0.25 * (Fhhupi *(ksz - qi)**(n-1) * exp(-sig**2 * (qi**2 - qi*(ksz - kz)))
                            + Fhhdni *(ksz + qi)**(n-1) * exp(-sig**2 *(qi**2 + qi*(ksz - kz)))
                            + Fhhups *(kz + qs)**(n-1) * exp(-sig**2 *(qs**2 - qs*(ksz - kz)))
                            + Fhhdns *(kz - qs)**(n-1) * exp(-sig**2 *(qs**2 + qs*(ksz - kz)))))
    
    #-- Shadowing function calculations

    if (thi==ths) and (phs==180): #i.e. working in backscatter mode
        ct = cot(theta)
        cts = cot(thetas)
        rslp = np.copy(rss)
        ctorslp = ct / sqrt(2) / rslp
        ctsorslp = cts / sqrt(2) / rslp
        shadf = 0.5 * (exp(-ctorslp**2) / sqrt(pi)/ctorslp - special.erfc(ctorslp))
        shadfs = 0.5 * (exp(-ctsorslp**2) / sqrt(pi)/ctsorslp - special.erfc(ctsorslp))
        ShdwS = 1/(1 + shadf + shadfs)
    else:
        ShdwS = 1
    
    #------- calculate the values of sigma_note --------------

    sigmavv = 0
    sigmahh = 0

    for n in range(1, Ts+1):
        a0 = wn[n-1] / factorial(n) * sig**(2*n)
        
        sigmavv = sigmavv + abs(Ivv[n-1])**2 * a0
        sigmahh = sigmahh + abs(Ihh[n-1])**2 * a0


    sigmavv = sigmavv * ShdwS * k**2 /2 * exp(-sig**2 * (kz**2 + ksz**2))
    sigmahh = sigmahh * ShdwS * k**2 /2 * exp(-sig**2 * (kz**2 + ksz**2))  

    sigma_0_vv = 10 * log10(sigmavv)
    sigma_0_hh = 10 * log10(sigmahh) 

    return sigma_0_vv, sigma_0_hh


def IEMX_model(fr, sig, L, theta_d, er, sp, xx, auto):
    

    # - fr: frequency in GHz
    # - sig: rms height of surface in cm
    # - L: correlation length of surface in cm
    # - theta_d: incidence angle in degrees
    # - er: relative permittivity
    # - sp: type of surface correlation function

    sig = sig * 100 ; # change to cm scale
    L = L * 100; # change to cm scale;

    error = 1.0e8

    k = 2 * pi * fr / 30  # wavenumber in free space. Speed of light is in cm/s
    theta = theta_d * pi / 180  # transform to radian
    
    ks = k * sig  # roughness parameter
    kl = k * L

    ks2 = ks**2
    kl2 = kl**2

    cs = cos(theta)
    s = sin(theta + 0.001)
    s2 = s**2

    #-- calculation of reflection coefficints
    rt = sqrt(er - s2)

    rv = (er *cs - rt) /(er*cs +rt)
    rh = (cs - rt)/(cs + rt)

    rvh = (rv - rh) /2

    #-- rms slope values
    sig_l = sig / L
    if sp==1:                #-- exponential correl func
        rss = sig_l
    
    if sp==2:                #-- Gaussian correl func
        rss = sig_l * sqrt(2)
    

    if sp==3:                #-- 1.5-power spectra correl func
        rss = sig_l * sqrt(2*xx)
    

    #--- Selecting number of spectral components of the surface roughness
    if auto == 0: 
        n_spec = 15 # number of terms to include in the surface roughness spectra


    if auto == 1:
        n_spec = 1 
        while error > 1.0e-8:
            n_spec = n_spec + 1
            error = (ks2 *(2*cs)**2 )**n_spec / factorial(n_spec); 
   

    #-- calculating shadow consideration in single scat (Smith, 1967)

    ct = cot(theta + 0.001)
    farg = ct / sqrt(2) / rss
    gamma = 0.5 * (exp(-farg**2) / 1.772 / farg - special.erfc(farg))
    Shdw = 1 / (1 + gamma)

    #-- calculating multiple scattering contribution
    #------ a double integration function


    # Need to be phi first in python (y, and x afterward)
    """svh, err = dblquad(lambda phi, r: xpol_integralfunc(r, phi, sp, xx, ks2, cs, s, kl2, L, er, rss, rvh, n_spec), 
                       0.1, 1, # bound for r 
                       lambda r: 0, # lower bound for phi 
                       lambda r: np.pi # higher bound for phi
                       )"""
    svh, err = dblquad(xpol_integralfunc_wrapped, 
                       0.1, 1, # bound for r   
                       lambda r: 0.0, lambda r: np.pi,    
                       args=(sp, xx, ks2, cs, s, kl2, L, er, rss, rvh, n_spec))
    
    sigvh = 10 * log10(svh * Shdw)

    return sigvh

def Rav_integration_wrapped(Zx, Zy, *params):
    return Rav_integration(Zy, Zx, *params)

def Rah_integration_wrapped(Zx, Zy, *params):
    return Rah_integration(Zy, Zx, *params)

def xpol_integralfunc_wrapped(phi, r, *params):    
    return xpol_integralfunc(r, phi, *params)

def Fppupdn_is_calculations(ud, iis, Rvi,Rhi,er,k,kz,ksz,s,cs,ss,css,cf,cfs,sfs):

    if iis==1:
        Gqi = ud * kz
        Gqti = ud * k * sqrt(er - s**2)
        qi = ud * kz
        
        c11 = k * cfs * (ksz - qi)

        c21 = (cs * (cfs * (k**2 * s * cf * (ss * cfs - s * cf) 
                            + Gqi * (k * css - qi)) 
                            + k**2 * cf  * s * ss * sfs**2))
        c31 = (k * s * (s * cf * cfs * (k * css - qi) 
                        - Gqi * (cfs * (ss * cfs - s * cf) 
                                 + ss * sfs**2)))
        
        c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))
        
        c12 = k * cfs *(ksz - qi)

        c22 = (cs * (cfs * (k**2 * s * cf * (ss * cfs - s * cf) 
                            + Gqti * (k * css - qi)) 
                            + k**2 * cf * s * ss * sfs**2))
        
        c32 = (k * s * (s * cf * cfs * (k * css - qi) 
                        - Gqti * (cfs * (ss * cfs - s * cf) 
                                  - ss * sfs**2)))
        
        c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))    


    if iis==2:
        Gqs = ud * ksz
        Gqts = ud * k * sqrt(er - ss**2)
        qs = ud * ksz
        
        c11 = k * cfs * (kz + qs)
        c21 = (Gqs * (cfs * (cs * (k * cs + qs) 
                             - k * s * (ss * cfs - s * cf)) 
                             - k * s * ss * sfs**2))
        c31 = k * ss * (k * cs * (ss * cfs - s * cf)+ s *(kz + qs))
        c41 = (k * css * (cfs * (cs * (kz + qs)
                                 -k * s * (ss * cfs - s* cf))
                                 -k * s * ss * sfs**2))
        c51 = -css * (k**2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))
            
        c12 = k * cfs * (kz + qs)
        c22 = (Gqts * (cfs * (cs * (kz + qs) 
                              - k * s * (ss * cfs - s * cf)) 
                              - k * s * ss * sfs**2))
        
        c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c42 = (k * css * (cfs * (cs * (kz + qs) 
                                 - k * s * (ss * cfs - s * cf)) 
                                 - k * s * ss * sfs**2))
        c52 = -css * (k**2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))


    q = np.copy(kz)
    qt = k * sqrt(er - s**2)

    vv = ((1 + Rvi) * (-(1 - Rvi) * c11 /q + (1 + Rvi) * c12 / qt)
          + (1 - Rvi) * ((1 - Rvi) * c21 /q - (1 + Rvi) * c22 / qt) 
          + (1 + Rvi) * ((1 - Rvi) * c31 /q - (1 + Rvi) * c32 / er /qt) 
          + (1 - Rvi) * ((1 + Rvi) * c41 /q - er * (1 - Rvi) * c42 / qt) 
          + (1 + Rvi) * ((1 + Rvi) * c51 /q - (1 - Rvi) * c52 / qt))

    hh = ((1 + Rhi) *((1 - Rhi) * c11 /q - er *(1 + Rhi) * c12 / qt) 
        - (1 - Rhi) * ((1 - Rhi) * c21 /q - (1 + Rhi) * c22 / qt) 
        - (1 + Rhi) *((1 - Rhi) * c31 /q - (1 + Rhi) * c32 / qt) 
        - (1 - Rhi) * ((1 + Rhi) * c41 /q - (1 - Rhi) * c42 / qt) 
        - (1 + Rhi) *( (1 + Rhi) * c51 /q - (1 - Rhi) * c52 / qt))


    return vv, hh

def cot(x): 
    return 1/np.tan(x)

def Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy):
    
    A = cs + Zx * s
    B = er * (1 + Zx**2 + Zy**2)
    CC = s2 - 2*Zx *s *cs + Zx**2 * cs**2 + Zy**2

    Rv = (er*A - sqrt(B-CC))/(er*A + sqrt(B-CC))

    pd = exp(-Zx**2 /(2*sigx**2) -Zy**2 /(2*sigy**2))
    Rav = Rv * pd

    return Rav

def Rah_integration(Zx, Zy, cs,s, er, s2, sigx, sigy):

    A = cs + Zx * s
    B = er * (1 + Zx**2 + Zy**2)
    CC = s2 - 2*Zx *s *cs + Zx**2 * cs**2 + Zy**2

    Rh = (A - sqrt(B-CC))/(A + sqrt(B-CC))

    pd = exp(-Zx**2/(2*sigx**2) -Zy**2/(2*sigy**2))
    Rah = Rh * pd

    return Rah

def x_exponential_spectrum(z, wvnb, L, n, xx):
    tmp = np.exp(-np.abs(z)**xx) * special.jv(0, z * wvnb * L / (n**(1/xx))) * z
    return tmp

def xpol_integralfunc(r, phi, sp, xx, ks2, cs,s, kl2, L, er, rss, rvh, n_spec):

    cs2 = cs**2

    r2 = r**2
    nr = len(r)

    sf = sin(phi)
    csf = cos(phi)
    rx = r * csf
    ry = r * sf

    #-- calculation of the field coefficients
    rp = 1 + rvh
    rm = 1 - rvh 

    q = sqrt(1.0001 - r2)
    qt = sqrt(er - r2)

    a = rp /q
    b = rm /q
    c = rp /qt
    d = rm /qt

    #--calculate cross-pol coefficient
    B3 = rx * ry /cs
    fvh1 = (b-c)*(1- 3 * rvh) - (b - c/er) * rp
    fvh2 = (a-d)*(1+ 3 * rvh) - (a - d * er) * rm
    Fvh = (abs((fvh1 + fvh2) * B3))**2


    #-- calculate shadowing func for multiple scattering 
    au = q /r /1.414 /rss
    fsh = (0.2821/au) * exp(-au**2) -0.5 *(1- erf(au))
    sha = 1./(1 + fsh)

    #-- calculate expressions for the surface spectra
    wn = spectrm1(sp, xx, kl2, L, rx, ry, s, n_spec, nr)
    wm = spectrm2(sp, xx, kl2, L, rx, ry, s, n_spec, nr)


    #--compute VH scattering coefficient
    acc = exp(-2* ks2 * cs2) / (16 * pi)
    vhmnsum = np.zeros((1, nr))
    for n in range(1, n_spec+1):
        for m in range(1, n_spec+1):
            vhmnsum = (vhmnsum + wn[n-1, :] * wm[m-1, :] 
                       * (ks2*cs2)**(n+m) / factorial(n) / factorial(m))

    VH = 4 * acc * Fvh * vhmnsum *r
    y = VH * sha

    return y 

def spectrm1(sp, xx, kl2, L, rx, ry, s, np, nr):

    wn = np.zeros(np, nr)

    if sp == 1:  # exponential 
        for n in range(1, np + 1):       
            wn[n-1, :] = n * kl2 /(n**2 + kl2 *((rx - s)**2 + ry**2))**1.5

    if sp == 2:  #  gaussian
        for n in range(1, np + 1):
            wn[n-1, :] = 0.5 * kl2 /n * exp(-kl2*((rx - s)**2 + ry**2)/(4 * n)) ;

    if sp== 3: # x-power
        for n in range(1, np + 1):
            wn[n-1, :] = (kl2 /(2**(xx * n - 1) * special.gamma(xx * n)) 
                          * (((rx - s)**2 + ry**2)*L)**(xx * n - 1) 
                          * special.kv(-xx * n + 1, L * ((rx - s)**2 + ry**2)))
    return wn

def spectrm2(sp, xx, kl2, L, rx, ry, s, np, nr):

    wm = np.zeros(np, nr)

    if sp == 1: # exponential
        for n in range(1, np + 1):
            wm[n-1, :] = n * kl2 /(n**2 + kl2 * ((rx + s)**2+ry**2))**1.5

    if sp == 2:  #  gaussian
        for n in range(1, np + 1):
            wm[n-1, :] = 0.5 * kl2/ n * exp(-kl2 * ((rx + s)**2 + ry**2)/(4 * n))

    if sp== 3: # x-power
        for n in range(1, np + 1):
            wm[n-1, :] = (kl2 / (2**(xx * n-1) * special.gamma(xx * n)) 
                          * (((rx + s)**2 + ry**2) * L)**(xx * n-1) 
                          * special.kv(-xx * n + 1, L * ((rx + s)**2 + ry**2)))

    return wm


def roughness_spectrum(sp, xx, wvnb, sig, L, Ts):

    wn = np.zeros((Ts, 1))  

    #-- exponential correl func
    if sp == 1:
        for n in range(1, Ts+1):
            wn[n-1] = L**2 / n**2 * (1 + (wvnb * L/n)**2)**(-1.5)
        rss = sig / L

    #-- gaussian correl func
    if sp == 2:
        for n in range(1, Ts+1):
            wn[n-1] = L**2 / (2 * n) * np.exp(-(wvnb * L)**2 / (4 * n))
        rss = np.sqrt(2) * sig / L

    #-- x-power correl func
    if sp == 3:
        for n in range(1, Ts+1):
            if wvnb == 0:
                wn[n-1] = L**2 / (3 * n - 2)
            else:
                wn[n-1] = (L**2 * (wvnb * L)**(-1 + xx * n) 
                           * special.kv(1 - xx * n, wvnb * L) 
                           / (2**(xx * n - 1) * special.gamma(xx * n)))

    #    if xx == 1.5
    #        rss  = sqrt(xx *2) *sig ./L;
    #    else
    #        rss = 0;
    #    end


    #-- x- exponential correl func

    if sp == 4:
        for n in range(1, Ts+1):
            tmp, err = quad(lambda z: x_exponential_spectrum(z, wvnb, L, n, xx), 0, 9)
            wn[n-1] = L**2 / n**(2/xx) * tmp
        rss  =  sig / L


    return wn, rss