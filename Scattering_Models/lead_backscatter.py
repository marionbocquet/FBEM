import numpy as np 
import warnings
from scipy.interpolate import CubicSpline
from scipy.constants import pi

from Scattering_Models.RelDielConst_SalineWater import RelDielConst_SalineWater

def lead_backscatter(Lambda, sigma_sw, T_sw, S_sw, beta_c):
    
    """
    # Models the backscattering coefficient from a lead surface, and
    # interpolates the scattering signature to a spline function

    # Input:
    # lambda = radar wavelength, m
    # sigma_sw = small-scale rms height of surface, m
    # T_sw = seawater temperature, C
    # S_sw = seawater salinity, psu
    # beta_c = Effective width of angular extent of coherent component (TUNING PARAMETER)

    # Output:
    # theta = angular sampling of the scattering signature
    # sigma_0_lead_surf = smoothing spline characterizing the scattering
    # signature of the lead surface, dB


    # Uses the following codes from external sources:
    # RelDielConst_SalineWater.m

    # (C) Jack Landy, University of Bristol, 2018
    """
    
    warnings.filterwarnings('ignore')

    ## Angular Sampling for Scattering Signature

    theta = np.logspace(np.log10(1e-6), np.log10(pi/2),200)

    ## Antenna Parameters

    c = 299792458 # speed of light, m/s
    f_c = c/Lambda # radar frequency, Hz
    k = (2 * pi)/Lambda # wavenumber

    ## Dielectric Properties

    # Seawater dielectrics
    epsr_sw, epsi_sw = RelDielConst_SalineWater(T_sw, f_c * 1e-9, S_sw) # permittivity of seawater
    eps_sw = epsr_sw + 1j * epsi_sw

    # Fresnel reflection coefficients
    epsr_a = 1 # relative permittivity of air

    eta_0 = 376.73031346177 # intrinsic impedance of free space, ohms
    eta_1 = eta_0/np.sqrt(epsr_a)
    eta_2 = eta_0/np.sqrt(np.real(eps_sw))
    theta_2 = np.arcsin(np.sin(theta)*(np.sqrt(epsr_a/np.real(eps_sw)))) # transmission angle

    rho_H = (eta_2*np.cos(theta) - eta_1*np.cos(theta_2))/(eta_2*np.cos(theta) + eta_1*np.cos(theta_2)) # reflection coeff
    rho_V = (eta_2*np.cos(theta_2) - eta_1*np.cos(theta))/(eta_2*np.cos(theta_2) + eta_1*np.cos(theta))# reflection coeff

    # tau_H = 1 + rho_H # transmission coeff
    # tau_V = (1 + rho_V)*(cos(theta)/cos(theta_2)) # transmission coeff

    #gamma_H = rho_H**2 # reflectivity (intensity)
    #gamma_V = rho_V**2 # reflectivity (intensity)
    gamma_H = np.abs(rho_H)**2
    gamma_V = np.abs(rho_V)**2

    ## Backscattering Coefficient of Snow-Ice Interface, sigma0

    # Calculate coherent vs. incoherent surface scattering ratio
    # psi = k*sigma_sw*cos(theta) # frequency-dependent roughness parameter
    # omega = exp(-4*psi.**2) # fractional coherent component

    # Calculate coherent reflected backscattering coefficient
    sigma_0_HH_coh = ((gamma_H/beta_c**2) 
                      * np.exp(-4 * k**2 * sigma_sw**2) 
                      * np.exp(-theta**2 / beta_c**2)) # coherent component of backscattering coefficient
    
    sigma_0_VV_coh = ((gamma_V/beta_c**2) 
                      * np.exp(-4 * k**2 * sigma_sw**2) 
                      * np.exp(-theta**2 / beta_c**2))# coherent component of backscattering coefficient

    # Calculate total co-polarized backscattering cofficients
    # Incoherent component ignored for radar smooth surface
    sigma_0_HH = 10*np.log10(sigma_0_HH_coh) # H-pol, dB
    sigma_0_VV = 10*np.log10(sigma_0_VV_coh) # V-pol, dB
    sigma_0_HH[np.isinf(sigma_0_HH)] = np.nan
    sigma_0_VV[np.isinf(sigma_0_VV)] = np.nan

    # Build spline interpolants (assumption that scattering is polarization-independent)
    # dB
    y_safe = np.nan_to_num((sigma_0_HH + sigma_0_VV)/2, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma_0_lead_surf = CubicSpline(theta, y_safe)

    warnings.filterwarnings('default') 


    return theta, sigma_0_lead_surf