import numpy as np 
import warnings
from scipy.interpolate import CubicSpline
from scipy.constants import pi
from Scattering_Models.I2EM_Backscatter_model import I2EM_Backscatter_model
from Scattering_Models.RelDielConst_PureWater import RelDielConst_PureWater

def pond_backscatter(Lambda, T_fw, beta_c, u_a):
    """ 
    Models the backscattering coefficient from a melt pond surface, and
    # interpolates the scattering signature to a spline function

    # Input:
    # Lambda = radar wavelen, m
    # T_fw = freshwater temperature, C
    # beta_c = Effective width of angular extent of coherent component (TUNING PARAMETER)
    # u_a = wind speed, m/s

    # Output:
    # theta = angular sampling of the scattering signature
    # sigma_0_lead_surf = smoothing spline characterizing the scattering
    # signature of the lead surface, dB


    # Uses the following codes from external sources:
    # RelDielConst_PureWater.m

    # (C) Jack Landy, University of Bristol, 2018

    """
    warnings.filterwarnings('ignore')
    
    ## Empirical relation between wind speed and pond roughness (Scharien et al. 2014)

    sigma_mp = (0.982 * u_a - 3.702)/1000
    l_mp = (2191 * u_a**(-2.621) + 12.79)/1000

    ## Angular Sampling for Scattering Signature

    theta = np.logspace(np.np.log10(1e-6), np.np.log10(pi/2), 200)

    ## Antenna Parameters

    c = 299792458 # speed of light, m/s
    f_c = c/Lambda # radar frequency, Hz
    k = (2 * pi)/Lambda # wavenumber

    # mss_np.exp_mp = (np.sqrt(2/pi)*sigma_mp/l_mp*np.sqrt(5*k*l_mp - atan(5*k*l_mp)))**2 # mean-square slope for np.exponential ACF (Dierking, 2000)

    # Validity criteria for IEM
    # max_sigma = 2/k # maximum viable rms height for IEM
    # 
    # rayleigh = Lambda/(8*np.cos(theta)) # surface = smooth if sigma < rayleigh criterion
    # fraunhofer = Lambda/(32*np.cos(theta)) # surface = smooth if sigma < fraunhofer criterion

    ## Dielectric Properties

    # Freshwater dielectrics
    [epsr_fw,epsi_fw] = RelDielConst_PureWater(T_fw,f_c*1e-9) # permittivity of freshwater
    eps_fw = epsr_fw + 1j*epsi_fw

    # Fresnel reflection coefficients
    epsr_a = 1 # relative permittivity of air

    eta_0 = 376.73031346177 # intrinsic impedance of free space, ohms
    eta_1 = eta_0/np.sqrt(epsr_a)
    eta_2 = eta_0/np.sqrt(np.real(eps_fw))
    theta_2 = np.arcsin(np.sin(theta)*(np.sqrt(epsr_a/np.real(eps_fw)))) # transmission angle

    rho_H = (eta_2*np.cos(theta) - eta_1*np.cos(theta_2))/(eta_2*np.cos(theta) + eta_1*np.cos(theta_2)) # reflection coeff
    rho_V = (eta_2*np.cos(theta_2) - eta_1*np.cos(theta))/(eta_2*np.cos(theta_2) + eta_1*np.cos(theta))# reflection coeff

    # tau_H = 1 + rho_H # transmission coeff
    # tau_V = (1 + rho_V)*(np.cos(theta)/np.cos(theta_2)) # transmission coeff

    gamma_H = rho_H**2 # reflectivity (intensity)
    gamma_V = rho_V**2 # reflectivity (intensity)


    ## Backscattering Coefficient of Snow-Ice Interface, sigma0

    # Calculate coherent vs. incoherent surface scattering ratio
    psi = k*sigma_mp*np.cos(theta) # frequency-dependent roughness parameter
    omega = np.exp(-4*psi**2) # fractional coherent component

    # Calculate coherent reflected backscattering coefficient (linear)
    sigma_0_HH_coh = (gamma_H/beta_c**2)*np.exp(-4*k**2*sigma_mp**2)*np.exp(-theta**2/beta_c**2) # coherent component of backscattering coefficient
    sigma_0_VV_coh = (gamma_V/beta_c**2)*np.exp(-4*k**2*sigma_mp**2)*np.exp(-theta**2/beta_c**2) # coherent component of backscattering coefficient

    # Calculate incoherent surface backscattering coefficient
    # Run single-scattering IEM for relevant range of facet incidence angles
    sigma_0_HH_inc = np.zeros((len(theta),1))
    sigma_0_VV_inc = np.zeros((len(theta),1))
    for i in range(1, len(theta)+1):
        sigma_0_VV_inc[i-1], sigma_0_HH_inc[i-1] = I2EM_Backscatter_model(f_c*1e-9, sigma_mp, l_mp, theta(i)*180/pi, eps_fw, 1, [])[0:2]
    

    # Calculate total co-polarized surface backscattering cofficients,
    # including coherent reflected power
    sigma_0_HH_mp_surf = 10*np.log10(sigma_0_HH_coh + 10**(sigma_0_HH_inc.T/10)*(1-omega)) # H-pol, dB
    sigma_0_VV_mp_surf = 10*np.log10(sigma_0_VV_coh + 10**(sigma_0_VV_inc.T/10)*(1-omega)) # V-pol, dB

    sigma_0_HH_mp_surf[np.isinf(sigma_0_HH_mp_surf)] = np.nan
    sigma_0_VV_mp_surf[np.isinf(sigma_0_VV_mp_surf)] = np.nan
    # Build spline interpolants (assumption that scattering is polarization-independent)
    # dB
    sigma_0_mp_surf = CubicSpline(theta,(sigma_0_HH_mp_surf + sigma_0_VV_mp_surf)/2)

    warnings.filterwarnings('default') 

    
    return theta, sigma_0_mp_surf