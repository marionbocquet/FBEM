import numpy as np 
from numpy import sin, cos, tan, sqrt, exp, log10
from scipy.constants import pi
from scipy.interpolate import CubicSpline
from Scattering_Models.RelDielConst_DrySnow import RelDielConst_DrySnow
from Scattering_Models.I2EM_Backscatter_model import I2EM_Backscatter_model 
from Scattering_Models.MieExtinc_DrySnow import MieExtinc_DrySnow
from Scattering_Models.I2EM_Backscatter_model import py_length
from tqdm import tqdm
from joblib import Parallel, delayed

def compute_sigma(theta_i, f_c, sigma_s, l_s, eps_ds):
    """Compute VV, HH, HV backscatter for a single angle theta_i in radians."""
    sigma_vv, sigma_hh, sigma_hv = I2EM_Backscatter_model(
        f_c*1e-9,           # frequency in GHz
        sigma_s,           # rms height in cm
        l_s,               # correlation length in cm
        theta_i*180/np.pi,  # incidence angle in degrees
        eps_ds,             # complex dielectric constant
        1,                  # correlation type
        []                  # additional coefficient
    )
    return sigma_vv, sigma_hh, sigma_hv

def snow_backscatter(Lambda,sigma_s, l_s, T_s, rho_s, r_s, h_s,beta_c):

    """
     Models the backscattering coefficients from the snow-ice interface and
     snow volume, and interpolates scattering signatures to spline functions
     (Suitable only for dry snow)

     Input:
     lambda = radar wavelength, m
     sigma_s = small-scale rms height of interface, m
     l_s = small-scale correlation length of interface, m
     T_s = snow bulk temperature, C
     rho_s = snow bulk density, kg/m**3
     r_s = snow grain size, m
     h_s = snow depth, m
     beta_c = Effective width of angular extent of coherent component


     Output:
     theta = angular sampling of the scattering signature
     sigma_0_snow_surf = smoothing spline characterizing the scattering
     signature of the air-snow interface, dB
     sigma_0_snow_vol = smoothing spline characterizing the scattering
     signature of the snow volume, dB
     kappa_e = extinction coefficient of snow volume, Np/m
     tau_snow = transmission coefficient at air-snow interface
     c_s = speed of light in snow, m/s
     epsr_ds = relative permittivity of dry snow


     Uses the following codes from external sources:
     I2EM_Backscatter_model.m (Fawwaz Ulaby)
     RelDielConst_DrySnow.m (Fawwaz Ulaby)
     MieExtinc_DrySnow.m (Fawwaz Ulaby)

     (C) Jack Landy, University of Bristol, 2018

    """

    ## Angular Sampling for Scattering Signature

    theta = np.logspace(log10(1e-6),log10(pi/2),200)

    ## Geophysical & Antenna Parameters

    c = 299792458 # speed of light, m/s
    f_c = c/Lambda # radar frequency, Hz
    k0 = (2*pi)/Lambda # wavenumber

    # mss_exp_s = (sqrt(2/pi)*sigma_s/l_s*sqrt(5*k*l_s - atan(5*k*l_s)))**2 # mean-square slope for exponential ACF (Dierking, 2000)

    # Reduced speed of light in snow, m/s
    c_s = c * (1 + 0.51 * rho_s * 1e-3)**(-1.5) 


    ## Dielectric Properties

    # Dry snow dielectrics
    print('---- Dry snow dielectrics ----')
    epsr_ds, epsi_ds = RelDielConst_DrySnow(T_s, rho_s*1e-3, f_c*1e-9)
    eps_ds = epsr_ds + 1j * epsi_ds

    # Fresnel reflection & transmission coefficients
    print('---- Fresnel reflection & transmission coefficients ----')

    epsr_a = 1 # relative permittivity of air

    eta_0 = 376.73031346177 # intrinsic impedance of free space, ohms
    eta_1 = eta_0/sqrt(epsr_a)
    eta_2 = eta_0/sqrt(epsr_ds)
    theta_2 = np.arcsin(np.sin(theta) * (sqrt(epsr_a / epsr_ds))) # transmission angle

    rho_H = ((eta_2 * cos(theta) - eta_1 * cos(theta_2))
             /(eta_2 * cos(theta) + eta_1 * cos(theta_2))) # reflection coeff
    rho_V = ((eta_2 * cos(theta_2) - eta_1 * cos(theta))
             /(eta_2 * cos(theta_2) + eta_1 * cos(theta))) # reflection coeff

    tau_H = 1 + rho_H # transmission coeff, H-pol
    tau_V = (1 + rho_V)*(cos(theta)/cos(theta_2)) # transmission coeff, V-pol
    tau_snow = CubicSpline(theta, (tau_H + tau_V)/2)

    gamma_H = rho_H**2 # reflectivity (intensity)
    gamma_V = rho_V**2 # reflectivity (intensity)


    ## Backscattering Coefficient of Air-Snow Interface, sigma0

    print('---- Backscattering Coefficient of Air-Snow Interface, sigma0 ----')

    # Calculate coherent vs. incoherent surface scattering ratio
    # psi = k0*sigma_s*cos(theta) # frequency-dependent roughness parameter
    # omega = exp(-4*psi.**2) # fractional coherent component

    # Calculate coherent reflected backscattering coefficient
    # sigma_0_HH_coh = ((gamma_H.*omega)/beta_c**2)*exp(-4*k0**2*sigma_s**2).*exp(-theta.**2/beta_c**2) # coherent component of backscattering coefficient
    # sigma_0_VV_coh = ((gamma_V.*omega)/beta_c**2)*exp(-4*k0**2*sigma_s**2).*exp(-theta.**2/beta_c**2) # coherent component of backscattering coefficient
    
    # Not commented in last version but doesn't work ...TODO
    sigma_0_HH_coh = 4 * ((gamma_H * 1)/beta_c**2)*exp(-4*k0**2*sigma_s**2)*exp(-4*theta**2/beta_c**2) # equation 6 of Fung and Eom 1983
    sigma_0_VV_coh = 4 * ((gamma_V * 1)/beta_c**2)*exp(-4*k0**2*sigma_s**2)*exp(-4*theta**2/beta_c**2)

    # Calculate incoherent surface backscattering coefficient
    # Run single-scattering IEM for relevant range of facet incidence angles
    sigma_0_HH_s_surf = np.zeros((len(theta),1))
    sigma_0_VV_s_surf = np.zeros((len(theta),1))
    sigma_0_HV_s_surf = np.zeros(len(theta))
    # Parallel execution across all angles
    results = Parallel(n_jobs=-1)(
        delayed(compute_sigma)(theta_i, f_c, sigma_s, l_s, eps_ds)
        for theta_i in tqdm(theta)
        )
    

    # Fill output arrays with results
    for i, (vv, hh, hv) in enumerate(results):
        sigma_0_VV_s_surf[i] = vv
        sigma_0_HH_s_surf[i] = hh
        sigma_0_HV_s_surf[i] = hv
    """
    for i in tqdm(range(len(theta))):
        sigma_0_VV_s_surf[i], sigma_0_HH_s_surf[i] = I2EM_Backscatter_model(f_c*1e-9, sigma_s, l_s, theta[i]*180/pi, eps_ds, 1, [])[0:2] 
    """
    
    
    # Calculate total co-polarized surface backscattering cofficients,
    # including coherent reflected power

    # TODO In previous version !!! -> Check what it involves no to have it ...
    sigma_0_HH_s_surf = 10*log10(sigma_0_HH_coh + 10**(sigma_0_HH_s_surf.T/10)) # H-pol, dB
    sigma_0_VV_s_surf = 10*log10(sigma_0_VV_coh + 10**(sigma_0_VV_s_surf.T/10)) # V-pol, dB

    # Assuming no coherent reflected power
    sigma_0_HH_s_surf[np.isinf(sigma_0_HH_s_surf)] = np.nan
    sigma_0_VV_s_surf[np.isinf(sigma_0_VV_s_surf)] = np.nan

    # Build spline interpolants (assumption that scattering is polarization-independent)
    # dB
    sigma_0_snow_surf = CubicSpline(theta, ((sigma_0_HH_s_surf + sigma_0_VV_s_surf)/2).flatten())


    ## Backscattering Coefficient of Snow Volume, sigma0

    # chi = sqrt(epsr_a)*(2*pi*r_s)/lambda # normalized circumference
    # n = sqrt(epsr_i/epsr_a)
    # rayleigh_approximation = abs(n*chi) # Rayleigh scattering appropriate if <0.5

    print('---- Backscattering Coefficient of Snow Volume, sigma0 ----')

    # Mie extinction coefficient in dry snow
    kappa_e = MieExtinc_DrySnow(rho_s * 1e-3, r_s, f_c * 1e-9, T_s)[2]

    # Two-way loss factor
    L_theta = exp((-2 * kappa_e * h_s)/cos(theta_2)) # including scattering

    # Penetration depth into snow cover (Ulaby et al 1984)
    # Ignores scattering losses (valid when grain size <0.0001 m)
    # delta_p = lambda/(4*pi)*(epsr_ds/2*(sqrt(1 + (epsi_ds/epsr_ds)**2) - 1))**(-1/2) # m
    # 
    # kappa_a2 = 1/delta_p # Alternative absorption coeff calculated from penetration depth
    # L_theta2 = exp((-2*kappa_a2*h_s)./cos(theta_2)) # ignoring scattering, following Nanden et al 2017

    # Backscattered power from snow layer, following Winebrenner et al 1992
    # TODO : exception if L_theta = 1 -> stupid log(0)...
    sigma_0_HH_s_vol = 10 * log10(tau_H**2 * (1 - L_theta))
    sigma_0_VV_s_vol = 10 * log10(tau_V**2 * (1 - L_theta))



    # Build spline interpolants (assumption that scattering is polarization-independent)
    # dB
    if abs(sigma_0_HH_s_vol.mean())==np.inf:
        sigma_0_snow_vol = np.zeros(py_length(sigma_0_HH_s_vol))
    else:
        sigma_0_snow_vol = CubicSpline(theta, (sigma_0_HH_s_vol + sigma_0_VV_s_vol)/2)

    return theta, sigma_0_snow_surf, sigma_0_snow_vol, kappa_e, tau_snow, c_s, epsr_ds