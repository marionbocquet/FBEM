## Shell Script for the Facet-based Radar Altimeter Echo Model for Sea Ice

# Shell script controlling application of the facet-based model for
# pulse-limited or SAR altimeter echo over snow-covered sea ice

# (c) J.C. Landy, University of Bristol, 2018

# Included Codes:
# Facet_Echo_Model.m
# RelDielConst_Brine.m
# rsgene2D_anisotrop.m
# synthetic_topo_shell.m
# snow_backscatter.m
# ice_backscatter.m
# lead_backscatter.m
# pond_backscatter.m

from Facet_Echo_Model import Facet_Echo_Model
from Facet_Echo_Model_2D import Facet_Echo_Model_2D


from Scattering_Models.RelDielConst_Brine import RelDielConst_Brine
from Scattering_Models.snow_backscatter import snow_backscatter
from Scattering_Models.ice_backscatter import ice_backscatter
from Scattering_Models.lead_backscatter import lead_backscatter
from Scattering_Models.pond_backscatter import pond_backscatter
from Synthetic_Topography.rsgene2D_anisotrop import rsgene2D_anisotrop
from Synthetic_Topography.synthetic_topo_shell import synthetic_topo_shell

# Uses the following codes from external sources:
# computeNormalVectorTriangulation.m (David Gingras)
# I2EM_Backscatter_model.m (Fawwaz Ulaby)
# MieExtinc_DrySnow.m (Fawwaz Ulaby)
# RelDielConst_PureIce.m (Fawwaz Ulaby)
# RelDielConst_DrySnow.m (Fawwaz Ulaby)
# RelDielConst_SalineWater.m (Fawwaz Ulaby)
# TVBmodel_HeterogeneousMix.m (Fawwaz Ulaby)
# artifical_surf.m (Mona Mahboob Kanafi)

from Scattering_Models.I2EM_Backscatter_model import I2EM_Backscatter_model
from Scattering_Models.MieExtinc_DrySnow import Mie_Rayleigh_ScatteringOfSpheres, MieExtinc_DrySnow
from Scattering_Models.RelDielConst_PureIce import RelDielConst_PureIce
from Scattering_Models.RelDielConst_DrySnow import RelDielConst_DrySnow
from Scattering_Models.RelDielConst_SalineWater import RelDielConst_SalineWater
from Scattering_Models.TVBmodel_HeterogeneousMix import TVBmodel_HeterogeneousMix
from Synthetic_Topography.computeNormalVectorTriangulation import computeNormalVectorTriangulation, centerTri3D
from Synthetic_Topography.artificial_surf import artificial_surf

import numpy as np 
from scipy.constants import pi
import sys


import pickle
import numpy as np
import types

def _is_serializable_value(v):
    """Filter useful types (avoid modules, functions, classes, etc.)."""
    # Common “workspace” types
    allowed = (int, float, bool, str, bytes, np.ndarray, list, tuple, dict, set)
    # Exclude functions, modules, classes
    if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type)):
        return False
    # Numpy scalar objects (np.float64, etc.) are allowed
    if isinstance(v, np.generic):
        return True
    # Containers: OK if pickle can handle them (usually yes)
    return isinstance(v, allowed)

def save_workspace_pickle(filename='workspace.pkl', namespace=None, include=None, exclude=None):
    """
    Save variables from the namespace into a pickle (.pkl) file.
    - filename: name of the file
    - namespace: dict (defaults to globals())
    - include: list of names to include (takes priority)
    - exclude: list of names to exclude
    """
    if namespace is None:
        namespace = globals()
    data = {}
    for k, v in namespace.items():
        if k.startswith('__') and k.endswith('__'):
            continue  # ignore system variables
        if include is not None and k not in include:
            continue
        if exclude is not None and k in exclude:
            continue
        if _is_serializable_value(v):
            data[k] = v
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol = pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(data)} variables to '{filename}'.")

def load_workspace_pickle(filename='workspace.pkl'):
    """Reload the workspace dictionary from a pickle file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)


## Model Variables (MODIFIABLE)

# Most AltiKa instrument parameters obtained from:
# https://directory.eoportal.org/web/eoportal/satellite-missions/s/saral

# Up to 3 variables can be vectors

# Geophysical parameters
sigma_s = 0.001 # snow rms height (default = 0.001 m)
l_s = 0.04 # snow correlation length (default = 0.04 m)
T_s = -20 # snow bulk temperature (default = -20 C)
rho_s = 350 # snow bulk density (default = 350 kg/m^3)
r_s = 0.001 # snow grain size (normal range from 0.0001 to 0.004 m, default 1 mm)
h_s = 0 # snow depth, m

sigma_si = np.concatenate([np.arange(0.0001, 0.0014 + 0.0001, 0.0001),    
                           np.arange(0.0015, 0.003 + 0.0005, 0.0005)]) 
# sea ice rms height (default = 0.002 m)
l_si = 0.02 # sea ice correlation length (default = 0.02 m)
T_si = -15 # sea ice bulk temperature (default = -15 C)
S_si = 6 # sea ice bulk salinity (default = 6 ppt)

sigma_sw = 0.00001 # lead rms height (default = 0.00001 m)
T_sw = 0 # temperature of seawater (default = 0 C)
S_sw = 34 # salinity of seawater (default = 34 ppt)

# Antenna parameters
Lambda = 0.00838580302 # radar wavelength (Ka-band AltiKa SARAL)

namespace = globals()
GP = [] # all parameters controlling scattering signatures
for name, obj in namespace.items():
    if name.startswith('__') and name.endswith('__'):        
        continue 
    size_info = ''    
    if isinstance(obj, np.ndarray):
        size_info = f"shape={obj.shape}, dtype={obj.dtype}"
        bytes_ = obj.nbytes
    else:        
        try:            
            size_info = f"len={len(obj)}"
        except Exception:            
            size_info = ''        
        bytes_ = sys.getsizeof(obj)    
    GP.append({        
        'name': name,        
        'type': type(obj).__name__,        
        'size_info': size_info,        
        'bytes': bytes_    
    })
        

op_mode = 1 # operational mode: 1 = pulse-limited, 2 = SAR (PL-mode only feasible on high memory machines)
beam_weighting = 1 # weighting on the beam-wise azimuth FFT: 1 = rectangular, 2 = Hamming (default = Hamming)
P_T = 2.188e-5 # transmitted peak power (default = 2.188e-5 watts)

pitch = 0 # antenna bench pitch counterclockwise (up to ~0.01 rads)
roll = 0 # antenna bench roll counterclockwise (up to ~0.005 rads)

h = 800000 # satellite altitude (default = 800000 m, AltiKa)
v = 7470 # satellite velocity (default = 7450 m/s, AltiKa)
N_b = 1 # no. beams in synthetic aperture (default = 1, AltiKa)
if op_mode==1: # single beam for PL echo
    N_b = 1

            
prf = 3800 # pulse-repetition frequency (default = 3800 Hz, AltiKa)
bandwidth = 480*10^6 # antenna bandwidth (default = 480*10^6 Hz, AltiKa) ### CHECK, SOME SOURCES STATE 500 MHz ###
G_0 = 49.3 # peak antenna gain, dB
D_0 = 1 # synthetic beam gain, dummy

# gammabar = 0.012215368000378016 # Cryosat-2 antenna pattern term 1
# gammahat = 0.0381925958945466 # Cryosat-2 antenna pattern term 2
# gamma1 = sqrt(2/(2/gammabar^2+2/gammahat^2)) # along-track antenna parameter
# gamma2 = sqrt(2/(2/gammabar^2-2/gammahat^2)) # across-track antenna parameter
beam_div = 0.605 # 3dB full beam divergence 0.605 deg AK
gamma1 = (beam_div * pi/180)/(2*np.sqrt(np.log(2)))
gamma2 = gamma1 # dummy

# Number of range bins
N_tb = 115 # (default = 70)

# Range bin at mean scattering surface, i.e. time = 0
t_0 = 30 # (default = 15)

# Time oversampling factor
t_sub = 5

# Parameters of synthetic topography
topo_type = 2 # type of surface: 1 = Gaussian, 2 = lognormal, 3 = fractal
sigma_surf = np.concatenate([np.arange(0, 0.009 + 0.001, 0.001),  
                                      np.arange(0.01, 0.09 + 0.01, 0.01),   
                                      np.arange(0.1, 1 + 0.025, 0.025)
                                      ])
l_surf = 20 # large-scale correlation length (default = 5 m)
H_surf = 0.5 # Hurst parameter (default = 0.5)
dx = 1 # resolution of grid for deriving the distribution of surface slopes, m (default = 1 m)

# # Lead parameters (optional)
# L_w = 0 # lead width (default = 100 m)
# L_h = 0 # lead depth (default = 0.2 m)
# D_off = 0 # distance off nadir (default = 0 m)
# 
# # Add melt ponds (optional)
# T_fw = 0 # temperature of freshwater in pond (default = 0 C)
# f_p = 0 # melt pond fraction (default = 0.5)
# u_a = 2 # boundary-layer wind speed (default = 4 m/s)

# Antenna Geometry
# epsilon_b = Lambda/(2*N_b*v*(1/prf)) # angular resolution of beams from full look crescent (beam separation angle) 
epsilon_b = Lambda/(2 * 64 * 7500 * (1/18182)) # KEEP SAME AS CRYOSAT-2 ANGLE

save_workspace_pickle('FEM_Simulations_AltiKa_Lognormal.pkl')

## Loop Echo Model

# Use parallel processing
# parpool

# Time domain, interval half t_sub
t = (0.5 / bandwidth) * (np.arange(1, N_tb + 1, 1 / t_sub) - t_0)

# Loop model over vector variables
rows = len(sigma_si)
cols = len(sigma_surf)
P_t_full_range = [[None for _ in range(cols)] for _ in range(rows)]
P_t_ml_range = [[None for _ in range(cols)] for _ in range(rows)]
EM_bias_range = np.full((rows, cols), np.nan)
for i in range(len(sigma_si)):
    
    # Effective width of angular extent of coherent component (TUNING PARAMETER FOR LEADS)
    beta_c = epsilon_b # same as Cryosat-2 angle
    # c = 299792458 # speed of light, m/s
    # beta_c = sqrt((c*1/bandwidth)/h) # from Carsey et al 1992 (pp 117)

    # Surface & volume backscattering properties
    theta, sigma_0_snow_surf, sigma_0_snow_vol, kappa_e, tau_snow, c_s, epsr_ds = snow_backscatter(Lambda,sigma_s,l_s,T_s,rho_s,r_s,h_s,beta_c)
    sigma_0_ice_surf = ice_backscatter(Lambda, sigma_si[i], l_si, T_si, S_si, h_s, beta_c, epsr_ds)[1]
    #     [~,sigma_0_lead_surf] = lead_backscatter(Lambda,sigma_sw,T_sw,S_sw,beta_c)
    #     [~,sigma_0_mp_surf] = pond_backscatter(Lambda,T_fw,beta_c,u_a)
    
    for j in range(len(sigma_si)):
        
        total = len(sigma_si) * len(sigma_surf)
        current = len(sigma_surf) * i + (j + 1)
        print(f"Simulation {current}/{total}")

        P_t_full, P_t_ml, em_bias = Facet_Echo_Model_2D(op_mode,Lambda,bandwidth,P_T,h,v,pitch,roll,prf,beam_weighting,G_0,D_0,gamma1,gamma2,N_b,t,sigma_0_snow_surf,sigma_0_snow_vol,kappa_e,tau_snow,c_s,h_s,sigma_0_ice_surf,sigma_surf(j),l_surf,H_surf,topo_type,dx)
        
        P_t_full_range[i][j]= P_t_full
        P_t_ml_range[i][j] = P_t_ml
        EM_bias_range[i][j] = em_bias
      


## Save Results


save_workspace_pickle('FEM_Simulations_AltiKa_Lognormal.pkl', include=['t','P_t_full_range','P_t_ml_range','EM_bias_range','beta_c'])



