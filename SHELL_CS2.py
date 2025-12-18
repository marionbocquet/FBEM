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
import numpy as np

import pickle
import types

from Plotting import plotting


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

def whos_py_old(namespace):
    """
    Simule MATLAB whos('*')
    """
    out = []
    for name, val in namespace.items():
        try:
            size = np.shape(val)
            if size == ():
                size = (1, 1)
        except Exception:
            size = (1, 1)

        out.append({
            'name': name,
            'size': size
        })
    return out

def whos_py(namespace):
    out = []
    for name, val in namespace.items():
        # Ignorer modules et fonctions
        if isinstance(val, types.ModuleType) or callable(val):
            continue
        
        # Déterminer le type et la "taille"
        try:
            size = val.shape  # pour les arrays
        except AttributeError:
            size = (1, 1)     # pour scalaires ou autres types
        
        out.append({
            'name': name,
            'size': size,
            'type': type(val)  # <- on garde le type pour filtrage
        })
    return out

def cell3(n1, n2, n3):
    return [[[None for _ in range(n3)] for _ in range(n2)] for _ in range(n1)]


# Geophysical parameters
sigma_s = 0.001 # snow rms height (default = 0.001 m)
l_s = 0.04 # snow correlation length (default = 0.04 m)
T_s = -20 # snow bulk temperature (default = -20 C)
rho_s = 350 # snow bulk density (default = 350 kg/m**3)
r_s = 0.001 # snow grain size (normal range from 0.0001 to 0.004 m, default 1 mm)
h_s = 0 # snow depth, m

sigma_si = 0.002 # sea ice rms height (default = 0.002 m)
l_si = 0.02 # sea ice correlation length (default = 0.02 m)
T_si = -15 # sea ice bulk temperature (default = -15 C)
S_si = 6 # sea ice bulk salinity (default = 6 ppt)

sigma_sw = 0.00001 # lead rms height (default = 0.00001 m)
T_sw = 0 # temperature of seawater (default = 0 C)
S_sw = 34 # salinity of seawater (default = 34 ppt)


# Antenna parameters
Lambda = 0.0221 # radar wavelength (default = 0.0221, Ku-band e.g. Cryosat-2)

GP = whos_py(globals())

op_mode = 2 # operational mode: 1 = pulse-limited, 2 = SAR (PL-mode only feasible on high memory machines)
beam_weighting = 2 # weighting on the beam-wise azimuth FFT: 1 = rectangular, 2 = Hamming (default = Hamming)
P_T = 2.188e-5 # transmitted peak power (default = 2.188e-5 watts)

pitch = 0 # antenna bench pitch counterclockwise (up to ~0.01 rads)
roll = 0 # antenna bench roll counterclockwise (up to ~0.005 rads)

h = 720000 # satellite altitude (default = 720000 m)
v = 7500 # satellite velocity (default = 7500 m/s)
N_b = 64 # no. beams in synthetic aperture (default = 64, e.g. Cryosat-2)
if op_mode==1: # single beam for PL echo
    N_b = 1

prf = 18182 # pulse-repetition frequency (default = 18182 Hz, e.g. Cryosat-2)
bandwidth = 320*10**6 # antenna bandwidth (default = 320*10**6 Hz, e.g. Cryosat-2)
G_0 = 42 # peak antenna gain, dB
D_0 = 36.12 # synthetic beam gain, 36.12 dB SAR mode (30.6 dB SARIn mode)

gammabar = 0.012215368000378016 # Cryosat-2 antenna pattern term 1
gammahat = 0.0381925958945466 # Cryosat-2 antenna pattern term 2
gamma1 = np.sqrt(2/(2/gammabar**2+2/gammahat**2)) # along-track antenna parameter
gamma2 = np.sqrt(2/(2/gammabar**2-2/gammahat**2)) # across-track antenna parameter

# Number of range bins
N_tb = 70 # (default = 70)

# Range bin at mean scattering surface, i.e. time = 0
t_0 = 15 # (default = 15)

# Time oversampling factor
t_sub = 1

# Parameters of synthetic topography
topo_type = 2 # type of surface: 1 = Gaussian, 2 = lognormal, 3 = fractal
sigma_surf = 0.1 # large-scale rms roughness height (default = 0.1 m)
l_surf = 5 # large-scale correlation length (default = 5 m)
H_surf = 0.5 # Hurst parameter (default = 0.5)
dx = 10 # resolution of grid, m (WARNING use dx>=10 for PL mode and dx>=5 for SAR mode)

# Lead parameters (optional)
L_w = 0 # lead width (default = 100 m)
L_h = 0 # lead depth (default = 0.2 m)
D_off = 0 # distance off nadir (default = 0 m)

# Add melt ponds (optional)
T_fw = 0 # temperature of freshwater in pond (default = 0 C)
f_p = 0 # melt pond fraction (default = 0.5)
u_a = 4 # boundary-layer wind speed (default = 4 m/s)


save_workspace_pickle('FEM_Simulations.pkl')

# Optional Plotting
topo_plot = 1 # example plot of tetrahedral surface mesh
echo_plot = 1 # example plots of modelled echoes

## Antenna Geometry

epsilon_b = Lambda/(2*N_b*v*(1/prf)) # angular resolution of beams from full look crescent (beam separation angle) 


## Loop Echo Model

# Use parallel processing
# parpool

# Identify vector variables
# all parameters controlling scattering signatures
PARAMETERS = whos_py(globals())
idS = [
    print(p) for i, p in enumerate(PARAMETERS)
    if isinstance(p['type'], type(np.ndarray)) and len(p['size']) > 1 and p['size'][1] > 1]
idG = [i for i, g in enumerate(GP) if g['size'][1] > 1]

nmG = [0] * max(1, len(idG))
for g_idx in idG:
    nmG = [
        i for i, p in enumerate(PARAMETERS)
        if p['name'] == GP[g_idx]['name']
    ]

match = [0, 0, 0]
for i in range(len(idS)):
    match[i] = (nmG == idS)

vec1 = vec2 = vec3 = 1
if len(idS) >= 1:
    vec1 = globals()[PARAMETERS[idS[0]]['name']]
if len(idS) >= 2:
    vec2 = globals()[PARAMETERS[idS[1]]['name']]
if len(idS) >= 3:
    vec3 = globals()[PARAMETERS[idS[2]]['name']]


# Loop model over vector variables

P_t_full_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_ml_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_full_comp_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_ml_comp_range = cell3(len(vec1), len(vec2), len(vec3))

counter = 0

for i in range(len(vec1)):
    for j in range(len(vec2)):
        for k in range(len(vec3)):

            # --- Affectation dynamique des paramètres (équivalent eval) ---
            idx_map = [i, j, k]
            for l, p_idx in enumerate(idS):
                globals()[PARAMETERS[p_idx]['name']] = [vec1, vec2, vec3][l][idx_map[l]]

            # --- Time domain ---
            t = (0.5 / bandwidth) * (
                np.arange(1, N_tb + 1, 1 / t_sub) - t_0
            )

            # --- Backscatter recomputation condition ---
            if counter < 1 or len(idG) > 0:

                beta_c = epsilon_b  # tuning parameter

                (
                    theta,
                    sigma_0_snow_surf,
                    sigma_0_snow_vol,
                    kappa_e,
                    tau_snow,
                    c_s,
                    epsr_ds
                ) = snow_backscatter(
                    Lambda, sigma_s, l_s, T_s, rho_s, r_s, h_s, beta_c
                )

                (
                    _,
                    sigma_0_ice_surf,
                    _
                ) = ice_backscatter(
                    Lambda, sigma_si, l_si, T_si, S_si, h_s, beta_c, epsr_ds
                )

                (
                    _,
                    sigma_0_lead_surf
                ) = lead_backscatter(
                    Lambda, sigma_sw, T_sw, S_sw, beta_c
                )

                (
                    _,
                    sigma_0_mp_surf
                ) = pond_backscatter(
                    Lambda, T_fw, beta_c, u_a
                )

            counter += 1

            # --- Initialisation ---
            itN = 1

            P_t_full = np.zeros((len(t), N_b, itN))
            P_t_ml = np.zeros((len(t), itN))
            P_t_full_comp = np.zeros((len(t), N_b, 4, itN))
            P_t_ml_comp = np.zeros((len(t), 4, itN))

            for l in range(itN):

                # --- Synthetic topography ---
                PosT, surface_type = synthetic_topo_shell(
                    op_mode, topo_type, pitch, roll,
                    sigma_surf, l_surf, H_surf,
                    dx, L_w, L_h, D_off, f_p
                )

                # --- Facet Echo Model ---
                (
                    P_t_full[:, :, l],
                    P_t_ml[:, l],
                    P_t_full_comp[:, :, :, l],
                    P_t_ml_comp[:, :, l]
                ) = Facet_Echo_Model(
                    op_mode, Lambda, bandwidth, P_T, h, v,
                    pitch, roll, prf, beam_weighting,
                    G_0, D_0, gamma1, gamma2, N_b,
                    t, PosT, surface_type,
                    sigma_0_snow_surf, sigma_0_snow_vol,
                    kappa_e, tau_snow, c_s, h_s,
                    sigma_0_ice_surf,
                    sigma_0_lead_surf,
                    sigma_0_mp_surf
                )

                sim_id = sum(
                    x is not None for plane in P_t_ml_range for row in plane for x in row
                ) + 1

                print(
                    f"Iteration {l+1}/{itN}, "
                    f"Simulation {sim_id}/{len(vec1)*len(vec2)*len(vec3)}"
                )

            # --- Moyennes (nanmean MATLAB) ---
            P_t_full_range[i][j][k] = np.nanmean(P_t_full, axis=2)
            P_t_ml_range[i][j][k] = np.nanmean(P_t_ml, axis=1)
            P_t_full_comp_range[i][j][k] = np.nanmean(P_t_full_comp, axis=3)
            P_t_ml_comp_range[i][j][k] = np.nanmean(P_t_ml_comp, axis=2)

            # --- Optional plotting ---
            if (topo_plot or echo_plot) > 0:
                Plotting(
                    topo_plot, echo_plot,
                    PosT, t,
                    P_t_ml_range[i][j][k],
                    P_t_full_range[i][j][k],
                    P_t_ml_comp_range[i][j][k],
                    N_b, epsilon_b
                )

            if match[2] > 0:
                counter = 0

        if match[1] > 0:
            counter = 0

    if match[0] > 0:
        counter = 0


## Save Results
if len(idS) == 0:
    vec1 = vec2 = vec3 = 1

elif len(idS) == 1:
    globals()[PARAMETERS[idS[0]]['name']] = vec1
    vec2 = vec3 = 1

elif len(idS) == 2:
    globals()[PARAMETERS[idS[0]]['name']] = vec1
    globals()[PARAMETERS[idS[1]]['name']] = vec2
    vec3 = 1

elif len(idS) == 3:
    globals()[PARAMETERS[idS[0]]['name']] = vec1
    globals()[PARAMETERS[idS[1]]['name']] = vec2
    globals()[PARAMETERS[idS[2]]['name']] = vec3

save_workspace_pickle('FEM_Simulations.pkl', 
                      include=['t','P_t_full_range','P_t_ml_range',
                               'P_t_full_comp_range','P_t_ml_comp_range'])

