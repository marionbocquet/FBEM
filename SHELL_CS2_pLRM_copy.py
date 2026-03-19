"""
Shell Script for the Facet-based Radar Altimeter Echo Model for Sea Ice
Faithful Matlab -> Python translation
(c) J.C. Landy, University of Bristol, 2018
"""

# ============================================================
# Imports (mêmes dépendances que Matlab)
# ============================================================

import numpy as np
from scipy.constants import pi
import types
import pickle

from Facet_Echo_Model_vori import Facet_Echo_Model
from Scattering_Models.snow_backscatter import snow_backscatter
from Scattering_Models.ice_backscatter import ice_backscatter
from Scattering_Models.lead_backscatter import lead_backscatter
from Scattering_Models.pond_backscatter import pond_backscatter
from Synthetic_Topography.synthetic_topo_shell import synthetic_topo_shell
from Plotting import plotting


# ============================================================
# Outils type Matlab
# ============================================================

def whos_py(namespace):
    """Simule Matlab whos('*')"""
    out = []
    for name, val in namespace.items():
        if name.startswith('__'):
            continue
        if isinstance(val, (types.ModuleType, types.FunctionType, type)):
            continue
        try:
            size = val.shape
        except Exception:
            size = (1, 1)
        out.append({'name': name, 'size': size, 'type': type(val)})
    return out


def cell3(n1, n2, n3):
    """cell(n1,n2,n3)"""
    return [[[None for _ in range(n3)] for _ in range(n2)] for _ in range(n1)]


def save_workspace_pickle(filename, namespace, include=None):
    """Sauvegarde façon save('file.mat')"""
    data = {}
    for k, v in namespace.items():
        if include is not None and k not in include:
            continue
        if isinstance(v, (types.ModuleType, types.FunctionType, type)):
            continue
        try:
            pickle.dumps(v)
            data[k] = v
        except Exception:
            pass
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# Paramètres géophysiques
# ============================================================

sigma_s = 0.001
l_s = 0.04
T_s = -20
rho_s = 350
r_s = 0.002
h_s = 0.2

sigma_si = 0.002
l_si = 0.02
T_si = -2
S_si = 6

sigma_sw = 1e-5
T_sw = 0
S_sw = 34


# ============================================================
# Paramètres antenne
# ============================================================

Lambda = 0.0221
op_mode = 1
beam_weighting = 1
P_T = 2.188e-5

pitch = 0.0
roll = 0.0

h = 720000.0
v = 7500.0
N_b = 1 if op_mode == 1 else 64

prf = 18182.0
bandwidth = 320e6
G_0 = 42.0
D_0 = 36.12

gammabar = 0.012215368000378016
gammahat = 0.0381925958945466
gamma1 = np.sqrt(2/(2/gammabar**2 + 2/gammahat**2))
gamma2 = np.sqrt(2/(2/gammabar**2 - 2/gammahat**2))


# ============================================================
# Domaine temporel
# ============================================================

N_tb = 70
t_0 = 15
t_sub = 2


# ============================================================
# Topographie synthétique
# ============================================================

topo_type = 2
sigma_surf = 0.2
l_surf = 5
H_surf = 0.5
dx = 20

L_w = 0
L_h = 0
D_off = 0

T_fw = 0
f_p = 0
u_a = 4


# ============================================================
# Géométrie antenne
# ============================================================

epsilon_b = Lambda / (2 * N_b * v * (1/prf))


# ============================================================
# Identification des variables vectorielles
# ============================================================

GP = whos_py(globals())
PARAMETERS = whos_py(globals())

idS = [
    i for i, p in enumerate(PARAMETERS)
    if len(p['size']) > 1 and p['size'][1] > 1
]

vec1 = vec2 = vec3 = [1]
if len(idS) >= 1:
    vec1 = globals()[PARAMETERS[idS[0]]['name']]
if len(idS) >= 2:
    vec2 = globals()[PARAMETERS[idS[1]]['name']]
if len(idS) >= 3:
    vec3 = globals()[PARAMETERS[idS[2]]['name']]

for name in ['vec1', 'vec2', 'vec3']:
    if not isinstance(locals()[name], (list, np.ndarray)):
        locals()[name] = [locals()[name]]


# ============================================================
# Allocation des sorties (cell arrays)
# ============================================================

P_t_full_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_ml_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_full_comp_range = cell3(len(vec1), len(vec2), len(vec3))
P_t_ml_comp_range = cell3(len(vec1), len(vec2), len(vec3))

counter = 0


# ============================================================
# Boucle principale (i, j, k)
# ============================================================

for i in range(len(vec1)):
    for j in range(len(vec2)):
        for k in range(len(vec3)):

            # Affectation dynamique (Matlab eval)
            idx_map = [i, j, k]
            for l, p_idx in enumerate(idS):
                globals()[PARAMETERS[p_idx]['name']] = [vec1, vec2, vec3][l][idx_map[l]]

            # Temps
            t = (0.5 / bandwidth) * (
                np.arange(1, N_tb + 1, 1 / t_sub) - t_0
            )

            # Backscatter (recalcul conditionnel)
            if counter < 1:
                beta_c = epsilon_b

                theta, sigma_0_snow_surf, sigma_0_snow_vol, kappa_e, tau_snow, c_s, epsr_ds = \
                    snow_backscatter(Lambda, sigma_s, l_s, T_s, rho_s, r_s, h_s, beta_c)

                _, sigma_0_ice_surf, _ = \
                    ice_backscatter(Lambda, sigma_si, l_si, T_si, S_si, h_s, beta_c, epsr_ds)

                _, sigma_0_lead_surf = \
                    lead_backscatter(Lambda, sigma_sw, T_sw, S_sw, beta_c)

                _, sigma_0_mp_surf = \
                    pond_backscatter(Lambda, T_fw, beta_c, u_a)

            counter += 1

            # Monte Carlo
            itN = 3

            P_t_full = np.zeros((len(t), N_b, itN))
            P_t_ml = np.zeros((len(t), itN))
            P_t_full_comp = np.zeros((N_b, len(t), 4, itN))
            P_t_ml_comp = np.zeros((len(t), 4, itN))

            for l in range(itN):

                PosT, surface_type = synthetic_topo_shell(
                    op_mode, topo_type, pitch, roll,
                    sigma_surf, l_surf, H_surf,
                    dx, L_w, L_h, D_off, f_p
                )

                (
                    P_t_full[:, :, l],
                    P_t_ml[:, l],
                    P_t_full_comp[:, :, :, l],
                    P_t_ml_comp[:, :, l]
                ) = Facet_Echo_Model(
                    op_mode, Lambda, bandwidth, P_T, h, v,
                    pitch, roll, prf, beam_weighting,
                    G_0, D_0,#, gamma1, gamma2,
                    N_b,
                    t, PosT, surface_type,
                    sigma_0_snow_surf, sigma_0_snow_vol,
                    kappa_e, tau_snow, c_s, h_s,
                    sigma_0_ice_surf,
                    sigma_0_lead_surf,
                    sigma_0_mp_surf
                )

                print(f"Iteration {l+1}/{itN}")

            # Moyennes
            P_t_full_range[i][j][k] = np.nanmean(P_t_full, axis=2)
            P_t_ml_range[i][j][k] = np.nanmean(P_t_ml, axis=1)
            P_t_full_comp_range[i][j][k] = np.nanmean(P_t_full_comp, axis=3)
            P_t_ml_comp_range[i][j][k] = np.nanmean(P_t_ml_comp, axis=2)

            topo_plot = 1 # example plot of tetrahedral surface mesh
            echo_plot = 1 # example plots of modelled echoes

            
            plotting(
                topo_plot, echo_plot,
                PosT, t, 
                P_t_ml_range[i][j][k],
                P_t_full_range[i][j][k],
                P_t_ml_comp_range[i][j][k],
                N_b, epsilon_b
                )

# ============================================================
# Sauvegarde
# ============================================================

save_workspace_pickle(
    'FEM_Simulations.pkl',
    globals(),
    include=[
        't',
        'P_t_full_range',
        'P_t_ml_range',
        'P_t_full_comp_range',
        'P_t_ml_comp_range'
    ]
)
