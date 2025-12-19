import numpy as np 
import warnings
from scipy.interpolate import CubicSpline
from scipy.constants import pi
from Synthetic_Topography.rsgene2D_anisotrop import rsgene2D_anisotrop
from Synthetic_Topography.artificial_surf import artificial_surf
from Synthetic_Topography.add_melt_ponds import add_melt_ponds

def synthetic_topo_shell(op_mode, topo_type, pitch, roll, sigma_surf, l_surf, H_surf, dx, L_w, L_h, D_off, f_p):

    """
    ## Generates xyz vectors of synthetic sea ice topography

    # Virtual sea ice surface topography with Gaussian, lognormal or fractal
    # roughness properties

    # Input:
    # op_mode: 1 = pulse-limited, 2 = SAR
    # topo_type: 1 = Gaussian, 2 = lognormal, 3 = fractal
    # sigma_surf = rms roughness height
    # l_surf = correlation length
    # H_surf = Hurst parameter for fractal surfaces
    # dx = grid resolution
    # L_w = lead width
    # L_h = lead depth
    # D_off = lead distance off nadir
    # f_p = melt pond fraction

    # Output:
    # PosT = n x 3 matrix of the xyz surface topography coordinate vertices
    # surface_type: 0 = lead/ocean, 1 = sea ice

    # (C) Jack Landy, University of Bristol, 2018
    """

    warnings.filterwarnings('ignore')

    ## Create grid

    # (Can be modified but these are default grid sizes)
    if op_mode == 1:
        L = 12000 # across-track diameter of grid, m
        W = 12000 # along-track diameter of grid, m
    else:
        L = 12000 # across-track diameter of grid, m
        W = 1200 # along-track diameter of grid, m
    
    Nx = int(round(W / dx))
    Ny = int(round(L / dx))
    xv = np.linspace(-W/2 + dx/2, W/2 - dx/2, Nx)
    yv = np.linspace(-L/2 + dx/2, L/2 - dx/2, Ny)
    x, y = np.meshgrid(xv, yv, indexing='xy')

    ## Generate topography

    if topo_type == 1: # Gaussian
        z = rsgene2D_anisotrop(W/dx, L/dx, W, L, sigma_surf, l_surf, l_surf, 0)[0]
    elif topo_type == 2: # Lognormal
        z = rsgene2D_anisotrop(W/dx, L/dx, W, L, sigma_surf, l_surf, l_surf, 1)[0]
    else: # Fractal
        z = artificial_surf(sigma_surf, H_surf, W, W/dx, L/dx, (2*pi)/(l_surf*dx))[0]
    
    ## Add lead

    surface_type = np.ones_like(z, dtype=np.int8) # sea ice = 1, lead/ocean = 0

    if L_w > 0:
        i0 = int((L/(2*dx) - 0.5*L_w/dx - D_off/dx))
        i1 = int((L/(2*dx) + 0.5*L_w/dx - D_off/dx))
        # Might need to use round()

        z[i0:i1 + 1, :] = -L_h
        surface_type[i0:i1 + 1, :] = 0
    

    ## Add melt ponds

    if f_p > 0:
        z, surface_type = add_melt_ponds(z, surface_type, f_p) # topo still referenced to mean height

    ## Finalize

    PosT = np.column_stack((x.ravel(), y.ravel(), z.ravel()))    
    surface_type = surface_type.ravel()

    warnings.filterwarnings('default') 



    return PosT, surface_type