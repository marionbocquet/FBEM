from scipy.constants import pi
import numpy as np
from numpy import sin, cos, tan, sqrt, exp, log10
from scipy.spatial import Delaunay
from Synthetic_Topography.computeNormalVectorTriangulation import computeNormalVectorTriangulation
from scipy.interpolate import interp1d

def Facet_Echo_Model(op_mode, Lambda, bandwidth, P_T, h, v, pitch, 
                     roll, prf, beam_weighting, G_0, D_0, gamma1, 
                     gamma2, N_b, t, PosT, surface_type, sigma_0_snow_surf, 
                     sigma_0_snow_vol, kappa_e, tau_snow, c_s, h_s, sigma_0_ice_surf, 
                     sigma_0_lead_surf, sigma_0_mp_surf):
    
    """
     Facet-based Radar Altimeter Echo Model for Sea Ice

     Simulates the backscattered echo response of a pulse-limited or synthetic
     aperture radar altimeter from snow-covered sea ice, over a facet-based
     triangular mesh of the sea ice surface topography

     Input (reference values)
     op_mode = operational mode: 1 = pulse-limited, 2 = SAR (PL-mode only feasible on high memory machines)
     Lambda = radar wavelength, 0.0221 m
     bandwidth = antenna bandwidth, Hz
     P_T = transmitted peak power, 2.188e-5 watts
     h = satellite altitude, 720000 m
     v = satellite velocity, 7500 m/s
     pitch = antenna bench pitch counterclockwise, rads (up to ~0.005 rads)
     roll = antenna bench roll counterclockwise, rads (up to ~0.005 rads)
     prf = pulse-repitition frequency, Hz
     beam_weighting = weighting function on beam pattern (1 = rectangular, 2 =
     Hamming)
     G_0 = peak antenna gain, dB
     gamma1 = along-track antenna parameter, 0.0116 rads
     gamma2 = across-track antenna parameter, 0.0129 rads
     N_b = no. beams in synthetic aperture, 64 (1 in PL mode)
     t = time, s
     PosT = surface facet xyz locations (n x 3 matrix)
     surface_type = surface facet type: 0 = lead/ocean, 1 = sea ice, 2 = melt
     pond (n x 3 matrix)
     theta = angular sampling of scattering signatures, rads
     sigma_0_snow_surf = backscattering coefficient of snow surface, dB
     sigma_0_snow_vol = backscattering coefficient of snow volume, dB 
     kappa_e = extinction coefficient of snow volume, Np/m
     tau_snow = transmission coefficient at air-snow interface
     c_s = speed of light in snowpack, m/s
     h_s = snow depth, m
     sigma_0_ice_surf = backscattering coefficient of ice surface, dB
     sigma_0_lead_surf = backscattering coefficient of lead surface, dB
     sigma_0_mp_surf = backscattering coefficient of pond surface, dB

     Output
     P_t_full = delay-Doppler map (DDM) of single look echoes, watts
     P_t_ml = multi-looked power waveform, watts
     P_t_full_comp = DDM for individual snow surface, snow volume, ice
     surface, and lead surface components, watts
     P_t_ml_comp = multi-looked power waveforms for individual snow surface,
     snow volume, ice surface, and lead surface components, watts

     Based on model equations introduced in Landy et al, TGARS, 2019
     Builing on theory of Wingham et al 2006, Giles et al 2007,
     Makynen et al 2009, Ulaby et al 2014

     Uses the following codes from external sources:
     computeNormalVectorTriangulation.m (David Gingras)

     (c) Jack Landy, University of Bristol, 2018

    """

    ## Antenna parameters

    c = 299792458 # speed of light, m/s
    Re = 6371 * 10**3 # earth's radius, m

    # f_c = c/Lambda # radar frequency, Hz
    k0 = (2 * pi) / Lambda # wavenumber

    delta_x = v / prf # distance between coherent pulses in synthetic aperture, m

    # delta_x_dopp = (h*prf*c)/(2*N_b*v*f_c) # along-track doppler-beam limited footprint size, m
    # delta_x_pl = 2*sqrt(c*(h/((Re+h)/Re))*(1/bandwidth)) # across-track pulse-limited footprint size, m
    # A_pl = pi*(delta_x_pl/2)**2 # area of each range ring (after waveform peak), m

    epsilon_b = Lambda / (2 * N_b * v * (1/prf)) # angular resolution of beams from full look crescent (beam separation angle) 

    # Antenna look geometry
    m = np.arange(-(N_b - 1) / 2, (N_b - 1) / 2 + 1)

    ## Triangulate surface

    # Triangulate
    TRI = Delaunay(PosT[:, :2]).simplices

    SURFACE_TYPE = surface_type[TRI[:, 0]]

    # Simplify triangulation to improve speed
    # simplication_factor = 0.8 # Fraction of facets remaining after simplification
    # figureDT=trisurf(TRI,PosTx,PosTy,PosTz) set(gcf,'Visible', 'off')
    # nfv=reducepatch(DT,simplification_factor)
    # TRI = nfv.faces
    # PosT = nfv.vertices
    # PosTx = PosT(:,1) PosTy = PosT(:,2) PosTz = PosT(:,3)

    # Compute normal vectors of facets
    NormalVx, NormalVy, NormalVz, PosVx, PosVy, PosVz = computeNormalVectorTriangulation(PosT, TRI, 'center-cells')
    # PosVz = PosVz*(std(PosT(:,3)/std(PosVz))) # to scale height distribution correctly?

    # Compute areas of facets
    
    P0 = PosT[TRI[:, 0], :]
    P1 = PosT[TRI[:, 1], :]
    P2 = PosT[TRI[:, 2], :]

    P10 = P1 - P0
    P20 = P2 - P0

    V = np.cross(P10, P20, axis=1)

    A_facets = 0.5 * np.sqrt(np.sum(V * V, axis=1))

    del P0, P1, P2, P10, P20, V

    # Alternative
    # [NormalVx, NormalVy, NormalVz, PosVx, PosVy, PosVz]=computeNormalVectorTriangulation(PosT,TRI,'vertices')
    # dx = 10
    # A_facets = ones(size(PosVz))*dx**2

    ## Radar Simulator Loop
    # Formulated for parallel processing

    P_t_full = np.zeros((len(m), len(t)))
    sigma_0_tracer = np.zeros((len(m), len(t), 4))
    for ii in range(len(m)):
        print('Beam number', ii+1 , 'of', len(m))
        # disp(ii)
        
        ## Angular geometry of surface
        
        # Antenna location
        x_0 = h * m[ii] * epsilon_b + h * np.tan(pitch)
        y_0 = h * np.tan(roll)
        
        # Calculate basic angles
        R_xy = (PosVx - x_0) ** 2 + (PosVy - y_0) ** 2
        R = np.sqrt((PosVz - h) ** 2 + R_xy * (1.0 + h / Re))

        THETA = np.pi / 2 + np.arctan2((PosVz - h), np.sqrt(R_xy))
        PHI = np.arctan2((PosVy - y_0), (PosVx - x_0))

        THETA_G = np.pi / 2 + np.arctan2(
            (PosVz - R), 
            np.sqrt((PosVx - x_0 + h * np.tan(pitch)) ** 2 +
                    (PosVy - y_0 + h * np.tan(roll)) ** 2)
        )
        

        PHI_G = np.arctan2((PosVy - y_0 + h * np.tan(roll)),
                           (PosVx - x_0 + h * np.tan(pitch)))
        
        # look angle of radar (synthetic beam pattern unaffected by mis-pointing, following beam steering)    
        theta_l = np.arctan(-(PosVx - x_0 + h * np.tan(pitch)) / (PosVz - h))

        # Compute angle between facet-normal vector and antenna-facet vector
        NormalAx = cos(PHI) * cos(pi/2 - THETA)
        NormalAy = sin(PHI) * cos(pi/2 - THETA)
        NormalAz = -sin(pi/2 - THETA)

        denom = (np.sqrt(NormalVx ** 2 + NormalVy ** 2 + NormalVz ** 2) *                 
                np.sqrt(NormalAx ** 2 + NormalAy ** 2 + NormalAz ** 2))
        cos_arg = (NormalVx * NormalAx + NormalVy * NormalAy + NormalVz * NormalAz) / denom

        theta_pr = np.pi - np.arccos(cos_arg)
        theta_pr = np.minimum(theta_pr, np.pi / 2)
        
        ## Compute gain functions
        
        # Antenna gain pattern (based on Cryosat-2)
        G = G_0 * np.exp(-THETA_G ** 2 *
                         (np.cos(PHI_G) ** 2 / (gamma1 ** 2) + 
                          np.sin(PHI_G) ** 2 / (gamma2 ** 2)))    
        P_m = [] 
        # Synthetic beam gain function
        if beam_weighting == 1: 
            # Rectangular
            arg = k0 * delta_x * np.sin(theta_l + m[ii] * epsilon_b)
            num = np.sin(N_b * arg) ** 2
            den = (N_b * np.sin(arg)) ** 2
            P_m = D_0 * (num / den)

        elif beam_weighting == 2:
            if op_mode == 1:
                #P_m = np.ones_like(N_b)
                P_m = np.ones_like(len(A_facets))
            else :
                # Apply hamming window to azimuthal response function
                n = np.arange(N_b)            
                w = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (N_b - 1))            
                a = (2j * k0 * v / prf) * (theta_l + m[ii] * epsilon_b)[:, None] * (n - (N_b - 1) / 2)[None, :]            
                P_m = np.abs(np.sum(w[None, :] * np.exp(a), axis=1)) ** 2
                

        ## Compute transmitted power envelope
        
        # Time offset
        tc = 2 * (np.sqrt(x_0 ** 2 * (1.0 + h / Re) + h ** 2) - h) / c # slant-range time correction
        T = (t + 2 * h / c + tc)[None, :] - (2 * R / c)[:, None]

        # Power envelope
        P_t = np.sinc(bandwidth * T) ** 2
        
        ## Compute linearized backscattering
        # Surface plus volume echo (following Arthern et al 2001, Kurtz et al 2014 procedure)
        
        theta_PR = theta_pr[:, None] * np.zeros_like(T)
        
        # Snow volume echo from IEM and Mie extinction
        vu_t_surf = (10.0 ** (sigma_0_snow_surf(theta_pr) / 10.0)) * (h_s != 0)

        mask_vol = (T >= -(2.0 * h_s) / c_s) & (T < 0.0)
        vu_t_vol = np.zeros_like(T, dtype=float)
        if mask_vol.any():
            sigma_vol_lin = 10.0 ** (sigma_0_snow_vol(theta_PR[mask_vol]) / 10.0)
        
            # Extinction
            atten = kappa_e * np.exp(-c_s * kappa_e * (T[mask_vol] + (2.0 * h_s) / c_s))
            vu_t_vol_masked = sigma_vol_lin * atten
            vu_t_vol[mask_vol] = vu_t_vol_masked

        # vu_t_surf_tracer : 
        x_old = t - 2.0 * h_s / c_s
        f_interp = interp1d(x_old, P_t, axis=1, kind="linear", bounds_error=False, fill_value=np.nan)
        P_t_shift = f_interp(t)
        vu_t_surf_tracer = P_t_shift * vu_t_surf #[:, None]

        vu_t_vol_tracer = np.zeros_like(P_t)
        if mask_vol.any():
            vu_t_vol_tracer[mask_vol] = vu_t_vol[mask_vol]
        vu_t_vol_tracer_p = P_t_shift * vu_t_vol_tracer
        

        # --- Ice surface echo from IEM ---        
        mu_t = np.zeros_like(theta_pr)  # (F,)        
        idx_ice = (SURFACE_TYPE == 1)  
        if idx_ice.any():      
            mu_t[idx_ice] = ((10.0 ** (sigma_0_ice_surf(theta_pr[idx_ice]) / 10.0)) * 
                            (tau_snow(theta_pr[idx_ice]) ** 2) *   
                            np.exp(-kappa_e * h_s / 2.0))   
        mu_t_si_tracer = P_t * mu_t[:, np.newaxis]
        # --- Leads / melt ponds ---        
        
        idx_lead = (SURFACE_TYPE == 0)       
        idx_mp = (SURFACE_TYPE == 2)   
        if idx_lead.any():     
            mu_t[idx_lead] = 10.0 ** (sigma_0_lead_surf(theta_pr[idx_lead]) / 10.0)        
        if idx_mp.any():
            mu_t[idx_mp] = 10.0 ** (sigma_0_mp_surf(theta_pr[idx_mp]) / 10.0)        
        
        mu_t = np.nan_to_num(mu_t, nan=0.0)        
        
        mu_t_ocean_tracer = P_t * mu_t[:, None] - mu_t_si_tracer        
        
        # --- Total echo (pre-convolved with transmitted pulse) ---        
        sigma_0_P_t = vu_t_surf_tracer + vu_t_vol_tracer_p + mu_t_si_tracer + mu_t_ocean_tracer  
        
        # --- Backscatter component fraction tracer ---        
        total_t = np.sum(sigma_0_P_t, axis=0)  # (M,)        
        total_t_safe = np.where(total_t == 0, np.finfo(float).eps, total_t)        
        f_surf = np.sum(vu_t_surf_tracer, axis=0) / total_t        
        f_vol  = np.sum(vu_t_vol_tracer_p,  axis=0) / total_t        
        f_ice  = np.sum(mu_t_si_tracer,   axis=0) / total_t        
        f_ocn  = np.sum(mu_t_ocean_tracer,axis=0) / total_t        
        sigma_0_tracer[ii, :, :] = np.stack([f_surf, f_vol, f_ice, f_ocn], axis=1)  # (M x 4)        
        
        vu_t_surf_tracer = np.array([])
        vu_t_vol_tracer = np.array([])
        vu_t_vol_tracer_p = np.array([])
        mu_t = np.array([])
        mu_t_si_tracer = np.array([])
        mu_t_ocean_tracer = np.array([])

        # --- Integrate power contributions from each facet ---        
        # Integrate radar equation
        const = ((Lambda ** 2) * P_T) / ((4.0 * np.pi) ** 3) * (0.5 * c * h)        
        # Weightning per facet       
        facet_w = (G ** 2) * P_m * A_facets  # (F,)        
               
        P_r = const * (sigma_0_P_t / (R[:, None] ** 4)) * facet_w[:, None]  # (F x M) 
        
        echo_t = np.nansum(P_r, axis=0)  # (M,)        
        
        #  Apply weighting to single-look echo stack
        P_t_full[ii, :] = np.real(echo_t)        
    
    P_t_ml = np.nansum(P_t_full, axis=0) 
    P_t_full_comp = sigma_0_tracer * P_t_full[:, :, None]
    P_t_full_T = P_t_full.T
    P_t_full_comp_perm = np.transpose(P_t_full_comp, (1, 0, 2))

    P_t_ml_comp = np.nansum(P_t_full_comp_perm, axis=1)

    return P_t_full_T, P_t_ml, P_t_full_comp_perm, P_t_ml_comp