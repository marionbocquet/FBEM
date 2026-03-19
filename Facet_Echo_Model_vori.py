import numpy as np
from numpy import sin, cos, tan, sqrt, exp
from scipy.constants import pi
from scipy.spatial import Delaunay
from Synthetic_Topography.computeNormalVectorTriangulation import computeNormalVectorTriangulation


def Facet_Echo_Model(
    op_mode, Lambda, bandwidth, P_T, h, v, pitch, roll, prf,
    beam_weighting, G_0, D_0, N_b, t, PosT, surface_type,
    sigma_0_snow_surf, sigma_0_snow_vol, kappa_e, tau_snow,
    c_s, h_s, sigma_0_ice_surf, sigma_0_lead_surf, sigma_0_mp_surf
):
    """
    Faithful Python translation of Matlab Facet_Echo_Model.m
    """

    # ------------------------------------------------------------------
    # Antenna parameters (Matlab section: Antenna parameters)
    # ------------------------------------------------------------------

    c = 299792458.0
    Re = 6371e3

    k = (2.0 * pi) / Lambda
    delta_x = v / prf

    epsilon_b = Lambda / (2.0 * N_b * v * (1.0 / prf))

    gammabar = 0.012215368000378016
    gammahat = 0.0381925958945466
    gamma1 = sqrt(2.0 / (2.0 / gammabar**2 + 2.0 / gammahat**2))
    gamma2 = sqrt(2.0 / (2.0 / gammabar**2 - 2.0 / gammahat**2))

    m = np.arange(-(N_b - 1) / 2.0, (N_b - 1) / 2.0 + 1.0)

    # ------------------------------------------------------------------
    # Triangulate surface
    # ------------------------------------------------------------------

    TRI = Delaunay(PosT[:, :2]).simplices
    SURFACE_TYPE = surface_type[TRI[:, 0]]

    NormalVx, NormalVy, NormalVz, PosVx, PosVy, PosVz = \
        computeNormalVectorTriangulation(PosT, TRI, 'center-cells')

    P0 = PosT[TRI[:, 0]]
    P1 = PosT[TRI[:, 1]]
    P2 = PosT[TRI[:, 2]]

    V = np.cross(P1 - P0, P2 - P0, axis=1)
    A_facets = 0.5 * np.sqrt(np.sum(V * V, axis=1))

    # ------------------------------------------------------------------
    # Beam weighting (Matlab: H)
    # ------------------------------------------------------------------

    if op_mode == 1 or beam_weighting == 1:
        H = np.ones(N_b)
    else:
        H = np.hamming(N_b)

    # ------------------------------------------------------------------
    # Radar simulator loop
    # ------------------------------------------------------------------

    P_t_full = np.zeros((len(m), len(t)))
    sigma_0_tracer = np.zeros((len(m), len(t), 4))

    for i in range(len(m)):

        # Antenna location
        x_0 = h * m[i] * epsilon_b + h * tan(pitch)
        y_0 = h * tan(roll)

        # Geometry
        R_xy = (PosVx - x_0)**2 + (PosVy - y_0)**2
        R = sqrt((PosVz - h)**2 + R_xy * (1.0 + h / Re))

        THETA = pi/2 + np.arctan2(PosVz - h, sqrt(R_xy))
        PHI = np.arctan2(PosVy - y_0, PosVx - x_0)

        THETA_G = pi/2 + np.arctan2(
            PosVz - R,
            sqrt((PosVx - x_0 + h*tan(pitch))**2 +
                 (PosVy - y_0 + h*tan(roll))**2)
        )

        PHI_G = np.arctan2(
            PosVy - y_0 + h*tan(roll),
            PosVx - x_0 + h*tan(pitch)
        )

        theta_l = np.arctan(-(PosVx - x_0 + h*tan(pitch)) / (PosVz - h))

        # Facet incidence angle
        NormalAx = cos(PHI) * cos(pi/2 - THETA)
        NormalAy = sin(PHI) * cos(pi/2 - THETA)
        NormalAz = -sin(pi/2 - THETA)

        denom = sqrt(NormalVx**2 + NormalVy**2 + NormalVz**2) * \
                sqrt(NormalAx**2 + NormalAy**2 + NormalAz**2)

        theta_pr = pi - np.arccos(
            (NormalVx*NormalAx + NormalVy*NormalAy + NormalVz*NormalAz) / denom
        )
        theta_pr = np.minimum(theta_pr, pi/2)

        # Antenna gain
        G = G_0 * exp(
            -THETA_G**2 *
            (cos(PHI_G)**2 / gamma1**2 + sin(PHI_G)**2 / gamma2**2)
        )

        # Synthetic beam pattern
        arg = k * delta_x * sin(theta_l + m[i]*epsilon_b)
        P_m = D_0 * (sin(N_b * arg)**2) / ((N_b * sin(arg))**2)

        # Time axis
        tc = 2.0 * (sqrt(x_0**2 * (1.0 + h/Re) + h**2) - h) / c
        T = (t + 2*h/c + tc)[None, :] - (2*R/c)[:, None]

        P_t = (sin(bandwidth*pi*T) / (bandwidth*pi*T))**2

        # ------------------------------------------------------------------
        # Backscattering
        # ------------------------------------------------------------------

        theta_PR = theta_pr[:, None] + np.zeros_like(T)

        # Snow surface
        mask_snow = (T >= -(2*h_s)/c_s) & (T < 0)
        vu_surf = np.zeros_like(T)
        vu_surf[mask_snow] = np.ravel(10.0**(sigma_0_snow_surf(theta_PR[mask_snow]) / 10.0))
        vu_surf *= P_t

        # Snow volume
        vu_vol = np.zeros_like(P_t)
        vu_vol[mask_snow] = (
            10.0**(sigma_0_snow_vol(theta_PR[mask_snow]) / 10.0) *
            kappa_e *
            exp(-c_s*kappa_e*(T[mask_snow] + (2*h_s)/c_s))
        )
        vu_vol *= P_t

        # Ice surface
        mu_t = np.zeros_like(theta_pr)
        idx_ice = (SURFACE_TYPE == 1)
        mu_t[idx_ice] = (
            10.0**(sigma_0_ice_surf(theta_pr[idx_ice]) / 10.0) *
            tau_snow(theta_pr[idx_ice])**2 *
            exp(-kappa_e*h_s/2)
        )

        mu_si = mu_t[:, None] * P_t

        # Leads / melt ponds
        idx_lead = (SURFACE_TYPE == 0)
        idx_mp = (SURFACE_TYPE == 2)

        mu_t[idx_lead] = 10.0**(sigma_0_lead_surf(theta_pr[idx_lead]) / 10.0)
        mu_t[idx_mp] = 10.0**(sigma_0_mp_surf(theta_pr[idx_mp]) / 10.0)

        mu_t = np.nan_to_num(mu_t)
        mu_ocn = mu_t[:, None]*P_t - mu_si

        sigma_0_P_t = vu_surf + vu_vol + mu_si + mu_ocn

        total = np.sum(sigma_0_P_t, axis=0)
        total[total == 0] = np.finfo(float).eps

        sigma_0_tracer[i, :, 0] = np.sum(vu_surf, axis=0) / total
        sigma_0_tracer[i, :, 1] = np.sum(vu_vol, axis=0) / total
        sigma_0_tracer[i, :, 2] = np.sum(mu_si, axis=0) / total
        sigma_0_tracer[i, :, 3] = np.sum(mu_ocn, axis=0) / total

        # Radar equation
        const = (Lambda**2 * P_T) / ((4*pi)**3) * (0.5*c*h)
        P_r = const * sigma_0_P_t / (R[:, None]**4) * (G**2 * P_m * A_facets)[:, None]

        P_t_full[i, :] = np.real(np.nansum(P_r, axis=0)) * H[i]

    # ------------------------------------------------------------------
    # Multi-looking and output
    # ------------------------------------------------------------------

    P_t_ml = np.nansum(P_t_full, axis=0)

    P_t_full_comp = sigma_0_tracer * P_t_full[:, :, None]
    #P_t_full = P_t_full.T
    #P_t_full_comp = np.transpose(P_t_full_comp, (1, 0, 2))
    P_t_ml_comp = np.nansum(P_t_full_comp, axis=1)

    return P_t_full, P_t_ml, P_t_full.T, P_t_ml_comp
