
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)
import matplotlib.tri as mtri

def plotting(topo_plot, echo_plot,
             PosT, t, P_t_ml, P_t_full, P_t_ml_comp, N_b, epsilon_b):
    """
    Python equivalent of the MATLAB function Plotting(...)

    Parameters
    ----------
    topo_plot : int
        If >0, plot the tetrahedral mesh of the snow-covered sea ice surface.
    echo_plot : int
        If >0, plot echo-related figures (multi-looked, single looks, DDM, components).
    PosT : ndarray (N, 3)
        Vertex positions of the mesh: columns are X, Y, Z.
    t : ndarray (M,)
        Time vector [s].
    P_t_ml : ndarray (M,)
        Multi-looked power echo vs time.
    P_t_full : ndarray (M, N_b)
        Single-look power echoes vs time for N_b looks.
    P_t_ml_comp : ndarray (M, K)
        Multi-looked component echoes (e.g., K=4: Snow Surf, Snow Vol, Ice Surf, Lead/Pond Surf).
    N_b : int
        Number of looks/beams.
    epsilon_b : float
        Look angle spacing (radians).
    """

    # --- Safety checks (optional but helpful) ---
    PosT = np.asarray(PosT)
    t = np.asarray(t).flatten()
    P_t_ml = np.asarray(P_t_ml).flatten()
    P_t_full = np.asarray(P_t_full)
    P_t_ml_comp = np.asarray(P_t_ml_comp)

    # Time in nanoseconds for plotting
    t_ns = t * 1e9

    # ---------------------------
    # Topography plot (3D trisurf)
    # ---------------------------
    if topo_plot > 0:
        fig = plt.figure(num=1); plt.clf()
        ax = fig.add_subplot(111, projection='3d')

        # Delaunay triangulation in 2D (x,y); trisurf with z
        tri = mtri.Triangulation(PosT[:, 0], PosT[:, 1])
        surf = ax.plot_trisurf(tri, PosT[:, 2], linewidth=0.0, antialiased=True, cmap='viridis')
        cbar = plt.colorbar(surf, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Z [m]')

        # Try to match MATLAB's view(-45, 80)
        ax.view_init(elev=80, azim=-45)

        ax.set_xlim([-1000, 1000])
        ax.set_ylim([-1000, 1000])

        ax.set_title('Snow-covered sea ice surface tetrahedral mesh')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        plt.tight_layout()
        plt.pause(0.001)
        plt.savefig('test_1.png')
    # ---------------------------
    # Echo plots
    # ---------------------------
    if echo_plot > 0:
        fig = plt.figure(num=2); plt.clf()

        # 1) Multi-looked power echo (normalized)
        ax1 = fig.add_subplot(2, 2, 1)
        max_ml = np.max(P_t_ml) if np.max(P_t_ml) != 0 else 1.0
        ax1.plot(t_ns, P_t_ml / max_ml, linewidth=2)
        ax1.set_xlim([-20, 80])
        ax1.set_ylim([0, 1])
        ax1.grid(True)
        ax1.set_title('Multi-looked power echo')
        ax1.set_xlabel('Time [ns]')
        ax1.set_ylabel('Normalized Power')

        # 2) Single-look power echoes (take columns 1:4:N_b in MATLAB -> Python 0-based)
        ax2 = fig.add_subplot(2, 2, 2)
        # MATLAB uses 1:4:N_b; Python 0-based so use indices: 0, 4, 8, ...
        idx = np.arange(0, N_b, 4)
        for i in idx:
            ax2.plot(t_ns, P_t_full[i, :], linewidth=1, label=f'{i+1}')  # label with MATLAB-like 1-based index
        ax2.set_xlim([-20, 80])
        ax2.grid(True)
        ax2.set_title('Single-look power echoes')
        ax2.set_xlabel('Time [ns]')
        ax2.set_ylabel('Power [W]')
        # Place legend "eastoutside" equivalent: use bbox_to_anchor
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

        # 3) Delay-Doppler Map: imagesc(t*1e9, m*epsilon_b*180/pi, P_t_full')
        ax3 = fig.add_subplot(2, 2, 3)
        # Create m = -(N_b-1)/2 : (N_b-1)/2 (integer sequence length N_b)
        m_center = (N_b - 1) / 2
        m = np.arange(-m_center, m_center + 1)  # floats if N_b even; works for plotting
        # Look angles in degrees
        look_deg = m * epsilon_b * 180.0 / np.pi

        # P_t_full' in MATLAB -> transpose in Python
        # Use imshow with extent to mimic imagesc. Ensure shape (N_b, M) after transpose.
        ddm = P_t_full.T  # shape: (N_b, M)
        extent = [t_ns.min(), t_ns.max(), look_deg.min(), look_deg.max()]
        im = ax3.imshow(ddm, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        ax3.set_xlim([-20, 80])
        ax3.set_title('Delay-Doppler Map')
        ax3.set_xlabel('Time [ns]')
        ax3.set_ylabel(f'Look Angle [\N{DEGREE SIGN}]')
        cbar3 = plt.colorbar(im, ax=ax3)
        cbar3.set_label('Power [W]')

        # 4) Multi-looked component echoes
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t_ns, P_t_ml_comp, linewidth=1)
        ax4.set_xlim([-20, 80])
        ax4.grid(True)
        ax4.set_title('Multi-looked component echoes')
        ax4.set_xlabel('Time [ns]')
        ax4.set_ylabel('Power [W]')

        # Legend with components names (assume 4 if provided)
        comp_names = ['Snow Surf', 'Snow Vol', 'Ice Surf', 'Lead/Pond Surf']
        # Use only as many labels as available in columns
        k = P_t_ml_comp.shape[1] if P_t_ml_comp.ndim == 2 else 1
        labels = comp_names[:k] if k <= len(comp_names) else [f'Comp {i+1}' for i in range(k)]
        ax4.legend(labels, loc='upper right')

        plt.tight_layout()
        plt.pause(0.001)
        plt.savefig('test_2.png')
