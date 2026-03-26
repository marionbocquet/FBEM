
import numpy as np
from numpy import sin, cos, tan, sqrt, exp, log10, log
from scipy.constants import pi
import numpy as npimport

def rsgene2D_anisotrop_old( N , M, rL , rW, h , Lx , Ly , LN):

    """
    # generates a square 2-dimensional random rough surface f(x,y) with NxN 
    # surface points. The surface has a Gaussian height distribution and 
    # exponential autocovariance functions (in both x and y), where rL is the 
    # length of the surface side, h is the RMS height and clx and cly are the 
    # correlation lengths in x and y. 
    #
    # Input:    N   - number of surface points (along square side)
    #           rL  - length of surface (along square side)
    #           h   - rms height
    #           L - correlation length (in x and y)
    #
    # Output:   f  - surface heights
    #           x  - surface points
    #           y  - surface points
    #

    # (C) Jack Landy & Alex Komarov, 2014 (adapted from code originally
    # developed by David Bergstr�m)
        
    """
    np.set_printoptions(precision=15, suppress=False)

    x = np.linspace(-rL/2, rL/2, int(N)) 
    y = np.linspace(-rW/2, rW/2, int(M))
    X, Y = np.meshgrid(x, y)

    if LN == 0:
        mu_norm = 0
        sigma_norm = h
    else:
        # calculate required normal parameters, np.mean = 1
        mu_norm = log(1**2/sqrt(h**2 + 1**2))
        sigma_norm = sqrt(log(h**2/1**2 + 1))

    Z = sigma_norm * np.random.randn(int(M), int(N)) + mu_norm # uncorrelated Gaussian random rough surface distribution                         # with rms height h
                                    
    dx = rL / (N-1)

    # Exponential correlation function
    C = exp(- sqrt((X / Lx)**2 + (Y / Ly)**2 ) )  
    #fft2_F = sqrt(np.fft.fft2(C))
    C_shifted = np.fft.ifftshift(C)
    fft2_C = np.fft.fft2(C_shifted)
    fft2_C[fft2_C < 0] = 0
    fft2_F = np.sqrt(fft2_C)
    # correlation of surface including convolution (faltung), inverse
    # Fourier transform and normalizing prefactors
    fft2_Z = np.fft.fft2(Z)

    f = np.real(np.fft.ifft(fft2_Z * fft2_F))     #using our method

    # scale to original parameter values
    f = f * sigma_norm/np.std(f[:])
    f = f - (np.mean(f[:]) - mu_norm)

    if LN == 1:
        f = exp(f) - 1

    return f, x, y

import numpy as np
from numpy import exp, sqrt, log

def rsgene2D_anisotrop(N, M, rL, rW, h, Lx, Ly, LN):
    """
    Generate a 2D random rough surface with Gaussian or lognormal height distribution
    and exponential autocorrelation (anisotropic).

    Inputs:
        N, M : number of points along x and y
        rL, rW : physical size along x and y (meters)
        h : rms height
        Lx, Ly : correlation lengths in x and y
        LN : 0 = Gaussian, 1 = Lognormal

    Outputs:
        f : surface height matrix (M x N)
        x, y : coordinates along x and y
    """
    N = int(round(N))
    M = int(round(M))    # coordinates
    x = np.linspace(-rL/2, rL/2, N)
    y = np.linspace(-rW/2, rW/2, M)
    X, Y = np.meshgrid(x, y)

    # lognormal params if needed
    if LN == 0:
        mu_norm = 0
        sigma_norm = h
    else:
        mu_norm = log(1**2 / sqrt(h**2 + 1**2))
        sigma_norm = sqrt(log(h**2/1**2 + 1))

    # uncorrelated Gaussian/Lognormal surface
    Z = sigma_norm * np.random.randn(M, N) + mu_norm

    # correlation function
    C = exp(-sqrt((X / Lx)**2 + (Y / Ly)**2))
    
    # **critical fix**: centre the correlation for FFT
    C = np.fft.ifftshift(C)

    # FFT convolution
    fft2_F = np.sqrt(np.fft.fft2(C))
    fft2_Z = np.fft.fft2(Z)
    f = np.real(np.fft.ifft2(fft2_Z * fft2_F))

    # normalize to original rms
    f = f * sigma_norm / np.std(f)
    f = f - (np.mean(f) - mu_norm)

    # lognormal transform if needed
    if LN == 1:
        f = np.exp(f) - 1

    return f, x, y