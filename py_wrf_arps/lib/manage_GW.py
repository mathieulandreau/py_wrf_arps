import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt

from ..lib import manage_time, manage_angle, manage_scipy, manage_images

def GW_variance(A, OMEGA, PHI, DT=600):
    """
    28/10/2024 : Mathieu Landreau
    Description
        Determine la variance théorique mesurée en un point fixe à l'oscillation d'une onde d'équation A*sin(OMEGA.T + PHI) pendant une durée DT (en principe inférieure à la période)
    Parameters
        A : ND numpy array or float : Amplitude of the oscillation
        OMEGA : ND numpy array or float : Frequency of the wave from an Eulerian frame (different from the intrinsic frequency often called omega)
        PHI : ND numpy array or float : phase of the oscillation 
    Optional
        DT : float : duration of the variance measurement (s), default 600 (10 min)
    Output
        varGW : same shape as A : variance
    """
    var1 = 0.5*A**2
    var2 =  - 0.25*A**2/(OMEGA*DT) * (np.sin(2*PHI+OMEGA*DT) - np.sin(2*PHI-OMEGA*DT))
    var3 = - (  A/(OMEGA*DT) * (np.cos(PHI+0.5*OMEGA*DT) - np.cos(PHI-0.5*OMEGA*DT))  )**2
    varGW = var1+var2+var3
    return varGW

def get_GWzone(W, CDI, DY, DX, threshold=1.45, display=False) : 
    """
    28/10/2024 : Mathieu Landreau
    Description
        Determine the area of the gravity waves
        Méthode : compute the spatially-averaged standard deviation of W, keep only offshore points, keep values above a threshold,
        and then use binary opening-closing methods to fill the shape
    Parameters
        W : numpy array of shape (NT, NY, NX) : Amplitude of vertical wind speed in space
        CDI : numpy array of shape (NY, NX) : distance from coast (computed with manage_images.get_COAST_DIST)
        DY, DX : float : horizontal grid size in meters
    Optional
        threshold : float : minimum std value to be in the mask
    Output
        velocities : numpy array of shape (NT-dit) : wave speed
        output_TIME : numpy array of shape (NT-dit) : average time of each correlation
    """
    # for idim in range(W.ndim - CDI.ndim) :
    #     CDI = np.expand_dims(CDI, axis=0)
    Ws = manage_scipy.gaussian_std(W, DX=DX, DY=DY, sigma=500)
    Ws[np.broadcast_to(CDI > -2000, Ws.shape)] = np.nan
    Ws = Ws / np.nanmean(Ws, axis=(-2, -1), keepdims=True)
    mask = 1*(Ws > threshold)
    """
    grad_y, grad_x = np.gradient(W2, DY, DX, axis=(-2, -1))
    GY = np.sqrt(manage_scipy.gaussian_smooth(grad_y**2, DX_smooth, DX))
    GX = np.sqrt(manage_scipy.gaussian_smooth(grad_x**2, DX_smooth, DX))
    angles = np.arctan2(GX, GY)

    grad_r = grad_x*np.cos(angles) + grad_y*np.sin(angles)
    grad_t = -grad_x*np.sin(angles) + grad_y*np.cos(angles)
    GR = np.sqrt(manage_scipy.gaussian_smooth(grad_r**2, DX_smooth, DX))
    GT = np.sqrt(manage_scipy.gaussian_smooth(grad_t**2, DX_smooth, DX))
    mask2 = GR/GT < 0.5
    """
    kernel = scipy.ndimage.generate_binary_structure(2, 1)
    if mask.ndim == 2 :
        mask = np.expand_dims(mask, axis=0)
    m = np.copy(mask)
    NT = len(m)
    for it in range(NT) :
        for i in range(1, 15, 3) :
            m[it] = scipy.ndimage.morphology.binary_opening(m[it], structure=kernel, iterations=i)
            m[it] = scipy.ndimage.morphology.binary_closing(m[it], structure=kernel, iterations=i)
        m[it] = scipy.ndimage.morphology.binary_fill_holes(m[it])
        m[it] = manage_images.get_main_area(m[it])
    if display :
        it = NT//2
        fig = plt.figure(figsize=[8, 8])
        plt.pcolor(W[it], vmin=-1, vmax=1, cmap="bwr")
        plt.colorbar()
        plt.contour(CDI, levels=[0], colors=["k"])
        plt.contour(m[it], levels=[0.5], colors=["r"], linewidths=[3])
        plt.contour(mask[it], levels=[0.5], colors=["g"], linewidths=[1])
    return np.squeeze(m)


def compute_average_amplitude(A_in, mask=None, display=False) :
    """
    25/10/2024 : Mathieu Landreau
    Description
        Determine the velocity of a moving 2D plane wave thanks to correlation between 2 consecutive images
        Written with chatGPT4
    Parameters
        A : numpy array of shape (..., NY, NX) : Amplitude of the plane wave in space
        DY, DX : float : horizontal grid size in meters
    Optional
        mask : same shape as A : area of the GW
    Output
        amplitude : float or numpy array with ndim = A.ndim-2 (X and Y dimensions vanish)
    """
    A = np.copy(A_in)
    A[mask==0] = np.nan
    amplitude = np.nanstd(A, axis=(-1, -2))*np.sqrt(2)
    return amplitude

def compute_local_amplitude(A, DY, DX, sigma=1000) :
    """
    25/10/2024 : Mathieu Landreau
    Description
        Determine the local amplitude of a plane wave by computing the spatial standard deviation.
        For a theoretical plane wave, standard deviation = amplitude/sqrt(2)
    Parameters
        A : numpy array of shape (..., NY, NX) : Amplitude of the plane wave in space
        DY, DX : float : horizontal grid size in meters
    Optional
        sigma : size of the kernel for std computation
    Output
        amplitude : float or numpy array with ndim = A.ndim-2 (X and Y dimensions vanish)
    """
    return manage_scipy.gaussian_std(A, DX=DX, DY=DY, sigma=sigma) * np.sqrt(2)
    
def compute_wavespeed(A, DY, DX, TIME=None,  dt=None, dit=1, lam=None, GWD=None, mask=None, display=False, fac=1, method="fit") :
    """
    25/10/2024 : Mathieu Landreau
    Description
        Determine the velocity of a moving 2D plane wave thanks to correlation between 2 consecutive images
        Written with chatGPT4
    Parameters
        A : numpy array of shape (NT, NY, NX) : Amplitude of the plane wave in space
        DY, DX : float : horizontal grid size in meters
    Optional
        TIME : 1D numpy arrays of shape (NT) of datetime.datetime : time vector
        dt : float : time step in seconds, used only if TIME is None
        dit : timestep to compute wave speed, default = 1 (a value of 1 is advised for a better fit)
        lam, GWD : numpy array of shape (NT) : Wavelenght in meters and wave direction (°), if absent : computed with compute_wavelength and compute_wave_orientation_gradient
        display : boolean : True to plot figures of correlation and fit, default : False
        method : string : "fit" for fitting corr(R) with a given function of "max" to get the distance corresponding to the correlation maximum
    Output
        velocities : numpy array of shape (NT-dit) : wave speed
        output_TIME : numpy array of shape (NT-dit) : average time of each correlation
    """
    NT, _, _ = A.shape
    if GWD is None :
        GWD = compute_wave_orientation_gradient(A, DY, DX)
    if lam is None and method == "fit" :
        lam = compute_wavelength(A, DY, DX)
    if TIME is None :
        if dt is None : dt = 1
        TIME = np.linspace(0, dt*(NT-1), NT)
    else : #assuming constant timestep
        dt = manage_time.timedelta_to_seconds(TIME[1] - TIME[0])
    output_TIME = TIME[:-dit] + (TIME[dit:] - TIME[:-dit])/(2*dit)
    A = np.copy(A - np.mean(A, axis=(-2, -1), keepdims=True))
    velocities = np.zeros(NT - dit)  # Stocker les vitesses à chaque pas de temps
    for it in range(NT - dit):
        if mask is not None :
            mx = np.sum(mask[it:it+dit+1], axis=(0, -2)); my = np.sum(mask[it:it+dit+1], axis=(0, -1))
            ix1 = np.nanargmax(mx>0); iy1 = np.nanargmax(my>0)
            ix2 = ix1 + np.nanargmin(mx[ix1:]>0); iy2 = iy1 + np.nanargmin(my[iy1:]>0)
            s = (slice(iy1, iy2), slice(ix1, ix2))
        else :
            s = (slice(None), slice(None))
        # Extraire les amplitudes à l'instant t et t+1
        A_t0 = A[it][s]
        A_t1 = A[it+dit][s]
        GWD_it = scipy.stats.circmean(GWD[it:it+dit+1], high=180)
        # Calcul de la corrélation croisée entre A_t0 et A_t1
        NY, NX = A_t0.shape
        YC, XC, RC, TC = get_XY_for_correlate2D(NY, NX, DY, DX, GWD_it)
        corr = scipy.signal.correlate2d(A_t1, A_t0, mode='full')
        #generate DataFrame
        df = pd.DataFrame()
        df["RC"] = RC.flatten()
        df["TC"] = TC.flatten()
        df["corr"] = corr.flatten()
        if method == "fit" :
            lam_it = np.mean(lam[it:it+dit+1])
            #select a specific area and fit
            df = df[np.abs(df["TC"]) < 0.2*lam_it]
            df = df[np.abs(df["RC"]) < fac*lam_it]
            RC2 = df["RC"]
            corr2 = df["corr"]
            def model_corr2D(x, a1, b2, shift):
                return a1*np.cos(2*np.pi*(x-shift)/lam_it) * np.exp(-((x-shift)/b2)**2)
            [a1, b2, shift], _ = scipy.optimize.curve_fit(model_corr2D, RC2, corr2, p0 = [np.var(A_t0)*NX*NY, lam_it, 100])
        elif method == "max" :
            max_corr_idx = np.unravel_index(np.argmax(corr), corr.shape)
            shift = RC[max_corr_idx]
        else :
            raise(Exception(f"error : unknown method ({method}) in manage_GW.compute_wavespeed"))
        velocities[it] = shift / (dit*dt)
        #display result if desired
        if display :
            plt.figure(figsize = [24, 8])
            plt.subplot(121)#--------------
            plt.pcolor(XC, YC, corr); plt.colorbar()
            plt.title(f"correlation2D of A, it1={it}, it2={it+dit}")
            if method == "fit" :
                mask_d = np.abs(TC) < 0.5*lam_it
                mask_d = np.logical_and(mask_d, np.abs(RC) < fac*lam_it)
                plt.contour(XC, YC, mask_d*1, levels=[0.5], colors=["r"])
            plt.subplot(122)#--------------
            if method == "fit" :
                plt.plot(RC2, corr2, ".", label="correlation2D")
                plt.plot(np.sort(RC2), model_corr2D(np.sort(RC2), a1, b2, shift), "r-", label=f"fit, a1={round(a1)}, b2={round(b2)}, shift={round(shift)}")
                plt.xlabel("radial coordinate (m)")
                plt.ylabel("correlation2D")
                plt.grid()
                plt.legend()
    return velocities, output_TIME

def compute_wave_orientation_gradient(A, DY, DX, mask=None):
    """
    25/10/2024 : Mathieu Landreau
    Description
        Compute plane wave orientation with a gradient
        Written with chatGPT4
    Parameters
        A : numpy array of shape (..., NY, NX) : Amplitude of the plane wave in space
        DY, DX : float : horizontal grid size
    Output
        mean_angle : average angle in degrees, in [0, 180] (modulo 180)
        std_angle : standard deviation of the angle (if std is too high, the mean value might be discarded)
    """
    grad_y, grad_x = np.gradient(A, DY, DX, axis=(-2, -1))
    grad = np.sqrt(grad_x**2 + grad_y**2)
    angles = manage_angle.UV2WD_deg(grad_x, grad_y)
    angles = angles%180
    if mask is not None :
        grad[np.logical_not(mask)] = np.nan
        angles[np.isnan(grad)] = np.nan
    shape = angles.shape
    new_shape = shape[:-2] + (shape[-2] * shape[-1],)
    angles = np.reshape(angles, new_shape)
    grad = np.reshape(grad, new_shape)
    mean_angle = manage_angle.circmean(angles, weights=grad, high=180, axis=-1, nan_policy='omit')
    std_angle = manage_angle.circstd(angles, weights=grad, high=180, axis=-1, nan_policy='omit')
    return mean_angle, std_angle

def compute_wavelength(A, DY, DX, expected_lambda=1700, GWD=None, mask=None, **kwargs) :
    """
    25/10/2024 : Mathieu Landreau
    Description
        Call compute_wavelength_2D for each timestep. See compute_wavelength_2D for more info
    Output
        lam : np.array of shape (NT) : wavelength over time
    """
    NT, NY, NX = A.shape
    lam = []
    if GWD is None :
        GWD = compute_wave_orientation_gradient(A, DY, DX)
    for it in range(NT):
        if mask is None :
            maskit = None
        else :
            maskit = mask[it]
        lam.append(compute_wavelength_2D(A[it], DY, DX, expected_lambda=expected_lambda, GWD=GWD[it], mask=maskit, **kwargs))
    return lam

def compute_wavelength_2D(A_2D, DY, DX, expected_lambda=1700, GWD=None, mask=None, display=False, method="fit") :
    """
    25/10/2024 : Mathieu Landreau
    Description
        Determine the wavelength of a 2D plane wave thanks to autocorrelation
        Written with chatGPT4
    Parameters
        A_2D : numpy array of shape (NY, NX) : Amplitude of the plane wave in space
        DY, DX : float : horizontal grid size
    Optional
        expected_lambda : expected wavelenght in meters
        GWD : float : wave direction (°), if absent : computed with compute_wave_orientation_gradient
        mask : 2D array of shape (NY, NX) : mask array of the GW area, used to zoom on the GW area
        display : boolean : True to plot figures of correlation and fit, default : False
    Output
        lam : float : wavelength
    """
    if expected_lambda < 6*DX :
        expected_lambda = 6*DX
    if GWD is None :
        GWD = compute_wave_orientation_gradient(A_2D, DY, DX)
    if mask is not None :
        mx = np.sum(mask, axis=1); my = np.sum(mask, axis=0)
        ix1 = np.nanargmax(mx>0); iy1 = np.nanargmax(my>0)
        ix2 = ix1 + np.nanargmin(mx[ix1:]>0); iy2 = iy1 + np.nanargmin(my[iy1:]>0)
        s = (slice(iy1, iy2), slice(ix1, ix2))
    else :
        s = (slice(None), slice(None))
    A_2D = np.copy(A_2D[s])
    # Compute autocorrelation
    corr = scipy.signal.correlate2d(A_2D, A_2D, mode='full')
    # Create radial and longitudinal coordinate
    NY, NX = A_2D.shape
    YC, XC, RC, TC = get_XY_for_correlate2D(NY, NX, DY, DX, GWD)
    # Keep points in a specific area with pandas 
    df = pd.DataFrame()
    df["RC"] = RC.flatten()
    df["TC"] = TC.flatten()
    df["corr"] = corr.flatten()
    mask_d = np.abs(TC) < 0.2*expected_lambda
    if method == "max" : 
        mask_d = np.logical_and(mask_d, RC < 1.5*expected_lambda)
        mask_d = np.logical_and(mask_d, RC > 0.5*expected_lambda)
    elif method == "fit" : 
        # Select a specific area and fit
        mask_d = np.logical_and(mask_d, np.abs(RC) < 1.5*expected_lambda)
    else :
        raise(Exception(f"unknown method in manage_GW.compute_wavelength_2D : {method}"))
    
    df["mask"] = mask_d.flatten()
    df = df[df["mask"]]
    RC2 = df["RC"]
    corr2 = df["corr"]
    if method == "max" :
        # find the maximum
        dfmax = RC2[corr2 == np.max(corr2)]
        lam = np.nan
        if dfmax.shape[0] > 0 :
            lam = float(dfmax.iloc[0])
    elif method == "fit" :
        # fit with the model (damped oscillation)
        def model_autocorr_R(x, a1, b1, b2):
            return a1*np.cos(2*np.pi*x/b1) * np.exp(-(x/b2)**2)
        [a1, lam, b2], _ = scipy.optimize.curve_fit(model_autocorr_R, RC2, corr2, p0 = [np.var(A_2D)*NX*NY, expected_lambda, expected_lambda])
        #display result if desired
    if display :
        plt.figure(figsize = [24, 8])
        plt.subplot(121)#--------------
        plt.pcolor(XC, YC, corr); plt.colorbar()
        plt.contour(XC, YC, mask_d*1, levels=[0.5], colors=["r"])
        plt.title("autocorrelation of A_2D")
        plt.subplot(122)#--------------
        if method == "max" : plt.axvline(x=lam, color="r")
        plt.plot(RC2, corr2, ".", label="autocorrelation")
        if method == "fit" : plt.plot(np.sort(RC2), model_autocorr_R(np.sort(RC2), a1, lam, b2), "r-", label=f"fit, a1={round(a1)}, lam={round(lam)}, b2={round(b2)}")
        plt.xlabel("radial coordinate (m)")
        plt.ylabel("autocorrelation")
        plt.grid()
        plt.legend()
    return lam

def get_XY_for_correlate2D(NY, NX, DY, DX, GWD=0):
    """
    25/10/2024 : Mathieu Landreau
    Description
        Get the shift matrix XS, YS to analyze the result of np.correlate2D
        Written with chatGPT4
    Parameters
        NY, NX : integers : size of the array
        DY, DX : float : horizontal grid size
        GWD : float : wave direction (°), to define the RC direction
    Output
        XS, YS : np.array of shape (NY, NX)
    """
    # Définir les matrices de décalage en x et y
    decalage_y = np.arange(-(NY-1), NY)*DY
    decalage_x = np.arange(-(NX-1), NX)*DX
    # Convertir en grilles 2D de décalages
    YC, XC = np.meshgrid(decalage_y, decalage_x, indexing='ij')
    alpha = np.deg2rad(GWD)
    RC =  -XC*np.sin(alpha) - YC*np.cos(alpha)
    TC =  -YC*np.sin(alpha) + XC*np.cos(alpha)
    return YC, XC, RC, TC

#################################################################################################################################
######  Not used anymore but might be usefull later, same idea as compute_wave_velocities with method = max
################################################################################################################################# 

def subpixel_shift(corr, max_idx):
    """
    25/10/2024 : Mathieu Landreau
    Description
        Quadratic interpolation for subpixel adjustment around the maximum.
        Written with chatGPT4
    """
    y, x = max_idx
    # Extraire les valeurs de corrélation autour du maximum (sur une fenêtre 3x3)
    window = corr[y-1:y+2, x-1:x+2]
    # Interpolation quadratique pour affiner la position du maximum
    dx = (window[1, 2] - window[1, 0]) / (2 * (2 * window[1, 1] - window[1, 2] - window[1, 0]))
    dy = (window[2, 1] - window[0, 1]) / (2 * (2 * window[1, 1] - window[2, 1] - window[0, 1]))
    return dx, dy

def compute_wave_velocity_subpixel(A, dy, dx, TIME=None, dt=None, dit=1):
    """
    25/10/2024 : Mathieu Landreau
    Description
        Compute plane wave velocity and direction using a spatial correlation between 2 consecutive horizontal cross-section
        Written with chatGPT4
    Parameters
        A : 3D numpy array of shape (NT, NY, NX)
        dy, dx : float : horizontal grid size
    Optionals
        TIME : 1D numpy arrays of shape (NT) of datetime.datetime : time vector
        dt : float : time step in seconds, used only if TIME is None
    Output
        velocities, directions, output_TIME : 1D numpy array of shape (NT-1)
    """
    NT, NY, NX = A.shape
    if TIME is None :
        if dt is None : dt = 1
        TIME = np.linspace(0, dt*(NT-1), NT)
    else :
        #assuming constant timestep
        dt = manage_time.timedelta_to_seconds(TIME[1] - TIME[0])
    output_TIME = TIME[:-dit] + (TIME[dit:] - TIME[:-dit])/dit
    velocities = np.zeros(NT - dit)  # Stocker les vitesses à chaque pas de temps
    directions = np.zeros(NT - dit)  # Stocker les directions à chaque pas de temps
    for t in range(NT - dit):
        # Extraire les amplitudes à l'instant t et t+1
        A_t0 = A[t]
        A_t1 = A[t+dit]
        # Calcul de la corrélation croisée entre A_t0 et A_t1
        corr = scipy.signal.correlate2d(A_t1, A_t0, mode='full')
        # Trouver le décalage maximum de corrélation (indice entier)
        max_corr_idx = np.unravel_index(np.argmax(corr), corr.shape)
        # Calculer le décalage subpixel
        dx_shift, dy_shift = subpixel_shift(corr, max_corr_idx)
        # Décalage de la position centrale (avec les corrections subpixel)
        shift_y = (max_corr_idx[0] - (NY - 1)) + dy_shift
        shift_x = (max_corr_idx[1] - (NX - 1)) + dx_shift
        # Calculer le déplacement physique
        displacement_x = shift_x * dx
        displacement_y = shift_y * dy
        # Calculer la distance totale du déplacement
        displacement_total = np.sqrt(displacement_x**2 + displacement_y**2)
        # Calcul de la vitesse à ce pas de temps et de la direction
        velocities[t] = displacement_total / (dit*dt)
        directions[t] = manage_angle.UV2WD_deg(displacement_x, displacement_y)
    return velocities, directions, output_TIME