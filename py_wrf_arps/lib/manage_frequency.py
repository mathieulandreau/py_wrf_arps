import numpy as np
import scipy
import datetime
import matplotlib.pyplot as plt
from ..lib.manage_time import *


def welch(TIME, U, window_minutes=30) :
    """
    Description
        Compute scipy.signal.welch from a time signal
    Parameters
        TIME : list or np.array of length NT, elements are float, datetime.datetime or datetime.timedelta : Time vector (assuming in seconds if type is float)
        U : list or np.array of length NT, elements are float : Velocity vector
    Optional 
        window_minutes : float : size of the windows in minutes
    Returns 
        int : index of the domain in self.tab_dom
    """
    NT = len(TIME)
    if type(TIME[0]) == datetime.datetime :
        T = manage_time.timedelta_to_seconds(TIME - TIME[0])
    elif type(TIME[0]) == datetime.timedelta :
        T = manage_time.timedelta_to_seconds(TIME)
    dt = T[1] - T[0]
    fs = 1/dt
    nperseg = (window_minutes*60)//dt
    frequencies, psd = scipy.signal.welch(U, fs=fs, nperseg=nperseg)
    return frequencies, psd

def fft2(Z, DX=None, DY=None, X=None, Y=None, plot=False, find_max=False):
    """
    Description
        Compute scipy.fftpack.fft2
    Parameters
        Z : 2D array of shape (NY, NX)
    Returns 
        KX, KY, D2 : 2D arrays of shape (NY, NX), D2 is the norm of the fft2 result
    """
    if DX is None :
        if X is None :
            DX = 1
        else :
            DX = np.mean(np.diff(X, axis=1))
    if DY is None :
        if Y is None :
            DY = DX
        else :
            DY = np.mean(np.diff(Y, axis=0))
    NY, NX = Z.shape
    kx = scipy.fft.fftfreq(NX, DX)
    ky = scipy.fft.fftfreq(NY, DY)
    KX, KY = np.meshgrid(kx, ky)
    D1 = scipy.fftpack.fftshift(scipy.fftpack.fft2(Z))
    D2 = np.abs(D1)
    KX = scipy.fftpack.fftshift(KX)
    KY = scipy.fftpack.fftshift(KY)
    
    # D2 = np.abs(D1)
    out = (KX, KY, D2)
    if plot :
        plt.figure()
        s = plt.scatter(KX, KY, c=np.log(np.abs(D1)), cmap="Spectral_r")
        plt.xlabel("kx")
        plt.ylabel("ky")
        cbar = plt.colorbar(mappable=s)
        cbar.set_label("np.log(np.abs(D1))")
    if find_max :
        i = np.nanargmax(D1)
        kxm, kym = KX.flatten()[i], KY.flatten()[i]
        lamx = 1/kxm
        lamy = 1/kym
        lam = np.sqrt(lamx**2 + lamy**2)
        angleNord = np.rad2deg(np.arctan2(lamy, lamx))
        out = out + (lamx, lamy, lam, angleNord)
    return out