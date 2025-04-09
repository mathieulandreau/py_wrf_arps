import numpy as np
import scipy

def peak_prominences_widths(MH, IZ):
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(MH, IZ)
    widths, _, left_ips, right_ips = scipy.signal.peak_widths(MH, IZ, prominence_data=(prominences, left_bases, right_bases))
    return prominences, widths, left_ips, right_ips

def detect_LLJ(MH_in, Z_in, IZ_in, DZ_in, zaxis, max_height=500, prom_abs=2, prom_rel=0.2, width=50, squeeze=False) :
    """ Detect the presence of LLJ by searching a peak in the wind speed profile below max_height meters, and return the core speed, height, and vertical index
        Partly based on Visich, Aleksandra, et Boris Conan. 2025. https://doi.org/10.1016/j.oceaneng.2025.120749.
    Parameters
        MH_in (np.array): Horizontal wind speed array of shape (..., NZ, ...)
        Z_in (np.array): Height array of shape (..., NZ, ...)
        IZ_in (1D np.array): Vertical indices array of shape (NZ)
        DZ_in (np.array): Vertical grid spacing array of shape (..., NZ, ...)
        zaxis (int): index of the Z axis
    Optional
        max_height (float): maximum height up to which the peak is searched, default=500
        prom_abs (float): minimum absolute prominence of the peak (see scipy.signal.find_peak), default=2 (see Visich and Conan, step 2)
        prom_rel (float): minimum prominence relative to the peak velocity (see scipy.signal.find_peak), default=0.2 (see Visich and Conan, step 2)
        width (float): minimum width of the peak in meters (see scipy.signal.find_peak), default=50
    Return
        LLJ (np.array): Status of the detection (0=no LLJ, 1=LLJ, 2=several peaks, 3=peak with small width, 4=peak with small prominence, 5=peak at higher altitude), shape=(..., 1, ...)
        LLJ_IZ (np.array): Vertical index of the LLJ core, shape=(..., 1, ...)
        LLJ_Z (np.array): LLJ core height, shape=(..., 1, ...)
        LLJ_MH (np.array): LLJ core speed, shape=(..., 1, ...)
        LLJ_PROM (np.array): Prominence of the peak found, shape=(..., 1, ...)
        LLJ_WIDTH (np.array): Width of the peak found, shape=(..., 1, ...)
    20/03/2025 : Mathieu Landreau
    """  
    # 
    DZ0 = np.take(DZ_in, 0, axis=zaxis)
    shape = DZ0.shape
    MH = np.insert(MH_in, 0, np.zeros(shape), axis=zaxis)
    Z = np.insert(Z_in, 0, np.zeros(shape), axis=zaxis)
    DZ = np.insert(DZ_in, 0, DZ0, axis=zaxis)
    IZ = np.append(IZ_in, np.nanmax(IZ_in)+1)
    
    # Roughly step 1 and 2 from Visich and Conan, 2025
    temp = np.apply_along_axis(peak_prominences_widths, zaxis, MH, IZ)
    PROM, IZ0, IZ1 = np.take(temp, 0, axis=zaxis), np.take(temp, 2, axis=zaxis), np.take(temp, 3, axis=zaxis)
    temp = np.apply_along_axis(peak_prominences_widths, zaxis, MH, IZ)
    IZ_to_Z = scipy.interpolate.interp1d(IZ, Z, axis=zaxis)
    Z0, Z1 = interpolate_along_axis(Z, IZ0, axis=zaxis), interpolate_along_axis(Z, IZ1, axis=zaxis)
    WIDTH = Z1 - Z0 #np.expand_dims(Z1 - Z0, axis=zaxis)
    m1 = 1*np.logical_or(PROM>=prom_abs, np.logical_and(PROM>=prom_rel*MH, PROM>=1)) # Roughly equivalent to their step 2
    m2 = 1*(WIDTH>=width)
    m3 = 1*(Z<=max_height)
    m4 = 1*(Z>max_height)*(Z<=max(2*max_height, 1000))
    LLJ = 1*(np.sum(m1*m2*m3, axis=zaxis, keepdims=True)==1) #there is only one valid peak
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m2*m3, axis=zaxis, keepdims=True)>1)] = 2 #there are several valid peaks
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m3, axis=zaxis, keepdims=True)>0)] = 3 #there is a valid peak with a too small width
    LLJ[np.logical_and(LLJ==0, np.sum(m2*m3, axis=zaxis, keepdims=True)>0)] = 4 #there is a valid peak with a too small prominence
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m2*m4, axis=zaxis, keepdims=True)>0)] = 5 #there is a valid peak at higher altitude
    LLJ = LLJ.astype(int)
    
    LLJ_IZ = np.expand_dims(np.nanargmax(m1*m2*m3, axis=zaxis), axis=zaxis)
    LLJ_Z = np.take_along_axis(Z, LLJ_IZ, axis=zaxis)
    LLJ_MH = np.take_along_axis(MH, LLJ_IZ, axis=zaxis)
    LLJ_PROM = np.take_along_axis(PROM, LLJ_IZ, axis=zaxis)
    LLJ_WIDTH = np.take_along_axis(WIDTH, LLJ_IZ, axis=zaxis)
    LLJ_IZ[LLJ != 1] = -1
    LLJ_Z[LLJ != 1] = LLJ_MH[LLJ != 1] = LLJ_PROM[LLJ != 1] = LLJ_WIDTH[LLJ != 1] = np.nan
    LLJ_IZ = LLJ_IZ-1
    if squeeze :
        LLJ = np.squeeze(LLJ)
        LLJ_IZ = np.squeeze(LLJ_IZ)
        LLJ_MH = np.squeeze(LLJ_MH)
        LLJ_Z = np.squeeze(LLJ_Z)
        LLJ_PROM = np.squeeze(LLJ_PROM)
        LLJ_WIDTH = np.squeeze(LLJ_WIDTH)
    return LLJ, LLJ_IZ, LLJ_Z, LLJ_MH, LLJ_PROM, LLJ_WIDTH

def interpolate_along_axis(data, indices, axis=0):
    """ Interpolate values in `data` along a given axis using fractional indices in `indices`.
    Author 
        ChatGPT-4-turbo
    Parameters
        data (np.ndarray): N-dimensional array of real values.
        indices (np.ndarray): Array of fractional indices (same shape as data) indicating where to interpolate.
        axis (int): Axis along which to interpolate.
    Returns
        interpolated (np.ndarray): Interpolated array (same shape as data).
    """
    data = np.asarray(data)
    indices = np.asarray(indices)
    # Ensure indices match shape
    if data.shape != indices.shape:
        raise ValueError("`indices` must have the same shape as `data`.")
    shape = data.shape
    ndim = data.ndim
    size_axis = shape[axis]
    # Clip indices to be within valid range for interpolation
    i0 = np.floor(indices).astype(int)
    i1 = i0 + 1
    i0 = np.clip(i0, 0, size_axis - 1)
    i1 = np.clip(i1, 0, size_axis - 1)
    w = indices - i0  # interpolation weights
    # Build indexing arrays
    slicer = [np.arange(s) for s in shape]
    grids = np.meshgrid(*slicer, indexing='ij')
    # Replace the axis-th dimension with i0 and i1
    idx0 = list(grids)
    idx1 = list(grids)
    idx0[axis] = i0
    idx1[axis] = i1
    val0 = data[tuple(idx0)]
    val1 = data[tuple(idx1)]
    return (1 - w) * val0 + w * val1