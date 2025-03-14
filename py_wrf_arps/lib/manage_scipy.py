from scipy.ndimage import _ni_support, gaussian_filter1d, uniform_filter1d
import operator
from collections.abc import Iterable
import numpy as np

# I cannot install the latest version of scipy (1.13.0), because of my other packages
# I only have the 1.7.3 version
# I needed the "axes" argument in the gaussian filter function to do horizontal smoothing of 3D variable (filter only two axis)
# So, I copied the latest version of the function from the github rep so that I can use it

def check_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    elif np.isscalar(axes):
        axes = (operator.index(axes),)
    elif isinstance(axes, Iterable):
        for ax in axes:
            axes = tuple(operator.index(ax) for ax in axes)
            if ax < -ndim or ax > ndim - 1:
                raise ValueError(f"specified axis: {ax} is out of range")
        axes = tuple(ax % ndim if ax < 0 else ax for ax in axes)
    else:
        message = "axes must be an integer, iterable of integers, or None"
        raise ValueError(message)
    if len(tuple(set(axes))) != len(axes):
        raise ValueError("axes must be unique")
    return axes

def my_gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0, *, radius=None,
                    axes=None):
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)

    axes = check_axes(axes, input.ndim)
    num_axes = len(axes)
    orders = _ni_support._normalize_sequence(order, num_axes)
    sigmas = _ni_support._normalize_sequence(sigma, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    radiuses = _ni_support._normalize_sequence(radius, num_axes)
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(num_axes) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output


def my_uniform_filter(input, size=3, output=None, mode="reflect",
                   cval=0.0, origin=0, *, axes=None):

    input = np.asarray(input)
    output = _ni_support._get_output(output, input,
                                     complex_output=input.dtype.kind == 'c')
    axes = check_axes(axes, input.ndim)
    num_axes = len(axes)
    sizes = _ni_support._normalize_sequence(size, num_axes)
    origins = _ni_support._normalize_sequence(origin, num_axes)
    modes = _ni_support._normalize_sequence(mode, num_axes)
    axes = [(axes[ii], sizes[ii], origins[ii], modes[ii])
            for ii in range(num_axes) if sizes[ii] > 1]
    if len(axes) > 0:
        for axis, size, origin, mode in axes:
            uniform_filter1d(input, int(size), axis, output, mode,
                             cval, origin)
            input = output
    else:
        output[...] = input[...]
    return output

def gaussian_smooth(A, X=None, DX=None, Y=None, DY=None, sigma=100e3) : 
    """
    Description
        General : Horizontally smooth an array with a gaussian filter
    Parameters
        A : 2D np.array of shape (NY, NX), or ND array of shape (..., NY, NX)
    Optional
        DX : float, default None, X grid size
        X : 2D np.arrays of shape (NY, NX) if ever DX is not given, it will be computed from X
        DY : float, default None, Y grid size
        Y : 2D np.arrays of shape (NY, NX) if ever DY is not given, it will be computed from Y. If neither Y or DY is input, DY=DX
        sigma : The standard deviation of the gaussian kernel
    Returns 
        smoothed A : same as A
    """
    if DX is None :
        if X is None :
            raise(Exception("error in manage_scipy.gaussian_smooth, X or DX must be in input"))
        else :
            DX = max(abs(X[0, 1] - X[0, 0]), abs(X[1, 0] - X[0, 0]))
    sigma_mask = sigma/DX
    if DY is not None :
        sigma_mask = (sigma_mask, sigma/DY)
    elif Y is not None :
        DY = max(abs(Y[0, 1] - Y[0, 0]), abs(Y[1, 0] - Y[0, 0]))
        sigma_mask = (sigma_mask, sigma/DY)
    return my_gaussian_filter(A, sigma=sigma_mask, truncate=5, axes=(-2, -1))

def gaussian_var(A, **k) : 
    """
    Description
        General : compute horizontally smoothed variance with a gaussian kernel
    Parameters
        A : 2D np.array of shape (NY, NX), or ND array of shape (..., NY, NX)
    Optional
        see gaussain_smooth
    Returns 
        smoothed variance : same type as A
    """
    return gaussian_smooth(A**2, **k) - gaussian_smooth(A, **k)**2

def gaussian_std(A, **k) : 
    """
    Description
        General : compute horizontally smoothed standard deviation with a gaussian kernel
    Parameters
        A : 2D np.array of shape (NY, NX), or ND array of shape (..., NY, NX)
    Optional
        see gaussain_smooth
    Returns 
        smoothed standard deviation : same type as A
    """
    return np.sqrt(gaussian_var(A, **k))