#!/usr/bin/env python3
import numpy as np
import scipy

def gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    taken from scipy source code : https://github.com/scipy/scipy/blob/main/scipy/ndimage/_filters.py
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x

def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0, *, radius=None,
                    axes=None):  
    """
    taken from scipy source code : https://github.com/scipy/scipy/blob/main/scipy/ndimage/_filters.py
    """ 
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)

    axes = _ni_support._check_axes(axes, input.ndim)
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
                              mode, cval, truncate, radius=radius)
            input = output
    else:
        output[...] = input[...]
    return output

def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):

    """
    taken from scipy source code : https://github.com/scipy/scipy/blob/main/scipy/ndimage/_filters.py
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)

def gradient_from_plane(X, Y, P, mask=None) :
    """ 
    Description
        The plane equation is P = a*X + b*Y + c
        This method fit the P points to a Plane. Then gradx = a and grady = b
    Parameters
        X, Y : ND arrays containing the coordinates of the points P
        P : ND array (same shape as X, Y) : values of P
    Optional
        mask : ND array (same shape as X, Y, P), Only the points with mask = 1 are used for the fit
    Returns 
        dPdx (a), dPdy (b), c : float 
        
    """
    if mask is None :
        mask = np.ones(P.shape)
    NP = int(np.sum(mask==1)) 
    points = np.zeros((NP, 3))
    points[:, 0] = X[mask==1].flatten()
    points[:, 1] = Y[mask==1].flatten()
    points[:, 2] = P[mask==1].flatten() 
    xs, ys, zs = zip(*points)
    A = np.hstack((points[:,:2], np.ones((len(xs),1))))
    b = points[:,2]
    res = scipy.optimize.lsq_linear(A, b)
    assert res.success
    a, b, c = res.x[0], res.x[1], res.x[2]
    return a, b, c #dPdx, dPdy, c