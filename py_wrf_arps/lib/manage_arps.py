#!/usr/bin/env python3

import numpy as np
from matplotlib import colors

def compute_gradient(X, arr):
    shape = X.shape
    Ndims = len(shape)
    assert arr.shape == shape
    s = (1,)
    s2 = (0,)
    axis_grad = -1
    axis = 0
    while axis_grad < 0 and axis < Ndims  :
        s = ()
        s2 = ()
        for i_temp in range(axis) :
            s = s + (0,)
            s2 = s2 + (0,)
        s = s + (1,)
        s2 = s2 + (0,)
        for i_temp in range(Ndims - axis - 1) :
            s = s + (0,)
            s2 = s2 + (0,)
        diff = X[s2] - X[s]
        if abs(diff) > 1e-5 :
            axis_grad = axis
        axis +=1
        print(s, s2, X[s2], X[s])
    assert axis_grad >= 0
    print(axis_grad)

def cat_weighted_mk(wght1, mk1, wght2, mk2):
    """
    Concatenate weighted moments.
    See biblio
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://doi.org/10.1007/s00180-015-0637-z

    Parameters
    ----------
    wght1: int, float or 1darray
        weight for mk1
        if int, assume weight of 1 for all mk
        if float, return a warning
    mk1: ndarray shape(k,...)
        k = 4 with k the moment
    wght2: int, float or 1darray
        weight for mk2
        if int, assume weight of 1 for all mk
        if float, return a warning
    mk2: ndarray shape(k,...)
        k = 4 with k the moment

    Optional
    ----------

    Returns
    ----------
    tuple: (wght, mk1)
        with wght and mk1 concatenated with wght2 and mk2

    Author(s)
    ----------
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """
    delete = False
    if type(wght1) in [int, float]:
        if int(wght1) == 0:
            delete = True
            wght1 = [0]
        elif type(wght1) is int:
            wght1 = np.ones(wght1)
        elif type(wght1) is float:
            print("WARNING: By giving a float as wght1, if each statistic '\
                  + 'doesn't have a weight of 1, the statistic computation will fail")
            wght1 = [wght1]
    if type(wght2) in [int, float]:
        if int(wght2) == 0:
            wght2 = [0]
        elif type(wght2) is int:
            wght2 = np.ones(wght2)
        elif type(wght2) is float:
            print("WARNING: By giving a float as wght2, if each statistic '\
                  + 'doesn't have a weight of 1, the statistic computation will fail")
            wght2 = [wght2]

    wght = np.concatenate((wght1, wght2))
    if delete:
        wght = np.delete(wght,0)

    wsum1 = np.sum(wght1)
    wsum2 = np.sum(wght2)
    wsum = wsum1 + wsum2
    w1_w = wsum1 / wsum
    w2_w = wsum2 / wsum
    c1_3 = 3.0
    c1_4 = 4.0
    c2_4 = 6.0

    delta = mk2[0,...] - mk1[0,...]

    mk1[0,...] = mk1[0,...] + w2_w*delta

    mk1[3,...] = mk1[3,...] + mk2[3,...] \
        + wsum1*(-w2_w*delta)**4 + wsum2*(w1_w*delta)**4 \
        + c1_4*delta    * ( mk1[2,...]*(-w2_w)    + mk2[2,...]*w1_w    ) \
        + c2_4*delta**2 * ( mk1[1,...]*(-w2_w)**2 + mk2[1,...]*w1_w**2 )

    mk1[2,...] = mk1[2,...] + mk2[2,...] \
        + wsum1*(-w2_w*delta)**3 + wsum2*(w1_w*delta)**3 \
        + c1_3*delta    * ( mk1[1,...]*(-w2_w)    + mk2[1,...]*w1_w    )

    mk1[1,...] = mk1[1,...] + mk2[1,...] \
        + wsum1*(-w2_w*delta)**2 + wsum2*(w1_w*delta)**2

    return wght, mk1

def calc_weighted_stats(wght, mk):
    """
    Calculate unbiased weighted statistic from moments.
    See biblio
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://doi.org/10.48550/arXiv.1304.6564

    Parameters
    ----------
    wght: int or 1darray
        weight for mk1
        if int, assume weight of 1 for all stats
    mk: ndarray shape(k,...)
        k = 4 with k the moment

    Optional
    ----------

    Returns
    ----------
    tuple: (wght, stats)
        with wght and stats the weight ans statistics.
        stats[0,...] = mean
        stats[1,...] = unbiased variance (squared standart deviation)
        stats[2,...] = unbiased skewness
        stats[3,...] = unbiased Fisher kurtosis (= 0 for normal law)

    Author(s)
    ----------
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """

    if type(wght) in [int, float]:
        if type(wght) is float:
            print("WARNING: By giving a float as weights, if each statistic '\
                  + 'doesn't have a weight of 1, the statistic computation will be wrong")
            wght = np.ones(int(wght))
        elif type(wght) is int:
            wght = np.ones(wght)

    wsum1 = np.sum(wght)
    wsum2 = np.sum(wght**2)
    wsum3 = np.sum(wght**3)
    wsum4 = np.sum(wght**4)
    three_two = 3.0 / 2.0

    # Unbiased moments
    mk[3,...] = wsum1 * (wsum1**4 - 3*wsum1**2*wsum2 \
                            + 2*wsum1*wsum3 + 3*wsum2**2 - 3*wsum4) \
                    / ( (wsum1**2 - wsum2) \
                      * (wsum1**4 - 6*wsum1**2*wsum2 + 8*wsum1*wsum3 \
                         + 3*wsum2**2 - 6*wsum4)) \
                    * mk[3,...] \
                  - 3* wsum1 * (2*wsum1**2*wsum2 - 2*wsum1*wsum3 \
                                 - 3*wsum2**2 + 3*wsum4) \
                    / ( (wsum1**2 - wsum2) \
                      * (wsum1**4 - 6*wsum1**2*wsum2 + 8*wsum1*wsum3 \
                         + 3*wsum2**2 - 6*wsum4)) \
                    * mk[1,...]**2

    mk[2,...] = wsum1**2 / (wsum1**3 - 3*wsum1*wsum2 + 2*wsum3) \
                  * mk[2,...]

    mk[1,...] = wsum1 / (wsum1**2 - wsum2) * mk[1,...]

    # Kurtosis and skewness
    mk[3,...] = mk[3,...] / mk[1,...]**2 - 3

    mk[2,...] = mk[2,...] / mk[1,...]**three_two

    return wght, mk

def cat_weighted_cov(wght1, mean1x, mean1y, cov1, wght2, mean2x, mean2y, cov2):
    """
    Concatenate weighted covariance.
    See biblio
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://doi.org/10.1007/s00180-015-0637-z

    Parameters
    ----------
    wght1: int, float or 1darray
        weight for cov1
        if int, assume weight of 1
        if float, return a warning
    mean1x: ndarray
        mean of x (could be mk1[0,...])
    mean1y: ndarray
        mean of y (could be mk1[0,...])
    cov1: ndarray
        covariance of xy
    wght2: int, float or 1darray
        weight for cov2
        if int, assume weight of 1
        if float, return a warning
    mean2x: ndarray
        mean of x (could be mk2[0,...])
    mean2y: ndarray
        mean of y (could be mk2[0,...])
    cov2: ndarray
        covariance of xy

    Optional
    ----------

    Returns
    ----------
    tuple: (wght, mean1x, mean1y, cov1)
        with wght1, mean1x, mean1y, cov1 concatenated with wght2, mean2x, mean2y, cov2

    Author(s)
    ----------
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """
    delete = False
    if type(wght1) in [int, float]:
        if int(wght1) == 0:
            delete = True
            wght1 = [0]
        elif type(wght1) is int:
            wght1 = np.ones(wght1)
        elif type(wght1) is float:
            print("WARNING: By giving a float as wght1, if the covariance '\
                  + 'doesn't have a weight of 1, the covariance computation will fail")
            wght1 = [wght1]
    if type(wght2) in [int, float]:
        if int(wght2) == 0:
            wght2 = [0]
        elif type(wght2) is int:
            wght2 = np.ones(wght2)
        elif type(wght2) is float:
            print("WARNING: By giving a float as wght2, if the covariance '\
                  + 'doesn't have a weight of 1, the covariance computation will fail")
            wght2 = [wght2]

    wght = np.concatenate((wght1, wght2))
    if delete:
        wght = np.delete(wght,0)

    wsum1 = np.sum(wght1)
    wsum2 = np.sum(wght2)
    wsum = wsum1 + wsum2
    w2_w = wsum2 / wsum
    w12_w = wsum1*wsum2 / wsum

    deltax = mean2x - mean1x
    deltay = mean2y - mean1y

    mean1x = mean1x + w2_w*deltax
    mean1y = mean1y + w2_w*deltay
    cov1 = cov1 + cov2 + deltax*deltay*w12_w

    return wght, mean1x, mean1y, cov1

def calc_weighted_cov(wght, cov):
    """
    Calculate unbiased weighted statistic from moments.
    See biblio
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        https://doi.org/10.48550/arXiv.1304.6564

    Parameters
    ----------
    wght: int or 1darray
        weight for mk1
        if int, assume weight of 1 for all stats
    mk: ndarray shape(k,...)
        k = 4 with k the moment

    Optional
    ----------

    Returns
    ----------
    tuple: (wght, stats)
        with wght and stats the weight ans statistics.
        stats[0,...] = mean
        stats[1,...] = unbiased variance (squared standart deviation)
        stats[2,...] = unbiased skewness
        stats[3,...] = unbiased Fisher kurtosis (= 0 for normal law)

    Author(s)
    ----------
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """

    if type(wght) in [int, float]:
        if type(wght) is float:
            print("WARNING: By giving a float as weights, if the covariance '\
                  + 'doesn't have a weight of 1, the covariance computation will be wrong")
            wght = np.ones(int(wght))
        elif type(wght) is int:
            wght = np.ones(wght)

    wsum = np.sum(wght)

    cov = cov / (wsum - 1)

    return wght, cov


def set_vmin_vmax_norm(data,fvmin,fvmax,Norm,varnb):
    """
    set vmin, vmax and norm for plot

    Parameters
    ----------
    data: numpy.ndarray
    fvmin: (list of) None, int or float
    fvmax: (list of) None, int or float
    Norm: (list of) matplotlib.colors.[Norm] or None
    varnb: int
        to loop through list

    Optional
    ----------

    Returns
    ----------
    data,vmin,vmax,norm

    Author(s)
    ----------
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """
    if fvmin is None:
        vmin = np.nanmin(data)
    elif type(fvmin) in (int,float):
        vmin = fvmin
    else:
        vmin = fvmin[varnb]

    if fvmax is None:
        vmax = np.nanmax(data)
    elif type(fvmax) in (int,float):
        vmax = fvmax
    else:
        vmax = fvmax[varnb]

    if Norm is None:
        norm = colors.Normalize
    elif type(Norm) is list:
        norm = Norm[varnb]
    else:
        norm = Norm

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    if norm is None:
        norm = colors.Normalize

    # due to log scale, remove very small values
    if norm is colors.LogNorm:
        data = np.where(data<0.1,0.1,data)
        vmin = 1.0

    return data, vmin, vmax, norm