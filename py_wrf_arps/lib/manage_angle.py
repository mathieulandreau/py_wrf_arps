import numpy as np
import scipy
from ..lib import manage_list

debug = False

def split_angles(TIME, WD, anglemin=-180, anglemax=180):
    """
    When ploting an angle in front of time with a line, since it is periodic, the line should go out of the plot to one side and
    come back from another side. This function is preparing the vector so that it does this instead of drawing a line crossing
    the whole plot.
    """
    NWD = len(WD)
    for iWD in range(NWD-1) :
        if np.abs(WD[iWD] - WD[iWD+1]) > (anglemax-anglemin)/2 :
            if WD[iWD] < WD[iWD+1] :
                dist1 = WD[iWD] - anglemin
                dist2 = anglemax - WD[iWD+1]
                wd1 = anglemin
                wd2 = anglemax
            else :
                dist1 = anglemax - WD[iWD]
                dist2 = WD[iWD+1] - anglemin
                wd1 = anglemax
                wd2 = anglemin
            dt = TIME[iWD+1] - TIME[iWD]
            t2 = TIME[iWD] + dist1/(dist1+dist2) * dt
            WD = np.insert(WD, iWD+1, wd1)
            WD = np.insert(WD, iWD+2, np.nan)
            WD = np.insert(WD, iWD+3, wd2)
            TIME = np.insert(TIME, iWD+1, t2)
            TIME = np.insert(TIME, iWD+2, t2)
            TIME = np.insert(TIME, iWD+3, t2)
            iWD += 3
    return TIME, WD

def wind2trigo_rad(angle) :
    """
    transform a radian angle from Wind (clockwise from North) to trigo (counter-clockwise from West)
    Note : the opposite transform is exactly the same so it also transform from trigo to wind
    """
    angle = 0.5*np.pi - angle # included in [-0.5*np.pi, 1.5*np.pi]
    return angle2pi(angle)

def trigo2wind_rad(angle) :
    return wind2trigo_rad(angle) #same operation
  
def wind2trigo_deg(angle) :
    """
    transform a degree angle from Wind (clockwise from North) to trigo (counter-clockwise from West)
    Note : the opposite transform is exactly the same so it also transform from trigo to wind
    """
    angle = 90 - angle # included in [90, 450]
    return angle360(angle)  

def trigo2wind_deg(angle) :
    return wind2trigo_deg(angle) #same operation

def UV2WD_rad(U, V) : 
    """
        get the Wind direction in radian from U and V
    """
    # calculating angle from East to windward direction (where air goes) counter clockwise oriented (trigo)
    WD = np.arctan2(V, U) # included in [-np.pi, np.pi]
    # reverting to get where the wind comes from
    WD = WD + np.pi
    # switching coordinates to wind ones 
    WD = trigo2wind_rad(WD)
    return WD

def UV2WD_deg(U, V) : 
    """
        get the Wind direction in degree from U and V
    """
    return angledeg(UV2WD_rad(U, V))
    

def angle_fit(x, y, return_beta=False) :
    """
    fit the best line from x and y data using an Orthognal Distance Regression (instead of the Y least square error of np.polyfit)
    This method is better for angles near 90° since Y errors become huge :
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.ODR.html#scipy.odr.ODR

    Mandatory
    x, y : numpy arrays or list of same size : the arrays are flatten

    Return
    an angle in degrees in [0, 360[ °
    """
    import scipy.odr # need to import like this : https://stackoverflow.com/questions/13581593/attributeerror-module-object-scipy-has-no-attribute-misc
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    def f(B, x):
        return B[0]*x + B[1]
    sx = np.std(x)
    sy = np.std(y)
    linear = scipy.odr.Model(f)
    mydata = scipy.odr.Data(x, y)
    myodr = scipy.odr.ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    if debug : myoutput.pprint()
    angle = angledeg(np.arctan(myoutput.beta[0]))
    good = myoutput.info == 1
    res = myoutput.sum_square/(sx*sy)
    if return_beta :
        return angle, good, res, myoutput.beta[1]
    else :
        return angle, good, res
    

def angledeg(angle) :
    return angle360(np.rad2deg(angle))

def anglerad(angle) :
    return angle2pi(np.deg2rad(angle))

def angle360(angle) :
    """
    set the range of an angle or array of angles to [0, 360[°
    https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    """
    try : 
        angle[angle<0] += 360
    except :
        angle = angle%360
    return angle

def angle2pi(angle) :
    """
    set the range of an angle or array of angles to [0, 2*np.pi[ rad
    https://stackoverflow.com/questions/37358016/numpy-converting-range-of-angles-from-pi-pi-to-0-2pi
    """
    try : 
        angle[angle<0] += 2*np.pi
    except :
        angle = angle%(2*np.pi)
    return angle

def angle180(angle) :
    """
    set the range of an angle or array of angles to ]-180, 180]°
    """
    angle = angle360(angle)
    if manage_list.is_iterable(angle) :
        angle[angle>180] -= 360
        return angle
    elif type(angle) in [int, float, np.int64, np.float32] :
        if angle > 180 :
            return angle - 360
        else :
            return angle
    else :
        raise(Exception("unknown type ("+str(type(angle))+") in manage_angle.angle180"))
        
def anglepi(angle) :
    """
    set the range of an angle or array of angles to ]-np.pi, np.pi] rad
    """
    angle = angle2pi(angle)
    if manage_list.is_iterable(angle) :
        angle[angle>np.pi] -= 2*np.pi
        return angle
    elif type(angle) in [int, float, np.int64, np.float32] :
        if angle > np.pi :
            return angle - 2*np.pi
        else :
            return angle
    else :
        raise(Exception("unknown type ("+str(type(angle))+") in manage_angle.anglepi"))
        
def circmean(samples, weights=None, high=2*np.pi, low=0, keepdims=False, axis=None, **kwargs) :
    """
    Description
        mean of an array of a cyclic variable (an angle for example) using scipy.stats.circmean
        added the possiblity of weighted mean
    Parameters 
        samples : an array or numpy array
    Optional
        high : maximum value, default=2*np.pi()
        low : minimum value, default=0
        axis 
        keepdims : default=False
    """
    if weights is None :
        return scipy.stats.circmean(samples, high=high, low=low, axis=axis, **kwargs)
    else :
        # Convertir les angles en radians
        angles_rad = 2*np.pi*(samples-low)/(high-low)
        # Calcul des sommes pondérées des parties cosinus et sinus
        sum_cos = np.nansum(weights * np.cos(angles_rad), keepdims=keepdims, axis=axis)
        sum_sin = np.nansum(weights * np.sin(angles_rad), keepdims=keepdims, axis=axis)
        # Calcul de l'angle moyen pondéré en radians
        mean_angle_rad = angle2pi(np.arctan2(sum_sin, sum_cos))
        # Reconvertir l'angle
        mean_angle = mean_angle_rad*(high-low)/(2*np.pi) + low
        return mean_angle

def circstd(samples, weights=None, high=2*np.pi, low=0, keepdims=False, axis=None, **kwargs) :
    """
    Description
        standard deviation of an array of a cyclic variable (an angle for example) using scipy.stats.circstd
        added the possiblity of weighted std
    Parameters 
        samples : an array or numpy array
    Optional
        high : maximum value, default=2*np.pi()
        low : minimum value, default=0
        axis 
        keepdims : default=False
    """
    if weights is None :
        return scipy.stats.circstd(samples, high=high, low=low, axis=axis, **kwargs)
    else :
        # Convertir les angles en radians
        angles_rad = 2*np.pi*(samples-low)/(high-low)
        # Sommes pondérées des parties cosinus et sinus
        if type(weights) in [int, float] :
            weights = np.ones(angles_rad.shape)
        if manage_list.is_iterable(weights) :
            weights[np.isnan(angles_rad)] = np.nan
        sum_cos = np.nansum(weights * np.cos(angles_rad), axis=axis, keepdims=keepdims)
        sum_sin = np.nansum(weights * np.sin(angles_rad), axis=axis, keepdims=keepdims)
        sum_weights = np.nansum(weights, axis=axis, keepdims=keepdims)
        # Longueur du vecteur résultant pondéré
        R = np.sqrt(sum_cos**2 + sum_sin**2) / sum_weights
        # Calcul de l'écart-type circulaire pondéré
        circ_std_rad = np.sqrt(-2 * np.log(R))
        # Reconvertir l'écart-type
        circ_std = circ_std_rad*(high-low)/(2*np.pi)
        return circ_std