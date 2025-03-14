import cartopy.crs as ccrs
import numpy as np
from wrf import CoordPair
from .constants import EARTH_RADIUS

if True :
    #après sim 12
    TRUELON = -2.516
    TRUELAT1 = 44.273
    TRUELAT2 = 50.273
    """
    TRUELON = -2.78
    TRUELAT1 = 44.0
    TRUELAT2 = 50.0
    """
else :
    #avant sim 12
    TRUELON = 3.0
    TRUELAT1 = 44.0
    TRUELAT2 = 49.0


LAT_REF = 47.43
LON_REF = -2.78
LAT_REF = 47.273
LON_REF = -2.516
CRStemp = ccrs.LambertConformal(central_longitude = TRUELON,
                    central_latitude = TRUELAT1,
                    false_easting = 0.0,
                    false_northing = 0.0,
                    standard_parallels = (TRUELAT1, TRUELAT2) )
FALSE_EASTING, FALSE_NORTHING = - CRStemp.transform_points(ccrs.PlateCarree(),np.array([LON_REF]),np.array([LAT_REF]))[0,:2]
del(CRStemp)
CRS = ccrs.LambertConformal(central_longitude = TRUELON,
                    central_latitude = TRUELAT1,
                    false_easting = FALSE_EASTING,
                    false_northing = FALSE_NORTHING,
                    standard_parallels = (TRUELAT1, TRUELAT2) )

def coord_to_points(coord_list) : 
    """
    coord_list :
        a wrf CoordPair
        a list of wrf CoordPair
    return : 
        a list of 2 float : [lat, lon]
        a list of lists of 2 floats : [[lat1, lon1], [lat2, lon2], ... ]
    """
    if type(coord_list) is not list :
        return [coord_list.lat, coord_list.lon]
    else :
        points_list = []
        for c in coord_list:
             points_list.append([c.lat, c.lon])
        return points_list

def points_to_coord(points_list) : 
    """
    points_list :
        a list of 2 float : [lat, lon]
        a list of lists of 2 floats : [[lat1, lon1], [lat2, lon2], ... ]
    return :
        a wrf CoordPair
        a list of wrf CoordPair
    """
    if type(points_list[0]) is not list :
        return CoordPair(lat=points_list[0], lon = points_list[1])
    else :
        coord_list = []
        for p in points_list:
             coord_list.append(CoordPair(lat=p[0], lon = p[1]))
        return coord_list

def get_str_location(lat, lon) :
    return "(lat="+str(float(round(lat,3)))+"°N, lon="+str(float(round(lon,3)))+"°E)"

def haversine(points1, points2):
    """
    return distance between 2 point [[lat,lon]]

    Parameters
    ----------
    points1: 2D array
        [[lat,lon]] in degrees
        Work with 1D array too
    points1: 2D array
        [[lat,lon]] in degrees
        Work with 1D array too

    Optional
    ----------

    Returns
    ----------

    Author(s)
    ----------
    See https://pypi.org/project/haversine/
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """

    # ensure arrays are numpy ndarrays
    if not isinstance(points1, np.ndarray):
        points1 = np.array(points1)
    if not isinstance(points2, np.ndarray):
        points2 = np.array(points2)

    # ensure will be able to iterate over rows by adding dimension if needed
    if points1.ndim == 1:
        points1 = np.expand_dims(points1, 0)
    if points2.ndim == 1:
        points2 = np.expand_dims(points2, 0)

    # unpack latitude/longitude
    lat1, lon1 = points1[:, 0], points1[:, 1]
    lat2, lon2 = points2[:, 0], points2[:, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # calculate haversine
    lat = lat2 - lat1
    lon = lon2 - lon1
    d = (np.sin(lat * 0.5) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(lon * 0.5) ** 2)

    # calculate direction
    x = np.cos(lat2)*np.sin(lon)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon)
    direction = np.arctan2(x,y)
    # Earth radius
    r = EARTH_RADIUS
    distance = 2 * r * np.arcsin(np.sqrt(d))
    return distance, direction

def inverse_haversine(point, distance, direction):
    """
    Return lat lon from a point (degrees) and distance (meters) and direction (radian)

    Parameters
    ----------
    point: tuple
        (lat,lon)
    distance: float or 1D array
        distance from initial point in meters
    direction: float or 1D array
        angle from initial direction in radian. North = 0, East = pi/2

    Optional
    ----------

    Returns
    ----------

    Author(s)
    ----------
    See https://pypi.org/project/haversine/
    Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """
    r = EARTH_RADIUS
    lat, lng = point
    lat, lng = map(np.radians, (lat, lng))
    d = distance
    brng = direction

    if type(d) == np.ndarray or type(d) == list:
        arr = np.zeros(np.size(d))
        arr[:] = lat
        lat = np.copy(arr)
        arr[:] = lng
        lng = np.copy(arr)

    return_lat = np.arcsin(np.sin(lat) * np.cos(d / r) + np.cos(lat) * np.sin(d / r) * np.cos(brng))
    return_lng = lng + np.arctan2(np.sin(brng) * np.sin(d / r) * np.cos(lat), np.cos(d / r) - np.sin(lat) * np.sin(return_lat))

    return_lat, return_lng = map(np.degrees, (return_lat, return_lng))
    return return_lat, return_lng

def ll_to_xy(LON, LAT, CRS=CRS):
    """
    Description
        Convert LONgitudes LATitudes to X, Y in Lambert Conformal defined by the CRS
    Parameters
        LON: 2D numpy.ndarray (NY, NX)
        LAT: 2D numpy.ndarray (NY, NX)
        CRS: some CRS (CRS for example)
        (NY, NX) : shape of LON, LAT , X and Y
    Returns
        X: 2D numpy.ndarray (NY, NX)
        Y: 2D numpy.ndarray (NY, NX)
    Author(s)
        Mathieu LANDREAU
    """ 
    if not type(LON) is np.array :
        LON = np.array(LON)
        LAT = np.array(LAT)
    cc = crsTransform(LON.flatten(), LAT.flatten(), ccrs.PlateCarree(), CRS)
    X = cc[:, 0].reshape(LON.shape)
    Y = cc[:, 1].reshape(LON.shape)
    return X, Y


def xy_to_ll(X, Y, CRS=CRS):
    """
    Description
        Convert LONgitudes LATitudes to X, Y in Lambert Conformal defined by the CRS
    Parameters
        LON: 2D numpy.ndarray (NY, NX)
        LAT: 2D numpy.ndarray (NY, NX)
        CRS: some CRS (CRS for example)
        (NY, NX) : shape of LON, LAT , X and Y
    Returns
        X: 2D numpy.ndarray (NY, NX)
        Y: 2D numpy.ndarray (NY, NX)
    Author(s)
        Mathieu LANDREAU
    """
    if not type(X) is np.array :
        X = np.array(X)
        Y = np.array(Y)
    cc = crsTransform(X.flatten(), Y.flatten(), CRS, ccrs.PlateCarree())
    LON = cc[:, 0].reshape(X.shape)
    LAT = cc[:, 1].reshape(X.shape)
    return LON, LAT

def crsTransform(lon, lat, orig_crs, target_crs):
    """
    Description
        Return lon,lat in orig_crs (ex: ccrs.PlateCarree()) to x,y in target_crs (ex: arps crs)
    Parameters
        lon: 1D numpy.ndarray : longitudes (or x)
        lat: 1D numpy.ndarray : latitudes (or y)
        orig_crs: cartopy.crs : A CRS from which input lat/lon are given
        target_crs: cartopy.crs : A CRS from which input lat/lon are returned
    Returns
        ndarray : [[x_i,y_i]] in new coordinate system
    Author(s)
        Benjamin LUCE (benjamin.luce[at]centrale-marseille.fr)
    """
    return target_crs.transform_points(orig_crs,np.array(lon),np.array(lat))[:,:2]
