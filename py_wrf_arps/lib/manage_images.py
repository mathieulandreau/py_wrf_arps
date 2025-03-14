import numpy as np
import scipy
from skimage import measure
from ..lib import manage_angle

##########################################################################################################
#### Find Zones
##########################################################################################################

def get_all_ZONE(LANDMASK) :
    """
    Description
        General : Separate a 2D binary image into zones
        For this case : From a LANDMASK, separate land, sea, lakes and islands into zones
        The algorithm start from a point, defined as zone 1, then it looks at its neighboors, if they are the same type, it defines them as 1 and add them in the queue
        if not and if no pixel has already been define as 2, it defines as zone 2 and add it in the queue
        Each pixel processed is removed from the queue
        The operation is repeted until the queue is finished
        If there are still undefined pixels, one is defined as zone 3 and so on.
    Parameters
        LANDMASK : binary 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Returns 
        ZONE : 2D array of shape (NY, NX), of dtype int, zone 1 is filled with 1, zone 2 is filled with 2, ..., zone Nzone is filled with Nzone
        LAND_ZONE : list of binary int (0 or 1) of length Nzone+1. LAND_ZONE[0] is nothing, LAND_ZONE[1] is type of zone 1, ... LAND_ZONE[Nzone] is type of zone Nzone
    """
    NY, NX = LANDMASK.shape
    LAND_ZONE = [0, LANDMASK[0, 0]]
    #create a 2D empty array, fill the first corner with a 1
    ZONE = np.zeros((NY, NX))
    ZONE[0, 0] = 1
    #old is the indices we are trying to fill, new will be the new one if we find a pixel from another type
    # it is important to not complete 3 zones at the same time because 2 of them might be the same
    # however with two zones, if we are sure that one is a "0" zone and the other is a "1" zone, they can't be the same
    old = 1
    new = 1
    loop = True
    #start a queue object with the first point
    queue = [[0, 0]]
    iy, ix = 0, 0
    while loop : #while there are still undefined pixels
        #check neighbor
        if iy > 0 and ZONE[iy-1, ix] == 0: #if undefined
            if LANDMASK[iy, ix] == LANDMASK[iy-1, ix] : #if same LANDMASK
                ZONE[iy-1, ix] = ZONE[iy, ix] #define as the same zone
                queue.append([iy-1, ix]) #add in the queue
            elif new == old : #else if no new zone has been defined
                new = new + 1 #create a new zone
                ZONE[iy-1, ix] = new #define the pixel as being in the new zone
                LAND_ZONE.append(LANDMASK[iy-1, ix]) #add the new type in LAND_ZONE
                queue.append([iy-1, ix]) #add it in the queue
        #repeat with other neighbors
        if iy < NY-1 and ZONE[iy+1, ix] == 0:
            if LANDMASK[iy, ix] == LANDMASK[iy+1, ix] :
                ZONE[iy+1, ix] = ZONE[iy, ix]
                queue.append([iy+1, ix])
            elif new == old :
                new = new + 1
                ZONE[iy+1, ix] = new
                LAND_ZONE.append(LANDMASK[iy+1, ix])
                queue.append([iy+1, ix])
        if ix > 0 and ZONE[iy, ix-1] == 0:
            if LANDMASK[iy, ix] == LANDMASK[iy, ix-1] :
                ZONE[iy, ix-1] = ZONE[iy, ix]
                queue.append([iy, ix-1])
            elif new == old :
                new = new + 1
                ZONE[iy, ix-1] = new
                LAND_ZONE.append(LANDMASK[iy, ix-1])
                queue.append([iy, ix-1])
        if ix < NX-1 and ZONE[iy, ix+1] == 0:
            if LANDMASK[iy, ix] == LANDMASK[iy, ix+1] :
                ZONE[iy, ix+1] = ZONE[iy, ix]
                queue.append([iy, ix+1])
            elif new == old :
                new = new + 1
                ZONE[iy, ix+1] = new
                LAND_ZONE.append(LANDMASK[iy, ix+1])
                queue.append([iy, ix+1])
        #remove the element from the queue
        queue.pop(0)
        if len(queue) > 0 :
            iy, ix = queue[0]
        #if no more element in the queue
        elif np.any(ZONE==0) : #check if there is still undefined pixels
            pos = np.where(ZONE==0) #find where and chose one undefined pixel
            iy = pos[0][0]
            ix = pos[1][0]
            old = new+1
            new = old
            ZONE[iy, ix] = old 
            LAND_ZONE.append(LANDMASK[iy, ix])
            #add it in the queue
            queue.append([iy, ix])
        else : #else break the loop
            loop = False     
    return ZONE, LAND_ZONE               

def get_main_LANDMASK(LANDMASK, ratio=50) :
    """
    Description
        General : from a binary image keep only largest areas
        For this case : From a LANDMASK, remove islands and lakes to get only the major coastlines
    Parameters
        LANDMASK : 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        ratio : float : keep the zones that are larger that "ratio" times lower than the largest zone from the same type
    Returns 
        LANDMASK2 : same as LANDMASK but modified
    """
    #get all different zone and if it is a land or a water zone
    ZONE, LAND_ZONE = get_all_ZONE(LANDMASK)
    #find the largest land zone and the largest sea zone
    index, counts = np.unique(ZONE, return_counts=True)
    c_land = 0
    c_sea = 0
    for i, ind in enumerate(index):
        ind = int(ind)
        mask = LAND_ZONE[ind]
        c = counts[i]
        if mask and  c > c_land :
            c_land = c
            ind_land = ind
        elif not mask and c > c_sea :
            c_sea = c
            ind_sea = ind
    #LANDMASK2 is the new LANDMASK, first fill with the biggest zone
    LANDMASK2 = np.copy(LANDMASK)
    #Inverse the type of small islands and small lakes
    for i, ind in enumerate(index):
        ind = int(ind)
        mask = LAND_ZONE[ind]
        c = counts[i]
        if mask and c <= c_land/ratio :
            LANDMASK2[ZONE==ind] = 0 #choose the opposite because it is inside a sea area
        elif not mask and c <= c_sea/ratio :
            LANDMASK2[ZONE==ind] = 1 #choose the opposite because it is inside a land area
    return LANDMASK2

def get_main_area(mask, land=True) :
    """
    Description
        General : from a binary image keep only largest 1 area
    Parameters
        mask : 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        ratio : float : keep the zones that are larger that "ratio" times lower than the largest zone from the same type
    Returns 
        LANDMASK2 : same as LANDMASK but modified
    """
    #get all different zone and if it is a land or a water zone
    ZONE, LAND_ZONE = get_all_ZONE(mask)
    #find the largest land zone and the largest sea zone
    index, counts = np.unique(ZONE, return_counts=True)
    c_land = 0
    c_sea = 0
    for i, ind in enumerate(index):
        ind = int(ind)
        mask = LAND_ZONE[ind]
        c = counts[i]
        if mask and  c > c_land :
            c_land = c
            ind_land = ind
        elif not mask and c > c_sea :
            c_sea = c
            ind_sea = ind
    if land :
        return 1*(ZONE == ind_land)
    else :
        return 1*(ZONE != ind_land)
    


##########################################################################################################
#### Gaussian Filter
##########################################################################################################

def get_COAST_ORIENT2(LANDMASK, X=None, DX=None, Y=None, DY=None, sigma=100e3, Xfac=1, Yfac=1):
    """
    Description
        General : from a binary image get the local coast orientationwith gradient of the landmask
        The angle is directed toward 1 values and is in WIND convention (see manage_angles.UV2WD_deg)
        The gradient is computed with a convolution product with a gaussian filter, the sigma value of the kernel is the standard deviation
    Parameters
        LANDMASK : 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        DX : float, default None, X grid size
        X : 2D np.arrays of shape (NY, NX) if ever DX is not given, it will be computed from X
        DY : float, default None, Y grid size
        Y : 2D np.arrays of shape (NY, NX) if ever DY is not given, it will be computed from Y. If neither Y or DY is input, DY=DX
        Xfac : factor on gradx, if ever the X axis is inverted use -1
        Yfac : factor on gradx, if ever the Y axis is inverted use -1 (in AROME for example)
        sigma : The standard deviation of the gaussian kernel
    Returns 
        COAST_ORIENT : same as LANDMASK but modified
    """
    if DX is None :
        if X is None :
            print("error in manage_images.get_COAST_ORIENT2, X or DX must be in input")
        else :
            DX = max(abs(X[0, 1] - X[0, 0]), abs(X[1, 0] - X[0, 0]))
    sigma_mask = sigma/DX
    if DY is not None :
        sigma_mask = (sigma_mask, sigma/DY)
    elif Y is not None :
        DY = max(abs(Y[0, 1] - Y[0, 0]), abs(Y[1, 0] - Y[0, 0]))
        sigma_mask = (sigma_mask, sigma/DY)
    gradx = Xfac*scipy.ndimage.gaussian_filter(LANDMASK, sigma=sigma_mask, order = (0,1), truncate = 8)
    grady = Yfac*scipy.ndimage.gaussian_filter(LANDMASK, sigma=sigma_mask, order = (1,0), truncate = 8)
    CO = manage_angle.UV2WD_deg(gradx, grady)
    return CO, gradx, grady

def get_LANDMASK_SIGMA(LANDMASK, X=None, DX=None, Y=None, DY=None, sigma=100e3):
    """
    Description
        General : from a binary image get the blurred image
        The blurred image is computed with a convolution product with a gaussian filter, the sigma value of the kernel is the standard deviation
    Parameters
        LANDMASK : 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        DX : float, default None, X grid size
        X : 2D np.arrays of shape (NY, NX) if ever DX is not given, it will be computed from X
        DY : float, default None, Y grid size
        Y : 2D np.arrays of shape (NY, NX) if ever DY is not given, it will be computed from Y. If neither Y or DY is input, DY=DX
        sigma : The standard deviation of the gaussian kernel
    Returns 
        LANDMASK_SIGMA : same as LANDMASK but modified but filled with reals between 0 and 1
    """
    if DX is None :
        if X is None :
            print("error in manage_images.get_COAST_ORIENT2, X or DX must be in input")
        else :
            DX = max(abs(X[0, 1] - X[0, 0]), abs(X[1, 0] - X[0, 0]))
    sigma_mask = sigma/DX
    if DY is not None :
        sigma_mask = (sigma_mask, sigma/DY)
    elif Y is not None :
        DY = max(abs(Y[0, 1] - Y[0, 0]), abs(Y[1, 0] - Y[0, 0]))
        sigma_mask = (sigma_mask, sigma/DY)
    LANDMASK_SIGMA = scipy.ndimage.gaussian_filter(LANDMASK, sigma=sigma_mask, order = (0,0), truncate = 8)
    return LANDMASK_SIGMA

##########################################################################################################
#### Coast CELLS (cells touching the interface) 
##########################################################################################################

def get_COASTCELL(LANDMASK) : 
    """
    Description
        General : get boundary pixels from a binary image (i.e. those who have neighbors different than them
        For this case : From a LANDMASK, get coast cells
        CAREFUL : Pixels in the limit of the image are never returned as coast cells here (this is good for our application)
    Parameters
        LANDMASK : binary 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Returns 
        COASTCELL : binary 2D array of shape (NY, NX), 1 for coast cells, 0 for others
    """
    NY, NX = LANDMASK.shape
    COASTCELL = np.zeros((NY, NX), dtype="int")
    for iy in range(1,NY-1):
        for ix in range(1,NX-1):
            COASTCELL[iy, ix] = (LANDMASK[iy, ix] + LANDMASK[iy-1, ix] + LANDMASK[iy+1, ix] + LANDMASK[iy, ix-1] + LANDMASK[iy, ix+1] not in [0, 5])
    return COASTCELL

def get_few_coastcells(X, Y, LANDMASK, Npoint=15, bdy_dist=None, coastmask=None) :
    """ 
    Author :
        22/03/2024 : Mathieu Landreau
    Description
        Return the indices of few coastcells in the domain 
    Parameters
        X : float 2D np.array (NY, NX)
        Y : float 2D np.array (NY, NX)
        LANDMASK : binary 2D np.array (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        Npoint : int : number of coastcells we want, default : 15
        bdy_dist : float : eliminate points too close from the boundary, default : 1/10 of the domain (keep only 80% in each direction)
        coastmask : binary 2D np.array (NY, NX) : eliminate points were mask = 0, for example where SBI == 0
    Returns 
        coords : list of Npoint list of 2 integers : [[iy1, ix1], [iy2,ix2], ..., [iyn, ixn]]
    """
    if bdy_dist is None :
        bdy_dist = abs(X[-1, -1] - X[0, 0])/10
    COASTCELL = get_COASTCELL(LANDMASK)
    BOUNDARY_DISTANCE = get_BOUNDARY_DISTANCE(X, Y)
    acceptable_COASTCELL = COASTCELL * (BOUNDARY_DISTANCE>bdy_dist)
    if coastmask is not None :
        acceptable_COASTCELL *= coastmask
    pos = np.where(acceptable_COASTCELL == 1)
    Npos = len(pos[0])
    Npoint = min(Npos, Npoint)
    coords = []
    for ip in np.linspace(0, Npos-1, Npoint):
        iy = pos[0][int(ip)]
        ix = pos[1][int(ip)]
        coords.append([iy, ix])
    return coords

def get_COASTDIST1(LANDMASK) :
    """ 
    Description
        Return the distance from COAST in pixel (positive in land and negative in sea), with a Manhattan distance (not a physical distance)
        Note : this algorithm might be faster with the queue method
    Parameters
        LANDMASK : binary 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Returns 
        COASTDIST : binary 2D array of shape (NY, NX)
    """
    NY, NX = LANDMASK.shape
    COASTDIST = np.zeros((NY, NX), dtype="int")
    COASTCELL = get_COASTCELL(LANDMASK)
    if np.all(COASTCELL==0) : return COASTDIST
    COASTDIST[COASTCELL==1] = 1
    index = 1
    while np.any(COASTDIST == 0) :
        for iy in range(1,NY-1):
            for ix in range(1,NX-1):
                if COASTDIST[iy, ix] == 0 and index in [COASTDIST[iy-1, ix], COASTDIST[iy+1, ix], COASTDIST[iy, ix-1], COASTDIST[iy+1, ix+1]] :
                    COASTDIST[iy, ix] = index + 1
        index = index+1
    return COASTDIST

def get_COAST_ORIENT(X, LANDMASK, DX_WINDOW, min_contour_length=10):
    """
    Description
        General : from a binary image find the coast points and get the orientation angle of that coast using points within a radius DX_WINDOW
        The angle is directed toward 0 values and is a TRIGONOMETRIC ANGLE
        returns NaN at non-coast points
        For each coast point, a square of size DX_WINDOW centered around this point is study. A mean gradient of the mask is calculated to get the orientation
    Parameters
        X, Y : 2D np.arrays of shape (NY, NX)
        LANDMASK : 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        DX_WINDOW : float, default 50000, scale at which the coast orientation is given
    Returns 
        COAST_ORIENT : same as LANDMASK but modified
    """
    DX = X[0, 1] - X[0, 0]
    window_size = DX_WINDOW/DX
    edge_pixels, curvature_values, direction_values = compute_curvature_profile(LANDMASK, window_size, min_contour_length=min_contour_length)
    COASTCELL = get_COASTCELL(LANDMASK)
    COAST_ORIENT = np.zeros(X.shape)
    for ic in range(len(edge_pixels)) :
        iy, ix = edge_pixels[ic]
        direc = direction_values[ic]
        if ix%1 == 0.5:
            iy1 = int(iy)
            iy2 = int(iy)
            ix1 = int(ix-0.5)
            ix2 = int(ix+0.5)
        elif iy%1 == 0.5:
            iy1 = int(iy-0.5)
            iy2 = int(iy+0.5)
            ix1 = int(ix)
            ix2 = int(ix)
        COAST_ORIENT[iy1, ix1] = direc
        COAST_ORIENT[iy2, ix2] = direc
    COAST_ORIENT[COASTCELL==0] = np.nan
    # The angle was a trigonometric angle (starting from West, positive counter-clockwise to the North) and was the along-coast angle
    # We want a wind angle (starting from North, positive clockwise to the West) and pointing the cross-coast sea side.
    COAST_ORIENT = np.pi - COAST_ORIENT 
    return mange_angle.angledeg(COAST_ORIENT)

def get_CURVATURE(X, LANDMASK, DX_WINDOW, min_contour_length=10):
    DX = X[0, 1] - X[0, 0]
    window_size = DX_WINDOW/DX
    edge_pixels, curvature_values, direction_values = compute_curvature_profile(LANDMASK, window_size, min_contour_length=min_contour_length)
    COASTCELL = get_COASTCELL(LANDMASK)
    CURVATURE = np.zeros(X.shape)
    NCURV = np.zeros(X.shape)
    for ic in range(len(edge_pixels)) :
        iy, ix = edge_pixels[ic]
        curv = curvature_values[ic]
        if ix%1 == 0.5:
            iy1 = int(iy)
            iy2 = int(iy)
            ix1 = int(ix-0.5)
            ix2 = int(ix+0.5)
        elif iy%1 == 0.5:
            iy1 = int(iy-0.5)
            iy2 = int(iy+0.5)
            ix1 = int(ix)
            ix2 = int(ix)
        CURVATURE[iy1, ix1] += curv
        CURVATURE[iy2, ix2] += curv
        NCURV[iy1, ix1] += 1
        NCURV[iy2, ix2] += 1
    CURVATURE = CURVATURE/NCURV / DX
    CURVATURE[COASTCELL==0] = np.nan
    return CURVATURE

##########################################################################################################
#### Coast POINTS (Points on the interface, on cells boundaries) 
##########################################################################################################

def get_COAST_LIST(LANDMASK) :
    NY, NX = LANDMASK.shape
    COAST_LIST = []
    for iy in range(NY):
        for ix in range(NX):
            if LANDMASK[iy, ix] == 0 :
                if iy > 0 and ix > 0 and LANDMASK[iy-1, ix-1] == 1 :
                    COAST_LIST.append([iy-0.5, ix-0.5])
                if iy > 0 and LANDMASK[iy-1, ix] == 1 :
                    COAST_LIST.append([iy-0.5, ix])
                if iy > 0 and ix < NX-1 and LANDMASK[iy-1, ix+1] == 1 :
                    COAST_LIST.append([iy-0.5, ix+0.5])
                    
                if ix > 0 and LANDMASK[iy, ix-1] == 1 :
                    COAST_LIST.append([iy, ix-0.5])
                if ix < NX-1 and LANDMASK[iy, ix+1] == 1 :
                    COAST_LIST.append([iy, ix+0.5])
                    
                if iy < NY-1 and ix > 0 and LANDMASK[iy+1, ix-1] == 1 :
                    COAST_LIST.append([iy+0.5, ix-0.5])
                if iy < NY-1 and LANDMASK[iy+1, ix] == 1 :
                    COAST_LIST.append([iy+0.5, ix])
                if iy < NY-1 and ix < NX-1 and LANDMASK[iy+1, ix+1] == 1 :
                    COAST_LIST.append([iy+0.5, ix+0.5])
    return COAST_LIST

def get_COAST_LIST2(LANDMASK) :
    contours = measure.find_contours(LANDMASK, 0.5)
    COAST_LIST = []
    for contour in contours:
        for i, point in enumerate(contour):
            COAST_LIST.append(point)
    return COAST_LIST

def get_COASTDIST2(LANDMASK, X=None, Y=None) :
    """ 
    Description
        Calculate the distance to the nearest coast from each cell center. Distance inland are positive and offshore are negative.
    Parameters
        LANDMASK : binary 2D np.array of shape (NY, NX), in WRF it is 1 for land and 0 for sea
    Returns 
        COASTDIST : 2D array of shape (NY, NX), with the distance in pixels to the nearest coast
    """
    NY, NX = LANDMASK.shape
    if X is None : 
        X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
    COAST_LIST = np.array(get_COAST_LIST2(LANDMASK))
    N_COAST = len(COAST_LIST)
    ALL_COASTDIST = np.zeros((NY, NX, N_COAST), dtype="int")
    pos = np.array([Y, X])
    pos = pos.transpose([1, 2, 0])
    pos = np.expand_dims(pos, axis=2)
    pos = np.concatenate((pos,)*N_COAST, axis=2)
    IYIX_to_X = scipy.interpolate.RegularGridInterpolator((np.arange(NY), np.arange(NX)), X)
    IYIX_to_Y = scipy.interpolate.RegularGridInterpolator((np.arange(NY), np.arange(NX)), Y)
    Xc = IYIX_to_X(COAST_LIST)
    Yc = IYIX_to_Y(COAST_LIST)
    ALL_COASTDIST = np.sqrt((pos[:, :, :, 0]-Yc)**2 + (pos[:, :, :, 1]-Xc)**2)
    COASTDIST = np.min(ALL_COASTDIST, axis=2)
    COASTDIST[LANDMASK==0] *= -1
    return COASTDIST

def compute_curvature_profile(mask, window_size, min_contour_length=10):
    """
    Adapted from :
    https://medium.com/@stefan.herdy/compute-the-curvature-of-a-binary-mask-in-python-5087a88c6288
    """
    contours = measure.find_contours(mask, 0.5)
    curvature_values = []
    direction_values = []
    edge_pixels = []
    for contour in contours:
        if contour.shape[0] > min_contour_length:
            for i, point in enumerate(contour):
                in_the_window = np.linalg.norm(contour-point, axis=1) < window_size
                start = 0
                end = len(contour)
                for i_point in range(i, 1, -1):
                    if in_the_window[i_point] and not in_the_window[i_point-1] :
                        start = i_point
                        break
                for i_point in range(i, len(contour)-1, 1):
                    if in_the_window[i_point] and not in_the_window[i_point+1] :
                        end = i_point+1
                        break
                neighborhood = contour[start:end]
                curvature, direction, offset = compute_curvature(point, neighborhood)
                curvature_values.append(curvature)
                direction_values.append(direction)
                edge_pixels.append(point)
    curvature_values = np.array(curvature_values)
    direction_values = np.array(direction_values)
    edge_pixels = np.array(edge_pixels)
    return edge_pixels, curvature_values, direction_values

def compute_curvature(point, neighborhood):
    """
    Adapted from :
    https://medium.com/@stefan.herdy/compute-the-curvature-of-a-binary-mask-in-python-5087a88c6288
    """
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]
    # Compute the tangent direction over the entire neighborhood and rotate the points
    grady = np.gradient(y_neighborhood)
    gradx = np.gradient(x_neighborhood)
    grady.fill(np.mean(grady))
    gradx.fill(np.mean(gradx))
    tangent_direction_original = np.arctan2(grady, gradx)
    # Translate the neighborhood points to the central point and apply rotation
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # You can compute the curvature using the formula: curvature = d2y/dx2 / (1 + (dy/dx)^2)^(3/2) = 1/Radius
    # Mazeran, P.E et al. 2005. « Curvature radius analysis for scanning probe microscopy ». 
    # Surface Science 585 (1): 25‑37. https://doi.org/10.1016/j.susc.2005.04.005.
    coeffs = np.polyfit(rotated_x, rotated_y, 2)
    dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    curvature = d2y_dx2 / np.power(1 + np.power(dy_dx, 2), 1.5)
    return np.mean(curvature), np.mean(tangent_direction_original), np.mean(coeffs[-1])

def compute_nearest_curvature(mask, nearest, window_size, min_contour_length):
    """
    Adapted from :
    https://medium.com/@stefan.herdy/compute-the-curvature-of-a-binary-mask-in-python-5087a88c6288
    """
    contours = measure.find_contours(mask, 0.5)
    dist = max(mask.shape)
    for contour in contours:
        if contour.shape[0] > min_contour_length:
            if np.nanmin(np.linalg.norm(contour-nearest, axis=1)) < dist:
                i_near = np.nanargmin(np.linalg.norm(contour-nearest, axis=1))
                point_near = contour[i_near]
                contour_near = contour
    #Select points within a radius distance
    in_the_window = np.linalg.norm(contour_near-point_near, axis=1) < window_size
    start = 0
    end = len(contour_near)
    for i_point in range(i_near, 1, -1):
        if in_the_window[i_point] and not in_the_window[i_point-1] :
            start = i_point
            break
    for i_point in range(i_near, len(contour_near)-1, 1):
        if in_the_window[i_point] and not in_the_window[i_point+1] :
            end = i_point+1
            break
    neighborhood = contour_near[start:end]
    curvature, direction, offset = compute_curvature(point_near, neighborhood)
    return point_near, curvature, direction, offset, neighborhood[:,1] , neighborhood[:,0]

##########################################################################################################
#### Other 
##########################################################################################################

def nearest_masked(iy, ix, mask) :
    index_list = np.argwhere(mask==1)
    distance = np.linalg.norm(index_list - np.array([iy, ix]), axis=1)
    return index_list[np.argmin(distance)]


def get_BOUNDARY_DISTANCE(X, Y) :
    """ 
    Author :
        22/03/2024 : Mathieu Landreau
    Description
        Return the indices of few coastcells in the domain 
    Parameters
        X : float 2D np.array (NY, NX)
        Y : float 2D np.array (NY, NX)
        LANDMASK : binary 2D np.array (NY, NX), in WRF it is 1 for land and 0 for sea
    Optional
        Npoint : int : number of coastcells we want, default : 15
        bdy_dist : float : eliminate points too close from the boundary, default : 1/10 of the domain (keep only 80% in each direction)
        coastmask : binary 2D np.array (NY, NX) : eliminate points were mask = 0, for example where SBI == 0
    Returns 
        coords : list of Npoint list of 2 integers : [[iy1, ix1], [iy2,ix2], ..., [iyn, ixn]]
    """
    Xmin = np.min(X)
    Xmax = np.max(X)
    Ymin = np.min(Y)
    Ymax = np.max(Y)
    return np.min(np.array([X-Xmin, Xmax-X, Y-Ymin, Ymax-Y]), axis=0)

