import numpy as np
from ..lib import manage_time
from scipy import interpolate
import time

def calculate_traj(X, Y, ZP, U, V, W, HT, DX, DY, DZ, TIME, xpos_init, ypos_init, zpos_init, forward=True):
    """
    Description
        Compute trajectory from simulation results. wind speed is supposed to be averaged over an output time step. 
        This method is based on 
            https://github.com/tomgowan/trajectories 
        ML : It is not perfect but it has been tested by the author. I adapted the code so that it takes into account the
        fact that Z is a variable in WRF
    Parameters
        X, Y : 2D numpy arrays of shape (NY, NX) : horizontal coordinates
        ZP, U, V, W : 4D numpy arrays of shape (NT, NZ, NY, NX) : respectively cartesian vertical coordinates (above 
                      sea level), and the three components of the velocity field
        HT : Terrain height
        DX, DY : float : horizontal grid spacing, supposed constant
        DZ : 4D numpy array of shape (NT, NZ, NY, NX) : vertical grid spacing
        TIME : numpy array of shape (NT) : time vector
        xpos_init, ypos_init, zpos_init : array of length Ntraj : initial position of the trajectories
    Optional
        forward : boolean : True for forward trajectory, False for backward, default is True
    Output
        Xpos, Ypos, zpos_heightASL : 2D numpy arrays of shape (NT, Ntraj) : coordinates of the trajectory
    21/08/2024
    """
    DT = manage_time.timedelta_to_seconds(TIME[1] - TIME[0])
    NT, NZ, NY, NX = U.shape
    x = np.arange(0,NX,1)
    y = np.arange(0,NY,1)
    z = np.arange(0,NZ,1)
    Ntraj = len(xpos_init)
    
    if forward :
        fac = 1 
        it0 = 0
        it1 = NT-1
        it_vec = range(NT)
    else :
        fac = -1 
        it0 = NT-1
        it1 = 0
        it_vec = range(NT-1, -1, -1)
        
    xpos = np.zeros((NT, Ntraj)) #x-location (grid points on staggered grid)
    ypos = np.zeros((NT, Ntraj)) #y-location (grid points on staggered grid)
    zpos = np.zeros((NT, Ntraj)) #z-location (grid points on staggered grid)
    zpos_heightASL = np.zeros((NT, Ntraj)) #Height above sea level (meters)
    zpos_vert_res = np.zeros((NT, Ntraj)) #Vertical grid spacing at parcel location (meters)
    # variable1 = np.zeros((NT, Ntraj)) #User specified variable to track
    Xpos = np.zeros((NT, Ntraj)) #X-location in meters
    Ypos = np.zeros((NT, Ntraj)) #Y-location in meters
    
    #Seed initial position
    xpos[it0,:] = xpos_init
    ypos[it0,:] = ypos_init 
    zpos[it0,:] = zpos_init
    
    #Get the actual inital height of the parcels in meters above sea level
    xloc = (xpos[it0,:]).flatten()
    yloc = (ypos[it0,:]).flatten()
    zloc = (zpos[it0,:]).flatten()
    coord_height = []
    for i in range(len(xloc)):
        coord_height.append((zloc[i], yloc[i], xloc[i]))
    zpos_heightASL[it0,:] = np.reshape(interpolate.interpn((z,y,x), ZP[it0], coord_height, method='linear', bounds_error=False, 
                                                           fill_value= 0), (Ntraj))
    
    #Loop over all time steps and compute trajectory
    for t in it_vec:
        start = time.time() #Timer
        #Get model data (set by user)
        u = U[t]
        v = V[t]
        w = W[t]
        # var1 = var[t]
        ############## Generate coordinates for interpolations ##############
        xloc = np.copy(xpos[t,:]).flatten()
        yloc = np.copy(ypos[t,:]).flatten()
        zloc = np.copy(zpos[t,:]).flatten()
        coord = []
        for i in range(len(xloc)):
            coord.append((zloc[i], yloc[i], xloc[i])) 

        ##########################################################################################################   
        ########################## Integrate to determine parcel's new location ##################################
        ##########################################################################################################   

        #########################   Calc new xpos in grdpts above surface  #######################################
        xpos[t+fac*1,:] = xpos[t,:] + fac*np.reshape(interpolate.interpn((z,y,x), u, coord,
                                  method='linear', bounds_error=False, fill_value=np.nan)*DT/DX, (Ntraj))

        #########################   Calc new ypos in grdpts above surface  #######################################
        ypos[t+fac*1,:]  = ypos[t,:] + fac*np.reshape(interpolate.interpn((z,y,x), v, coord,
                                   method='linear', bounds_error=False, fill_value=np.nan)*DT/DX, (Ntraj))

        #########################   Calc new zpos in meters above sea level ######################################
        zpos_heightASL[t+fac*1,:]  = zpos_heightASL[t,:] + fac*np.reshape(interpolate.interpn((z,y,x), w, coord, 
                                             method='linear', bounds_error=False, fill_value= 0)*DT, (Ntraj))

        ############# Convert zpos from meters above sea level to gridpts abve surface for interpolation #########
        #Get vertical grid spacing at each parcel's location
        zpos_vert_res[t,:] = np.reshape(interpolate.interpn((z,y,x), DZ[t], coord, 
                               method='linear', bounds_error=False, fill_value= np.nan), (Ntraj))

        #Calculate change in surface height and change in parcel height, and real position
        xloc = np.copy(xpos[t,:]).flatten()
        yloc = np.copy(ypos[t,:]).flatten()
        coord_HT1 = []
        for i in range(len(xloc)):
            coord_HT1.append((yloc[i], xloc[i]))

        xloc = np.copy(xpos[t+fac*1,:]).flatten()
        yloc = np.copy(ypos[t+fac*1,:]).flatten()
        coord_HT2 = []
        for i in range(len(xloc)):
            coord_HT2.append((yloc[i], xloc[i]))

        Xpos[t, :] = np.reshape(interpolate.interpn((y,x), X, coord_HT2, method='linear', bounds_error=False, 
                                                       fill_value=np.nan), (Ntraj))
        Ypos[t, :] = np.reshape(interpolate.interpn((y,x), Y, coord_HT2, method='linear', bounds_error=False, 
                                                       fill_value=np.nan), (Ntraj))

        #Change in surface height over last timestep
        HT1 = interpolate.interpn((y,x), HT, coord_HT1, method='linear', bounds_error=False, fill_value= np.nan)
        HT2 = interpolate.interpn((y,x), HT, coord_HT2, method='linear', bounds_error=False, fill_value= np.nan)
        HT_change = HT2-HT1

        #Change in parcel height over last times step
        zpos_heightASL_change = zpos_heightASL[t+fac*1,:].flatten()-zpos_heightASL[t,:].flatten()

        #Calculate zpos in grdpts above surface
        zpos[t+fac*1,:] = zpos[t,:] + np.reshape((zpos_heightASL_change - HT_change)/zpos_vert_res[t,:].flatten(), 
                                                 (Ntraj))
        ##########################################################################################################

        #Prevent parcels from going into the ground
        zpos = zpos.clip(min=0)
        zpos_heightASL = zpos_heightASL.clip(min=0)

        #Calculate value of variable at each parcel's location
        # variable1[t,:] = np.reshape(interpolate.interpn((z,y,x), var1, coord, method = 'linear', bounds_error=False, fill_value= np.nan), (Ntraj))  

        #Timer
        stop = time.time()
    print("Integration {:01d} took {:.2f} seconds".format(t, stop-start))
    #Load variable data
    t = it1

    #Get get x, y, and z positions from scalar grid
    xloc = np.copy(xpos[t,:]).flatten()
    yloc = np.copy(ypos[t,:]).flatten()
    zloc = np.copy(zpos[t,:]).flatten()

    coord_zs2 = []
    for i in range(len(xloc)):
        coord_zs2.append((yloc[i], xloc[i])) 
    Xpos[t, :] = np.reshape(interpolate.interpn((y,x), X, coord_zs2, method='linear', bounds_error=False, 
                                                   fill_value=np.nan), (Ntraj))
    Ypos[t, :] = np.reshape(interpolate.interpn((y,x), Y, coord_zs2, method='linear', bounds_error=False, 
                                                   fill_value=np.nan), (Ntraj))

    coord = []
    for i in range(len(xloc)):
        coord.append((zloc[i], yloc[i], xloc[i])) 
    #Variables
    # variable1[t,:] = np.reshape(interpolate.interpn((z,y,x), T[t], coord, method = 'linear', bounds_error=False, fill_value= np.nan), (Ntraj))
    
    return Xpos, Ypos, zpos_heightASL #, variable1