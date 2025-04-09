from ..lib import manage_time
import numpy as np
from scipy import interpolate
import time

def calculate_double_traj(X, Y, ZP, U, V, W, HT, DX, DY, DZ, TIMEin, ix_init, iy_init, iz_init, it_init_in, dt_small="10m"):
    """
    Description
        Compute trajectory from simulation results. wind speed is sup_ted to be averaged over an output time step. 
        This method is based on 
            https://github.com/tomgowan/trajectories 
        ML : It is not perfect but it has been tested by the author. I adapted the code so that 
            - it takes into account the fact that Z is a variable in WRF 
            - it computes the trajectory both forward and backward
            - it divides the input time step in more time step to avoid diverging because of the Euler time-integration.
    Parameters
        X, Y (np.arrays of shape (NY, NX)): horizontal coordinates
        ZP, U, V, W (np.arrays of shape (NT, NZ, NY, NX)): respectively vertical coordinates (above sea level), and the three components of the velocity field
        HT : Terrain height
        DX, DY (float): horizontal grid spacing, supposed constant
        DZ (np.array of shape (NT, NZ, NY, NX)): vertical grid spacing
        TIME (np.array of shape (NT)) : time vector
        ix_init, iy_init, iz_init, it_init_in (array of length Ntraj): initial position and time of the trajectories
    Optional
        dt_small (anything readable by manage_time.to_datetime): subdivision of the input timestep for better results
        forward : boolean : True for forward trajectory, False for backward, default is True
    Output
        X_t, Y_t, ZP_t : 2D numpy arrays of shape (Ntraj, NT) : coordinates of the trajectory
    21/08/2024
    """
    # Dims
    NTin, NZ, NY, NX = U.shape
    Ntraj = len(ix_init)

    # Time in
    DTin = manage_time.timedelta_to_seconds(TIMEin[1] - TIMEin[0])
    DELTAin = manage_time.timedelta_to_seconds(TIMEin - TIMEin[0])

    # Smaller time step
    DT = manage_time.timedelta_to_seconds(manage_time.to_timedelta(dt_small))
    TIME = manage_time.to_date_list(TIMEin[0], TIMEin[-1], f"{int(DT)}s")
    DIT = int(DTin/DT)
    NT = (NTin-1) * DIT + 1
    assert(NT == len(TIME))
    DELTA = manage_time.timedelta_to_seconds(TIME - TIMEin[0])
    sliceout = slice(0, NT, DIT)
    it_init = it_init_in*DIT
    if it_init.ndim == 1 : it_init = np.expand_dims(it_init, axis=1)
    if iz_init.ndim == 1 : iz_init = np.expand_dims(iz_init, axis=1)
    if iy_init.ndim == 1 : iy_init = np.expand_dims(iy_init, axis=1)
    if ix_init.ndim == 1 : ix_init = np.expand_dims(ix_init, axis=1)
    
    IX = np.arange(0,NX,1)
    IY = np.arange(0,NY,1)
    IZ = np.arange(0,NZ,1)
    ITin = np.arange(0,NTin,1)
    IT = np.arange(0,NT,1)

    IX_t = np.zeros((Ntraj, NT)) #x-location (grid points on staggered grid)
    IY_t = np.zeros((Ntraj, NT)) #y-location (grid points on staggered grid)
    IZ_t = np.zeros((Ntraj, NT)) #z-location (grid points on staggered grid)
    ZP_t = np.zeros((Ntraj, NT)) #Height above sea level (meters)
    DZ_t = np.zeros((Ntraj, NT)) #Vertical grid spacing at parcel location (meters)
    X_t = np.zeros((Ntraj, NT))*np.nan #X-location in meters
    Y_t = np.zeros((Ntraj, NT))*np.nan #Y-location in meters

    #Seed initial position
    np.put_along_axis(IX_t, it_init, ix_init, axis=1)
    np.put_along_axis(IY_t, it_init, iy_init, axis=1)
    np.put_along_axis(IZ_t, it_init, iz_init, axis=1)
    coord_2D = []
    for itraj in range(Ntraj):
        coord_2D.append((iy_init[itraj, 0], ix_init[itraj, 0]))
    X_init = np.expand_dims(interpolate.interpn((IY,IX), X, coord_2D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)
    Y_init = np.expand_dims(interpolate.interpn((IY,IX), Y, coord_2D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)
    np.put_along_axis(X_t, it_init, X_init, axis=1)
    np.put_along_axis(Y_t, it_init, Y_init, axis=1)

    #Get the actual inital height of the parcels in meters above sea level
    coord_4D = []
    for itraj in range(Ntraj):
        coord_4D.append((it_init[itraj, 0]/DIT, iz_init[itraj, 0], iy_init[itraj, 0], ix_init[itraj, 0]))
    ZP_init = np.expand_dims(interpolate.interpn((ITin,IZ,IY,IX), ZP, coord_4D, method='linear', bounds_error=False, fill_value= 0), axis=1)
    np.put_along_axis(ZP_t, it_init, ZP_init, axis=1)
    
    def loop(fac, IT_p, IZ_p, IY_p, IX_p, ZP_p, X_t, Y_t, ZP_t, IX_t, IY_t, IZ_t):
        #Loop over all time steps and compute trajectory
        for it in range(NT):
            # check if finished
            IT_p += fac
            if np.all(IT_p >= NT) or np.all(IT_p < 0) :
                break
            ############## Generate coordinates for interpolations ##############
            coord_4D = []
            for itraj in range(Ntraj):
                coord_4D.append((IT_p[itraj, 0]/DIT, IZ_p[itraj, 0], IY_p[itraj, 0], IX_p[itraj, 0]))
            coord_2D_old = []
            for itraj in range(Ntraj):
                coord_2D_old.append((np.copy(IY_p[itraj, 0]), np.copy(IX_p[itraj, 0])))

            ##########################################################################################################   
            ########################## Integrate to determine parcel's new location ##################################
            ##########################################################################################################   
            #########################   Calc new IX_t in grdpts above surface  #######################################
            #########################   Calc new IY_t in grdpts above surface  #######################################
            #########################   Calc new IZ_t in meters above sea level and grdpts ###########################
            IX_p += fac*np.expand_dims(interpolate.interpn((ITin,IZ,IY,IX), U, coord_4D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)*DT/DX
            IY_p += fac*np.expand_dims(interpolate.interpn((ITin,IZ,IY,IX), V, coord_4D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)*DT/DY
            ZP_p_change = fac*np.expand_dims(interpolate.interpn((ITin,IZ,IY,IX), W, coord_4D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)*DT
            ZP_p += ZP_p_change
            #Get vertical grid spacing at each parcel's location
            DZ_p = np.expand_dims(interpolate.interpn((ITin,IZ,IY,IX), DZ, coord_4D, method='linear', bounds_error=False, fill_value= np.nan), axis=1)
            
            coord_2D = []
            for itraj in range(Ntraj):
                coord_2D.append((IY_p[itraj, 0], IX_p[itraj, 0]))

            X_p = np.expand_dims(interpolate.interpn((IY,IX), X, coord_2D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)
            Y_p = np.expand_dims(interpolate.interpn((IY,IX), Y, coord_2D, method='linear', bounds_error=False, fill_value=np.nan), axis=1)

            #Change in surface height over last timestep
            HT1 = np.expand_dims(interpolate.interpn((IY,IX), HT, coord_2D_old, method='linear', bounds_error=False, fill_value= np.nan), axis=1)
            HT2 = np.expand_dims(interpolate.interpn((IY,IX), HT, coord_2D, method='linear', bounds_error=False, fill_value= np.nan), axis=1)
            HT_change = HT2-HT1
            
            #Calculate IZ_t in grdpts above surface
            IZ_p += (ZP_p_change - HT_change)/DZ_p

            #Prevent parcels from going into the ground
            IZ_p = IZ_p.clip(min=0)
            ZP_p = ZP_p.clip(min=0)

            which = np.logical_and(IT_p[:, 0] < NT, IT_p[:, 0] >= 0)
            def mask_put_along_axis(X_t, IT_p, X_p, which) :
                temp = X_t[which, :]
                np.put_along_axis(temp, IT_p[which], X_p[which], axis=1)
                X_t[which, :] = temp
            mask_put_along_axis(X_t, IT_p, X_p, which)
            mask_put_along_axis(Y_t, IT_p, Y_p, which)
            mask_put_along_axis(IX_t, IT_p, IX_p, which)
            mask_put_along_axis(IY_t, IT_p, IY_p, which)
            mask_put_along_axis(IZ_t, IT_p, IZ_p, which)
            mask_put_along_axis(ZP_t, IT_p, ZP_p, which)
    
    # forward loop
    start = time.time() #Timer
    loop(1, np.copy(it_init), np.copy(iz_init).astype(float), np.copy(iy_init).astype(float), np.copy(ix_init).astype(float), np.copy(ZP_init), X_t, Y_t, ZP_t, IX_t, IY_t, IZ_t)
    print(f"Forward loop: {round(time.time()-start)} s")
    # backward loop
    start = time.time() #Timer
    # print(it_init)
    loop(-1, np.copy(it_init), np.copy(iz_init).astype(float), np.copy(iy_init).astype(float), np.copy(ix_init).astype(float), np.copy(ZP_init), X_t, Y_t, ZP_t, IX_t, IY_t, IZ_t)
    print(f"Backward loop: {round(time.time()-start)} s")
    return X_t[:,sliceout], Y_t[:,sliceout], ZP_t[:,sliceout], IX_t[:,sliceout], IY_t[:,sliceout], IZ_t[:,sliceout] #, variable1
    


def calculate_traj(X, Y, ZP, U, V, W, HT, DX, DY, DZ, TIMEin, ix_init, iy_init, iz_init, dt_small="10m", forward=True):
    """
    Description
        Compute trajectory from simulation results. wind speed is sup_ted to be averaged over an output time step. 
        This method is based on 
            https://github.com/tomgowan/trajectories 
        ML : It is not perfect but it has been tested by the author. I adapted the code so that 
            - it takes into account the fact that Z is a variable in WRF
            - it divides the input time step in more time step to avoid diverging because of the Euler time-integration.
    Parameters
        X, Y : 2D numpy arrays of shape (NY, NX) : horizontal coordinates
        ZP, U, V, W : 4D numpy arrays of shape (NT, NZ, NY, NX) : respectively cartesian vertical coordinates (above 
                      sea level), and the three components of the velocity field
        HT : Terrain height
        DX, DY : float : horizontal grid spacing, supposed constant
        DZ : 4D numpy array of shape (NT, NZ, NY, NX) : vertical grid spacing
        TIMEin : numpy array of shape (NT) : time vector
        ix_init, iy_init, iz_init : array of length Ntraj : initial position of the trajectories
    Optional
        dt_small (anything readable by manage_time.to_datetime): subdivision of the input timestep for better results
        forward : boolean : True for forward trajectory, False for backward, default is True
    Output
        X_t, Y_t, ZP_t : 2D numpy arrays of shape (Ntraj, NT) : coordinates of the trajectory
    21/08/2024
    """
    DTin = manage_time.timedelta_to_seconds(TIMEin[1] - TIMEin[0])
    NTin, NZ, NY, NX = U.shape
    DT = manage_time.timedelta_to_seconds(manage_time.to_timedelta(dt_small))
    # Smaller time step
    TIME = manage_time.to_date_list(TIMEin[0], TIMEin[-1], f"{int(DT)}s")
    NT = (NTin-1) * int(DTin/DT) + 1
    assert(NT == len(TIME))
    DELTAin = manage_time.timedelta_to_seconds(TIMEin - TIMEin[0])
    DELTA = manage_time.timedelta_to_seconds(TIME - TIMEin[0])
    sliceout = slice(0, NT, int(DTin/DT))
    
    IX = np.arange(0,NX,1)
    IY = np.arange(0,NY,1)
    IZ = np.arange(0,NZ,1)
    IT = np.arange(0,NTin,1)
    Ntraj = len(ix_init)
    if forward :
        fac = 1 
        it0 = 0
        it0in = 0
        it1 = NT-1
        it_vec = range(NT-1)
    else :
        fac = -1 
        it0 = NT-1
        it0in = NTin-1
        it1 = 0
        it_vec = range(NT-1, 0, -1)
        
    ix_t = np.zeros((Ntraj, NT)) #x-location (grid points on staggered grid)
    iy_t = np.zeros((Ntraj, NT)) #y-location (grid points on staggered grid)
    iz_t = np.zeros((Ntraj, NT)) #z-location (grid points on staggered grid)
    ZP_t = np.zeros((Ntraj, NT)) #Height above sea level (meters)
    dz_t = np.zeros((Ntraj, NT)) #Vertical grid spacing at parcel location (meters)
    # variable1 = np.zeros((Ntraj, NT)) #User specified variable to track
    X_t = np.zeros((Ntraj, NT))*np.nan #X-location in meters
    Y_t = np.zeros((Ntraj, NT))*np.nan #Y-location in meters
    
    #Seed initial position
    ix_t[:,it0] = ix_init
    iy_t[:,it0] = iy_init 
    iz_t[:,it0] = iz_init
    X_t[:,it0] = X[iy_init, ix_init]
    Y_t[:,it0] = Y[iy_init, ix_init]
    
    #Get the actual inital height of the parcels in meters above sea level
    ix_p = (ix_t[:,it0]).flatten()
    iy_p = (iy_t[:,it0]).flatten()
    iz_p = (iz_t[:,it0]).flatten()
    coord_height = []
    for i in range(len(ix_p)):
        coord_height.append((iz_p[i], iy_p[i], ix_p[i]))
    ZP_t[:,it0] = interpolate.interpn((IZ,IY,IX), ZP[it0in], coord_height, method='linear', bounds_error=False, fill_value= 0)
                   
    #Loop over all time steps and compute trajectory
    for t in it_vec:
        start = time.time() #Timer
        #Get model data (set by user)
        # u = U[t]
        # v = V[t]
        # w = W[t]
        # var1 = var[t]
        ############## Generate coordinates for interpolations ##############
        ix_p = np.copy(ix_t[:,t]).flatten()
        iy_p = np.copy(iy_t[:,t]).flatten()
        iz_p = np.copy(iz_t[:,t]).flatten()
        coord = []
        for i in range(len(ix_p)):
            coord.append((t*DT/DTin, iz_p[i], iy_p[i], ix_p[i])) 

        ##########################################################################################################   
        ########################## Integrate to determine parcel's new location ##################################
        ##########################################################################################################   

        #########################   Calc new ix_t in grdpts above surface  #######################################
        #########################   Calc new iy_t in grdpts above surface  #######################################
        #########################   Calc new iz_t in meters above sea level and grdpts ###########################
        ix_t[:,t+fac*1] = ix_t[:,t] + fac*interpolate.interpn((IT,IZ,IY,IX), U, coord, method='linear', bounds_error=False, fill_value=np.nan)*DT/DX
        iy_t[:,t+fac*1] = iy_t[:,t] + fac*interpolate.interpn((IT,IZ,IY,IX), V, coord, method='linear', bounds_error=False, fill_value=np.nan)*DT/DY
        ZP_t[:,t+fac*1]  = ZP_t[:,t] + fac*interpolate.interpn((IT,IZ,IY,IX), W, coord, method='linear', bounds_error=False, fill_value= 0)*DT
        #Get vertical grid spacing at each parcel's location
        dz_t[:,t] = interpolate.interpn((IT,IZ,IY,IX), DZ, coord, method='linear', bounds_error=False, fill_value= np.nan)

        #Calculate change in surface height and change in parcel height, and real position
        ix_p = np.copy(ix_t[:,t]).flatten()
        iy_p = np.copy(iy_t[:,t]).flatten()
        coord_HT1 = []
        for i in range(len(ix_p)):
            coord_HT1.append((iy_p[i], ix_p[i]))

        ix_p = np.copy(ix_t[:,t+fac*1]).flatten()
        iy_p = np.copy(iy_t[:,t+fac*1]).flatten()
        coord_HT2 = []
        for i in range(len(ix_p)):
            coord_HT2.append((iy_p[i], ix_p[i]))

        X_t[:,t+fac*1] = interpolate.interpn((IY,IX), X, coord_HT2, method='linear', bounds_error=False, fill_value=np.nan)
        Y_t[:,t+fac*1] = interpolate.interpn((IY,IX), Y, coord_HT2, method='linear', bounds_error=False, fill_value=np.nan)

        #Change in surface height over last timestep
        HT1 = interpolate.interpn((IY,IX), HT, coord_HT1, method='linear', bounds_error=False, fill_value= np.nan)
        HT2 = interpolate.interpn((IY,IX), HT, coord_HT2, method='linear', bounds_error=False, fill_value= np.nan)
        HT_change = HT2-HT1

        #Change in parcel height over last times step
        ZP_t_change = ZP_t[:,t+fac*1].flatten()-ZP_t[:,t].flatten()

        #Calculate iz_t in grdpts above surface
        iz_t[:,t+fac*1] = iz_t[:,t] + (ZP_t_change - HT_change)/dz_t[:,t].flatten()
        
        ##########################################################################################################

        #Prevent parcels from going into the ground
        iz_t = iz_t.clip(min=0)
        ZP_t = ZP_t.clip(min=0)

        #Calculate value of variable at each parcel's location
        # variable1[:,t] = interpolate.interpn((IZ,IY,IX), var1, coord, method = 'linear', bounds_error=False, fill_value= np.nan) 

        #Timer
        stop = time.time()
    print("Integration {:01d} took {:.2f} seconds".format(t, stop-start))
    #Load variable data
    t = it1

    #Get get ix, iy, and iz positions from scalar grid
    ix_p = np.copy(ix_t[:,t]).flatten()
    iy_p = np.copy(iy_t[:,t]).flatten()
    iz_p = np.copy(iz_t[:,t]).flatten()

    coord_zs2 = []
    for i in range(len(ix_p)):
        coord_zs2.append((iy_p[i], ix_p[i])) 
    X_t[:,t] = interpolate.interpn((IY,IX), X, coord_zs2, method='linear', bounds_error=False, fill_value=np.nan)
    Y_t[:,t] = interpolate.interpn((IY,IX), Y, coord_zs2, method='linear', bounds_error=False, fill_value=np.nan)

    # coord = []
    # for i in range(len(ix_p)):
    #     coord.append((iz_p[i], iy_p[i], ix_p[i])) 
    # variable1[:,t] = np.reshape(interpolate.interpn((IZ,IY,IX), T[t], coord, method = 'linear', bounds_error=False, fill_value= np.nan), (Ntraj))
    
    return X_t[:,sliceout], Y_t[:,sliceout], ZP_t[:,sliceout], ix_t[:,sliceout], iy_t[:,sliceout], iz_t[:,sliceout] #, variable1

def pos_to_var(ix_p, iy_p, iz_p, VAR, z=None, y=None, x=None):
    NT, NZ, NY, NX = VAR.shape
    if None in [x, y, z] :
        x = np.arange(NX)
        y = np.arange(NY)
        z = np.arange(NZ)
    return np.array([interpolate.interpn((z,y,x), VAR[it], (iz_p[it], iy_p[it], ix_p[it]), method='linear', bounds_error=False, fill_value=np.nan) for it in range(NT)])

def traj_to_var2D(ix_t, iy_t, VAR, z=None, y=None, x=None):
    ndim = VAR.ndim
    if ndim == 3 :
        NT, NY, NX = VAR.shape
    else :
        NY, NX = VAR.shape
        Ntraj, NT = ix_t.shape
    if None in [x, y, z] :
        x = np.arange(NX)
        y = np.arange(NY)
    coord = np.array([iy_t, ix_t]) #shape(2, NP, NT)
    if ndim == 3 :
        return np.array([interpolate.interpn((y,x), VAR[it], coord[:,:,it].T, method='linear', bounds_error=False, fill_value=np.nan) for it in range(NT)]).T #shape(NP, NT)
    else :
        return np.array([interpolate.interpn((y,x), VAR, coord[:,:,it].T, method='linear', bounds_error=False, fill_value=np.nan) for it in range(NT)]).T #shape(NP, NT)

def traj_to_var(ix_t, iy_t, iz_t, VAR, z=None, y=None, x=None):
    NT, NZ, NY, NX = VAR.shape
    if None in [x, y, z] :
        x = np.arange(NX)
        y = np.arange(NY)
        z = np.arange(NZ)
    coord = np.array([iz_t, iy_t, ix_t]) #shape(3, NP, NT)
    return np.array([interpolate.interpn((z,y,x), VAR[it], coord[:,:,it].T, method='linear', bounds_error=False, fill_value=np.nan) for it in range(NT)]).T #shape(NP, NT)

def traj_to_profile(ix_t, iy_t, VAR, z=None, y=None, x=None):
    NT, NZ, NY, NX = VAR.shape
    if None in [x, y, z] :
        x = np.arange(NX)
        y = np.arange(NY)
        z = np.arange(NZ)
    coord = np.array([iy_t, ix_t]) #shape(3, NP, NT)
    return np.swapaxes(np.swapaxes(  #shape(NP, NT, NZ)
        np.array([np.array([interpolate.interpn((y,x), VAR[it, iz], coord[:,:,it].T, method='linear', bounds_error=False, fill_value=np.nan) for iz in range(NZ)]) for it in range(NT)]), 1,2), 0,1)