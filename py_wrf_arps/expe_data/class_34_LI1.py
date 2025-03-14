import sys

from .class_expe import Expe
from .class_09_CRO import CRO
from ..lib import manage_projection, manage_time, manage_angle, manage_path
from ..class_variables import *

import numpy as np
import scipy
import netCDF4
import datetime
import os
from scipy import spatial #for nearest neighbor

debug = False

rejected_files = ["WLS100s-162_2020-05-17_05-52-30_ppi_96_50m.nc", "WLS100s-162_2020-05-17_09-52-25_ppi_96_50m.nc", #lack of data
                  "WLS100s-162_2020-05-18_09-53-16_ppi_96_50m.nc"] # wrong data

class LI1(Expe):
    code = "LI1"
    name = "LI1"
    fmt = "%Y-%m-%d_%H-%M-%S"
    def __init__(self):
        super().__init__()
        
    def get_output_filenames(self):
        folder_list = manage_path.LI1_folder_list
        for folder in folder_list :
            if os.path.exists(folder) :
                self.folder = folder
        filename_list = []
        for filename in os.listdir(self.folder):
            if filename not in rejected_files and filename.startswith("WLS"):
                filename_list.append(self.folder+filename)
        self.filename_list = np.sort(filename_list)
        
        self.postprocdir = self.folder + "post/"
        if not os.path.exists(self.postprocdir):
            print('Create postproc directory in {0}'.format(self.postprocdir))
            os.makedirs(self.postprocdir, exist_ok=True)
        filename_list = []
        for filename in os.listdir(self.postprocdir):
            if filename not in rejected_files and filename.startswith(self.name+"_post"):
                filename_list.append(self.postprocdir+filename)
        self.postproc_filename_list = np.sort(filename_list)
            
    def get_other_params(self):
        self.dataset_dict = {}
        for filename in self.filename_list :
            self.dataset_dict[filename] = True
        self.postproc_dataset_dict = {}
        for filename in self.postproc_filename_list :
            self.postproc_dataset_dict[filename] = True
        # time
        self.date_list = []
        for filename in self.filename_list:
            N = len(self.folder)
            self.date_list.append(manage_time.to_datetime(filename[N+12:N+31]))
        self.date_list = np.array(self.date_list)
        self.NT = len(self.date_list)
        self.postproc_date_list = []
        for filename in self.postproc_filename_list:
            N = len(self.postprocdir)
            self.postproc_date_list.append(manage_time.to_datetime(filename[N+9:N+28]))
        self.postproc_date_list = np.array(self.postproc_date_list)
        self.postproc_NT = len(self.postproc_date_list)
        if self.postproc_NT > 0 and not np.all(self.date_list == self.postproc_date_list) :
            print("Warning in LI1, postprocfile_date_list is not equal to date_list : ", self.postproc_NT, self.NT)
        self.max_time_correction = 2*(self.date_list[1] - self.date_list[0])
        # space
        self.loc_mat_virtuel = [47.273,  -2.516]
        self.loc_LiDAR = [47.2857, -2.517058]
        self.lat_lidar, self.lon_lidar = self.loc_LiDAR
        self.ZPl = 21 #m
        self.init_from_file(0)
        self.R, self.A = np.meshgrid(self.Rvec, self.Avec)
        self.A_deg = np.rad2deg(self.A)
        self.NR = len(self.Rvec)
        self.NA = len(self.Avec) 
        xl, yl = manage_projection.ll_to_xy(self.loc_LiDAR[1], self.loc_LiDAR[0], manage_projection.CRS)
        self.X = xl + self.R * np.sin(self.A)
        self.Y = yl + self.R * np.cos(self.A)
        self.LAT, self.LON = manage_projection.xy_to_ll(self.X, self.Y, manage_projection.CRS)
        self.range_max = np.max(self.Rvec)
        self.azimuth_min, self.azimuth_max = np.min(self.Avec), np.max(self.Avec)
        #variables
        self.init_variables()
        self.prefix = ""
        
    def init_from_file(self, it):
        self.i_sweep = 1
        if False :
            filename = self.filename_list[it]
            with netCDF4.Dataset(filename) as file :
                sweep_name = list(file.groups.keys())[self.i_sweep]
                sweep = file[sweep_name]
                self.Avec_deg = sweep["azimuth"][:]
                self.Rvec = sweep["range"][:]
                self.Avec = np.deg2rad(self.Avec_deg)
        else : 
            # Finalement on définit manuellement plutôt qu'en lisant les fichier parce que selon
            # le sens de balayage, les angles sont décalés de 3°
            # Comme ça on est à des positions fixées, c'est mieux
            self.Avec_deg = np.arange(159, 201.1, 3)
            self.Rvec = np.arange(100, 3001, 100)
            self.Avec = np.deg2rad(self.Avec_deg)
            # Check that it is correct :
            filename = self.filename_list[it]
            with netCDF4.Dataset(filename) as file :
                sweep_name = list(file.groups.keys())[self.i_sweep]
                sweep = file[sweep_name]
                Avec_deg_file = sweep["azimuth"][:]
                Rvec_file = sweep["range"][:]
                if np.max(np.abs(Avec_deg_file - self.Avec_deg)) > 3 :
                    raise(Exception(f"self.Avec_deg doesn't correspond to the Avec_deg from the file : {self.Avec_deg}, {Avec_deg_file}"))
                if np.max(np.abs(Rvec_file - self.Rvec)) > 100 :
                    raise(Exception(f"self.Rvec doesn't correspond to the Rvec from the file : {self.Rvec}, {Rvec_file}"))
        
    def init_variables(self):
        self.VARIABLES = {}
        # self.VARIABLES[varname] = VariableRead(legend_short, legend_long, latex_units, units, dim, dataset_dict, is_time_file, var_key,\
        #                                        cmap=cmap, keep_open=self.keep_open, fmt=fmt, shape=shape, index=index)
        self.VARIABLES["RWS"] = VariableRead("RWS", "Radial Wind Speed", "m.s^{-1}", "m.s-1", 2, self.dataset_dict, True, "radial_wind_speed",\
                                            cmap=0, keep_open=False, fmt="NETCDF", shape=(self.NA, self.NR), i_sweep=self.i_sweep)
        self.VARIABLES["CNR"] = VariableRead("CNR", "CNR", "", "", 2, self.dataset_dict, True, "cnr",\
                                            cmap=0, keep_open=False, fmt="NETCDF", shape=(self.NA, self.NR), i_sweep=self.i_sweep)
        self.VARIABLES["R"] = VariableSave("", "", "", "", 0, self.R)
        self.VARIABLES["A"] = VariableSave("", "", "", "", 0, self.A)
        self.VARIABLES["X"] = VariableSave("", "", "", "", 0, self.X)
        self.VARIABLES["Y"] = VariableSave("", "", "", "", 0, self.Y)
        self.VARIABLES["NY"] = VariableSave("", "", "", "", 0, self.NR)
        self.VARIABLES["NX"] = VariableSave("", "", "", "", 0, self.NA)
        self.VARIABLES["NR"] = VariableSave("", "", "", "", 0, self.NR)
        self.VARIABLES["NA"] = VariableSave("", "", "", "", 0, self.NA)
        self.VARIABLES["R_KM"] = VariableSave("", "", "", "", 0, self.R/1000)
        self.VARIABLES["A_DEG"] = VariableSave("", "", "", "", 0, self.A_deg)
        self.VARIABLES["X_KM"] = VariableSave("", "", "", "", 0, self.X/1000)
        self.VARIABLES["Y_KM"] = VariableSave("", "", "", "", 0, self.Y/1000)
        self.VARIABLES["LAT"] = VariableSave("", "", "", "", 0, self.LAT)
        self.VARIABLES["LON"] = VariableSave("", "", "", "", 0, self.LON)
        self.VARIABLES["CRS"] = VariableSave("", "", "", "", 0, manage_projection.CRS)
        self.VARIABLES["ZERO"] = VariableSave("", "", "", "", 0, self.X*0)
        llgrid = np.array([self.LAT, self.LON])
        llgrid = llgrid.reshape(2, -1).T
        tree = spatial.cKDTree(llgrid) #needed for nearest neighbor
        self.VARIABLES['tree'] = VariableSave("", "", "", "", 0, tree)
        if self.postproc_NT > 0 :
            filename = self.postproc_filename_list[0]
            with netCDF4.Dataset(filename) as file :
                for v in file.variables :
                    # self.VARIABLES[varname] = VariableRead(legend_short, legend_long, latex_units, units, dim, dataset_dict, is_time_file, var_key,\
                    #                                            cmap=cmap, keep_open=self.keep_open, fmt=fmt, shape=shape, index=index)
                    var = file[v]
                    shape = var.shape
                    self.VARIABLES[v] = VariableRead(var.standard_name, var.long_name, var.latex_units, var.units, len(shape), self.postproc_dataset_dict, True, v,\
                                                    cmap=0, keep_open=False, fmt="NETCDF", shape=shape)
        
    
    def get_data(self, varname, itime=None, time_slice=None, crop=None, n_procs=1, saved=None, **kwargs):
        if saved is None : 
            saved = {}
        if varname in saved : 
            return saved[varname]
        self.prefix += "|"
        if time_slice is None :
            time_slice = manage_time.get_time_slice(itime, self.date_list, self.max_time_correction)
        crop = self.prepare_crop_for_get(crop, varname)
        if varname in self.VARIABLES :
            if debug : print(self.prefix, "variable", varname)
            data = self.VARIABLES[varname].get_data(time_slice, crop, None, n_procs=n_procs)
        elif varname == "TIME":
            data = np.array(self.date_list[time_slice])
        else :
            data = self.calculate(varname, itime=itime, time_slice=time_slice, crop=crop, n_procs=n_procs, saved=saved, **kwargs)
        saved[varname] = data
        self.prefix = self.prefix[:-1]
        return data
    
    def calculate(self, varname, **kwargs) :
        if varname in ["LANDMASK"]:
            return self.get_data("ZERO", **kwargs)
        elif varname == "RWSC" :
            RWS = self.get_data("RWS", **kwargs)
            CNR = self.get_data("CNR", **kwargs)
            RWS[CNR < -28] = np.nan
            return RWS
        elif varname in ["WD180"]: #Wind direction in degrees [-180, 180]
            WD = self.get_data(varname[:-3], **kwargs)
            return manage_angle.angle180(WD)
        if varname in ["U", "V", "WD", "MH", "R2", "QIND"] :
            TIME = self.get_data("TIME", **kwargs)
            RWS = self.get_data("RWSC", **kwargs)
            A = self.get_data("A_DEG", **kwargs)
            try :
                NT = len(TIME)
            except :
                NT = 1
                A = np.expand_dims(A, axis=0)
                RWS = np.expand_dims(RWS, axis=0)
            p = kwargs["saved"]
            p["U"] = np.zeros((NT))
            p["V"] = np.zeros((NT))
            p["MH"] = np.zeros((NT))
            p["WD"] = np.zeros((NT))
            p["R2"] = np.zeros((NT))
            p["QIND"] = np.zeros((NT))
            for it in range(NT) :
                _, p["U"][it], p["V"][it], p["WD"][it], p["MH"][it], p["R2"][it], p["QIND"][it] = self.fitting_cosinus(str(it), 5, A[it].flatten(), RWS[it].flatten(), silent=True)
            if NT == 1 :
                for v in ["U", "V", "WD", "MH", "R2", "QIND"] :
                    p[v] = np.squeeze(p[v])
            return p[varname]
        else :
            raise(Exception("error in class_34_LI1.get_data, the variable name : " + str(varname) + " doesn't exist"))
            
    def prepare_crop_for_get(self, crop, varname) :
        """
        Decription
            Prepare crop. The function is called in self.get_data so that all crop values are lists
        Parameters
            self : Dom
            crop = (cropz, cropy, cropx). cropz, cropy, cropx can be
                str : "ALL", all layers are selected
                int : a single layer is selected
                list of 2 int : [i1, i2], the layers between i1 and i2 are selected, i1 and i2 included 
        Optional 
            zmin, zmax : float : minimale and maximale accepted levels
        Output
            new crop
        """
        NZ = 1
        NY = self.NA
        NX = self.NR
        if crop is not None :
            cropz, cropy, cropx = crop
            if cropz == "ALL" or cropz is None  :
                cropz = [0, NZ]
            elif type(cropz) in [int, np.int64] :
                cropz = [cropz, cropz+1]
            if cropy == "ALL" or cropy is None  :
                cropy = [0, NY]
            elif type(cropy) in [int, np.int64] :
                cropy = [cropy, cropy+1]
            if cropx == "ALL" or cropx is None :
                cropx = [0, NX]
            elif type(cropx) in [int, np.int64] :
                cropx = [cropx, cropx+1]
        else :
            cropz = [0, NZ]
            cropy = [0, NY]
            cropx = [0, NX]
        return (cropz, cropy, cropx)
    
    
    
    def get_label(self) :
        return "Croisic LiDAR"
    
    def get_limits(self) : #to plot on 2DH map
        xl, yl = manage_projection.ll_to_xy(self.lon_lidar, self.lat_lidar, manage_projection.CRS)
        xc1 = xl + self.range_max * np.sin(self.azimuth_min)
        yc1 = yl + self.range_max * np.cos(self.azimuth_min)
        xc2 = xl + self.range_max * np.sin(self.azimuth_max)
        yc2 = yl + self.range_max * np.cos(self.azimuth_max)
        lon1, lat1 = manage_projection.xy_to_ll(xc1, yc1, manage_projection.CRS)
        lon2, lat2 = manage_projection.xy_to_ll(xc2, yc2, manage_projection.CRS)
        return np.array([lon1, lonl, lon2]), np.array([lat1, latl, lat2])
    
    def detect_corrupted_files(self):
        for it in range(self.NT) :
            try : 
                RWS = self.get_data("RWS", itime=it)
            except :
                print(it, self.date_list[it], self.filename_list[it])
        
          
#################################################################################################################################
######  MANAGE_MESH
#################################################################################################################################
   
    def nearest_index_from_self_grid(self, points, **kwargs):
        """
        for each coordinate pair of points, it return the index of the nearest self grid points
        in
        points : 2D array of shape (N, 2) : [[lat1, lon1],[lat2, lon2],[lat3,lon3],...]
        out
        iy, ix : 1D arrays of length N : [iy1, iy2, iy3, ...] and [ix1, ix2, ix3, ...]
        """
        if type(points[0])is CoordPair :
            for ip in range(len(points)):
                points[ip] = [points[ip].lat, points[ip].lon]
        tree = self.get_data("tree")
        I = tree.query(points)
        iy, ix = np.unravel_index(I[1], (self.get_data("NY"), self.get_data("NX")))
        return iy, ix
    
    def get_line(self, points):
        """
        return the coordinates y1, y2, x1, x2 to draw a line on a 2D horizontal plot
        dom : the domain on which we want to draw a line
            
        points : something to deduce the index, can be
            - already the index (a list of 4 integers) [y1, y2, x1, x2]
            - two coordinates pairs : the upper-left and the lower_right (a list of 2 lists of 2 reals) [[lat1, lon1], [lat2, lon2]]
            - the coordinates of the center, the length and direction [[lat_c, lon_c], L, beta]
        """
        
        if type(points) is list :
            if len(points) == 4 : 
                return points
            else :
                if len(points) == 3: 
                    center = points[0]
                    c_lat, c_lon = center
                    distance = points[1]
                    direction = points[2]
                    lat1, lon1 = manage_projection.inverse_haversine([c_lat, c_lon], distance/2, direction+math.pi)
                    lat2, lon2 = manage_projection.inverse_haversine([c_lat, c_lon], distance/2, direction)
                    iy1, ix1 = self.nearest_index_from_self_grid([lat1, lon1])
                    iy2, ix2 = self.nearest_index_from_self_grid([lat2, lon2])
                elif len(points) == 2 :
                    iy1, ix1 = self.nearest_index_from_self_grid(points[0])
                    iy2, ix2 = self.nearest_index_from_self_grid(points[1])
                X = self.get_data("X")
                Y = self.get_data("Y")
                x1 = X[iy1, ix1]
                x2 = X[iy2, ix2]
                y1 = Y[iy1, ix1]
                y2 = Y[iy2, ix2]
                return y1, y2, x1, x2
        else :
            raise(Exception("error in dom.get_line, the key " + str(points) + " is not a list"))
       
    def get_point(self, point):
        """
        return the coordinates y, x, to plot a point on a 2D horizontal plot
        dom : the domain on which we want to draw the point
            
        point : something to deduce the index, can be
            - already the location (a tuple of 2 reals) (y, x)
            - a coordinates pair : (a list of 2 reals) [lat, lon]
        """
        if type(point) is list :
            iy, ix = self.nearest_index_from_self_grid(point)
            Y = self.get_data("Y")
            X = self.get_data("X")
            y = Y[iy, ix]
            x = X[iy, ix]
            print("-------")
            print(iy, ix)
            print(Y, X)
            print(y, x)
            return y, x
        elif type(point) is tuple :
            return point
        else :
            raise(Exception("error in dom.get_point, the key " + str(point) + " is not a list or a tuple"))
    
        
          
#################################################################################################################################
######  POST_PROCESS
#################################################################################################################################  
    
    def fitting_cosinus(self, timestamp, minData, azimuth, RWS_h, silent=False):
        # =============================================================================
        # Inputs : - minData (int) = nombre minimum de données pour réaliser le scan
        #          - timestamp (str) = date de la mesure
        #          - azimuth (liste de int) = liste des azimuth pour chaque mesure en DEGRE
        #          - RWS (liste de int) = liste des Radial Wind Speed pour chaque mesure
        #
        # Output : réalise le fitting en cosinus des mesures et renvoie
        #          [ date (str) , U en m/s (float), V en m/s (float), Wind Direction en °(float),
        #            horizontal Wind Speed en m/s (float) , coefficient determination R2 (float), Quality Index QIND (float) ]
        #
        # Sea Visich, Alexandra, et Boris Conan. 2024. 
        # Measurement and analysis of high altitude wind profiles over the sea in a coastal zone using a scanning doppler LiDAR: Application to wind energy
        # https://hal.science/hal-04734469.
        # =============================================================================
        #Modifier le programme pour introduire l'élévation
        def func(azi, U, V):
            return U*np.sin(azi)+V*np.cos(azi)
            #return U*np.sin(azi)*np.cos(elev)+V*np.cos(azi)*np.cos(elev)

        #On vérifie qu il y a assez de données
        mask = np.isfinite(RWS_h)
        RWS_h = RWS_h[mask]
        azimuth = azimuth[mask]
        if len(RWS_h) >= minData :  
            # Réalisation du fitting
            params, params_covariance = scipy.optimize.curve_fit(func, np.deg2rad(azimuth), RWS_h)       
            U = params[0]
            V = params[1]
            # Calcul du R2 et QIND
            RWS_fit = func(np.deg2rad(azimuth), U, V)
            residuals = RWS_h - RWS_fit
            ss_res = np.sum(residuals**2)
            r_squared = np.power(np.corrcoef(RWS_h, RWS_fit), 2)
            QIND = 1 - np.sqrt(np.sum((RWS_h - RWS_fit)**2)/len(RWS_h)) / np.max(np.abs(RWS_h))
            # Calcul de la direction du vent et du horizontal wind speed 
            WD = manage_angle.UV2WD_deg(U, V)
            WS = np.sqrt(U**2+V**2) 

            ff = [timestamp, U, V, WD, WS, r_squared[1,0], QIND]
            if not silent :
                plt.figure()
                plt.plot(azimuth, RWS_h, '.r', label='Data')
                plt.plot(azimuth, RWS_fit, '--k', label='Fitting')
                plt.text(min(azimuth), min(RWS_h), 'WD ='+str(np.round(WD,2))+'° ,'+'WS ='+str(np.round(WS,2))+' m/s, '+'$R^2$ ='+str(np.round(r_squared[1,0],2)))
                plt.grid()
                plt.legend()
                plt.xlim([0,360])
                plt.xlabel("Azimuth [°]")
                plt.ylabel("RWS [m/s]")
                # print(ff)

            return ff #[timestamp, U, V, WD, WS, r_squared, QIND]
        else: #s'il n'y a pas assez de données
            return [timestamp, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
           
    def postproc_cosinus_fit(self, crop) :
        """
            dom.postproc_cosinus_fit(crop=("ALL", "ALL", [10, 30]))
            will save U_10_30, V_10_30, ... in postproc files
        """
        for it in range(self.NT) :
            if it%10 == 0 :
                print(it)
            kw_get_i = {
                "itime" : it,
                "crop" : crop
            }
            crop = self.prepare_crop_for_get(crop, "U")
            ir1, ir2 = crop[2]
            suffix = "_" + str(ir1) + "_" + str(ir2)
            U = np.array([self.get_data("U", **kw_get_i)])
            V = np.array([self.get_data("V", **kw_get_i)])
            WD = np.array([self.get_data("WD", **kw_get_i)])
            MH = np.array([self.get_data("MH", **kw_get_i)])
            R2 = np.array([self.get_data("R2", **kw_get_i)])
            QIND = np.array([self.get_data("QIND", **kw_get_i)])
            self.write_postproc("U"+suffix, U, (), itime=it, long_name="U"+suffix, standard_name="U"+suffix, units="m.s-1", latex_units="m.s^{-1}")
            self.write_postproc("V"+suffix, V, (), itime=it, long_name="V"+suffix, standard_name="V"+suffix, units="m.s-1", latex_units="m.s^{-1}")
            self.write_postproc("WD"+suffix, WD, (), itime=it, long_name="Wind direction"+suffix, standard_name="WD"+suffix, units="°", latex_units="°")
            self.write_postproc("MH"+suffix, MH, (), itime=it, long_name="Wind speed"+suffix, standard_name="MH"+suffix, units="m.s-1", latex_units="m.s^{-1}")
            self.write_postproc("R2"+suffix, R2, (), itime=it, long_name="R-squared"+suffix, standard_name="R2"+suffix, units="", latex_units="")
            self.write_postproc("QIND"+suffix, QIND, (), itime=it, long_name="Quality Index"+suffix, standard_name="QIND"+suffix, units="", latex_units="")
            
            
    def postproc_cosinus_fit_10min_avg(self, crop) :
        """
            dom.postproc_cosinus_fit_10min_avg(crop=("ALL", "ALL", [10, 30]))
            will save U_10_30, V_10_30, ... in postproc files
        """
        crop = self.prepare_crop_for_get(crop, "U")
        if type(crop[2]) in [list, tuple, np.array] :
            ir1, ir2 = crop[2]
            suffix = f"_{ir1}_{ir2}"
            suffix2 =  f"fit with range ir in [{ir1}:{ir2}]"
        elif type(crop[2]) in [int, float] :
            ir = int(crop[2])
            suffix = f"_{ir}"
            suffix2 =  f"fit with range ir == {ir}"
        RWS = self.get_data("RWSC", itime="ALL_TIMES", crop=crop)
        A = self.get_data("A_DEG", itime="ALL_TIMES", crop=crop)
        
        cro = CRO()
        TIME10 = cro.date_list
        DT10 = cro.DT #10 min
        NT10 = cro.NT #10 min
        TIME = self.date_list
        initial_time = manage_time.to_datetime(TIME10[0])
        DELTA = manage_time.timedelta_to_seconds(TIME-initial_time)
        DELTAbin = manage_time.timedelta_to_seconds(TIME10-initial_time)
        digitized = np.digitize(DELTA, DELTAbin)
    
        RWS10 = [np.array([])]*NT10
        A10 = [np.array([])]*NT10
        N10 = [0]*NT10
        # print(p)
        for it, t in enumerate(TIME) :
            it10 = digitized[it]
            if it%20 == 0 :
                print(it, t, it10, TIME10[it10])
            if it10 >= 0 and it10 < NT10:
                RWS10[it10] = np.concatenate((RWS10[it10], RWS[it].flatten()))
                A10[it10] = np.concatenate((A10[it10], A[it].flatten()))
                N10[it10] += 1
            else :
                print(TIME[it])
        U10 = np.zeros((NT10)) * np.nan
        V10 = np.zeros((NT10)) * np.nan
        MH10 = np.zeros((NT10)) * np.nan
        WD10 = np.zeros((NT10)) * np.nan
        R210 = np.zeros((NT10)) * np.nan
        QIND10 = np.zeros((NT10)) * np.nan
        for it in range(NT10) :
            if N10[it] > 0 :
                _, U10[it], V10[it], WD10[it], MH10[it], R210[it], QIND10[it] = self.fitting_cosinus(it, 5, A10[it].flatten(), RWS10[it].flatten(), silent=True)
        
        cro.write_postproc("U21"+suffix, U10, ("time",), long_name="21m U"+suffix2, standard_name="U21"+suffix, units="m.s-1", latex_units="m.s^{-1}", typ=np.float32)
        cro.write_postproc("V21"+suffix, V10, ("time",), long_name="21m V"+suffix2, standard_name="V21"+suffix, units="m.s-1", latex_units="m.s^{-1}", typ=np.float32)
        cro.write_postproc("WD21"+suffix, WD10, ("time",), long_name="21m WD"+suffix2, standard_name="WD21"+suffix, units="°", latex_units="°", typ=np.float32)
        cro.write_postproc("MH21"+suffix, MH10, ("time",), long_name="21m MH"+suffix2, standard_name="MH21"+suffix, units="m.s-1", latex_units="m.s^{-1}", typ=np.float32)
        cro.write_postproc("R221"+suffix, R210, ("time",), long_name="R-squared of the fit", standard_name="R221"+suffix, units="", latex_units="", typ=np.float32)
        cro.write_postproc("QIND21"+suffix, N10, ("time",), long_name="Quality Index", standard_name="QIND21"+suffix, units="", latex_units="", typ=np.float32)
        cro.write_postproc("N21"+suffix, N10, ("time",), long_name="Number of scans for the fit", standard_name="N21"+suffix, units="", latex_units="", typ=np.float32)
        return U10, V10, WD10, MH10, R210, QIND10, N10
        
    def write_postproc(self, varname, var, dims, itime=0, long_name="", standard_name="", units="", latex_units="", typ=np.float32) :
        """
        Description
            Write a variable in the postproc datafile
        Parameters
            varname : str : name of the variable, ex : "X", "LAT", "U_XSTAG"
            var : np.array : data
            dims : tuple of str : dimensions of the variable ; ex : ("x"), ("y", "x"), (,)
        Optional
            itime : int : index of the date in self.date_list ; ex : 0, 2, -1 ; default : 0
            long_name : str : long name of the variable ; ex : "South-West coordinate" ; default : ""
            standard_name : str : standard name of the variable ; ex : "X coordinate" ; default : ""
            units : str : units name ; ex : "m", "Pa.s-1" ; default : ""
            latex_units : str : units in latex format ; ex : "m", "Pa.s^{-1}" ; default : ""
        Returns 
            Dom 
        """
        date = self.date_list[itime]
        filename = self.postprocdir + self.name + "_post_" + manage_time.date_to_str(date, self.fmt) + ".nc"
        if standard_name == "" :
            standard_name = long_name 
        if long_name == "" :
            long_name = standard_name
            
        if not os.path.exists(filename):
            print("creating postproc file : ", filename)
            init = True
            ncfout = netCDF4.Dataset(filename, mode="w", format='NETCDF4_CLASSIC')
        else :
            print("opening file : ", filename)
            init = False
            ncfout = netCDF4.Dataset(filename, mode="a", format='NETCDF4_CLASSIC')
        if init :
            NA, NR = self.NA, self.NR

            ncfout.Title        = "py_wrf_arps postproc"
            ncfout.Institution  = "LHEEA-DAUC"
            ncfout.FMTVER       = "NetCDF 4.0 Classic"
            ncfout.Source       = "py_wrf_arps/expe_data/class_34_LI1.py"
            ncfout.References   = "See class_34_LI1.py"
            ncfout.Comment      = "Postproc data saved in file"
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            ncfout.History      = "Creation date: {date}".format(date=dt_string)
            ncfout.createDimension('time', NA)
            ncfout.createDimension('range', NR)
            
        if varname in ncfout.variables :
            if debug : print("opening with r+")
            ncfout.close()
            ncfout = netCDF4.Dataset(filename, mode="r+", format='NETCDF4_CLASSIC')
            ncout = ncfout[varname]
        else :
            ncout = ncfout.createVariable(varname, typ, dims)
        ncout.long_name = long_name
        ncout.standard_name = standard_name
        ncout.units = units
        ncout.latex_units = latex_units
        ncout.stagger = ""
        ncout[:] = var[:]
        if debug : print(np.min(var[:]), np.max(var[:]))
        if debug : print(np.min(ncout[:]), np.max(ncout[:]))
        ncfout.close()