#!/usr/bin/env python3
import sys
from .class_dom import Dom
from ..class_variables import VariableSave, Variable, VariableRead
from ..lib import manage_images, manage_time, constants, manage_projection

import os
import numpy as np
from netCDF4 import Dataset
import collections #to sort dictionnaries
import copy #for dictionnaries kwargs in calculate
import datetime
import pygrib

"""
    This class has been designed to extract AROME data from Telem's dataset (2023). See Boris Conan or Sandrine Aubrun to get more information on it
    The dataset covers a specific area from 47.1°N to 47.7°N and from -3.65°E (i.e. 3.65°W) to -2.125°E
    To initialize this object the 4 following 2D numpy arrays have to be generated and saved in the dataset directory :
        AROME_static_HT.npy
        AROME_static_LANDMASK
        AROME_static_LAT
        AROME_static_LON
    See the function generate_numpy_arrays to know more about these
    The AROME_VARIABLES list can be obtained by opening an AROME grib file with 'file=pygrib.open(filename)' and type 'for var in file : print(var)'
    See class_variableRead for more informations about reading grib file
"""
    
def print_variables_from_file(file_path="/data/data-mod/mlandreau/04_data/03_raw/AROME/01_Telem/T64200_AROME_0025_20160630230000.grb"):
    """
    Author(s)
        06/03/2024 : Mathieu LANDREAU
    Description :
        print the variables of a grib file
    Note : 
        There are 2D variables and 3D variables.
        The 3D variables are defined on a constant grid with 24 levels :
        #Z_vec = np.array([20, 35, 50, 75, 100, 150, 200, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375, 1500, 1750, 2000, 2250, 2500, 2750, 3000])
        these levels corresponds to height above surface and not altitudes ("Z" and not "ZP" in py_wrf_arps naming)
    Input :
        static_file_path : the path to the grib file
    """
    with pygrib.open(file_path) as file :
        for var in file : 
            print(var)
        print("---- trying to extract U")
        print("U = file.select(name='U component of wind')")
        U = file.select(name='U component of wind')
        print("type(U) : ", type(U), ", len(U) : ", len(U))
        print("--printing all elements of U")
        for U_i in U :
            print(U_i)
        print("get the 2D array of a single vertical levels iz with 'U[iz].levels'")
        print("U[10].data()[0].shape : ", U[10].data()[0].shape)
        print("---------------------------")
        print("---- trying to extract Temperature")
        print("T = file.select(name='Temperature')")
        T = file.select(name='Temperature')
        print("type(T) : ", type(T), ", len(T) : ", len(T))
        for T_i in T :
            print(T_i)
        print("-- For temperature and some other fields, we need to remove a supplementary level (0 m) to get the same shape as U")
        print("-- see class_variableRead for more infos")   

class DomAROME(Dom):
    software = "AROME"
    x_order = 1 #X are in ascending order
    y_order = -1 #Y are in DESCENDING order
    suffix_length = 18 #T64200_AROME_0025_20160630230000.grb
    lonmin_arome, latmax_arome  = -12, 55.4
    AROME_VARNAMES = {
        "U10" : 1,
        "V10" : 2,
        "WD10" : 3,
        "MH10" : 4,
        #"URAF10" : 5,
        #"VRAF10" : 6,
        #"MHRAF10" : 7,
        #"T2MIN" : 8,
        #"T2MAX" : 9,
        "T2" : 10,
        "RH2" : 11,
        "TD2" : 12,
        "QV2" : 13,
        "T" : [14, 37],
        "RH100" : [38, 61],
        "U" : [62, 85],
        "V" : [86, 109],
        "WD" : [110, 133],
        "MH" : [134, 157], 
        "P" : [158, 181],
        #"QV" : [182, 205],
        "PSL" : 206,
        "Total Cloud Cover" : 207,
        "Total Precipitation" : 208,
        "Liquid precipitation" : 209,
        "Snow melt" : 210,
        "Graupel" : 211,
        "Low cloud cover" : 212,
        "FLSOLAIRE_D" : 213,
        "High cloud cover" : 214,
        "Medium cloud cover" : 215,
        "Evaporation" : 216,
        "???1" : 217,
        "???2" : 218,
        "SWDNB" : 219, #FLTHERM_D in AROME documentation, SWDNB is the WRF varname corresponding (ShortWave DowNward at Bottom)
        "PSFC" : 220,
        "T_SKIN" : 221,
        "CAPE" : 222,
        "Z_PBL" : 223,
        "???3" : 224,
        "Experimental product 1" : 225,
        "Experimental product 2" : 226,
        "???4" : 227,
        "FLSOLAIRE_D2" : 228,
        "USTR" : 229,
        "VSTR" : 230,
    }
    
    def __init__(self, *args, **kwargs):
        kwargs["keep_open"] = False #For the moment
        super().__init__(*args, **kwargs)
    
    def get_output_filenames(self):
        """
        Author(s)
            06/03/2024 : Mathieu LANDREAU
        """
        self.output_filenames = {
            "hist" : {},
            "constant" : {},
            "post" : {},
        }             
        for filename in os.listdir(self.output_data_dir):
            if filename.startswith(self.output_prefix) :
                self.output_filenames['hist'][self.output_data_dir + filename] = True
                self.FLAGS['hist'] = True
            elif filename.startswith("CONSTANT_AROME_EUR") : 
                #these files have been found in :
                #https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=131&id_rubrique=51
                self.output_filenames["constant"][self.output_data_dir + filename] = True
                self.FLAGS["constant"] = True
            
        for filename in os.listdir(self.postprocdir):
            if filename.startswith(self.software + self.i_str + "_post_" ):
                self.output_filenames['post'][self.postprocdir + filename] = True
                self.FLAGS['post'] = True
                
        if os.path.exists(self.output_data_dir+"index") :
            self.output_filenames['index'] = self.output_data_dir+"index"
        else :
            print("error : no index file has been found in ", self.output_data_dir)
            print("please create a file named index and add the following with the values corresponding to your domain :")
            print("lat1   lat2   lon1   lon2   res")
            print("47.1   47.7   -3.65  -2.125 0.025")
            print("the values are the minimum latitude, maximum latitude, minimum longitude, maximum longitude, and resolution in degree (0.025 or 0.01)")
            raise
            
        if not self.FLAGS['hist'] :
            print("error in dom_"+self.software+self.i_str+", no file starting with "+self.output_prefix+" found in "+self.output_data_dir)
            raise
        if not self.FLAGS["constant"] :
            print("error : no static file has been found in ", self.output_data_dir)
            print("please download the files 'CONSTANT_AROME_EUR..' on https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=131&id_rubrique=51")
            raise
        # sort 
        self.output_filenames['hist'] = collections.OrderedDict(sorted(self.output_filenames['hist'].items()))
        self.output_filenames['post'] = collections.OrderedDict(sorted(self.output_filenames['post'].items()))
    
    def init_raw_variables(self):
        """
        Get variables names and units over files and store them in a temporary dictionnary "RAW_VARIABLES"
        Parameters
            self: DOM
        Author(s)
            06/03/2024 : Mathieu LANDREAU
        """
        self.attributes = {}
        for key in self.output_filenames: 
            if key in ["constant", "index"] :
                continue
            if self.FLAGS[key]:
                filename_list = list(self.output_filenames[key].keys())
                if len(filename_list) > 0 :
                    filename = filename_list[0]
                    is_time_file = (len(filename_list) > 1)
                    with pygrib.open(filename) as grbs : 
                        shape = grbs[1].values.shape
                        for k in self.AROME_VARNAMES:
                            ind = self.AROME_VARNAMES[k]
                            self.RAW_VARIABLES[k] = VariableRead(k, k, None, None, None, self.output_filenames[key], is_time_file, k, file=grbs, cmap=0, 
                                                                 keep_open=False, fmt="GRIB", shape=shape, index=ind)
                    
                        
    def init_saved_variables(self):
        """
        Description
            Routine to initialize all the variables that have been defined as 'saved' in ARPS_WRF_VARNAMES
            Additionnal variables can be saved.
        Author(s)
            06/03/2024 : Mathieu LANDREAU
        """
        with open(self.output_filenames["index"]) as index :
            index.readline()
            l = index.read()
            lat1, lat2, lon1, lon2, res = l.split(" ")
            lat1, lat2, lon1, lon2, res = float(lat1), float(lat2), float(lon1), float(lon2), float(res)
            print("lat1, lat2, lon1, lon2, res : ", lat1, lat2, lon1, lon2, res)
        if res not in [0.025, 0.01] :
            print("error : unknown resolution for AROME : ", res, ", should be either 0.025 or 0.01")
            raise 
        
        constant_filename = ""
        extension = "S40" if res == 0.025 else "S100"
        for filename in self.output_filenames["constant"] :
            if extension in filename :
                constant_filename = filename
        if constant_filename == "" :
            print("error, no CONSTANT_AROME with the extension '"+extension+"' found in : ", self.output_data_dir)
            raise
        
        ilat1 = int((self.latmax_arome-lat1)//res)+2
        ilat2 = int((self.latmax_arome-lat2)//res)+1
        ilon1 = int((lon1-self.lonmin_arome)//res)+1
        ilon2 = int((lon2-self.lonmin_arome)//res)+2
        self.ilat1, self.ilat2, self.ilon1, self.ilon2, self.res = ilat1, ilat2, ilon1, ilon2, res
        
        with pygrib.open(constant_filename) as constant_file :
            varnames = ["Land-sea mask", "Geometrical height"]
            constant_vars = constant_file.select(name=varnames)
            self.constant_LANDMASK = constant_vars[1].values
            self.constant_HT = constant_vars[0].values
            self.constant_LAT, self.constant_LON = constant_vars[0].latlons()
        
        self.VARIABLES['LANDMASK'].value = LANDMASK = self.constant_LANDMASK[ilat2:ilat1, ilon1:ilon2] 
        self.VARIABLES['HT'].value = HT = self.constant_HT[ilat2:ilat1, ilon1:ilon2]
        self.VARIABLES['LAT'].value = LAT = self.constant_LAT[ilat2:ilat1, ilon1:ilon2]
        self.VARIABLES['LON'].value = LON = self.constant_LON[ilat2:ilat1, ilon1:ilon2]
    
        #Getting dimensions
        #Z_vec is found in doc_arome_pour-portail_20190903-_250.pdf [https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=131&id_rubrique=51]
        Z_vec = np.array([20, 35, 50, 75, 100, 150, 200, 250, 375, 500, 625, 750, 875, 1000, 1125, 1250, 1375, 1500, 1750, 2000, 2250, 2500, 2750, 3000])
        NY, NX = LAT.shape
        NZ = len(Z_vec)
        Z = np.tensordot(Z_vec, np.ones((NY, NX)), 0)
        HT3 = np.tensordot(np.ones(NZ), HT, 0)
        ZP = Z + HT3
        self.VARIABLES['NX'].value = NX
        self.VARIABLES['NY'].value = NY
        self.VARIABLES['NZ'].value = NZ
        self.VARIABLES['NX_XSTAG'].value = NX + 1
        self.VARIABLES['NY_YSTAG'].value = NY + 1
        self.VARIABLES['NZ_ZSTAG'].value = NZ + 1
        self.VARIABLES["Z"].value = Z
        self.VARIABLES["ZP"].value = ZP
        self.VARIABLES["Z_vec"] = VariableSave("", "", "", "", 0, Z_vec) 
        self.VARIABLES["HT3"] = VariableSave("", "", "", "", 0, HT3) 
        
        #Need this ? Getting simulation initial time
        #self.VARIABLES['INITIAL_TIME'].value = self.str2date(file.START_DATE, self.software)
        #self.VARIABLES["SIM_INITIAL_TIME"] = VariableSave("", "", "", "", 0, self.str2date(self.get_data("SIMULATION_START_DATE"), self.software))
                         
        #Getting projection and central coordinates
        self.VARIABLES['MAPPROJ'].value = 6 #Latitude longitude in wrf Proj
        self.VARIABLES['DX_PROJ'].value = DLON = np.round(LON[0, 1] - LON[0, 0], 5)*np.pi*constants.EARTH_RADIUS/180 #meters
        self.VARIABLES['DY_PROJ'].value = DLAT = -np.round(LAT[0, 0] - LAT[1, 0], 5)*np.pi*constants.EARTH_RADIUS/180 #degrees
        self.VARIABLES['CTRLAT'].value = CTRLAT = np.round(np.mean(LAT[:,0]), 5)
        self.VARIABLES['CTRLON'].value = CTRLON = np.round(np.mean(LON[0,:]), 5)
        self.VARIABLES['TRUELAT1'].value = CTRLAT
        self.VARIABLES['TRUELAT2'].value = CTRLAT
        self.VARIABLES['TRUELON'].value = CTRLON
        print(DLON, DLAT, CTRLAT, CTRLON)

        #Setting the chosen projection to match between all simulations
        CRS = manage_projection.CRS
        self.VARIABLES['CRS'] = VariableSave("", "", "", "", 0, CRS)

        #Calculating new X, Y with chosen projection
        self.VARIABLES['X'].value, self.VARIABLES['Y'].value = X, Y = manage_projection.ll_to_xy(LON, LAT, CRS)
        #Be careful DX and DY don't have really fixed values because the AROME projection is regular_ll and the CRS is Lambert Conformal
        self.VARIABLES['DX'].value = np.mean(np.diff(X, axis=1))
        self.VARIABLES['DY'].value = np.mean(np.diff(Y, axis=0))
                   
    def calculate(self, varname, **kwargs):
        """
        Description :
            See Dom.calculate
        Author(s)
            06/03/2024 : Mathieu LANDREAU
        """
        if varname == "PT" : 
            P = self.get_data("P", **kwargs)
            T = self.get_data("T", **kwargs)
            return constants.T_to_PT(T, P)
        elif varname in ["TV", "PTV"] : #WARNING : doesn't take into account snow, rain or ice
            T = self.get_data(varname[:-1], **kwargs)
            QV = self.get_data("QV", **kwargs)
            return T * (1 + constants.EPSILON*QV) 
        elif varname in ["QV"] : #WARNING : doesn't take into account snow, rain or ice
            #Warning : QV here is mixing ratio whereas in Stull 2017 p.88, q is specific humidity and r is mixing ratio
            T = self.get_data("T", **kwargs)
            RH = self.get_data("RH", **kwargs)
            P = self.get_data("P", **kwargs)
            es = constants.E0 * np.exp(constants.LV/constants.RV * (1/273.15 - 1/T))
            e = RH*es
            return constants.EPSILON2*e/(P-e) #Mixing ratio in Stulle 2017 p.88
        elif varname in ["RH"] :
            return self.get_data("RH100", **kwargs)/100
        elif varname == "LANDMASK2" : #Continental mask
            LANDMASK = self.constant_LANDMASK
            LANDMASK2 =  manage_images.get_main_LANDMASK(LANDMASK)
            self.constant_LANDMASK2 = LANDMASK2
            self.VARIABLES[varname] = VariableSave("Continental mask", "Continental mask", "", "", 2, LANDMASK2[self.ilat2:self.ilat1, self.ilon1:self.ilon2], cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname.startswith("COR") or varname.startswith("CGX") or varname.startswith("CGY") : #Coast orient, Coast gradx, Coast grady
            LANDMASK = self.constant_LANDMASK
            LAT = self.constant_LAT
            earth_circ = constants.EARTH_RADIUS * 2*np.pi
            one_degree = earth_circ/360
            DY = self.res * one_degree
            DX = DY * np.cos(np.deg2rad(LAT))
            if len(varname) > 3 :
                sigma = int(varname[3:])
            else :
                sigma = 100
            COR, CGX, CGY = manage_images.get_COAST_ORIENT2(LANDMASK, DX=np.mean(DX), DY=DY, sigma=sigma*1000, Yfac=self.y_order)
            self.constant_COR = COR
            self.constant_CGX = CGX
            self.constant_CGY = CGY
            self.VARIABLES["COR"+varname[3:]] = VariableSave("Coast orientation", "Coast orientation", "°", "°", 2, COR[self.ilat2:self.ilat1, self.ilon1:self.ilon2], cmap=2)
            self.VARIABLES["CGX"+varname[3:]] = VariableSave("Coast x gradient", "Coast x gradient", "", "", 2, CGX[self.ilat2:self.ilat1, self.ilon1:self.ilon2], cmap=3)
            self.VARIABLES["CGY"+varname[3:]] = VariableSave("Coast y gradient", "Coast y gradient", "", "", 2, CGY[self.ilat2:self.ilat1, self.ilon1:self.ilon2], cmap=3)
            return self.get_data(varname, **kwargs)
        elif varname in ["COASTDIST", "COASTDIST_KM"] :
            self.get_data("LANDMASK2") #on continental mask
            LANDMASK2 = self.constant_LANDMASK2
            X, Y = manage_projection.ll_to_xy(self.constant_LAT, self.constant_LON, self.get_data("CRS"))
            COASTDIST = manage_images.get_COASTDIST2(LANDMASK2, X, Y)
            self.constant_COASTDIST = COASTDIST
            self.VARIABLES[varname] = VariableSave("Distance from coast", "Distance from coast", "m", "m", 2, COASTDIST[self.ilat2:self.ilat1, self.ilon1:self.ilon2], cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["US", "VS", "PS"]: #synoptic Velocites and Pressure
            TIME = self.get_data("TIME", **kwargs)
            date = manage_time.to_day(TIME)
            if type(date) == datetime.datetime :
                date = np.array([date])
            unique_date, unique_inverse = np.unique(date, return_inverse=True)
            temp_list = []
            for date_i in unique_date :
                #for each day, compute in the entire domain, save, and then get the saved data
                new_varname = varname+manage_time.date_to_str(date_i)
                if not new_varname in self.VARIABLES :
                    new_kwargs = copy.deepcopy(kwargs)
                    new_kwargs["time_slice"] = None
                    new_kwargs["saved"] = {}
                    new_kwargs["itime"] = (date_i, date_i+manage_time.to_timedelta(1, "d"))
                    if varname == "PS" :
                        new_kwargs["crop"] = ("ALL", "ALL", "ALL")
                        new_kwargs["hinterp"] = {
                            "levels" : 2999, #I arbitrary chose a height of 2999m above SEA level because AROME is limited to 3000m above sea level.
                            "ZP" : "ZP", #geopotential height is necessary
                        }
                    else :
                        new_kwargs["crop"] = (self.get_data("NZ")-1, "ALL", "ALL") # last level is 3000 m above ground level
                    var = self.get_data(varname[0], **new_kwargs)
                    var = np.mean(var, axis=0)
                    cmap = self.VARIABLES[varname[:-1]].cmap
                    latex_units = self.VARIABLES[varname[:-1]].latex_units
                    units = self.VARIABLES[varname[:-1]].units
                    self.VARIABLES[new_varname] = VariableSave(new_varname, new_varname, latex_units, units, 2, var, cmap=cmap) #save
                var = self.get_data(new_varname, crop=kwargs["crop"]) #read the saved variable and crop if necessary
                temp_list.append(var)
            temp_list = np.array(temp_list)
            return np.squeeze(temp_list[unique_inverse])
        else :
            return Dom.calculate(self, varname, **kwargs)
        
    