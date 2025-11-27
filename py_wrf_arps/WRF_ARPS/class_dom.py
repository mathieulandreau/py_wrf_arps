#!/usr/bin/env python3
import sys
from . import *
from ..class_variables import *
from ..lib import manage_display, manage_time, manage_projection, manage_GW, manage_angle, manage_scipy, manage_images, constants, manage_list, manage_dict

from collections.abc import Iterable
import os
import numpy as np
import math
import copy
import pandas as pd
from itertools import product
from netCDF4 import Dataset
import netCDF4
import datetime
import cartopy.crs as ccrs
from wrf import vertcross, CoordPair, interplevel, WrfProj, interpline
from wrf.extension import _slp as wrf_slp
from multiprocessing import Pool, cpu_count
from scipy import spatial #for nearest neighbor
import scipy
import collections #to sort dictionnaries

debug = False

class Dom():
    """
    Be careful : lon = x, lat = y
    """
    software = "WRForARPS"
    x_order = 1 #X are in ascending order
    y_order = 1 #Y are in ascending order (not true in AROME)
    suffix_length = 0
    

#################################################################################################################################
######  Init
#################################################################################################################################
    def __init__(self, i_str, output_data_dir, output_prefix, aerradii=None, keep_open=False, test="", **kwargs):
        """ Extract data for a domain WRF or ARPS
        
        Parameters
            self (Dom)
            proj_dir (str): Global path toward the whole simulation, ex: #"scratch/data-mod/mlandreau/03_simulation/"
            data_dir (str): Name of the simulation directory, ex: #"02_20200519/"
            tab_dom_str (list of str): List of domaine names, ex: #["05", "04", "03"]
                
        Optional
            aerradii : specific to ARPS development by Benjamin Luce
            keep_open : boolean : True if the files that are read are kept open (faster but needs more memory)
            test : str : subdivision of a simulation, useful to compare the same domain of two distinct run.

        06/01/2023 : Mathieu Landreau
        """  
        #attributs
        self.FLAGS = {
            'base': False,
            'trn': False,
            'sfc': False,
            'hist': False,
            'soil': False,
            'stats': False,
            'post': False,
            'post_static': False,
            'df': False,
            "diag" : False,
            "diag2" : False,
            "traj" : False,
            "ts" : False,
            'verbose': True,
            'conc':False
        }
        if "postprocdir" in kwargs :
            self.postprocdir = kwargs["postprocdir"]
        self.prefix = ""
        self.max_time_correction = 0 #temp
        self.output_filenames = {}
        self.aerrho = None
        self.aertyp = None
        self.i_str = i_str #DD
        self.test = test
        self.output_data_dir = output_data_dir # "/scratch/.../03_simulation/03_output/02_WRF/XX_YYYYMMDD/"
        self.output_prefix = output_prefix #XXDD
        self.aerradii = aerradii
        self.RAW_VARIABLES = {} #temporary dictionnary that contains variables directly read in files. Used only in initialisation but saved in case we need it
        self.VARIABLES = {} #contains first only variables defined in ARPS_WRF_VARNAMES.py and then all the known variables
        self.name = self.software+self.i_str+self.test
        self.keep_open=keep_open
        
        #init by methods
        self.get_output_filenames()
        self.get_avg_variables = self.FLAGS["stats"]
        self.date_list = []
        self.date_list_stats = []
        self.init_raw_variables()
        self.init_variables()
        self.init_saved_variables()
        self.init_date_list()
        self.init_saved_variables_common() #Things that can be the same for all childs class
        self.init_hardcoded_variables()
        #check if all saved have values
        for k in self.VARIABLES:
            if isinstance(self.VARIABLES[k], VariableSave) and self.VARIABLES[k].value is None :
                if debug : print("warning : the variable '" + k + "' is defined as saved variable but has no value") 
        self.set_aerradii()
        self.VARIABLES = {key: value for key, value in sorted(self.VARIABLES.items())}
        self.RAW_VARIABLES = {key: value for key, value in sorted(self.RAW_VARIABLES.items())}
    
    def get_output_filenames(self):
        pass
    
    def init_raw_variables(self):
        """
        Warning : This routine is redefined in class_domAROME
        Description
            Get variables names and units over files and store them in a temporary dictionnary "RAW_VARIABLES"
        Parameters
            self : Dom
        Author(s)
            Benjamin LUCE
            06/01/2023 : Highly modified by Mathieu LANDREAU
        """
        self.attributes = {}
        for key in self.output_filenames: 
            if key in ["ts", "index", "constant", "df"] :
                continue
            if self.FLAGS[key]:
                filename_list = list(self.output_filenames[key].keys())
                if len(filename_list) > 0 :
                    filename = filename_list[0]
                    is_time_file = (len(filename_list) > 1)
                    ncfile = self.output_filenames[key][filename] if self.keep_open else netCDF4.Dataset(filename, "r", format='NETCDF4_CLASSIC')
                    for k, v in ncfile.variables.items():
                        if k.isupper() or k == 'Time':
                            self.RAW_VARIABLES[k] = VariableRead(k, k, None, None, None, self.output_filenames[key], is_time_file, k, ncfile, keep_open=self.keep_open) 
                    for k in ncfile.ncattrs() :
                        if not k in self.RAW_VARIABLES :
                            self.RAW_VARIABLES[k] = VariableSave(k, k, "", "", 0, None)
                            self.RAW_VARIABLES[k].value = ncfile.getncattr(k)
                    if not self.keep_open :
                        ncfile.close()
                        
    def init_variables(self):
        #Creating variables as specified in ARPS_WRF_VARNAMES.py
        if(self.software in ["WRF", "ARPS", "WPS", "WRFinput", "AROME"]) :
            for varname in ARPS_WRF_VARNAMES_DICT:
                legend_short, legend_long, latex_units, units, dim, cmap,\
                wrf_var_key, wrf_how_to_get, \
                arps_var_key, arps_how_to_get, \
                wps_var_key, wps_how_to_get, \
                wrfinput_var_key, wrfinput_how_to_get, \
                arome_how_to_get = \
                ARPS_WRF_VARNAMES_DICT[varname]
                if self.software == "WRF" :
                    var_key, how_to_get = wrf_var_key, wrf_how_to_get
                elif self.software == "ARPS" :
                    var_key, how_to_get = arps_var_key, arps_how_to_get
                elif self.software == "WPS" :
                    var_key, how_to_get = wps_var_key, wps_how_to_get
                elif self.software == "WRFinput" :
                    var_key, how_to_get = wrfinput_var_key, wrfinput_how_to_get
                elif self.software == "AROME" :
                    var_key, how_to_get = varname, arome_how_to_get
                if how_to_get == "save" :
                    self.VARIABLES[varname] = VariableSave(legend_short, legend_long, latex_units, units, dim, cmap=cmap)
                elif how_to_get in ["read"] :
                    if var_key not in self.RAW_VARIABLES :
                        if debug : print("error : the key '" + var_key + "' was not found in files to initialize the variable " + varname, " in domain ", self.name,)
                        if False :
                            string = "self.RAW_VARIABLES : \n"
                            for k in self.RAW_VARIABLES:
                                string = string + k + "\n"
                            raise(Exception(string))
                    else : 
                        dataset_dict = self.RAW_VARIABLES[var_key].dataset_dict
                        is_time_file = self.RAW_VARIABLES[var_key].is_time_file
                        fmt = self.RAW_VARIABLES[var_key].fmt
                        shape = self.RAW_VARIABLES[var_key].shape
                        index = self.RAW_VARIABLES[var_key].index
                        self.VARIABLES[varname] = VariableRead(legend_short, legend_long, latex_units, units, dim, dataset_dict, is_time_file, var_key,\
                                                               cmap=cmap, keep_open=self.keep_open, fmt=fmt, shape=shape, index=index)
                elif how_to_get in ["try"] :
                    if var_key not in self.RAW_VARIABLES :
                        #print(var_key, " not found")
                        if debug : print(var_key, "not found in self.RAW_VARIABLES, the variable is initialized as calc")
                        self.VARIABLES[varname] = Variable(legend_short, legend_long, latex_units, units, dim, cmap=cmap)
                    else : 
                        #print(var_key, "found")
                        dataset_dict = self.RAW_VARIABLES[var_key].dataset_dict
                        is_time_file = self.RAW_VARIABLES[var_key].is_time_file
                        fmt = self.RAW_VARIABLES[var_key].fmt
                        shape = self.RAW_VARIABLES[var_key].shape
                        self.VARIABLES[varname] = VariableRead(legend_short, legend_long, latex_units, units, dim, dataset_dict, is_time_file, var_key,\
                                                               cmap=cmap, keep_open=self.keep_open, fmt=fmt, shape=shape)
                elif how_to_get == "calc" :
                    self.VARIABLES[varname] = Variable(legend_short, legend_long, latex_units, units, dim, cmap=cmap)
                elif how_to_get == "ignore" :
                    continue
                else :
                    print("error, unkown how_to_get key : " + how_to_get)
                    
        #Adding the Raw variables read in files to self.VARIABLES without replacing the existing variables
        for varname in self.RAW_VARIABLES:
            if varname not in self.VARIABLES :
                self.VARIABLES[varname] = self.RAW_VARIABLES[varname]
    
    def init_saved_variables(self):
        '''
        Description
            Routine to initialize all the variables that have been defined as 'saved' in ARPS_WRF_VARNAMES
            Aditionnal variables can be saved.
            This routine must be redefined in inherited class
        Parameters
            self : Dom
        '''
        pass
    
    def init_date_list(self):
        '''
        Description
            Routine to initialize date_list and date_list_stats
        Parameters
            self : Dom
        '''
        if 'hist' in self.output_filenames.keys() and len(self.output_filenames['hist']) > 0 :
            for filename in self.output_filenames['hist'] :
                self.date_list.append(self.str2date(filename[-self.suffix_length:], self.software))
        else :
            self.date_list.append(self.VARIABLES['INITIAL_TIME'].value)
        self.date_list.sort()
        NT_HIST = len(self.date_list)
        if NT_HIST > 1 : 
            DT_HIST = self.date_list[1] - self.date_list[0]
            self.max_time_correction = manage_time.to_datetime64(self.date_list[1]) - manage_time.to_datetime64(self.date_list[0])
        else : 
            DT_HIST = datetime.timedelta(days=0)
            self.max_time_correction = np.timedelta64(1000, 'D')
        self.date_list = np.array(self.date_list)
        if not manage_time.is_regular(self.date_list) : 
            print("---", self.name)
            print(manage_time.find_missing_date(self.date_list))
            raise(Exception(f"non-regular date_list"))
        self.VARIABLES["DT_HIST"] = VariableSave("", "", "", "", 0, DT_HIST)
        self.VARIABLES["NT_HIST"] = VariableSave("", "", "", "", 0, NT_HIST)
            
        if 'stats' in self.output_filenames.keys() and len(self.output_filenames['stats']) > 0 :
            for filename in self.output_filenames['stats'] :
                self.date_list_stats.append(self.str2date(filename[-self.suffix_length:], self.software))
            self.date_list_stats.sort() 
            NT_STATS = len(self.date_list_stats)
            if NT_STATS > 1 :
                DT_STATS = self.date_list_stats[1] - self.date_list_stats[0]
            else :
                DT_STATS = datetime.timedelta(days=0)
            self.date_list_stats = np.array(self.date_list_stats)
            if not manage_time.is_regular(self.date_list_stats) : 
                raise(Exception(f"non-regular date_list_stats"))
            if NT_STATS != NT_HIST or not np.all(self.date_list_stats == self.date_list) :
                print(self.name)
                print("NT_HIST", NT_HIST, "NT_STATS", NT_STATS)
                print("self.date_list", self.date_list)
                print("self.date_list_stats", self.date_list_stats)
                raise(Exception(f"Error: date_list is not equal to date_list_stats, this is not allowed yet in calculate and would be very complicated"))
            self.VARIABLES["DT_STATS"] = VariableSave("", "", "", "", 0, DT_STATS)
            self.VARIABLES["NT_STATS"] = VariableSave("", "", "", "", 0, NT_STATS)
               
        if 'post' in self.output_filenames.keys() and len(self.output_filenames['post']) > 0 :
            temp = []
            for filename in self.output_filenames['post'] :
                temp.append(self.str2date(filename[-self.suffix_length-3:-3], self.software))
            temp.sort() 
            NT_temp = len(temp)
            temp = np.array(temp)
            if not manage_time.is_regular(temp) : 
                raise(Exception(f"non-regular date_list_post"))
            if NT_temp != NT_HIST or not np.all(temp == self.date_list) :
                print(self.name)
                print("NT_HIST", NT_HIST, "NT_POST", NT_temp)
                print("self.date_list", self.date_list)
                print("self.date_list_stats", temp)
                raise(Exception(f"Error: date_list is not equal to date_list_post, this is not allowed yet"))
                                          
    
    def init_saved_variables_common(self):
        '''
        Desciption
            Routine to initialize some saved variables that are common to all type of domains
        Parameters
            self : Dom
        '''
        LON = self.get_data("LON")  
        LAT = self.get_data("LAT") 
        NX = self.get_data("NX")
        NY = self.get_data("NY")
        NZ = self.get_data("NZ")
        self.VARIABLES["ZERO"] = VariableSave("", "", "", "", 3, np.zeros((NZ, NY, NX)))
        NX_XSTAG = self.get_data("NX_XSTAG")
        NY_YSTAG = self.get_data("NY_YSTAG")
        NZ_ZSTAG = self.get_data("NZ_ZSTAG")
        self.VARIABLES["ZERO_XSTAG"] = VariableSave("", "", "", "", 3, np.zeros((NZ, NY, NX_XSTAG)))
        self.VARIABLES["ZERO_YSTAG"] = VariableSave("", "", "", "", 3, np.zeros((NZ, NY_YSTAG, NX)))
        self.VARIABLES["ZERO_ZSTAG"] = VariableSave("", "", "", "", 3, np.zeros((NZ_ZSTAG, NY, NX)))
        
        LON_MIN = np.min(LON)
        LON_MAX = np.max(LON)
        LAT_MIN = np.min(LAT)
        LAT_MAX = np.max(LAT)
        """
        LOWER_LEFT_LAT_LON = CoordPair(lat=LAT_MIN,lon=LON_MIN)
        UPPER_LEFT_LAT_LON = CoordPair(lat=LAT_MAX,lon=LON_MIN)
        UPPER_RIGHT_LAT_LON = CoordPair(lat=LAT_MIN,lon=LON_MAX)
        LOWER_RIGHT_LAT_LON = CoordPair(lat=LAT_MAX,lon=LON_MAX)
        """
        LOWER_LEFT_LAT_LON  = CoordPair(lat=LAT[0, 0], lon=LON[0, 0])
        UPPER_LEFT_LAT_LON  = CoordPair(lat=LAT[NY-1, 0], lon=LON[NY-1, 0])
        UPPER_RIGHT_LAT_LON = CoordPair(lat=LAT[NY-1, NX-1], lon=LON[NY-1, NX-1])
        LOWER_RIGHT_LAT_LON = CoordPair(lat=LAT[0, NX-1], lon=LON[0, NX-1])
        llgrid = np.array([LAT, LON])
        llgrid = llgrid.reshape(2, -1).T
        tree = spatial.cKDTree(llgrid) #needed for nearest neighbor
        self.VARIABLES['LON_MIN'] = VariableSave("", "", "", "", 0,LON_MIN)
        self.VARIABLES['LON_MAX'] = VariableSave("", "", "", "", 0,LON_MAX)
        self.VARIABLES['LAT_MIN'] = VariableSave("", "", "", "", 0,LAT_MIN)
        self.VARIABLES['LAT_MAX'] = VariableSave("", "", "", "", 0,LAT_MAX)
        self.VARIABLES['LOWER_LEFT_LAT_LON'] = VariableSave("", "", "", "", 0, LOWER_LEFT_LAT_LON)
        self.VARIABLES['UPPER_LEFT_LAT_LON'] = VariableSave("", "", "", "", 0, UPPER_LEFT_LAT_LON)
        self.VARIABLES['UPPER_RIGHT_LAT_LON'] = VariableSave("", "", "", "", 0, UPPER_RIGHT_LAT_LON)
        self.VARIABLES['LOWER_RIGHT_LAT_LON'] = VariableSave("", "", "", "", 0, LOWER_RIGHT_LAT_LON)
        self.VARIABLES['tree'] = VariableSave("", "", "", "", 0, tree)
        
        FC_un = 2*constants.OMEGA*np.sin(np.deg2rad(LAT))
        FC_ZSTAG = np.tensordot(np.ones(NZ_ZSTAG), FC_un, 0)
        FC = np.tensordot(np.ones(NZ), FC_un, 0)
        self.VARIABLES['FC'] = VariableSave("f_c", "f_c", "rad.s^{-1}", "rad.s-1", 3, FC)
        
        X = self.get_data("X")
        Y = self.get_data("Y")
        self.VARIABLES['X_KM'] = VariableSave("X", "X", "km", "km", 2, X/1000)
        self.VARIABLES['Y_KM'] = VariableSave("Y", "Y", "km", "km", 2, Y/1000)
        X3 = np.tensordot(np.ones(NZ), X, 0)
        Y3 = np.tensordot(np.ones(NZ), Y, 0)
        self.VARIABLES['X3'] = VariableSave("X", "X", "m", "m", 3, X3)
        self.VARIABLES['Y3'] = VariableSave("Y", "Y", "m", "m", 3, Y3)
    
    def init_hardcoded_variables(self):
        pass
    
    def set_aerradii(self):
        '''
        Desciption
            Routine that was used by Benjamin Luce in ARPS to initialize the aerosols radii
        Parameters
            self : Dom
        '''
        pass

#################################################################################################################################
######  Get data 
#################################################################################################################################  
    def copy_kw_get(self, kwargs, saved=False):
        new_kwargs = {}
        for key in kwargs.keys() :
            if key not in ["saved"] :
                new_kwargs[key] = copy.deepcopy(kwargs[key])
        if saved :
            new_kwargs["saved"] = copy.deepcopy(kwargs["saved"])
        else :
            new_kwargs["saved"] = {}
        return new_kwargs

    def get_data(self, varname, itime = None, time_slice = None, crop = None, zoom=None, i_unstag = None, hinterp=None, vinterp=None, DX_smooth=None, n_procs=1, avg=None, avg_time=None, avg_area=None, sigma=50, squeeze=True, avg_stats=True, avg_deriv=True, quick_deriv=False, saved=None, save=True, print_level=0, return_val=True):
        """
        Description
            Get data thanks to variable name and time
        Parameters
            self : Dom
            varname (str, or list, array, tuple of str) : name of the variable(s) 
            ----
            varname must be present in self.VARIABLES
            There are different kind of variables :
            - calc : defined in ARPS_WRF_VARNAMES.py. The basic informations are stored in the object Variable
                     When calling self.VARIABLES[varname].get_data(), the return value is None, meaning it has to be calculated in self.calculate
            - read : defined in ARPS_WRF_VARNAMES.py. The basic informations are stored in the object VariableRead
                     The object contain also the name of the file in which read the data and the name of the variable in that file
                     When calling self.VARIABLES[varname].get_data(), the data is read in the file
                     In case it is a time changing variable, the string itime is the suffix of the file corresponding to the time
            - save : defined in ARPS_WRF_VARNAMES.py and calculated at the initialization
                     the value is stored in the object VariableSave
                     It cannot be a time changing variable
            - raw  : not defined in ARPS_WRF_VARNAMES.py but found during init_raw_variables. The very basic info are stored in the object VariableRead
                     It is read the same way than the variables read
        Optional
            time_slice (slice) : slice of the TIME vector required
                       (int) : index of the TIME vector required
                       (list) : list of required indices
                            Note : if time_slice is not given, it is built from itime
            itime : date to construct time_slice thanks to the method manage_time.get_time_slice, can be of type :
                datetime.datetime : a single date 
                np.datetime64 : a single date
                int : a single date, index of the file in self.date_list
                list : list of any type listed before
                tuple : (first_date, last_date, timestep), first_date and last_date can be any type liste before, timestep can be anything accepted by manage_time.to_timedelta(). timestep is optional.
                str : "ALL_TIMES"  or anything accepted by manage_time.to_datetime (single date)
                None : first file
            crop : tuple : (cropz, cropy, cropx), slice in z, y and x direction. cropz, cropy and cropx can be (see prepare_crop_for_get):
                "ALL" : all layers (default)
                int : single layer
                list of 2 int : [iz1, iz2] is equivalent to slice(iz1, iz2) i.e. iz2 is not included
                *Note : if hinterp or vinterp are defined, or if the variable is 2D cropz has no impact*
                *Note : if vinterp or zoom is defined, cropy and cropx have no impact"
            zoom : something to determine cropy and cropx, see Dom.get_zoom_index, and Proj.get_zoom_index
            i_unstag (int) : which dimension will be unstaggered in Variable.get_data, in practice, it is not useful for the user
            hinterp (dict) : allow the horizontal cross section to a given level or several levels, see hinterp
            vinterp (dict) : allow the vertical cross section at selected levels, see vinterp
            DX_smooth (float) : apply a 2D horizontal gaussian filter on the desired variable, DX_smooth is the kernel size in kilometers
            n_procs (int): number of procs used to read the data, it is not recommended to set higher than 1 for netcdf files. It is useful for AROME grib files
            avg (boolean) : True if you want averaged variables output. If ever in your output files you have W and W_AVG, it will take W_AVG. This is not well done. This has been used because in first version, I didn't have averaged values but then I implemented the average calculation in WRF so the available variables have changed. The logic of self.calculate and DomWRF.calculate have changed.
            avg_time (float) : The variable is averaged over avg_time for each required timestep. avg_time must be a multiple of dt_hist. The behavior depends on avg_stats, avg_deriv
            avg_area (see Dom.get_rectangle): if defined, the variable is averaged over the desired area, that should be a rectangle. avg_area can be any type accepted by zoom. Once again
            avg_stats (boolean): 
                if True : The time_avg is not computed on the output variable but on variables named XX_AVG read in output files. Therefore, the covariances are not averaged but the 1st and 2nd order non-centered moments are averaged. The derivatives are not averaged as well but are computed from averaged variables, ...
                          The avg_area, dx_smooth, 
            quick_deriv (boolean) : 
                False : always compute centered derivatives (if you want to load DX_U in a cropped area (cropz, cropy, [20, 30]), it will read U in (cropz, cropy, [19, 31])
                True : compute non-centered derivatives at the edge of the selected area (save a lot of space and time if you load many variables in a huge area)
            sigma (float) : kernel size in kilometer for landmask, coast orientation, ... It is used in sea-breeze and cross-coast velocities, ...
            squeeze (boolean) : if True, the dimensions of size 1 are squeezed. Default is True
            saved : dict : Saved can help to save a lot of time. Every intermediate variable used in the calculation is saved in this dictionary. If a dictionary is passed as argument out of the function, the user can have access to these variable. In the following example, since U and V are needed to calculate both MH and WD, the calculation of WD will be faster with saved :
                ```
                    saved = {}
                    MH = dom.get_data("MH", itime=("2020-05-17", "2020-05-17_02"), saved=saved)
                    WD = dom.get_data("WD", itime=("2020-05-17", "2020-05-17_02"), saved=saved)
                ```
                *Note : if you change one argument of get_data, saved must be also changed. The following example, WD2 = WD1 which is not what was expected :
                ```
                    saved = {}
                    MH = dom.get_data("MH", itime=("2020-05-17", "2020-05-17_02"), saved=saved)
                    WD1 = dom.get_data("WD", itime=("2020-05-17", "2020-05-17_02"), saved=saved)
                    WD2 = dom.get_data("WD", itime="2020-05-17", crop=("ALL", 10, 12), saved=saved)
                    WD3 = dom.get_data("WD", itime="2020-05-17", crop=("ALL", 10, 12))
                ```
            save : If save is set to False, the intermediate variables are not saved at all. The calculation might be (sometimes a lot) longer but it saves some space.
            print_level (int): if print_level > 0 it will print how it gets the variable, if print_level > 1, it will also print how it gets the variables needed to calculate the final variable, ...
            return_val (boolean): if False returns nothing, but the results can still be saved in "saved"
        Output 
            data : the data expected, can be a float or a numpy array, shape order is (NT, NZ, NY, NX)
                    
        10/01/2023 : Mathieu LANDREAU
        """
        if type(varname) == str : 
            print_this = print_level > len(self.prefix)
            self.prefix = "|" + self.prefix
            if avg is None : 
                avg = self.get_avg_variables
            if saved is None : 
                saved = {}
            if varname in saved : #temporary saved values, useful if you don't want to read twice the same data
                if debug or print_this : print(self.prefix, "saved", varname)
                if squeeze and type(saved[varname]) in [np.array, np.ndarray, list, np.ma.core.MaskedArray]:
                    self.prefix = self.prefix[1:]
                    return np.squeeze(np.array(saved[varname])) if return_val else None
                else :
                    self.prefix = self.prefix[1:]
                    return saved[varname] if return_val else None
            if zoom is not None :
                iy1, iy2, ix1, ix2  = self.get_zoom_index(zoom)
                if crop is None : 
                    crop = ("ALL", [iy1, iy2], [ix1, ix2])
                else :
                    crop = crop[:-2] + ([iy1, iy2], [ix1, ix2])
                zoom = None
            crop = self.prepare_crop_for_get(crop, varname)
            if time_slice is None : 
                if self.FLAGS["stats"] and ( varname == "TIME_STATS" or varname.endswith("AVG") or self.is_statistics(varname) or avg==True or ( "stats" in self.output_filenames and varname in self.VARIABLES and isinstance(self.VARIABLES[varname], VariableRead) and next(iter(self.VARIABLES[varname].dataset_dict.items()))[0] in self.output_filenames["stats"] )):
                    time_slice = manage_time.get_time_slice(itime, self.date_list_stats, self.max_time_correction)
                else:
                    time_slice = manage_time.get_time_slice(itime, self.date_list, self.max_time_correction)
            
            
            if varname == "TIME":
                self.prefix = self.prefix[1:]
                return self.date_list[time_slice] if return_val else None
            if varname == "TIME_STATS":
                self.prefix = self.prefix[1:]
                return self.date_list_stats[time_slice] if return_val else None
            
            # for WD or COV or ... , it is better to interpolate or average other quantities and then calculate, so we skip this now and go to calculate first
            not_avg = varname in ["X", "Y", "X_KM", "Y_KM"]
            avg_before = (varname.startswith("WD") or varname.startswith("MH") or varname.startswith("SCORER") or self.is_statistics(varname, avg_stats) or self.is_normdir(varname) or varname.upper() in ["DGWRDT", "GAMMA", "IT", "IZ", "IY", "IX", "MG"])
            interp_before = avg_before or varname.startswith("X2DV")
            smooth_before = interp_before or ("TIME" in varname or varname in ["X", "Y", "LON", "LAT"] or varname[:2] in ["DX", "DY"] \
                             or "LANDMASK" in varname or "COASTDIST" in varname or varname[:3] in ["COR", "CGX", "CGY", "CDI"]\
                             or self.is_derivative(varname, avg_deriv))
            
            kw_get = dict(time_slice=time_slice, crop=crop, i_unstag=i_unstag, avg=avg, avg_time=avg_time, DX_smooth=DX_smooth, n_procs=n_procs, sigma=sigma, 
                          saved=saved, squeeze=False, avg_stats=avg_stats, avg_deriv=avg_deriv, quick_deriv=quick_deriv, save=save, print_level=print_level)
            # Averaging, filtering and interpolation
            if avg_area is not None and not avg_before and not not_avg and self.get_dim(varname) > 1 :
                if debug or print_this : print(self.prefix, "avg_area varname =", varname, " avg_area = ", avg_area)
                data = self.avg_area(varname, avg_area, hinterp=hinterp, **kw_get)
                
            elif hinterp is not None and not interp_before and not not_avg and self.get_dim(varname) > 2 :
                if debug or print_this : print(self.prefix, "hinterp", varname)
                data = self.hinterp(varname, **kw_get, **hinterp)
                
            elif vinterp is not None and not interp_before and self.get_dim(varname) > 0 :
                if debug or print_this : print(self.prefix, "vinterp", varname)
                data = self.vinterp(varname, **kw_get, **vinterp)
                
            elif DX_smooth is not None and not smooth_before and not not_avg and self.get_dim(varname) > 1 : 
                # if WD in varname, it is better to smooth U and V and then calculate WD, so we skip this and go to calculate
                if debug or print_this : print(self.prefix, "DX_smooth", varname)
                data = self.smooth(varname, **kw_get)
                
            elif avg_time is not None and not avg_before :
                print(self.prefix, "averaging varname =", varname, " avg_time = ", avg_time)
                data = self.time_avg(varname, **kw_get)
             
            # Calculate some variables based on other variables that we have
            elif not varname in self.VARIABLES or type(self.VARIABLES[varname]) not in [VariableRead, VariableSave] :
                if debug or print_this : print(self.prefix, "calculate", varname)
                data = self.calculate(varname, **kw_get, hinterp=hinterp, vinterp=vinterp, avg_area=avg_area)
            # Get from file
            else :
                if debug or print_this : print(self.prefix, "variable", varname)
                data = self.VARIABLES[varname].get_data(time_slice, crop, i_unstag, n_procs=n_procs)
           
            # Save the variable
            if save : saved[varname] = data
            # Squeeze at the end
            if squeeze and type(data) in [np.array, np.ndarray, list, np.ma.core.MaskedArray]:
                data = np.squeeze(np.array(data))
            self.prefix = self.prefix[1:]
            return data if return_val else None
        
        # If several variables, call get data for each
        elif type(varname) in (list, tuple, np.array, np.ndarray) :
            out = ()
            for varname_i in varname :
                out = out + (self.get_data(varname_i, itime=itime, time_slice=time_slice, crop=crop, zoom=zoom, i_unstag=i_unstag, hinterp=hinterp, vinterp=vinterp, DX_smooth=DX_smooth, n_procs=n_procs, avg=avg, avg_time=avg_time, avg_area=avg_area, sigma=sigma, squeeze=np.copy(squeeze), avg_stats=avg_stats, avg_deriv=avg_deriv, quick_deriv=quick_deriv, saved=saved, save=save, print_level=print_level, return_val=return_val),)
            return out if return_val else None
        else :
            raise(Exception("error, unknown type for varname in Dom.get_data : " + str(type(varname)) + ", varname = " + varname))
    
    def avg_area(self, varname, avg_area, saved={}, crop=None, **kw_get) :
        """
        Description
            Compute the horizontal averaging of a quantity over an area defined with avg_area
        Parameters
            self : Dom
            varname : str : name of the variable 
            avg_area : averaging area, see self.get_zoom_index
        Optional
            saved : see self.get_data
            crop : see self.get_data
            kw_get : dictionnary : all the parameters to call again get_data
        Output
            the area-averaged data
        """
        if not "before_avg_area" in saved :
            saved["before_avg_area"] = {}
        cropz = "ALL" if crop is None else crop[0]
        iy1, iy2, ix1, ix2 = self.get_zoom_index(avg_area)
        new_crop = [cropz, [iy1, iy2], [ix1, ix2]]
        data = self.get_data(varname, crop=new_crop, saved=saved["before_avg_area"], **kw_get)
        return np.mean(data, axis=(-2, -1), keepdims=True)
        
    def time_avg(self, varname, avg_time=None, time_slice=None, saved={}, **kw_get) :
        """
        Description
            Compute the time-average of a quantity over the period avg_time, this quantity must already be an average over a certain period of time
            for each date in self.get_data("TIME", time_slice=time_slice), the quantity is averaged between date-avg_time and date.
        Parameters
            self : Dom
            varname : str : name of the variable 
            avg_time : averaging period of time in minutes, must be a multiple of self.get_data("DT_STATS")
            time_slice : see self.get_data
        Optionals
            kw_get : dictionnary : all the parameters to call again get_data
        Output
            the time-averaged data
        """
        avg_time_sec = avg_time*60
        DT_STATS = self.get_data("DT_STATS").seconds
        DT_HIST = self.get_data("DT_HIST").seconds
        if DT_STATS != DT_HIST and not varname.endswith("_AVG") :
            raise(Exception(f"The output time step {DT_HIST} is not equal to the averaging time step {DT_STATS}. The time_avg might be wrong"))
        if (avg_time_sec/DT_STATS)%1 != 0 : 
            raise(Exception("error, with varname =" + varname + ", DT_STATS :" + str(DT_STATS)+ "s is not compatible with avg_time = " + str(avg_time) + "s, reste : " + str((avg_time_sec/DT_STATS.seconds)%1)))
        elif DT_STATS <= avg_time_sec / 2 : #DT_STATS should be at most half the desired stats period
            Navg = int(avg_time_sec/DT_STATS) #how many time do we average together
            it_list = np.arange(5000, dtype="int")[time_slice]
            if type(it_list) in [int, np.int64] :
                it_list = np.array([it_list])
            NT = len(it_list)
            it_list2 = []
            for it in it_list :
                if it < Navg :
                    raise(Exception("error : cannot average at time it = " + str(it) + "with avg_time = " + str(avg_time) + "min, Navg = " + str(Navg)))
                for i_avg in range(Navg) :
                    it_list2.append(it-Navg+i_avg+1)
            NT_tot = NT*Navg
            # The following is averaged over the time DT_STATS
            var = self.get_data(varname, time_slice=it_list2, saved={}, **kw_get)
            if debug : print(self.prefix, "avg_time, before np.mean, varname = ", varname, ", shape = ", var.shape)
            # We need to average over avg_time minutes
            old_shape = var.shape
            new_shape = (NT, Navg) + old_shape[1:]
            var_avg_time = np.mean(np.reshape(var, new_shape), axis=1)
            if NT == 1 :
                var_avg_time = var_avg_time[0]
            if debug : print(self.prefix, "avg_time, after np.mean, varname = ", varname, ", shape = ", var_avg_time.shape)
            return var_avg_time
        elif DT_STATS == avg_time_sec : 
            avg_time = None
            return self.get_data(varname, time_slice=time_slice, saved=saved, **kw_get)
        else : raise(Exception(DT_STATS, avg_time))
    
    def smooth(self, varname, DX_smooth=None, crop=None, truncate=5, saved={}, **kw_get):
        """
        Description
            Smooth horizontally in model coordinates with a Gaussian filter. Assuming a constant value of DX.
        Parameters
            self : Dom
            varname : str : name of the variable 
            DX_smooth : size of the kernel in kilometers, assuming DX is in meters
        Optionals
            kw_get : dictionnary : all the parameters to call again get_data
        Output
            the horizontally smoothed data
        """
        if varname.endswith("STAG") and kw_get["i_unstag"] == None :
            raise(Exception("error in Proj.smooth : cannot smooth horizontally with staggered variable : " + varname + "\n"+ 
                            "Modify the code if you need to do this"))
        
        #get size of the kernel
        NX, NY, DX, DY = self.get_data(["NX", "NY", "DX", "DY"])
        if type(DX) in [np.array, np.ndarray] : DX = np.nanmean(DX)
        if type(DY) in [np.array, np.ndarray] : DY = np.nanmean(DY) #for AROME
        sigma_mask = (DX_smooth*1000/DX, DX_smooth*1000/DY) if DY != DX else DX_smooth*1000/DX
        
        #extend the crop to limit the boundary effect
        crop = self.prepare_crop_for_get(crop, varname) #in case crop is None but it shouldn't
        new_crop = self.prepare_crop_for_get((crop[0], "ALL", "ALL"), varname)
        #To get faster if several time the same variables
        if not "before_smooth" in saved :
            saved["before_smooth"] = {}
        
        if debug : print(self.prefix, "smooth before get data : varname :", varname, ", crop :", crop, ", new_crop :", new_crop)
        #get the variable
        data = self.get_data(varname, crop=new_crop, saved=saved["before_smooth"], **kw_get)
        
        #smooth the variable
        data_smooth = manage_scipy.my_gaussian_filter(data, sigma=sigma_mask, truncate=truncate, axes=(-2, -1)) #see lib.manage_scipy
        
        #To get faster if several time the same variables in different kw_get
        # if not "after_smooth" in saved :
        #     saved["after_smooth"] = {}
        # saved["after_smooth"][varname] = data_smooth
        
        if debug : print(self.prefix, "smooth before crop : varname :", varname, ", shape :", data_smooth.shape)
        #crop after
        s = ()
        for dim in range(data_smooth.ndim - 2) :
            s += (slice(None),)
        for c in crop[-2:] :
            if type(c) == list :
                s += (slice(c[0], c[1]),)
            elif type(c) in [int, np.int64] :
                s += (c,)
        data_smooth = data_smooth[s]
        if debug : print(self.prefix, "smooth after crop : varname :", varname, ", shape :", data_smooth.shape)
        return data_smooth
         
    def vinterp(self, varname, points, levels=None, ZP="ZP", crop=None, saved={}, **kw_get):
        """
        Description
            Get vertically interpolated data thanks to wrf.vertcross
            https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.vertcross.html
            Note that the name is not well chosen, vcross would have been more clear
        Parameters
            self : Dom
            varname : str : name of the variable 
            points : something to deduce the coordinates, can be
                - already two coordinates pairs : the upper-left and the lower_right (a list of 2 lists of 2 reals) [[lat1, lon1], [lat2, lon2]]
                - The points index (a list of 4 integers) [iy1, iy2, ix1, ix2]
                - the coordinates of the center, the length and direction [[lat_c, lon_c], L, beta]
        Optionals
            levels :
                float : 1 level of interpolation
                1D sequence : levels of interpolation, ex : [50, 100, 2000], default : None (a set of levels is provided by wrf.vertcross)
                tuple : (start, end, step), see get_cropz_for_interp
            ZP : str : name of the vertical variable of which levels are defined, default is "ZP",  ex : "ZP", "Z", "P"
            kw_get : dict :  can contains : "time_slice", "crop", "i_unstag" : see self.get_data 
        Output
            data on the vertical cross-section
        """
        if "WD" in varname : 
            print(self.prefix, "WARNING : THE VARIABLE", varname, "WILL BE INTERPOLATED. TO INTERPOLATE WIND DIRECTION IT WOULD BE BETTER TO\
            FIRST INTERPOLATE U AND V AND THEN COMPUTE THE WIND DIRECTION OF THE INTERPOLATION.")
        if varname.endswith("STAG") and i_unstag == None and not ZP.endswith("STAG"):
            raise(Exception("error in Proj.vinterp : cannot interp horizontally with staggered variable : " + varname + ", because Z (" + ZP + ") is not staggered \n Modify the code if you need to do this"))
            
        #Managing Horizontal crop :
        points = self.get_points_for_vcross(points)
        ind = manage_projection.points_to_coord(points)
        iy1, iy2, ix1, ix2 = self.get_zoom_index(points)
        iy1 = max(iy1-4, 0); 
        iy2 = min(iy2+4, self.get_data("NY"))
        ix1 = max(ix1-4, 0)
        ix2 = min(ix2+4, self.get_data("NX"))        
        
        #Reading levels
        crop_ZP = ([0, self.get_data("NZ")], [iy1, iy2], [ix1, ix2])
        time_slice_ZP = 0
        dim = self.get_dim(varname)
        if dim is None :
            # we get data a first time to know the dimension, on a single time step
            dim = self.get_data(varname, time_slice=time_slice_ZP, crop=crop_ZP).ndim
        is3D = (dim == 3)
        if debug : print(self.prefix, var_temp.shape)
        if is3D :
            ZP_data = self.get_data(ZP, time_slice=time_slice_ZP, crop=crop_ZP)
            zmax = np.max(ZP_data[-1])-1
            zmin = max(np.min(ZP_data[0]), 0)
            ZP_max = zmax #for next section
            ZP_min = zmin #for next section
            dz_levels = 100
            if levels is None : 
                levels = (zmin, zmax, dz_levels) 
            elif type(levels) in [tuple, list, np.array, np.ndarray] :
                pass
            else : #assuming it is int or float
                zmax = min(zmax, levels)
                levels = (zmin, zmax, dz_levels)  
            cropz, levels = self.get_cropz_for_interp(levels, ZP_max, ZP_min, ZP_data, zmin, zmax)
        else :
            cropz = 0
        crop_data = (cropz, [iy1, iy2], [ix1, ix2])
        proj_params = {
            'MAP_PROJ': self.get_data('MAPPROJ'),
            'TRUELAT1': self.get_data('TRUELAT1'),
            'TRUELAT2': self.get_data('TRUELAT2'),
            'MOAD_CEN_LAT': self.get_data('CTRLAT'),
            'STAND_LON': self.get_data('TRUELON'),
            'POLE_LAT': 90,
            'POLE_LON': 0,
            'DX': self.get_data('DX_PROJ'),
            'DY': self.get_data('DY_PROJ'),
        }
        proj = WrfProj(**proj_params)
                           
        LAT = self.get_data("LAT", crop=crop_data)
        LON = self.get_data("LON", crop=crop_data)
        ll_point = manage_projection.points_to_coord([LAT.flatten()[0], LON.flatten()[0]])
        
        #To get faster if several time the same variables
        if not "before_vinterp" in saved :
            saved["before_vinterp"] = {}
            
        #Interpolating data
        if is3D : ZP_data = self.get_data(ZP, crop=crop_data, saved=saved["before_vinterp"], **kw_get)
        var_data = self.get_data(varname, crop=crop_data, saved=saved["before_vinterp"], **kw_get)
        #it is better to interpolate logP than P
        if self.is_Pressure(varname) : var_data = np.log(var_data)
        
        #Probably useless : data must be 3D arrays for interplevel and 2D with interpline so we need to convert to 3D/2D array
        # n_dim = var_data.ndim
        # nt = len(var_data) if type(kw_get["time_slice"]) is slice else 1 
        # expected_ndim = 3 if is3D else 2
        # if nt > 1 : expected_ndim +=1
        # for i_dim in range(expected_ndim-n_dim):
        #     if is3D : ZP_data = np.expand_dims(ZP_data, axis=-1)
        #     var_data = np.expand_dims(var_data, axis=-1)
            
        if is3D : #calling wrf.vertcross
            data = vertcross(var_data, ZP_data, 
                              levels=levels, start_point=ind[0],end_point=ind[1],
                              projection=proj, ll_point=ll_point)
        else : #if 2D calling interpline
            data = interpline(var_data, start_point=ind[0], end_point=ind[1], latlon=True,
                              projection=proj, ll_point=ll_point)
            
        if self.is_Pressure(varname) : data = np.exp(data)
        return np.array(data)
    
    def hinterp(self, varname, levels=None, ZP="ZP", crop=None, saved={}, **kw_get):
        """
        Decription
            Get horizontally interpolated data thanks to wrf.interplevel
            https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.interplevel.html
            Note that the name is not well chosen since the interpolation is only vertical. hcross would have been better.
        Parameters
            self : Dom
            varname : str : name of the variable 
        Optional
            levels :
                float : 1 level of interpolation
                1D sequence : levels of interpolation, ex : [50, 100, 2000], default : None (a set of levels is provided by wrf.vertcross)
                tuple : (start, end, step), see get_cropz_for_interp
            ZP : str : name of the vertical variable of which levels are defined, default is "ZP" ex : "ZP", "Z", "P"
            kw_get : dict :  can contains : "time_slice", "crop", "i_unstag" : see self.get_data 
        Output
            data on the horizontal cross section
        """
        if "WD" in varname : 
            print(self.prefix, "WARNING : THE VARIABLE", varname, "WILL BE INTERPOLATED. TO INTERPOLATE WIND DIRECTION IT WOULD BE BETTER TO\
            FIRST INTERPOLATE U AND V AND THEN COMPUTE THE WIND DIRECTION OF THE INTERPOLATION.")
                  
        if varname.endswith("STAG") and i_unstag == None and not ZP.endswith("STAG"):
            raise(Exception("error in Proj.hinterp : cannot interp with staggered variable : ", varname, ", because Z is not staggered \nModify the code if you need to do this"))
            
        crop_ZP = None if crop is None else ("ALL", crop[-2], crop[-1])
        time_slice_ZP = 0
        ZP_data = self.get_data(ZP, time_slice=time_slice_ZP, crop=crop_ZP, saved={})
        ZP_max = np.min(ZP_data[-1])
        ZP_min = np.min(ZP_data[0])
        if ZP_max < ZP_min :
            temp = ZP_max
            ZP_max = ZP_min
            ZP_min = temp
            descending = True
        cropz, levels = self.get_cropz_for_interp(levels, ZP_max, ZP_min, ZP_data, ZP_min, 20000)
        crop_data = (cropz, "ALL", "ALL") if crop is None else (cropz, crop[-2], crop[-1])
        
        #To get faster if several time the same variables
        if not "before_hinterp" in saved :
            saved["before_hinterp"] = {}
            
        #Interpolating data 
        ZP_data = self.get_data(ZP, crop=crop_data, saved=saved["before_hinterp"], **kw_get)
        var_data = self.get_data(varname, crop=crop_data, saved=saved["before_hinterp"], **kw_get)
        #it is better to interpolate logP than P
        if self.is_Pressure(varname) : var_data = np.log(var_data)
        if self.is_Pressure(ZP) : 
            ZP_data = np.log(ZP_data)
            levels = np.log(levels)
            
        #I don't know why but it doesn't work if shape_y = 1
        shape_y = ZP_data.shape[-2]
        if shape_y == 1 :
            ZP_data = np.concatenate((ZP_data, ZP_data), axis=-2)
            var_data = np.concatenate((var_data, var_data), axis=-2)
        #calling wrf.interplevel
        data = interplevel(var_data, ZP_data, levels, squeeze=False)
        if shape_y == 1 :
            y_axis = var_data.ndim - 2
            if y_axis == 1 :
                data = data[:, :1]
            elif y_axis == 2 :
                data = data[:, :, :1]
        if self.is_Pressure(varname) : data = np.exp(data)
        return np.array(data)
    
    def prepare_crop_for_get(self, crop, varname=None) :
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
        if varname is not None and "ZSTAG" in varname :
            NZ = self.VARIABLES["NZ_ZSTAG"].value
        elif varname is not None and ("SOIL" in varname or varname in ["TSLB", "SMOIS"]) :
            NZ = 4
        else :
            NZ = self.VARIABLES["NZ"].value
        NY = self.VARIABLES["NY"].value
        NX = self.VARIABLES["NX"].value
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
    
    def get_cropz_for_interp(self, levels, ZP_max, ZP_min, ZP_data, zmin=0, zmax=20000):
        """
        Decription
            Prepare cropz for vinterp and hinterp
        Parameters
            self : Dom
            levels :
                float : 1 level of interpolation
                1D sequence : levels of interpolation, ex : [50, 100, 2000]
                tuple : (start, end, step), see get_cropz_for_interp
            ZP_max, ZP_min : float : maximale and minimale values of ZP
            ZP_data : array of shape (.., NZ, NY, NX) : ZP data
        Optional 
            zmin, zmax : float : minimale and maximale accepted levels
        """
        #checking if some levels are out of the domain :
        if type(levels) is tuple :
            if len(levels) == 1 :
                zmax = min(zmax, levels[0])
            elif len(levels) == 2 :
                zmin = max(levels[0], zmin)
                zmax = min(zmax, levels[1])
            elif len(levels) == 3 :
                zmin = max(levels[0], zmin)
                zmax = min(zmax, levels[1])
                dz_levels = levels[2]
            levels = np.arange(zmin, zmax, dz_levels)
        if type(levels) in [list, np.array, np.ndarray] : 
            levels = np.array(levels)
            if np.any(levels > ZP_max) :
                for i_l in range(len(levels)-1, -1, -1) : 
                    if levels[i_l] > ZP_max : np.delete(levels, i_l)
            if np.any(levels < ZP_min) : 
                for i_l in range(len(levels)-1, -1, -1) : 
                    if levels[i_l] < ZP_min : np.delete(levels, i_l)
            if len(levels) == 0 :
                raise(Exception("error in vinterp : all levels (" +str(levels) + ") are above the maximum : ZP = " +str(ZP_max) + "or below the minimum : ZP = " + str(ZP_min)))
            else :
                levels_max = np.max(levels)
                levels_min = np.min(levels)
        elif levels > ZP_max or levels < ZP_min : # only one level
            raise(Exception("error in vinterp : levels (" + str(levels) + ") is above the maximum : ZP = " + str(ZP_max) + "or below the minimum : ZP = " + str(ZP_min)))
        else : # only one level
            levels_min = levels_max = levels 
        #Setting crop values for z to save time, read only the interesting altitudes
        temp = np.argwhere(np.logical_and(ZP_data>=levels_min, ZP_data<=levels_max))[:, 0]
        while len(temp) == 0 :
            levels_max = levels_max + 100
            levels_min = levels_min - 100
            temp = np.argwhere(np.logical_and(ZP_data>=levels_min, ZP_data<=levels_max))[:, 0]
        iz_min = max(np.min(temp)-2, 0) #-2 for security because Z can change over time in WRF
        iz_max = min(np.max(temp)+2, len(ZP_data)) #+2 for same reason
        return [iz_min, iz_max], levels
    
    def is_vectorial(self, varname):
        """
        Description
            Find if a variable is a vector operation (rotation, angle, norm)
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if vectorial variable, False otherwise
        25/07/2024 : Mathieu LANDREAU
        """
        return varname.startswith("NORM_") \
            or varname.startswith("DIR_") \
            or varname.startswith("CC_") \
            or (varname.startswith("AD") and "_" in varname[3:6]\
            or varname.startswith("GWM2_")\
            or varname.startswith("GWM4_")\
               )
    
    def is_normdir(self, varname):
        """
        Description
            Find if a variable is a norm or dir of a vector
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if vectorial variable, False otherwise
        25/07/2024 : Mathieu LANDREAU
        """
        return varname.startswith("NORM_") or varname.startswith("DIR_")
    
    def is_statistics(self, varname, avg_stats=False):
        """
        Description
            Find if a variable is a statistics variance/covariance, ... and thus cannot be directly averaged or interpolated.
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if statistics variable, False otherwise
        25/07/2024 : Mathieu LANDREAU
        """
        return not avg_stats and ( 
               (varname.startswith("COV") 
             or varname.startswith("TKE") 
             or varname.startswith("M2") 
             or varname.startswith("STD") 
             or varname.startswith("M3")) 
            and not "_SFS" in varname)
    
    def is_derivative(self, varname, avg_deriv=False):
        """
        Description
            Find if a variable is a derivative
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if derivative variable, False otherwise
        25/07/2024 : Mathieu LANDREAU
        """
        return not avg_deriv and ( 
            varname.startswith("DTW_") or varname.startswith("DETA_") or varname.startswith("DYW_") or varname.startswith("DXW_")\
            or varname.startswith("DTC_") or varname.startswith("DZ_")   or varname.startswith("DYC_") or varname.startswith("DXC_")\
            or varname.startswith("DIV_") or varname.startswith("ROT_")  or varname.startswith("GRAD") or varname.startswith("WDGRAD")\
            or varname.startswith("DCC_") or varname.startswith("DTL_")  or varname.startswith("DTS_")  or varname == "SHEAR"
        )
    
    def is_using_derivative(self, varname):
        """
        Description
            Find if a variable is a using a derivative in the calculation
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if using a derivative, False otherwise
        25/07/2024 : Mathieu LANDREAU
        """
        raise(Exception("error : the function Dom.is_using_derivative is not finished"))
        if varname in ["RI", "COVWPTV_SGS", "NBV", "NBV2"] :
            return True
        if varname in "RIF" and self.get_data("KM_OPT") == 2 :
            return True
        if varname in "KMW" and self.get_data("BL_PBL_PHYSICS") == 2 :
            return True
        return False
    
    def is_Pressure(self, varname) :
        """
        Description
            Find if a variable is a Pressure variable (used in hinterp and vinterp)
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if Pressure variable, False otherwise
        25/03/2024 : Mathieu LANDREAU
        """
        return varname in ["P", "P_AVG", "P_BASE", "P_PERT", "P_PERT_AVG", "P_HYD", "P_HYD_AVG"] or (varname[0:2] == "P_" and "_TS" in varname)
        
    def is_stress_tensor_bilan_term(self, varname) :
        """
        Description
            Find if a variable is a term of the bilan of the stress tensor Rij = <ui'uj'>
            See DomWRF.calculate_stress_tensor_bilan
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            boolean : True if stress tensor bilan term variable, False otherwise, ex : PUU, PVW, AUU1, PTKE or WPTKE, WPUU1, ...
        25/03/2024 : Mathieu LANDREAU
        """
        return (
            varname[0] in ["P", "A", "T", "D", "B", "S", "N", "R", "G", "K", "Z"] and (
                len(varname) >= 4 and varname[1:4] == "TKE" or 
                (len(varname) >= 3 and (varname[1] in ["U", "V", "W"] and varname[2] in ["U", "V", "W"]))
            )
        or
            varname[0] in "W" and self.is_stress_tensor_bilan_term(varname[1:])
        )
        
    def find_axis(self, axis, dim=None, varname=None, time_slice=None, crop=None, i_unstag=None, itime=None, **kwargs):
        """
        Description
            find the dimension corresponding to 'x', 'y', 'z' or 't' based on kwargs_get_data and a variable
            WARNING : For spatial dimension, assuming 1D is ('z'), 2D is ('y', 'x'), 3D is ('z', 'y', 'x')
            WARNING : i_unstag is not used for the moment
        Parameters
            self : Dom
            axis : str : 'x', 'y', 'z' or 't'
        Optional
            dim : int : the SPATIAL dimension of the variable
            varname : str : name of the variable to find the dim if dim is not mentioned
            time_slice, crop, i_unstag : see get_data
            kwargs : if ever one want to use **kwargs_get_data instead of explicitely setting time_slice and crop 
        Output
            int : 0, 1, 2 or 3, maybe 4 but not for the moment
        22/03/2024 : Mathieu LANDREAU
        """
        axis = axis.lower()
        if dim not in [0, 1, 2, 3, 4, 5] :
            if varname is None :
                raise(Exception("error in Dom.find_axis, either dim or varname must be defined"))
            dim = self.get_dim(varname)
        count_dim = 0
        # time slice can be int or slice
        # is there a time dimension ?
        if time_slice is None : #when find_axis is called in Dom.calculate, this shouldn't happen
            time_slice = manage_time.get_time_slice(itime, self.date_list, self.max_time_correction)
        if type(time_slice) in [int, np.int64] : #No
            if axis == "t" : 
                return None
        else : #Yes
            if axis == "t" : return 0
            count_dim += 1
        if varname is not None: crop = self.prepare_crop_for_get(crop, varname) #when find_axis is called in Dom.calculate, this shouldn't be necessary
        cropz, cropy, cropx = crop
        # cropz can be int or list
        # is there a z dimension ?
        if type(cropz) in [int, np.int64] or dim in [0, 2] : #No
            if axis == "z" : return None
        else : #Yes
            if axis == "z" : return count_dim
            count_dim += 1
        # cropy can be int or list
        # is there a y dimension ?
        if type(cropy) in [int, np.int64] or dim in [0, 1] : #No
            if axis == "y" : return None
        else : #Yes
            if axis == "y" : return count_dim
            count_dim += 1
        # cropx can be int or list
        # is there a y dimension ?
        if type(cropx) in [int, np.int64] or dim in [0, 1] : #No
            if axis == "x" : return None
        else : #Yes
            if axis == "x" : return count_dim
        raise(Exception("error in Dom.find_axis : the axis type : ", axis, "cannot be found"))
        
    def get_dim(self, varname, return_status=False) :
        """
        Description
            get the number of spatial dimensions of a variable
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            int : 0, 1, 2 or 3, maybe 4 but not for the moment
        22/03/2024 : Mathieu LANDREAU
        """
        if varname in self.VARIABLES :
            return self.VARIABLES[varname].dim
        elif varname.endswith("_TS") : 
            return 1
        elif varname in ["GWM2X1", "GWM2Y1", "GWM2RMAX", "GWM2LMAX", "ZCBL", "ZCI"] \
            or (varname.startswith("GW") and (varname.endswith("LAM") or varname.endswith("LAMM") or varname.endswith("DIR") or varname.endswith("S")
                                              or varname.endswith("SM") or varname.endswith("D"))) :
            return 0
        elif varname in ["GWM2R", "GWM2L", "GWM2D", "SBZC", "LBZC", "CIBLZC", "SIBLZC", "CIBLZC2", "SIBLZC2", "CIBLZC3", "SIBLZC3", "PSLAS", "CIBLW", "SIBLW"] or varname.startswith("GWMASK") \
        or varname.startswith("LLJ_") or varname.startswith("LLJ2_"):
            return 2
        else : 
            varname2 = self.find_similar_variable(varname)
            if varname2 is None :
                if return_status :
                    return None
                else :
                    print(self.prefix, "warning in Dom.get_dim : ", varname, " not in self.VARIABLES, cannot find the dimension, assuming 3")
                    return 3
            else :
                return self.get_dim(varname2)
    
    def find_similar_variable(self, varname):
        """
        Description
            for variables that are not in self.Variables, find a know variable that is similar.
            Used in get_cmap, get_dim, (maybe others)
        Parameters
            self : Dom
            varname : str : name of the variable 
        Output
            similar_var_name : str : name of the similar variable
            None if no similar varname is found
        22/03/2024 : Mathieu LANDREAU
        """
        if varname[:3] in ["DZ_", "CC_"] : 
            return varname[3:]
        elif varname[:4] in ["DXC_", "DXW_", "DYC_", "DYW_", "DIV_", "ROT_", "DCC_", "DIR_", "DTW_", "DTC_", "LLJ_", "LOG_", "EXP_"] : 
            return varname[4:]
        elif varname[:5] in ["DETA_", "GRAD_", "NORM_", "GWM2_", "GWM4_", "SQRT_", "LLJ2_"] : 
            return varname[5:]
        elif varname[:4] in ["GRAD"] and varname[5] == "_" : 
            return varname[6:]
        elif varname[:4] in ["GRAD"] and len(varname)>6 and varname[6] == "_" : 
            return varname[7:]
        elif varname[:4] in ["GRAD"] and len(varname)>7 and varname[7] == "_" : 
            return varname[8:]
        elif varname[:7] in ["WDGRAD_"] : 
            return varname[7:]
        elif varname[:8] in ["SQUARED_"] : 
            return varname[8:]
        elif varname[-4:] in ["_AVG"] :
            return varname[:-4]
        elif varname[-2:] in ["_C"] :
            return varname[:-2]
        elif varname[-3:] in ["_KM"] :
            return varname[:-3]
        elif varname.startswith("AD") :
            if varname[3] == "_" :
                return varname[4:]
            elif varname[4] == "_" :
                return varname[5:]
            elif varname[5] == "_" :
                return varname[6:]
            else :
                return None
        elif varname == "WD180" : 
            return "WD"
        elif "LANDMASK" in varname :
            return "LANDMASK"
        else : 
            return None
    
    def get_X_for_vinterp(self, kw_get, Z=None, KM=False) :
        """
        Decription
            Create the horizontal coordinate for vertical cross-section (vinterp only return the data)
        Parameters
            self : Dom
            kw_get : the kwargs dictionnary containing the parameters of get_data
        Optional 
            Z : the data, needed just to get the shape (the number of points)
            KM : boolean : if True returns the distance in kilometers, otherwise in meters
        Output
            the X data i.e. the distance from the center of the cross-section, positive in one direction and negative in the other.
        """
        if not "vinterp" in kw_get or kw_get["vinterp"] is None :
            raise(Exception("error in Sim.get_X_for_vinterp : no vinterp in kwargs_get_data"))
        else :
            points = kw_get["vinterp"]["points"]
        if Z is None :
            Z = self.get_data("HT", **kw_get)
        NX = np.shape(Z)[-1]
        distance, direction = manage_projection.haversine(points[0], points[1])
        if KM :
            distance = distance/1000
        return np.linspace(-distance/2, distance/2, NX)[:,0]
    
    def get_circle_mask(self, R, iy, ix) : 
        """
        Get a mask for all points below a certain radius R from iy, ix
        Parameters
            self : Dom
            R : float : radius of the circle
            iy, ix : int : position of the center
        Returns 
            mask : numpy array of shape (NY, NX) : 1 for points within the circle, 0 for others
        01/2024 : Mathieu LANDREAU
        """
        DX = self.get_data("DX")
        DY = self.get_data("DY")
        ny = int(R//DY)
        NX, NY = self.get_data("NX"), self.get_data("NY")
        mask = np.zeros((NY, NX))
        for iy2 in range(ny) :
            y2 = iy2*DY
            x2 = np.sqrt(R**2-y2**2)
            ix2 = int(x2//DX)
            ix3 = ix2
            if ix - ix3 < 0 :
                ix3 = ix
            if ix + ix2 >= NX :
                ix2 = NX - ix - 1
            if iy+iy2 < NY :
                mask[iy+iy2, ix-ix3:ix+ix2+1] = 1
            if iy-iy2 >= 0 :
                mask[iy-iy2, ix-ix3:ix+ix2+1] = 1
        return mask
    
    def get_few_coastcells(self, Npoint=15, bdy_dist=None, coastmask=None, **kwargs) :
        X = self.get_data("X", **kwargs)
        Y = self.get_data("Y", **kwargs)
        LANDMASK = self.get_data("LANDMASK2", **kwargs)
        return manage_images.get_few_coastcells(X, Y, LANDMASK, Npoint=Npoint, bdy_dist=bdy_dist, coastmask=coastmask)
    
    def get_SB_t0(self, iy, ix, itime="ALL_TIMES", R=50e3): 
        """
        Get the initial time of the SB defined as when the sensible heat flux on land surface and on sea surface are equal within a given radius
        Parameters
            self : Dom
            itime : same thing as in get_data, should be (date1, date2) or "ALL_TIMES", shouldn't be a single date
            iy, ix : int : position of the center
            R : float : radius for get_circle_mask
        Returns 
            mask : numpy array of shape (NY, NX) : 1 for points within the circle, 0 for others
        01/2024 : Mathieu LANDREAU
        """
        X = self.get_data("X")
        Y = self.get_data("Y")
        SH_FLX = self.get_data("SH_FLX", itime=itime)
        LANDMASK = self.get_data("LANDMASK")
        TIME = self.get_data("TIME", itime=itime)
        circlemask = self.get_circle_mask(10e3, iy, ix)
        land = np.logical_and(LANDMASK == 1, circlemask == 1)
        sea = np.logical_and(LANDMASK == 0, circlemask == 1)
        nland = np.sum(land)
        nsea = np.sum(sea)
        SH_LAND = np.mean(SH_FLX[:, land], axis=1)
        SH_SEA = np.mean(SH_FLX[:, sea], axis=1)
        PHI = SH_LAND - SH_SEA
        it0 = np.argmax(np.diff(PHI>0))+1
        return TIME[it0]
    
    def get_ZT(self, **kwargs):
        """
        Description
            Get Z and TIME for a ZT plot
        Parameters
            self : Dom
        Optional
            kwargs : parameters for get data
        Returns 
            Z, t : 2D numpy arrays of shape (NZ, NT)
            Z_vec : 1D numpy array of shape NZ
            t_vec : 1D numpy array of shape NT
        01/2024 : Mathieu LANDREAU
        """
        Z = self.get_data("Z", **kwargs)
        t_vec = self.get_data("TIME", **kwargs)
        Z_vec = Z[0] #just needed to expand t_vec 
        _, t = np.meshgrid(Z_vec, t_vec) 
        return Z, t, Z_vec, t_vec
   
    def nearest_z_index(self, z_in, itime=0, crop=("ALL", 0, 0), return_Z=False, return_diff=False):
        """
        Description
            Find the nearest z index to a given height
        Parameters
            self : Dom
            z_in : the expected height
        Optional
            itime, crop : see self.get_data
            return_Z : boolean : if True, return also the real z value
            return_diff : boolean : if True, return the difference 
        Returns 
            Z, t : 2D numpy arrays of shape (NZ, NT)
            Z_vec : 1D numpy array of shape NZ
            t_vec : 1D numpy array of shape NT
        01/2024 : Mathieu LANDREAU
        """
        Z_vec = self.get_data("Z", itime=itime, crop=crop)
        diff = np.abs(z_in - Z_vec)
        iz = np.argmin(diff)
        result = (iz,) if return_Z or return_diff else iz
        if return_Z :
            result += (Z_vec[iz],)
        if return_diff :
            result += (diff[iz],)
        return result
    
    def get_title(self, varname, *args, **kwargs) :
        """
        Description
            Create a title for figures. This is not used but replaced by Proj.compute_title
        Parameters
            varname : str : name of the variable 
        Optional
            kwargs : Parameters of self.get_data
        04/2023 : Mathieu LANDREAU
        """
        var_legend = self.VARIABLES[varname].get_legend(long=True, latex=True)
        domain_legend = self.name
        at_legend = kwargs["at"] if "at" in kwargs else ""
        crop_legend = ", crop=" +str(kwargs["crop"]) if "crop" in kwargs else ""
        itime_legend = ", t=" +str(kwargs["itime"]) if "itime" in kwargs else ""
        return var_legend +" "+ at_legend + ", dom="+domain_legend+crop_legend+itime_legend
    
    def get_units(self, varname, *args, **kwargs) :
        """
        Description
            Not used
            Get units of a variable, call self.VARIABLES[varname].get_units
        Parameters
            varname : str : name of the variable 
        04/2023 : Mathieu LANDREAU
        """
        if varname in self.VARIABLES :
            return self.VARIABLES[varname].get_units(*args, **kwargs)
        else :
            return ""
    
    def get_legend(self, varname, *args, **kwargs) :
        """
        Description
            Get legend of a variable
        Parameters
            varname : str : name of the variable 
        01/02/2023 : Mathieu LANDREAU
        """
        if varname in self.VARIABLES :
            return self.VARIABLES[varname].get_legend(*args, **kwargs)
        elif varname.endswith("_C") :
            varname_temp = varname[:-2]
            return self.get_legend(varname_temp, *args, units="°C", **kwargs)
        elif varname.endswith("_KM") :
            varname_temp = varname[:-3]
            return self.get_legend(varname_temp, *args, units="km", **kwargs)
        elif varname in ["WD180"] :
            varname_temp = self.find_similar_variable(varname)
            return self.get_legend(varname_temp, *args, **kwargs)
        else :
            return varname
    
    def get_cmap(self, varname, *args, **kwargs) :
        """
        Description
            Get colormap of a variable
        Parameters
            varname : str : name of the variable 
        Output
            integer : see manage_plot for the correspondance between int and colormaps
        01/02/2023 : Mathieu LANDREAU
        """
        if varname in self.VARIABLES :
            return self.VARIABLES[varname].get_cmap(*args, **kwargs)
        else : 
            varname2 = self.find_similar_variable(varname)
            if varname2 is None :
                return 0
            else :
                return self.get_cmap(varname2)
        
           
    def calculate(self, varname, **kwargs):
        """
        Description
            Calculate the value of a variable if not directly in a file
            Note : for inherited class (like DomARPS, DomWRF, ...) the inherited method (DomWRF.calculate, ...) is called first and 
                   if the variable isn't defined in the inherited method, then Dom.calculate is called
        Parameters
            varname : str : name of the variable 
        Optional
            kwargs : all the kwargs from get_data
        01/02/2023 : Mathieu LANDREAU
        """
        if varname.upper() == "IT" :
            it = np.arange(5000, dtype="int")[kwargs["time_slice"]]
            return np.expand_dims(it, axis=(-3, -2, -1))
        elif varname.upper() == "IZ" :
            s = slice(kwargs["crop"][0][0], kwargs["crop"][0][1])
            iz = np.arange(5000, dtype="int")[s]
            return np.expand_dims(iz, axis=(0, -2, -1))
        elif varname.upper() == "IY" :
            s = slice(kwargs["crop"][1][0], kwargs["crop"][1][1])
            iy = np.arange(5000, dtype="int")[s]
            return np.expand_dims(iy, axis=(0, 1, -1))
        elif varname.upper() == "IX" :
            s = slice(kwargs["crop"][2][0], kwargs["crop"][2][1])
            ix = np.arange(5000, dtype="int")[s]
            return np.expand_dims(ix, axis=(0, 1, 2))
        elif varname in ["IX2D", "IY2D"] :
            IX, IY = self.get_data(["IX", "IY"], **kwargs)
            IX2D, IY2D = np.meshgrid(IX, IY)
            if varname == "IX2D" : return IX2D
            elif varname == "IY2D" : return IY2D
        elif varname.startswith("SQUARED_"):
            return self.get_data(varname[8:], **kwargs)**2
        elif varname.startswith("SQRT_"):
            return np.sqrt(self.get_data(varname[5:], **kwargs))
        elif varname.startswith("LOG_"):
            return np.log(self.get_data(varname[4:], **kwargs))
        elif varname.startswith("EXP_"):
            return np.exp(self.get_data(varname[4:], **kwargs))
        elif varname in ["Z_KM"] :
            return self.get_data("Z", **kwargs)/1e3
        elif varname == "MTIME" : 
            # not used anymore. The objective was to get the time of the middle of the averaged window for averaged variables
            TIME_STATS = np.array(self.get_data("TIME_STATS", **kwargs))
            return TIME_STATS[:-1] + (TIME_STATS[1:] - TIME_STATS[:-1])/2 
        elif varname in ["M"]: # Wind magnitude
            U, V, W = self.get_data(["U", "V", "W"], **kwargs)
            return np.sqrt( U**2 + V**2 + W**2 )
        elif varname in ["MH", "MHS", "MHAS"]: #Horizontal wind magnitude, S=synoptic, AS=deviation from S
            U, V = self.get_data(["U"+varname[2:], "V"+varname[2:]], **kwargs)
            return np.sqrt( U**2 + V**2 )
        elif varname in ["WD", "WDS", "WDAS"]: #Wind direction in degrees [0, 360], S=synoptic, AS=deviation from S
            # note : Wind direction is clockwise from North and points where wind comes FROM
            WD = manage_angle.UV2WD_deg(self.get_data("U"+varname[2:], **kwargs), self.get_data("V"+varname[2:], **kwargs)) # see lib/manage_angle
            return WD
        elif varname in ["WD180", "WDS180", "WDAS180"]: #Wind direction in degrees [-180, 180], S=synoptic, AS=deviation from S
            return manage_angle.angle180(self.get_data(varname[:-3], **kwargs))
        elif varname in ["US", "VS", "PS", "PSLS"]: #synoptic Velocites, Pressure, and surface pressure
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
                    new_kwargs = self.copy_kw_get(kwargs)
                    new_kwargs["time_slice"] = None
                    new_kwargs["saved"] = {}
                    new_kwargs["itime"] = (date_i, date_i+manage_time.to_timedelta(1, "d"))
                    new_kwargs["crop"] = ("ALL", "ALL", "ALL")
                    if varname == "PS" :
                        new_kwargs["hinterp"] = {
                            "levels" : 3000, #I arbitrary chose a height of 3000m above SEA level.
                            "ZP" : "ZP", #geopotential height is necessary
                        }
                    elif varname == "PSLS" :
                        new_kwargs["crop"] = (0, "ALL", "ALL")
                    else :
                        new_kwargs["hinterp"] = {
                            "levels" : 2000, #I arbitrary chose a height of 3000m above GROUND level.
                            "ZP" : "Z",
                        }
                    var = self.get_data(varname[:-1], **new_kwargs)
                    var = np.mean(var, axis=0)
                    cmap = self.VARIABLES[varname[:-1]].cmap
                    latex_units = self.VARIABLES[varname[:-1]].latex_units
                    units = self.VARIABLES[varname[:-1]].units
                    self.VARIABLES[new_varname] = VariableSave(new_varname, new_varname, latex_units, units, 2, var, cmap=cmap) #save
                var = self.get_data(new_varname, crop=kwargs["crop"]) #read the saved variable and crop if necessary
                temp_list.append(var)
            temp_list = np.array(temp_list)
            return np.squeeze(temp_list[unique_inverse])
        elif varname in ["WS"]: #synoptic W = 0
            return self.get_data("ZERO", **kwargs)
        elif varname in ["UAS", "VAS", "WAS", "PAS"]: #deviation from synoptic 
            varS = self.get_data(varname[:-2]+"S", **kwargs)
            var = self.get_data(varname[:-2], **kwargs)
            zaxis = self.find_axis("z", varname=varname[:-2], **kwargs)
            varS = np.expand_dims(varS, axis=zaxis)
            return var - varS
        elif varname in ["PSLAS"]: #deviation from synoptic 
            PSL, PSLS = self.get_data(["PSL", "PSLS"], **kwargs)
            return PSL - PSLS
        elif varname in ["URHO", "VRHO", "WRHO"]: # Product rho*u
            RHO, U = self.get_data(["RHO", varname[-1]], **kwargs)
            return RHO*U
        elif varname in ["PRHO"]: # Potential density (equivalent to potential temperature)
            # Warning : does it really exist ?
            RHO, P = self.get_data(["RHO", "P"], **kwargs)
            return constants.RHO2PRHO(RHO, P)
        
        elif self.is_derivative(varname): # Every derivative
            return self.calculate_derivative(varname, **kwargs)
        elif self.is_statistics(varname) :
            return self.calculate_statistics(varname, **kwargs)
        elif self.is_vectorial(varname): #Calculate rotation, norm, direction of vectors
            return self.calculate_rotation(varname, **kwargs)
        elif self.is_stress_tensor_bilan_term(varname) :  # production, advection, pressure transport, turbulent diffusion terms of the stress tensor
            return self.calculate_stress_tensor_bilan_term(varname, **kwargs)  
        
        elif varname.endswith("_C"): #convert from Kelvin to Celsius degrees
            varname_K = varname[:-2]
            var_K = self.get_data(varname_K, **kwargs)
            return constants.Kelvin_to_Celsius(var_K) 
        elif varname == "LOGP" : #ln(P)
            P = self.get_data("P", **kwargs)
            return np.log(P)
        elif varname == "P_BASE2" : 
            #useless : tried to decompose P = P_BASE2 + P_PERT2 with P_BASE2 contant with height
            ZP = self.get_data("ZP", **kwargs)
            return 106398 * np.exp(-0.000116625 * ZP) - 5655.92
        elif varname in ["P_PERT2"] :
            #useless : tried to decompose P = P_BASE2 + P_PERT2 with P_BASE2 contant with height
            return self.get_data("P", **kwargs) - self.get_data("P_BASE2", **kwargs)
        elif varname in ["PSL"] : 
            #Fictive Sea level pressure (extrapolation of surface pressure from height ZP=HT to 0mASL)
            #see https://github.com/NCAR/wrf-python/blob/main/src/wrf/extension.py#L238
            #see method : DCOMPUTESEAPRS https://github.com/NCAR/wrf-python/blob/develop/fortran/wrf_user.f90
            kwargs_slp = self.copy_kw_get(kwargs)
            if "crop" in kwargs_slp :
                crop_slp = ("ALL", kwargs_slp["crop"][1], kwargs_slp["crop"][2])
            else :
                crop_slp = ("ALL", "ALL", "ALL")
            kwargs_slp["crop"] = crop_slp
            ZP, T, P, QV = self.get_data(["ZP", "T", "P", "QV"], **kwargs_slp)
            print(self.prefix, ZP.shape, T.shape, P.shape, QV.shape)
            return wrf_slp(ZP, T, P, QV)*100
        elif varname in ["PSL24"]:
            # PSL averaged over 1 day (t-12h, t+12h)
            print("warning : this function return PSL24 for every time at every location. It takes some time to calculate, \
                  you should better write_postproc of PSL (timestep by timestep) and then of PSL24 (all together)\n\
                  Then, it will be cropped and selected as usual")
            PSL = self.get_data("PSL", itime="ALL_TIMES")
            return manage_list.moving_average2(PSL, axis=0)
        elif varname in ["PSLD"] :
            # Deviation of PSL from PSL24
            PSL, PSL24 = self.get_data(["PSL", "PSL24"], **kwargs)
            return PSL-PSL24
        elif varname in ["AVO", "PVO"] :
            #Horizontal Vorticity : it can be added using the following method :
            raise(Exception("https://github.com/NCAR/wrf-python/blob/develop/fortran/wrf_pvo.f90"))
        elif varname == "Q" :
            #Total mixing ratio
            QV,QC,QR,QI,QS,QH,QG = self.get_data(["QV","QC","QR","QI","QS","QH","QG"], **kwargs)
            return QV + QC + QR + QI + QS + QH + QG
        elif varname == "RHO" :
            #Total density
            P,T,Q,QV = self.get_data(["P","T","Q","QV"], **kwargs)
            #if QI, QC, QR > 0 then Q != QV then RHO_D + RHO_V < RHO
            #However since ice, rain, ... are not gaz, then they do not directly act on state equation but their presence decrease partial density of air and vapor
            #Using Stull (2017) p.15, eq. 1.22 and eq. 1.23 and replacing r_v, r_i, ... by RHOV/RHO, RHOI/RHO, one can deduce the equation below
            # P = (RHOD*RD + RHOV*RV) T with RHOV = QV*RHO and RHOD = (1-QV-QC-QI-...)*RHO
            return P/(T*(QV*constants.RV + (1-Q)*constants.RD))
        elif varname in ["RHOV", "RHOC", "RHOR", "RHOI", "RHOS", "RHOH", "RHOG"] :
            #Partial densities
            var_Q,RHO = self.get_data(["Q"+varname[-1],"RHO"], **kwargs)
            return RHO*var_Q
        elif varname == "RHOD" :
            #Dry air density
            Q = self.get_data("Q", **kwargs)
            RHO = self.get_data("RHO", **kwargs)
            return RHO*(1-Q)
        elif varname in ["NBV2"] : #squared Brunt-Väisälä frequency, WARNING : defined again in DomWRF
            # PTV = np.squeeze(self.get_data("PTV", **kwargs))
            # DZ_PTV = np.squeeze(self.get_data("DZ_PTV", **kwargs))
            PTV,DZ_PTV = self.get_data(["PTV","DZ_PTV"], **kwargs)
            return constants.G/PTV * DZ_PTV
        elif varname in ["NBV"] : #Brunt-Väisälä frequency, WARNING : NBV2 is defined again in DomWRF
            NBV2 = self.get_data("NBV2", **kwargs)
            return np.sqrt(NBV2)
        
        elif varname in ["D11", "D12", "D13", "D21", "D22", "D23", "D31", "D32", "D33"]:
            # Components of the tensor Dij = Ui,j+Uj,i (as defined in WRF)
            i = int(varname[1])-1
            j = int(varname[2])-1
            varnamei = ["U", "V", "W"][i]
            varnamej = ["U", "V", "W"][j]
            derivi = ["DXC_", "DYC_", "DZ_"][i]
            derivj = ["DXC_", "DYC_", "DZ_"][j]
            dUidXj = self.get_data(derivj+varnamei, **kwargs)
            if i == j : #do not calculate twice the same (derivatives are expensive)
                return 2*dUidXj
            else :
                dUjdXi = self.get_data(derivi+varnamej, **kwargs)
                return dUidXj + dUjdXi
        elif varname in ["TAU11", "TAU12", "TAU22", "TAU21"] : 
            #horizontal components of the stress tensor in K theory, CAREFUL : ONLY THE SUBGRID PART
            KMH = self.get_data("KMH", **kwargs) 
            Dij = self.get_data("D"+varname[3:], **kwargs)
            return -KMH*Dij
        elif varname in ["TAU13", "TAU23", "TAU33", "TAU31", "TAU32"] : 
            #Vertical components of the stress tensor in K theory, CAREFUL : ONLY THE SUBGRID PART
            KMV = self.get_data("KMV", **kwargs) 
            Dij = self.get_data("D"+varname[3:], **kwargs)
            return -KMV*Dij
        
        # Forces in m/s/h
        elif varname in ["FTURB1", "FTURB2", "FTURB3"] : 
            #Mixing force (TAUij,j)
            DXC_TAUi1 = self.get_data("DXC_TAU"+varname[-1]+"1", **kwargs)
            DYC_TAUi2 = self.get_data("DYC_TAU"+varname[-1]+"2", **kwargs)
            DZ_TAUi3 = self.get_data("DZ_TAU"+varname[-1]+"3", **kwargs)
            return -(DXC_TAUi1 + DYC_TAUi2 + DZ_TAUi3) * 3600 #m/s/h
        elif varname in ["FP1", "FPS1", "FPAS1", "FP2", "FPS2", "FPAS2", "FP3", "FPS3", "FPAS3"] : 
            #Pressure force (P=total, PS=synoptic, PAS=asynoptic)
            i = int(varname[-1])
            derivi = ["DXC_", "DYC_", "DZ_"][i-1]
            RHO = self.get_data("RHO", **kwargs)
            DXi_P = self.get_data(derivi+varname[1:-1], **kwargs)
            return -DXi_P/RHO * 3600 #m/s/h
        elif varname in ["FADV11", "FADV12", "FADV13", "FADV21", "FADV22", "FADV23", "FADV31", "FADV32", "FADV33"] : 
            #Advection Force separating each direction
            i = int(varname[4])
            j = int(varname[5])
            varnamei = ["U", "V", "W"][i-1]
            varnamej = ["U", "V", "W"][j-1]
            derivj = ["DXC_", "DYC_", "DZ_"][j-1]
            return self.get_data(varnamej, **kwargs) * self.get_data(derivj + varnamei, **kwargs) * 3600 #m/s/h
        elif varname in ["FADV1", "FADV2", "FADV3"] : 
            #Total Advection Force
            return self.get_data(varname+"1", **kwargs) + self.get_data(varname+"2", **kwargs) + self.get_data(varname+"3", **kwargs) * 3600 #m/s/h
        elif varname in ["FCOR1", "FCORS1", "FCORAS1"] :
            #X Coriolis force (=total, S=synoptic, AS=asynoptic)
            V = self.get_data("V"+varname[4:-1], **kwargs)
            FC = self.get_data("FC", **kwargs)
            return FC*V * 3600 #m/s/h
        elif varname in ["FCOR2", "FCORS2", "FCORAS2"] : 
            #Y Coriolis force (=total, S=synoptic, AS=asynoptic)
            U = self.get_data("U"+varname[4:-1], **kwargs)
            FC = self.get_data("FC", **kwargs)
            return -FC*U * 3600 #m/s/h
        elif varname in ["FGRAV"] :
            #Gravity force
            ZERO = self.get_data("ZERO", **kwargs) #just to get the right shape
            return -constants.G * np.ones(U.shape) * 3600 #m/s/h
        elif varname in ["FGEO1", "FGEO2", "FGEOS1", "FGEOS2", "FGEOAS1", "FGEOAS2"] : 
            #Geostrophic force (pressure - coriolis)
            FP = self.get_data("FP"+varname[4:], **kwargs)
            FCOR = self.get_data("FCOR"+varname[4:], **kwargs)
            return FP + FCOR * 3600 #m/s/h
        elif varname in ["FGEO3", "FGEOS3", "FGEOAS3"] : 
            #Hydrostatic force (pressure - gravity)
            FP = self.get_data("FP"+varname[4:], **kwargs)
            FGRAV = self.get_data("FGRAV", **kwargs)
            return FP + FGRAV * 3600 #m/s/h
        
        elif varname in ["UG", "VG", "MHG", "WDG", \
                         "UGSL", "VGSL", "MHGSL", "WDGSL",\
                         "UGS", "VGS", "MHGS", "WDGS",\
                         "UGSLS", "VGSLS", "MHGSLS", "WDGSLS"] : #Geostrophic velocities
            if varname.startswith("UG") : #x component
                RHO,FC,dPdy = self.get_data(["RHO","FC","DYC_P"+varname[2:]], **kwargs)
                return -dPdy/(RHO*FC)
            elif varname.startswith("VG") : #y component
                RHO,FC,dPdx = self.get_data(["RHO","FC","DXC_P"+varname[2:]], **kwargs)
                return dPdx/(RHO*FC)
            elif varname.startswith("MHG"): #total
                RHO,FC,dPdx,dPdy = self.get_data(["RHO","FC","DXC_P"+varname[3:],"DYC_P"+varname[3:]], **kwargs)
                return np.sqrt(dPdx**2 + dPdy**2)/(RHO*FC)
            elif varname.startswith("WDG"): #direction
                dPdx,dPdy = self.get_data(["DXC_P"+varname[3:],"DYC_P"+varname[3:]], **kwargs)
                WDG = manage_angle.UV2WD_deg(-dPdy, dPdx)
                return WDG
        elif varname in ["PBARO"]:
            NZ = self.get_data("NZ")
            Z1 = 1500
            kwargs_Z1 = copy.copy(kwargs)
            kwargs_Z1["saved"] = {}
            kwargs_Z1["hinterp"] = {"levels" : Z1,}
            P1 = self.get_data("P", **kwargs_Z1)
            cropz, cropy, cropx = kwargs["crop"]
            if cropz == [0, NZ] :
                kwargs_Z = kwargs
            else :
                kwargs_Z = copy.copy(kwargs)
                kwargs_Z["saved"] = {}
                kwargs_Z["crop"] = [[0, NZ], cropy, cropx]
            T, ZP, DZ = self.get_data(["T", "ZP", "DZ"], **kwargs_Z)
            zaxis = self.find_axis("z", dim=3, **kwargs_Z)
            iz = np.expand_dims(np.nanargmin(np.abs(Z1 - ZP), axis=zaxis), axis=zaxis)
            A = np.cumsum(DZ/T, axis=zaxis) - 0.5*DZ/T #correction middle of the cell
            A = A - np.take_along_axis(A, iz, axis=zaxis)
            DZiz = np.take_along_axis(ZP, iz, axis=zaxis) - Z1
            Tiz = np.take_along_axis(T, iz, axis=zaxis)
            A = A + DZiz/Tiz
            A *= -constants.G/constants.RD
            return np.take(P1*(np.exp(A)-1), np.arange(cropz[0], cropz[1]), axis=zaxis)
        elif varname in ["X2DV", "X2DV_KM"] :
            KM = varname.endswith("_KM")
            return self.get_X_for_vinterp(kwargs, KM=KM)
        elif varname == "LANDMASK2" : #Continental mask
            LANDMASK = self.get_data("LANDMASK")
            LANDMASK2 =  manage_images.get_main_LANDMASK(LANDMASK)
            self.VARIABLES[varname] = VariableSave("Continental mask", "Continental mask", "", "", 2, LANDMASK2, cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname == "COASTCELL" :
            LANDMASK2 = self.get_data("LANDMASK2") #on continental mask (no islands)
            COASTCELL = manage_images.get_COASTCELL(LANDMASK2)
            self.VARIABLES[varname] = VariableSave("Coast cells mask", "Coast cells mask", "", "", 2, COASTCELL, cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["COASTDIST", "COASTDIST_KM", "CDI"] :
            if varname == "CDI": varname = "COASTDIST_KM"
            LANDMASK2 = self.get_data("LANDMASK2") #on continental mask (no islands)
            X = self.get_data("X"+varname[9:])
            Y = self.get_data("Y"+varname[9:])
            COASTDIST = manage_images.get_COASTDIST2(LANDMASK2, X, Y)
            self.VARIABLES[varname] = VariableSave("Distance from coast", "Distance from coast", "m", "m", 2, COASTDIST, cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["COR", "CGX", "CGY"] : #COAST_ORIENT, landmask_gradient,
            sigma = int(kwargs["sigma"]*1000)
            sigma_str = str(sigma)
            if not varname+sigma_str in self.VARIABLES : 
                LANDMASK = self.get_data("LANDMASK")
                DX = self.get_data("DX")
                COR, CGX, CGY = manage_images.get_COAST_ORIENT2(LANDMASK, DX=DX, sigma=sigma)
                self.VARIABLES["COR"+sigma_str] = VariableSave("Coast orientation", "Coast orientation", "°", "°", 2, COR, cmap=2)
                self.VARIABLES["CGX"+sigma_str] = VariableSave("Coast x gradient", "Coast x gradient", "", "", 2, CGX, cmap=3)
                self.VARIABLES["CGY"+sigma_str] = VariableSave("Coast y gradient", "Coast y gradient", "", "", 2, CGY, cmap=3)
            return self.get_data(varname+sigma_str, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["CDI_SIGMA"] : #COAST_ORIENT, landmask_gradient,LANDMASK2 = self.get_data("LANDMASK2") #on continental mask (no islands)
            sigma = int(kwargs["sigma"]*1000)
            sigma_str = str(sigma)
            if not "CDI"+sigma_str in self.VARIABLES : 
                LANDMASK2 = self.get_data("LANDMASK_SIGMA", sigma=sigma/1000) #on continental mask (no islands)
                LANDMASK2[LANDMASK2 > 0.5] = 1
                LANDMASK2[LANDMASK2 <= 0.5] = 0
                LANDMASK2 = LANDMASK2.astype(int)
                X = self.get_data("X_KM")
                Y = self.get_data("Y_KM")
                CDI_SIGMA = manage_images.get_COASTDIST2(LANDMASK2, X, Y)
                self.VARIABLES["CDI"+sigma_str] = VariableSave("Distance from coast sigma="+sigma_str, "CDI"+sigma_str, "m", "m", 2, CDI_SIGMA, cmap=0)
            return self.get_data("CDI"+sigma_str, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["LANDMASK_SIGMA"] :
            #Blurry landmask with a gaussian filter
            sigma = int(kwargs["sigma"]*1000)
            sigma_str = str(sigma)
            if not varname+sigma_str in self.VARIABLES : 
                LANDMASK = self.get_data("LANDMASK")
                DX = self.get_data("DX")
                LANDMASK_SIGMA = manage_images.get_LANDMASK_SIGMA(LANDMASK, DX=DX, sigma=sigma)
                cmap = self.get_cmap("LANDMASK")
                self.VARIABLES["LANDMASK_SIGMA"+sigma_str] = VariableSave("Blurred land mask", "Blurred land mask", "", "", 2, LANDMASK_SIGMA, cmap=cmap)
            return self.get_data(varname+sigma_str, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["BDY_DIST", "BDY_DIST_KM"] :
            km = varname.endswith("_KM")
            X = self.get_data("X"+km*"_KM")
            Y = self.get_data("Y"+km*"_KM")
            BDY_DIST = manage_images.get_BOUNDARY_DISTANCE(X, Y)
            self.VARIABLES[varname] = VariableSave("Distance from domain boundary", "Distance from domain boundary", km*"k"+"m", km*"k"+"m", 2, BDY_DIST, cmap=0)
            return self.get_data(varname, **kwargs) #call again get data, if ever there is a crop, it will be done in VariableSave
        elif varname in ["SBI", "LBI"] :
            new_kwargs = self.copy_kw_get(kwargs)
            crop = kwargs["crop"]
            new_kwargs["crop"] = ("ALL", crop[1], crop[2]) 
            new_kwargs["hinterp"] = None
            Z,WD,COR = self.get_data(["Z","WD","COR"], **new_kwargs)
            zaxis = self.find_axis("z", dim=3, **new_kwargs)
            return self.get_BI(WD, COR, typ=varname[:3], zaxis=zaxis, Z=Z)
        elif varname in ["SBS", "LBS"] :
            crop = kwargs["crop"]
            if crop[0] in ["ALL", [0, self.get_data("NZ")]] :
                new_kwargs = kwargs
            else :
                new_kwargs = self.copy_kw_get(kwargs)
                new_kwargs["crop"] = ("ALL", crop[1], crop[2]) 
            Z,CC_U = self.get_data(["Z","CC_U"], **new_kwargs)
            zaxis = self.find_axis("z", dim=3, **new_kwargs)
            BS = self.get_BS(CC_U, typ=varname[:3], zaxis=zaxis, Z=Z)
            return BS
        elif varname in ["Z_SB", "Z_LB"] :
            crop = kwargs["crop"]
            if crop[0] in ["ALL", [0, self.get_data("NZ")]] :
                new_kwargs = kwargs
            else :
                new_kwargs = self.copy_kw_get(kwargs)
                new_kwargs["crop"] = ("ALL", crop[1], crop[2]) 
            Z = self.get_data("Z", **new_kwargs)
            CC_U = self.get_data("CC_U", **new_kwargs)
            zaxis = self.find_axis("z", dim=3, **new_kwargs)
            return self.get_Z_SB_LB(CC_U, Z, typ=varname[-2:], zaxis=zaxis)
        elif varname in ["SBZC", "LBZC", "CIBLZC", "SIBLZC", "CIBLZC2", "SIBLZC2", "CIBLZC3", "SIBLZC3", "SBZC2", "LBZC2", "SBMH", "LBMH", "SBMHR", "LBMHR"] :
            crop = kwargs["crop"]
            if crop[0] in ["ALL", [0, self.get_data("NZ")]] or (type(crop[0]) is list and crop[0][0] == 0 and crop[0][1] > 10) :
                new_kwargs = kwargs
            else :
                new_kwargs = self.copy_kw_get(kwargs)
                new_kwargs["crop"] = ("ALL", crop[1], crop[2]) 
            Z = self.get_data("Z", **new_kwargs)
            zaxis = self.find_axis("z", dim=3, **new_kwargs)
            if varname in ["SBZC", "LBZC"] :
                AD225_U = self.get_data("AD225_U", **new_kwargs)
                return self.get_Z_SB_LB(AD225_U, Z, typ=varname[:2], zaxis=zaxis)
            elif varname in ["SBZC2", "LBZC2"] :
                CC_U = self.get_data("CC_U", **new_kwargs)
                return self.get_Z_SB_LB(CC_U, Z, typ=varname[:2], zaxis=zaxis)
            elif varname in ["SBMH", "LBMH"] :
                CC_U = self.get_data("CC_U", **new_kwargs)
                return self.get_SBMH(CC_U, Z, typ=varname[:2], zaxis=zaxis)
            elif varname in ["SBMHR", "LBMHR"] :
                CC_U = self.get_data("CC_U", **new_kwargs)
                return self.get_SBMHR(CC_U, Z, typ=varname[:2], zaxis=zaxis)
            elif varname in ["CIBLZC", "SIBLZC"]:
                RI = self.get_data("RI", **new_kwargs)
                return self.get_Z_TIBL(RI, Z, typ=varname[:4], zaxis=zaxis)
            elif varname in ["CIBLZC2", "SIBLZC2"]:
                NBV2 = self.get_data("NBV2", **new_kwargs)
                return self.get_Z_TIBL2(NBV2, Z, typ=varname[:4], zaxis=zaxis)
            elif varname in ["CIBLZC3", "SIBLZC3"]:
                NBV2 = self.get_data("NBV2", **new_kwargs)
                return self.get_Z_TIBL2(NBV2, Z, typ=varname[:4], zaxis=zaxis, threshold=2e-4)
        elif varname in ["CIBLW", "SIBLW"]: #w* TIBL
            ZC, SH_FLX = self.get_data([varname[:4]+"ZC", "SH_FLX"], **kwargs)
            W_STAR = (ZC*constants.BETA*SH_FLX/(1.2*constants.CP))**(1/3) #rho=1.2
            W_STAR[W_STAR >1e2] = 0.
            W_STAR[np.isnan(W_STAR)] = 0.
            return W_STAR
        elif varname == "W_STAR" :
            ZCBL, SH_FLX = self.get_data(["ZCBL", "SH_FLX"], **kwargs)
            W_STAR = (ZCBL*constants.BETA*SH_FLX/(1.2*constants.CP))**(1/3) #rho=1.2
            W_STAR[np.isnan(W_STAR)] = 0
            return W_STAR
        elif varname == "T_STAR" :
            kwargs_sfc = self.copy_kw_get(kwargs)
            if "crop" in kwargs :
                _, cropy, cropx = kwargs["crop"]
                kwargs_sfc["crop"] = (0, cropy, cropx)
            else :
                kwargs_sfc["crop"] = (0, "ALL", "ALL")
            SH_FLX = self.get_data("SH_FLX", **kwargs)
            RHO = self.get_data("RHO", **kwargs_sfc)
            zaxis = self.find_axis("z", dim=3, **kwargs_sfc)
            RHO = np.squeeze(RHO, axis=zaxis)
            U_STAR = self.get_data("U_STAR", **kwargs)
            return -SH_FLX/ (RHO * constants.CP * U_STAR)
        elif varname in ["RI", "RIB", "RIF", "LMO", "LMO_INV", "ZETA_MO", "ZETA_MO_10", "STAB", "STAB_10"] or varname.startswith("SCORER"):
            return self.calculate_stability(varname, **kwargs)
        elif varname == "WD_SB" : 
            return self.calculate_WD_SB()
        elif varname in ["GWD", "GWLAM", "GWS", "GWSM", "GWM2D", "GWM2X1", "GWM2Y1", "GWM2RMAX", "GWM2LMAX","GWM4LAMM", "GWM4SM", "GWM4D",
                        "GWM2R", "GWM2L", "GWM2U", "GWM2V", "GWMASK2", "GWMASK3", "GWMASK4", "GWA", "GWR", "GWR_INST", "DGWRDT", "GWITKE",
                        "GWCTKE", "GWCTKE_RES"]\
            or varname.startswith("GWAVG") or varname.startswith("GWA") or varname.startswith("GWIM2") or varname.startswith("GWCM2")\
            or varname.startswith("GWCSTD") :
            return self.calculate_GW(varname, **kwargs)
        elif varname.startswith("LLJ_") or varname.startswith("LLJ2_"):
            return self.calculate_LLJ(varname, **kwargs)
        elif varname in ["CS"] : #Sound velocity
            TV = self.get_data("TV", **kwargs)
            return np.sqrt(constants.GAMMA * constants.RD * TV)
        elif varname in ["GAMMA"] : #Gamma factor in Gossard and Earl, Waves in the atmosphere, 1975
            RHO, DZ_RHO, CS = self.get_data(["RHO", "DZ_RHO", "CS"], **kwargs)
            return -0.5*constants.G/RHO * DZ_RHO + constants.G/CS**2
        
        elif varname in ["ZCBL", "ZCI"] :
            # CBL height of Capping inversion height, calculated in post.manage_ABL
            if self.FLAGS["df"] :
                df = list(self.output_filenames["df"].values())[0]
                out = df[varname][kwargs["time_slice"]]
                if type(out) in [int, float, np.int64, np.float64]:
                    return out
                elif type(out) in [list, np.array, np.ndarray, pd.core.series.Series] :
                    out = np.expand_dims(np.array(out), axis=(-3, -2, -1)) #(NT, NZ, NY, NX) (will be squeezed later in get_data)
                    return out
                else :
                    raise(Exception(f"Unknow type {type(out)}"))
            else :
                raise(Exception(f"error in Dom.calculate {varname} : please add the online calculation of {varname} or calculate before and save in df (see post.manage_ABL)"))
        
        elif kwargs["avg"] and not varname.endswith("_AVG") :
            if varname[0] != "Q" :
                print(self.prefix, "warning : searching ", varname+"_AVG instead of "+varname)
            new_kwargs = copy.copy(kwargs)
            new_kwargs["save"] = False
            return self.get_data(varname+"_AVG", **new_kwargs)
        elif varname in ["QC", "QR", "QI", "QS", "QH", "QG", "QC_AVG", "QR_AVG", "QI_AVG", "QS_AVG", "QH_AVG", "QG_AVG"]: #Will be called only if varname is not found in files
            return self.get_data("ZERO", **kwargs) #The default value is then 0
        print(self.prefix, "unkown varname in Dom.get_data : ", varname)

    def calculate_statistics(self, varname, **kwargs):
        # Defined again in DomWRF
        print(self.prefix, "unkown varname in Dom.calculate_statistics : ", varname)
    
    def calculate_stress_tensor_bilan_term(self, varname, p="", **kwargs):
        # Defined again in DomWRF
        pass
    
    def calculate_stability(self, varname, **kwargs):
        if varname == "RI" : #Richardson
            num = constants.BETA * self.get_data("DZ_PTV", **kwargs)
            den = self.get_data("DZ_U", **kwargs)**2 + self.get_data("DZ_V", **kwargs)**2
            return num/den
        elif varname == "RIF" : #Flux Richardson : NEED TO BE CHANGED FOR LES
            # IN RANS I don't have <w'.theta'>, I can try with KH, and KM
            print("warning, change definition of RIF for LES")
            KH,KM,RI = self.get_data(["KH","KM","RI"], **kwargs)
            return KH*RI/KM
        elif varname == "RIB" : #Bulk Richardson
            #1- WHAT IS PT0 ?
            #2- I should use average PTV
            raise(Exception("RIB is not written yet in Dom.calculate_stability"))
        elif varname in ["LMO", "LMO_INV"] :
            kwargs_sfc = self.copy_kw_get(kwargs)
            U_STAR,T_STAR = self.get_data(["U_STAR","T_STAR"], **kwargs)
            if varname == "LMO" : 
                return U_STAR**2 / (constants.KARMAN * constants.BETA * T_STAR)
            elif varname == "LMO_INV" : #compute the inverse so that neutral is 0 and not infinity
                return (constants.KARMAN * constants.BETA * T_STAR) / U_STAR**2
        elif varname in ["ZETA_MO", "ZETA_MO_10"] :
            #It can crash because LMO is 2D and Z is 3D
            LMO_INV = self.get_data("LMO_INV", **kwargs)
            Z = self.get_data("Z", **kwargs) if varname == "ZETA_MO" else 10
            return Z * LMO_INV
        elif varname in ["STAB", "STAB_10"] : #From Rodrigo et al., ref from Visich 2022
            ZETA_MO = self.get_data("ZETA_MO"+varname[4:], **kwargs)
            STAB = np.zeros(ZETA_MO.shape)
            STAB[ZETA_MO > 0] += 1
            STAB[ZETA_MO > 0.02] += 1
            STAB[ZETA_MO > 0.2] += 1
            STAB[ZETA_MO > 0.6] += 1
            STAB[ZETA_MO > 2] += 1
            STAB[ZETA_MO < 0] -= 1
            STAB[ZETA_MO < -0.02] -= 1
            STAB[ZETA_MO < -0.2] -= 1
            STAB[ZETA_MO < -0.6] -= 1
            STAB[ZETA_MO < -2] -= 1
            return STAB
        elif varname.startswith("SCORER") :
            #Scorer parameter for gravity waves
            angle = varname[6:]
            NBV2, U, DZ_DZ_U = self.get_data(["NBV2", "AD"+angle+"_U", "DZ_DZ_AD"+angle+"_U"], **kwargs)
            return NBV2/U**2 - DZ_DZ_U/U
     
    def calculate_rotation(self, varname, **kwargs):
        """
        varname = prefix+varname1
        prefix can be :
            CC_ : get the cross-coast component
            ADxx_ : get the component along direction xx in meteo convention (ex. AD225_U = wind component coming from south-west)
            GWM2_ or GWM4_ : get the component across direction GWM2D or GWM4D which is perpendicular to the gravity waves
            ADWD_ : get the component along wind direction
            ADLLJ_ : get the component along LLJ wind direction
            NORM_ : get the norm of the vector (NORM_U = np.sqrt(U**2 + V**2))
            DIR_ : get the direction of the vector in degree in meteo convention (DIR_U = manage_angle.UV2WD_deg(U, V))
        varname 1 can be :
            U, URHO, DXC_xx, M2U, xx1, COVUW, ... : component along the chosen direction
            V, VRHO, DYC_xx, M2V, xx2, COVVW, ... : component perpendicular to the chosen direction (ex. AD0_V = U)
            COVUV : calculate the covariance in the rotated referential frame
            M3XXX : calculate the 3rd order statistics in the rotated referential frame
        """
        ind = 2 if varname[2] == "_" else 3 if varname[3] == "_" else 4 if varname[4] == "_" else 5 if varname[5] == "_" else 6 if varname[6] == "_" else 7
        varname1 = varname[ind+1:]
        prefix = varname[:ind+1]
        angle_rad=0
        if varname.startswith("CC_"):  
            COR = self.get_data("COR", **kwargs)
            angle_rad = np.deg2rad(COR)
            if self.get_dim(varname1) == 3 :
                zaxis = self.find_axis("z", dim=3, **kwargs)
                angle_rad = np.expand_dims(angle_rad, axis=zaxis)
        elif varname.startswith("GWM2_") or varname.startswith("GWM4_"): 
            angle = self.get_data(varname[:4]+"D", **kwargs)
            angle_rad = np.deg2rad(angle)
            # if self.get_dim(varname1) == 3 :
            #     zaxis = self.find_axis("z", dim=3, **kwargs)
            #     angle_rad = np.expand_dims(angle_rad, axis=zaxis)
        elif varname.startswith("ADWD_"):    
            angle_rad = np.deg2rad(self.get_data("WD", **kwargs))
        elif varname.startswith("ADLLJ_"):    
            angle_rad = np.deg2rad(self.get_data("LLJ_WD", **kwargs))
        elif varname.startswith("AD") :    
            angle_deg = int(varname[2:ind])
            angle_rad = np.deg2rad(angle_deg)
            
        if self.is_stress_tensor_bilan_term(varname1) :
            if prefix in ["NORM_", "DIR_"] :
                raise(Exception(f"error : norm and direction of the covariance of {varname1} doesn't make sense"))
            return self.calculate_stress_tensor_bilan_term(varname1, p=prefix, **kwargs)
        
        s = np.sin(angle_rad)
        c = np.cos(angle_rad)
        fac = 1
        if varname1 in ["COVUV", "COVVU", "M2U", "M2V", "COVUW", "COVVW", "COVWU", "COVWV"] or varname1.startswith("COVU") or varname1.startswith("COVV"):
            varname1 = "COVUW" if varname1 == "COVWU" else "COVVW" if varname1 == "COVWV" else varname1
            if varname1.startswith("NORM_") or varname.startswith("DIR_"):
                raise(Exception(f"error : norm and direction of {varname1} doesn't make sense"))
            elif varname1 in ["M2U", "M2V", "COVUV", "COVVU"]:
                M2U, M2V, COVUV = self.get_data(["M2U", "M2V", "COVUV"], **kwargs)
                if varname1 == "M2U" :
                    return M2U*s*s + M2V*c*c + 2*COVUV*c*s
                elif varname1 == "M2V" :
                    return M2U*c*c + M2V*s*s - 2*COVUV*c*s
                else :
                    return -M2U*c*s + M2V*c*s + COVUV*(s*s - c*c)
            elif varname1.startswith("COVU") or varname1.startswith("COVV"):
                COVU, COVV = self.get_data(["COVU"+varname1[4:], "COVV"+varname1[4:]], **kwargs)
                if varname1.startswith("COVU") :
                    return -s*COVU -c*COVV
                elif varname1.startswith("COVV")  :
                    return c*COVU -s*COVV
            else:
                raise(Exception("How did you get here ?"))
        if varname1.startswith("M3") :
            # Note, cannot rotate the M3 with anything else than U, V, W for now
            # Einstein notation
            # Change of referential frame : vi = Ail.ul (sum on l in [1, 2, 3])
            # Then the product of 3 velocity components is : vi.vj.vk = Ail.ul.Ajm.um.Akn.un (sum on (l, m, n) in [1, 2, 3]^3)
            # It can be shown that : <vi'vj'vk'> = Ail.Ajm.Akn.<ul'um'un'> (sum on (l, m, n) in [1, 2, 3]^3)
            # We can do the same for covariances and variances and we should find the same result
            vi = {"U":0, "V":1, "W":2}
            iv = ["U", "V", "W"]
            # rotation matrix
            A = np.array([[-s, -c, 0],
                          [ c, -s, 0],
                          [ 0,  0, 1]])
            temp = [varname1[2], varname1[3], varname1[4]]
            temp.sort()
            v1, v2, v3 = temp
            i, j, k = vi[v1], vi[v2], vi[v3]
            out = 0
            # There might exist a matrix operator for this but I don't know it
            for l in range(3): 
                for m in range(3):
                    for n in range(3):
                        fac = A[i,l] * A[j,m] * A[k,n]
                        if np.abs(fac) > 1e-5 : # out of the 27 terms, we skip those with a coefficient of 0 (e.g. for <w'^3> there is only one term)
                            temp = [iv[l], iv[m], iv[n]]
                            temp.sort()
                            u1, u2, u3 = temp
                            out += fac*self.get_data("M3"+u1+u2+u3, **kwargs)
            return out
        else :
            if varname1[-1] == "1" :
                varname2 = varname1[:-1] + "2"
                fac=-1
            elif varname1[-1] == "2" :
                varname2 = varname1[:-1] + "1"
            elif varname1[0] == "U" :
                varname2 = "V" + varname1[1:]
                fac=-1
            elif varname1[0] == "V" :
                varname2 = "U" + varname1[1:]
            elif varname1[:4] in ["COVU", "DZ_U"] :
                varname2 = varname1[:3] + "V" + varname1[4:]
                fac=-1
            elif varname1[:4] in ["COVV", "DZ_V"] :
                varname2 = varname1[:3] + "U" + varname1[4:]
            elif varname1[:4] in ["DXC_", "DXW_"] :
                varname2 = "DY" + varname1[2] + "_" + varname1[4:]
                fac=-1
            elif varname1[:4] in ["DYC_", "DYW_"] :
                varname2 = "DX" + varname1[2] + "_" + varname1[4:]
            else :
                print(f"warning in Dom.calculate rotation, unknown varname1 = {varname1}, cannot apply {prefix}, returning the original data")
                return self.get_data(varname1, **kwargs)
            var1 = self.get_data(varname1, **kwargs)
            var2 = self.get_data(varname2, **kwargs)

            if varname.startswith("NORM_"):
                return np.sqrt(var1**2 + var2**2)
            elif varname.startswith("DIR_"):    
                if fac == -1 : #var1 is X and var2 is Y
                    return manage_angle.UV2WD_deg(var1, var2)
                else : #var1 is Y and var2 is X
                    return manage_angle.UV2WD_deg(var2, var1)
            else :
                #x_r = -s.x - c.y
                #y_r =  c.x - s.y
                # with (x_r, y_r) the rotated referential frame coordinates, (x, y) the classical referential frame, and s=sin(alpha), c=cos(alpha)
                # alpha is the direction of x_r in meteorological convention, so that an angle of 270° gives x=x_r, y=y_r
                return -s*var1 + fac*c*var2
            
    def calculate_derivative(self, varname, **kwargs):
        """ Calculate the spatial derivative of a value
            Note: for inherited class (like DomARPS, DomWRF, ...) the inherited method (DomWRF.calculate, ...) is called first and 
                   if the variable isn't defined in the inherited method, then Dom.calculate is called
            Warning: the computation is very slow and could be improved
        varname (str): name of the variable 
        kwargs: all the kwargs from get_data
        27/02/2023 : Mathieu LANDREAU
        28/06/2023 : ML - add derivatives for non-WRF domains
        """ 
        #Time derivative in domains referential frame (t, x, y, ?)
        # Valid for all
        if varname.startswith("DTW_") :
            varname2 = varname[4:]
            time_slice = kwargs["time_slice"]
            it_list = np.arange(5000, dtype="int")[time_slice]
            squeeze = False
            if type(it_list) in [int, np.int64] :
                it_list = np.array([it_list])
                squeeze = True
            if len(it_list) > 1 and np.any(np.diff(it_list) != it_list[1] - it_list[0]) :
                raise(Exception("error in DTW, needs a constant time_step, time_slice =" + str(time_slice) + ", varname = " + varname))
            it1 = it_list[0]
            it2 = it_list[-1]
            dit = it_list[1] - it_list[0] if it2 > it1 else 1
            if it1 < dit or kwargs["quick_deriv"]:
                adjust_first = True
                new_it1 = it1
            else :
                adjust_first = False
                new_it1 = it1 - dit
            if it2 >= self.get_data("NT_HIST")-dit or kwargs["quick_deriv"] :
                adjust_last = True
                new_it2 = it2
            else :
                adjust_last = False
                new_it2 = it2 + dit
            
            new_kwargs = self.copy_kw_get(kwargs)
            new_time_slice = slice(new_it1, new_it2+1, dit)
            if debug : print(self.prefix, "old time_slice : ", time_slice)
            if debug : print(self.prefix, "new_time_slice : ", new_time_slice)
            new_kwargs["time_slice"] = new_time_slice
            if adjust_first and adjust_last :
                new_kwargs["saved"] = kwargs["saved"]
            TIME = self.get_data("TIME", **new_kwargs)
            DELTA = manage_time.timedelta_to_seconds(TIME - TIME[0])
            var = self.get_data(varname2, **new_kwargs)
            ndim = var.ndim
            taxis = 0
            if DELTA.ndim != ndim :
                idim = 0
                while idim < ndim :
                    len_idim = var.shape[idim]
                    if idim == DELTA.ndim or DELTA.shape[idim] != len_idim :
                        #X = np.tensordot(np.ones(len_idim), X, 0)
                        DELTA = np.expand_dims(DELTA, axis=idim)
                        DELTA = np.concatenate((DELTA,)*len_idim, axis=idim)
                    idim = idim + 1
            if debug : print(self.prefix, "new_time_slice : ", new_time_slice)
            if debug : print(self.prefix, "compare DELTA shape : ", DELTA.shape, var.shape, varname)
            if debug : print(self.prefix, "taxis : ", taxis)
            DELTA_diff = np.diff(DELTA, axis=taxis)
            var_diff = np.diff(var, axis=taxis)
            e1 = np.delete(DELTA_diff, -1, axis=taxis)
            e2 = np.delete(DELTA_diff, 0, axis=taxis)
            f1 = np.delete(var_diff, -1, axis=taxis)
            f2 = np.delete(var_diff, 0, axis=taxis)
            den = e1*e2*(e1 + e2)
            num = e2*e2*f1 + e1*e1*f2
            data = num/den
            if adjust_first :
                NT_temp = DELTA_diff.shape[taxis]
                DELTA_diff1 = np.delete(DELTA_diff, range(1,NT_temp), axis=taxis)
                var_diff1 = np.delete(var_diff, range(1,NT_temp), axis=taxis)
                temp = (var_diff1/DELTA_diff1)
                temp = temp.squeeze(axis=taxis)
                data = np.insert(data, 0, temp, axis=taxis)
            if adjust_last :
                NT_temp = DELTA_diff.shape[taxis]
                DELTA_diffm1 = np.delete(DELTA_diff, range(NT_temp-1), axis=taxis)
                var_diffm1 = np.delete(var_diff, range(NT_temp-1), axis=taxis)
                temp = (var_diffm1/DELTA_diffm1)
                data = np.append(data, temp, axis=taxis)
            if squeeze :
                data = np.squeeze(data, axis=taxis)
            return data
        
        #Z derivative in cartesian referential frame (t, x, y, z)
        # Valid for AROME, maybe ARPS and WRF
        elif varname.startswith("DZ_") :
            varname2 = varname[3:]
            cropz, cropy, cropx = kwargs["crop"]
            cropz1, cropz2 = cropz
            if cropz1 == 0  or kwargs["quick_deriv"] :
                adjust_first = True
                new_cropz1 = cropz1
            else :
                adjust_first = False
                new_cropz1 = cropz1 - 1
            if cropz2 == self.get_data("NZ") or kwargs["quick_deriv"] :
                adjust_last = True
                new_cropz2 = cropz2
            else :
                adjust_last = False
                new_cropz2 = cropz2 + 1
            new_kwargs = self.copy_kw_get(kwargs)
            new_crop = ([new_cropz1, new_cropz2], cropy, cropx)
            if debug : print(self.prefix, "new_crop : ", new_crop)
            new_kwargs["crop"] = new_crop
            if adjust_first and adjust_last :
                new_kwargs["saved"] = kwargs["saved"]
            ZP = self.get_data("ZP", **new_kwargs)
            var = self.get_data(varname2, **new_kwargs)
            ndim = var.ndim
            count = 0
            zaxis = self.find_axis("z", varname=varname2, **new_kwargs)
            # adjusting ZP shape to var shape
            if ZP.ndim != ndim :
                idim = 0
                while idim < ndim :
                    len_idim = var.shape[idim]
                    if idim == ZP.ndim or ZP.shape[idim] != len_idim :
                        ZP = np.expand_dims(ZP, axis=idim)
                        ZP = np.concatenate((ZP,)*len_idim, axis=idim)
                    idim = idim + 1
                    
            if debug : print(self.prefix, "zaxis : ", zaxis)
            if debug : print(self.prefix, varname2)
            if debug : print(self.prefix, "compare ZP shape : ", ZP.shape, var.shape)
            ZP_diff = np.diff(ZP, axis=zaxis)
            var_diff = np.diff(var, axis=zaxis)
            e1 = np.delete(ZP_diff, -1, axis=zaxis)
            e2 = np.delete(ZP_diff, 0, axis=zaxis)
            f1 = np.delete(var_diff, -1, axis=zaxis)
            f2 = np.delete(var_diff, 0, axis=zaxis)
            data = (e2*e2*f1 + e1*e1*f2) / (e1*e2*(e1 + e2))
            if debug : print(self.prefix, "data.shape : ", data.shape)
            if adjust_first :
                NZ_temp = ZP_diff.shape[zaxis]
                temp = np.expand_dims( np.take(var_diff, 0, axis=zaxis)/np.take(ZP_diff, 0, axis=zaxis), axis=zaxis )
                data = np.concatenate((temp, data), axis=zaxis)
            if adjust_last :
                NZ_temp = ZP_diff.shape[zaxis]
                temp = np.expand_dims( np.take(var_diff, NZ_temp-1, axis=zaxis)/np.take(ZP_diff, NZ_temp-1, axis=zaxis), axis=zaxis )
                data = np.append(data, temp, axis=zaxis)
            #data = np.squeeze(data)
            if debug : print(self.prefix, "data.shape : ", data.shape)
            return data
          
        #X derivative in domains referential frame (t, x, y, ?)
        # Valid for all
        elif varname.startswith("DXW_") :
            varname2 = varname[4:]
            cropz, cropy, cropx = kwargs["crop"]
            if type(cropx) in [int, np.int64] :
                cropx1 = cropx
                cropx2 = cropx+1
            else :
                cropx1, cropx2 = cropx
                
            if cropx1 == 0 or kwargs["quick_deriv"] :
                adjust_first = True
                new_cropx1 = cropx1
            else :
                adjust_first = False
                new_cropx1 = cropx1 - 1
            if cropx2 == self.get_data("NX") or kwargs["quick_deriv"] :
                adjust_last = True
                new_cropx2 = cropx2
            else :
                adjust_last = False
                new_cropx2 = cropx2 + 1
            new_kwargs = self.copy_kw_get(kwargs)
            new_crop = (cropz, cropy, [new_cropx1, new_cropx2])
            if debug : print(self.prefix, "new_crop : ", new_crop)
            new_kwargs["crop"] = new_crop
            if adjust_first and adjust_last :
                new_kwargs["saved"] = kwargs["saved"]
            X = self.get_data("X", **new_kwargs)
            var = self.get_data(varname2, **new_kwargs)
            ndim = var.ndim
            xaxis = self.find_axis('x', varname=varname2, **new_kwargs)
            if X.ndim != ndim :
                idim = 0
                while idim < ndim :
                    len_idim = var.shape[idim]
                    if idim == X.ndim or X.shape[idim] != len_idim :
                        #X = np.tensordot(np.ones(len_idim), X, 0)
                        X = np.expand_dims(X, axis=idim)
                        X = np.concatenate((X,)*len_idim, axis=idim)
                    idim = idim + 1
            if debug : print(self.prefix, "new_kwargs[crop] : ", new_kwargs["crop"])
            if debug : print(self.prefix, "compare X shape : ", X.shape, var.shape, varname)
            if debug : print(self.prefix, "xaxis : ", xaxis)
            X_diff = np.diff(X, axis=xaxis)
            var_diff = np.diff(var, axis=xaxis)
            e1 = np.delete(X_diff, -1, axis=xaxis)
            e2 = np.delete(X_diff, 0, axis=xaxis)
            f1 = np.delete(var_diff, -1, axis=xaxis)
            f2 = np.delete(var_diff, 0, axis=xaxis)
            den = e1*e2*(e1 + e2)
            num = e2*e2*f1 + e1*e1*f2
            data = num/den
            if adjust_first :
                NX_temp = X_diff.shape[xaxis]
                X_diff1 = np.delete(X_diff, range(1,NX_temp), axis=xaxis)
                var_diff1 = np.delete(var_diff, range(1,NX_temp), axis=xaxis)
                temp = (var_diff1/X_diff1)
                temp = temp.squeeze(axis=xaxis)
                data = np.insert(data, 0, temp, axis=xaxis)
            if adjust_last :
                NX_temp = X_diff.shape[xaxis]
                X_diffm1 = np.delete(X_diff, range(NX_temp-1), axis=xaxis)
                var_diffm1 = np.delete(var_diff, range(NX_temp-1), axis=xaxis)
                temp = (var_diffm1/X_diffm1)
                data = np.append(data, temp, axis=xaxis)
            return data
        
        #Y derivative in domain referential frame (t, x, y, ?)
        # Valid for All
        elif varname.startswith("DYW_") :
            varname2 = varname[4:]
            cropz, cropy, cropx = kwargs["crop"]
            if type(cropy) in [int, np.int64] :
                cropy1 = cropy
                cropy2 = cropy+1
            else :
                cropy1, cropy2 = cropy
            if cropy1 == 0 or kwargs["quick_deriv"] :
                adjust_first = True
                new_cropy1 = cropy1
            else :
                adjust_first = False
                new_cropy1 = cropy1 - 1
            if cropy2 == self.get_data("NY") or kwargs["quick_deriv"] :
                adjust_last = True
                new_cropy2 = cropy2
            else :
                adjust_last = False
                new_cropy2 = cropy2 + 1
            new_kwargs = self.copy_kw_get(kwargs)
            new_crop = (cropz, [new_cropy1, new_cropy2], cropx)
            if debug : print(self.prefix, "new_crop : ", new_crop)
            new_kwargs["crop"] = new_crop
            if adjust_first and adjust_last :
                new_kwargs["saved"] = kwargs["saved"]
            Y = self.get_data("Y", **new_kwargs)
            var = self.get_data(varname2, **new_kwargs)
            ndim = var.ndim
            yaxis = self.find_axis('y', varname=varname2, **new_kwargs)
            if Y.ndim != ndim :
                idim = 0
                while idim < ndim :
                    len_idim = var.shape[idim]
                    if idim == Y.ndim or Y.shape[idim] != len_idim :
                        #X = np.tensordot(np.ones(len_idim), X, 0)
                        Y = np.expand_dims(Y, axis=idim)
                        Y = np.concatenate((Y,)*len_idim, axis=idim)
                    idim = idim + 1
            if debug : print(self.prefix, "compare Y shape : ", Y.shape, var.shape)
            if debug : print(self.prefix, "yaxis : ", yaxis)
            Y_diff = np.diff(Y, axis=yaxis)
            var_diff = np.diff(var, axis=yaxis)
            e1 = np.delete(Y_diff, -1, axis=yaxis)
            e2 = np.delete(Y_diff, 0, axis=yaxis)
            f1 = np.delete(var_diff, -1, axis=yaxis)
            f2 = np.delete(var_diff, 0, axis=yaxis)
            den = e1*e2*(e1 + e2)
            num = e2*e2*f1 + e1*e1*f2
            data = num/den
            if adjust_first :
                NY_temp = Y_diff.shape[yaxis]
                Y_diff1 = np.delete(Y_diff, range(1,NY_temp), axis=yaxis)
                var_diff1 = np.delete(var_diff, range(1,NY_temp), axis=yaxis)
                temp = (var_diff1/Y_diff1)
                temp = temp.squeeze(axis=yaxis)
                data = np.insert(data, 0, temp, axis=yaxis)
            if adjust_last :
                NY_temp = Y_diff.shape[yaxis]
                Y_diffm1 = np.delete(Y_diff, range(NY_temp-1), axis=yaxis)
                var_diffm1 = np.delete(var_diff, range(NY_temp-1), axis=yaxis)
                temp = (var_diffm1/Y_diffm1)
                data = np.append(data, temp, axis=yaxis)
            return data  
        
        #TIME derivative in cartesian referential frame (t, x, y, z)
        # Valid for AROME (maybe ARPS), redefined in WRF
        elif varname.startswith("DTC_") : 
            varname2 = varname[4:]
            DTW_var = self.get_data("DTW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DTW_var
            else : 
                DTW_ZP = self.get_data("DTW_ZP", **kwargs)
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DTC", DTW_var.shape, DTW_ZP.shape, DZ_var.shape)
                return DTW_var - DTW_ZP*DZ_var
            
        # LAGRANGIAN TIME derivative
        elif varname.startswith("DTL_") : 
            varname2 = varname[4:]
            DTC_var = self.get_data("DTC_"+varname2, **kwargs)
            U, V = self.get_data(["U", "V"], **kwargs)
            DXC_var, DYC_var = self.get_data(["DXC_"+varname2, "DYC_"+varname2], **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DTC_var + U*DXC_var + V*DYC_var
            else : 
                W, DZ_var = self.get_data(["W", "DZ_"+varname2], **kwargs)
                return DTC_var + U*DXC_var + V*DYC_var + W*DZ_var
            
        # Stokes TIME derivative
        # Same as Lagrangian derivatives but velocities are spatially averaged to filter gravity waves
        elif varname.startswith("DTS_") : 
            varname2 = varname[4:]
            DTC_var = self.get_data("DTC_"+varname2, **kwargs)
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["DX_smooth"] = 5 #arbitrary chosen to filter gravity waves
            #interdépendance pour gagner de l'espace et du temps, je ne suis pas sûr que ça marche
            if not "stokes" in kwargs["saved"]:
                kwargs["saved"]["stokes"] = {}
            new_kwargs["saved"] = kwargs["saved"]["stokes"]
            if not "before_smooth" in new_kwargs["saved"] :
                new_kwargs["saved"]["before_smooth"] = kwargs["saved"]
            U, V = self.get_data(["U", "V"], **new_kwargs)
            DXC_var, DYC_var = self.get_data(["DXC_"+varname2, "DYC_"+varname2], **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DTC_var + U*DXC_var + V*DYC_var
            else : 
                W = self.get_data("W", **new_kwargs)
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                return DTC_var + U*DXC_var + V*DYC_var + W*DZ_var
            
        #Y derivative in cartesian referential frame (t, x, y, z)
        # Valid for AROME (maybe ARPS), redefined in WRF
        elif varname.startswith("DYC_") : 
            varname2 = varname[4:]
            DYW_var = self.get_data("DYW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DYW_var
            else : 
                DYW_ZP = self.get_data("DYW_ZP", **kwargs)
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DYC", DYW_var.shape, DYW_ZP.shape, DZ_var.shape,)
                return DYW_var - DYW_ZP*DZ_var
            
        
        #X derivative in cartesian referential frame (t, x, y, z)
        # Valid for AROME (maybe ARPS), redefined in WRF
        elif varname.startswith("DXC_") :
            varname2 = varname[4:]
            DXW_var = self.get_data("DXW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DXW_var
            else : 
                DXW_ZP = self.get_data("DXW_ZP", **kwargs)
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DXC", DXW_var.shape, DXW_ZP.shape, DZ_var.shape)
                return DXW_var - DXW_ZP*DZ_var
        
        # Cross-coast derivative in cartesian referential frame (t, x, y, z)
        # Valid for All
        elif varname.startswith("DCC_") :
            return self.get_data("CC_DXC_"+varname[4:], **kwargs)
        
        # Horizontal divergence in cartesian referential frame (t, x, y, z)
        # Valid for All
        elif varname.startswith("DIV_") :
            varname2 = varname[4:]
            if varname2.startswith("U") :
                Uname = varname2
                Vname = "V"+varname2[1:]
            else :
                raise(Exception("cannot compute div from this variable in Dom.calculate_derivative"))
            return self.get_data("DXC_"+Uname, **kwargs) + self.get_data("DYC_"+Vname, **kwargs)
        
        # Horizontal 2D curl in cartesian referential frame (t, x, y, z)
        # Valid for All
        elif varname.startswith("ROT_") :
            varname2 = varname[4:]
            if varname2.startswith("U") :
                Uname = varname2
                Vname = "V"+varname2[1:]
            else :
                raise(Exception("cannot compute rot from this variable in Dom.calculate_derivative"))
            return self.get_data("DXC_"+Vname, **kwargs) - self.get_data("DYC_"+Uname, **kwargs)
        
        #norm of the gradient of the velocity vector
        # Valid for All
        elif varname == "GRAD2U" :
            DXC_U = self.get_data("DXC_U", **kwargs)
            DYC_U = self.get_data("DYC_U", **kwargs)
            DXC_V = self.get_data("DXC_V", **kwargs)
            DYC_V = self.get_data("DYC_V", **kwargs)
            return np.sqrt(DXC_U**2 + DYC_U**2 + DXC_V**2 + DYC_V**2)
        
        #like GRAD50_U, GRAD220_PT : component GRAD(var) with an angle of angle_deg ° from North
        # if no angle is given (like "GRAD_P"), the norm of the gradient is returned
        # Valid for All
        elif varname.startswith("GRAD") :
            if varname[4] == "_" : #ABSOLUTE VALUE
                varname2 = varname[5:]
                DXC = self.get_data("DXC_"+varname2, **kwargs)
                DYC = self.get_data("DYC_"+varname2, **kwargs)
                return np.sqrt(DXC**2 + DYC**2)
            elif varname[5] == "_" :
                angle_deg = int(varname[4:5])
                varname2 = varname[6:]
            elif varname[6] == "_" :
                angle_deg = int(varname[4:6])
                varname2 = varname[7:]
            elif varname[7] == "_" :
                angle_deg = int(varname[4:7])
                varname2 = varname[8:]
            else : 
                raise(Exception("unknown angle for grad in Dom.calculate_derivative : ", varname))
            DXC = self.get_data("DXC_"+varname2, **kwargs)
            DYC = self.get_data("DYC_"+varname2, **kwargs)
            angle_rad = np.deg2rad(angle_deg)
            return -DXC*np.sin(angle_rad) - DYC*np.cos(angle_rad)
        
        elif varname.startswith("WDGRAD_") :
            varname2 = varname[7:]
            DXC = self.get_data("DXC_"+varname2, **kwargs)
            DYC = self.get_data("DYC_"+varname2, **kwargs)
            return manage_angle.UV2WD_deg(DXC, DYC)
        
        elif varname == "SHEAR" :
            return np.sqrt( self.get_data("DZ_U", **kwargs)**2 + self.get_data("DZ_V", **kwargs)**2 )

#################################################################################################################################
######  POST_PROCESS_SEA_BREEZES
#################################################################################################################################
    def get_BI(self, WD, phi, typ="SBI", zaxis=0, Z=None, ZPBL=3000) :
        """
        Description
            Calculate SB index or LB index from Hallgren, Christoffer, Heiner Körnich, Stefan Ivanell, et Erik Sahlée. 2023. 
            « A Single-Column Method to Identify Sea and Land Breezes in Mesoscale-Resolving NWP Models ». Weather and Forecasting 38 (6): 1025‑39. 
            https://doi.org/10.1175/WAF-D-22-0163.1.
        Parameters
            WD : np.array of N dimension : the Wind direction array
            phi : the coastline angle (COR = COAST_ORIENT)
        Optional
            typ : str : "SBI" or "LBI" for land or sea breeze index
            zaxis : if WD is more than 1D, we need to know which axis correspond to "z"
            Z : np.array, same dimension as WD : if Z is present, only the points below ZPBL are used
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 3000m
        Returns 
            BI : np.array of N-1 dimension : the z dimension is squeezed, there is a single value of SBI or LBI per column
        """
        alpha = np.expand_dims(np.take(WD, 0, zaxis), axis=zaxis)
        phi = np.expand_dims(phi, axis=zaxis)
        beta = np.copy(WD)
        if Z is not None :
            beta[Z>ZPBL] = np.nan
        else :
            print(self.prefix, "warning in get_SBI : no height vector, assuming all heights are in the PBL")
        angle1 = manage_angle.anglerad(alpha - phi)
        angle2 = manage_angle.anglerad(beta - phi)
        angle3 = manage_angle.anglerad(alpha - beta)
        #default is SBI
        s1 = np.logical_or(angle1 < 0.5*np.pi, angle1 > 1.5*np.pi) #10m wind is going from sea to land
        s2 = np.logical_and(angle2 > 0.5*np.pi, angle2 < 1.5*np.pi) #higher wind is going from land to sea
        s3 = np.logical_and(angle3 > 0.5*np.pi, np.abs(angle3) < 1.5*np.pi) #higher wind is opposed to 10m wind
        BI = np.cos(angle1)*np.cos(angle3+np.pi)
        #LBI
        if typ == "LBI" : 
            s1 = np.logical_not(s1)
            s2 = np.logical_not(s2)
            BI = -BI
        BI[np.logical_not(np.logical_and(np.logical_and(s1, s2), s3))] = 0
        BI = np.nanmax(BI, axis=zaxis)
        return BI
    
    def get_BS(self, CC_U_in, typ="SBS", zaxis=0, Z=None, ZPBL=2000) :
        """
        Description
            Calculate SB strength (self made formula) 
        Parameters
            CC_U : np.array of N dimension : the cross-coast velocity array
        Optional
            typ : str : "SBS" or "LBS" for land or sea breeze strength
            zaxis : if CC_U is more than 1D, we need to know which axis correspond to "z"
            Z : np.array, same dimension as CC_U : if Z is present, only the points below ZPBL are used
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 3000m
        Returns 
            BS : np.array of N-1 dimension : the z dimension is squeezed, there is a single value of SBS or LBS per column
        """
        CC_U = np.copy(CC_U_in)
        if Z is not None :
            notPBL = Z > ZPBL
        else :
            print(self.prefix, "warning in get_SBS : no height vector, assuming all heights are in the PBL")
            notPBL = False
        CC_U[notPBL] = 0
        a = np.nanmax(CC_U, axis=zaxis)
        b = np.nanmin(CC_U, axis=zaxis)
        c = np.nanargmax(CC_U, axis=zaxis)
        d = np.nanargmin(CC_U, axis=zaxis)
        CC_U0 = np.take(CC_U, 0, zaxis)
        if typ == "LBS" :
            mask = np.logical_and(c>d, CC_U0<0)
        else : #default is SBS
            mask = np.logical_and(c<d, CC_U0>0)
        mask = np.logical_and(np.logical_and(a>0, b<0), mask)
        BS = np.min(np.abs(np.array([a, b])), axis=0)
        BS[np.logical_not(mask)] = 0
        BS = np.expand_dims(BS, axis=zaxis)
        return BS
    
    def get_Z_SB_LB(self, CC_U, Z, typ="SB", zaxis=0, ZPBL=2000) :
        """
        Description
            Calculate SB height from cross-coast velocity profile (height at which CC_U=0) 
        Parameters
            CC_U : np.array of N dimension : the cross-coast velocity array
            Z : np.array of N dimension : Height above ground in meters
        Optional
            typ : str : "SB" or "LB" for land or sea breeze height
            zaxis : if CC_U is more than 1D, we need to know which axis correspond to "z"
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 2000m
        Returns 
            Z_SB : np.array of N dimension (length = 1 for zaxis)
        """
        SBS = self.get_BS(CC_U, typ=typ+"S", zaxis=zaxis, Z=Z, ZPBL=ZPBL)
        Z_SB = np.nanargmin(CC_U>0, axis=zaxis)
        Z_SB = np.expand_dims(Z_SB, axis=zaxis)
        Z_SB0 = np.take_along_axis(Z, Z_SB-1, axis=zaxis)
        Z_SB1 = np.take_along_axis(Z, Z_SB, axis=zaxis)
        CC_U0 = np.take_along_axis(CC_U, Z_SB-1, axis=zaxis)
        CC_U1 = np.take_along_axis(CC_U, Z_SB, axis=zaxis)
        pos = Z_SB != 0
        Z_SB[pos] = -CC_U1[pos] * ((Z_SB0[pos] - Z_SB1[pos])/(CC_U0[pos] - CC_U1[pos])) + Z_SB1[pos]
        Z_SB[np.logical_not(pos)] = Z_SB1[np.logical_not(pos)]
        # Z_SB = np.squeeze(Z_SB, axis=zaxis)
        Z_SB[np.logical_or(np.isnan(SBS), SBS<1e-5)] = .0
        return Z_SB
    
    def get_SBMH(self, CC_U_in, Z, typ="SB", zaxis=0, ZPBL=2000) :
        """
        Description
            Calculate SB maximal velocity in the gravity current
        Parameters
            CC_U : np.array of N dimension : the cross-coast velocity array
            Z : np.array of N dimension : Height above ground in meters
        Optional
            typ : str : "SB" or "LB" for land or sea breeze height
            zaxis : if CC_U is more than 1D, we need to know which axis correspond to "z"
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 2000m
        Returns 
            SBMH : np.array of N dimension (length = 1 for zaxis)
        """
        CC_U = np.copy(CC_U_in)
        Z_SB = self.get_Z_SB_LB(CC_U, Z, typ=typ, zaxis=zaxis, ZPBL=ZPBL)
        notSBG = np.where(Z > Z_SB-1e-5)
        CC_U[notSBG] = 0
        SBMH = -np.nanmin(CC_U, axis=zaxis) if typ == "LB" else np.nanmax(CC_U, axis=zaxis)
        return np.expand_dims(SBMH-1e-5, axis=zaxis)
    
    def get_SBMHR(self, CC_U_in, Z, typ="SB", zaxis=0, ZPBL=2000) :
        """
        Description
            Calculate SB maximal velocity in the return current
        Parameters
            CC_U : np.array of N dimension : the cross-coast velocity array
            Z : np.array of N dimension : Height above ground in meters
        Optional
            typ : str : "SB" or "LB" for land or sea breeze height
            zaxis : if CC_U is more than 1D, we need to know which axis correspond to "z"
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 2000m
        Returns 
            SBMHR : np.array of N dimension (length = 1 for zaxis)
        """
        CC_U = np.copy(CC_U_in)
        Z_SB = self.get_Z_SB_LB(CC_U, Z, typ=typ, zaxis=zaxis, ZPBL=ZPBL)
        notPBL = np.where(Z > ZPBL)
        CC_U[notPBL] = 0
        notSB = np.where(Z*Z_SB < 1)
        CC_U[notSB] = 0
        SBMHR = np.nanmax(CC_U, axis=zaxis) if typ == "LB" else -np.nanmin(CC_U, axis=zaxis)
        return np.expand_dims(SBMHR, axis=zaxis)
    
    def get_Z_TIBL(self, RI, Z, typ="CIBL", zaxis=0, ZPBL=2000) :
        """
        Description
            Calculate TIBL height from Richardson profile (height at which Ri=0) 
        Parameters
            RI : np.array of N dimension : the Richardson profile
            Z : np.array of N dimension : Height above ground in meters
        Optional
            typ : str : "CIBL" or "SIBL" for convective or stable TIBL height, respectively
            zaxis : if RI is more than 1D, we need to know which axis correspond to "z"
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 2000m
        Returns 
            TIBLZ : np.array of N dimension
        """
        if typ == "SIBL" :
            TIBLIZ = np.nanargmax(RI<0, axis=zaxis)
        else :
            TIBLIZ = np.nanargmax(RI>0, axis=zaxis)
        TIBLIZ = np.expand_dims(TIBLIZ, axis=zaxis)
        TIBLZ0 = np.take_along_axis(Z, TIBLIZ-1, axis=zaxis)
        TIBLZ1 = np.take_along_axis(Z, TIBLIZ, axis=zaxis)
        RI0 = np.take_along_axis(RI, TIBLIZ-1, axis=zaxis)
        RI1 = np.take_along_axis(RI, TIBLIZ, axis=zaxis)
        TIBLZ = -RI1 * ((TIBLZ0 - TIBLZ1)/(RI0 - RI1)) + TIBLZ1
        # Z_SB = np.squeeze(Z_SB, axis=zaxis)
        TIBLZ[TIBLIZ==0] = 0
        return TIBLZ
    
    def get_Z_TIBL2(self, NBV2, Z, typ="CIBL", zaxis=0, ZPBL=2000, threshold=1e-4) :
        """
        Description
            Calculate TIBL height from NBV2 profile (height at which NBV2=0) 
        Parameters
            RI : np.array of N dimension : the Richardson profile
            Z : np.array of N dimension : Height above ground in meters
        Optional
            typ : str : "CIBL" or "SIBL" for convective or stable TIBL height, respectively
            zaxis : if RI is more than 1D, we need to know which axis correspond to "z"
            ZPBL : float, int or np.array, same dimension as WD and Z : if Z is present, only the points below ZPBL are used, default : 2000m
        Returns 
            TIBLZ : np.array of N dimension
        """
        if typ == "SIBL" :
            TIBLIZ = np.nanargmax(NBV2<threshold, axis=zaxis)
        else :
            TIBLIZ = np.nanargmax(NBV2>threshold, axis=zaxis)
        TIBLIZ = np.expand_dims(TIBLIZ, axis=zaxis)
        TIBLZ0 = np.take_along_axis(Z, TIBLIZ-1, axis=zaxis)
        TIBLZ1 = np.take_along_axis(Z, TIBLIZ, axis=zaxis)
        N0 = np.take_along_axis(NBV2, TIBLIZ-1, axis=zaxis)
        N1 = np.take_along_axis(NBV2, TIBLIZ, axis=zaxis)
        a = (TIBLZ0 - TIBLZ1)/(N0 - N1)
        b = TIBLZ1 - N1 * a
        TIBLZ = a*threshold + b
        TIBLZ[TIBLIZ==0] = 0
        return TIBLZ
    
    def detect_SB1(self, dic={}) :
        print("detect SB1")
        
        if not "test1" in dic :
            Z = self.get_data("Z", itime=0)
            cropz = self.get_cropz_for_interp([1, 1499], 1500, 0, Z)
            crop = (cropz, "ALL", "ALL")
            itime = "ALL_TIMES"
            kw_get = {
                "crop" : crop, 
                "itime" : itime,
                "saved" : {}
            }

            COR = self.get_data("COR")

            print("read U")
            U = self.get_data("U", **kw_get)
            print("read V")
            V = self.get_data("V", **kw_get)
            NT, NZ, NY, NX = U.shape

            print("process")
            angle_rad = 1.5*np.pi - COR + 1*np.pi
            U_AD = -U*np.sin(angle_rad) - V*np.cos(angle_rad)
            m = np.ma.masked_array(U_AD, np.isnan(U_AD))
            a = np.nanmax(m, axis=1)
            b = np.nanmin(m, axis=1)
            c = np.nanargmax(m, axis=1)
            d = np.nanargmin(m, axis=1)
            test1 = np.logical_and(a > 2, b < -2) 
            test_SB = np.logical_and(test1, c < d) #SB
            #test_LB = np.logical_and(np.logical_not(test1), c < d)#LB
            dic["test1"] = test_SB
        else :
            test1 = dic[test1]
            NT, NY, NX = test1.shape
        print("write postproc")
        for itime in range(NT):
            self.write_postproc("SB1",    test_SB[itime],    ('y', 'x'), itime=itime, long_name="SB1 mask on coast",           standard_name="SB1",    units=" ",     latex_units=" ", typ="i1")

    def calculate_WD_SB(self) :
        cropz = self.get_cropz_for_interp([1, 1499], 1500, 0, self.get_data("ZP", crop = ("ALL","ALL","ALL"), itime=0))
        kw_get = {
            "itime" : "ALL_TIMES",
            "crop" : (cropz, "ALL", "ALL"),
        }
        U = self.get_data("U", **kw_get)
        V = self.get_data("V", **kw_get)
        Z = self.get_data("Z", **kw_get)
        TIME = self.get_data("TIME", **kw_get)
        NT, NZ, NY, NX = U.shape
        WD_SB = np.zeros((NT, NY, NX))
        R2 = np.zeros((NT, NY, NX))
        SB2 = np.zeros((NT, NY, NX))
        COASTCELL = self.get_data("COASTCELL")
        COR = self.get_data("COR")
        for iy in range(NY):
            for ix in range(NX):
                if COASTCELL[iy, ix] == 1 :
                    for it in range(NT):
                        iz = np.sum(Z[it, :, iy, ix] < 1300)
                        angle1, good, res = manage_angle.angle_fit(U[it, :, iy, ix], V[it, :, iy, ix])
                        if U[it, 3, iy, ix] < 0 :
                            angle1 = angle1 + 180
                        angle = (270 - angle1)%360
                        angle_rad = manage_angle.anglerad(angle)
                        
                        WD_SB[it, iy, ix] = angle
                        R2[it, iy, ix] = res
                        
                        U_AD = - U[it, :, iy, ix]*np.sin(angle_rad) - V[it, :, iy, ix]*np.cos(angle_rad)
                        a = np.nanmax(U_AD)
                        b = np.nanmin(U_AD)
                        c = np.nanargmax(U_AD)
                        d = np.nanargmin(U_AD)
                        test1 = a > 1 and b < -1 and a-b > 5
                        test_angle = c < d ^ ( (angle_rad-COR[iy, ix]+0.5*np.pi)%(2*np.pi) > np.pi )
                        SB2[it, iy, ix] = (test1 and test_angle) * 1
        return WD_SB, R2, SB2            
        for itime in range(NT):
            self.write_postproc("WD_SB", WD_SB[itime], ('y', 'x'), itime=itime, long_name="SB direction", standard_name="WD_SB", units="°", latex_units="°")
            self.write_postproc("WD_SB_R2", R2[itime], ('y', 'x'), itime=itime, long_name="R2 on SB direction", standard_name="WD_SB_R2", units=" ", latex_units=" ")
            self.write_postproc("SB2", SB2[itime], ('y', 'x'), itime=itime, long_name="SB2 mask on coast", standard_name="SB2", units=" ", latex_units=" ", typ="i1")
        
        return WD_SB

        
#################################################################################################################################
######  POST_PROCESS_GRAVITY_WAVES
#################################################################################################################################
        
    def calculate_GW(self, varname, **kwargs):
        if varname in ["GWD", "GWLAM", "GWS", "GWSM", 
                         "GWM2D", "GWM2X1", "GWM2Y1", "GWM2RMAX", "GWM2LMAX",
                         "GWM4LAMM", "GWM4SM", "GWM4D"] :
            # Gravity wave direction, wavelength, speed on the 17th May 2020
            # This variable is specific to this event
            if self.FLAGS["df"] :
                df = list(self.output_filenames["df"].values())[0]
                out = df[varname][kwargs["time_slice"]]
                if type(out) in [int, float, np.int64, np.float64]:
                    return out
                elif type(out) in [list, np.array, np.ndarray, pd.core.series.Series] :
                    out = np.expand_dims(np.array(out), axis=(-3, -2, -1)) #(NT, NZ, NY, NX) (will be squeezed later in get_data)
                    return out
                else :
                    raise(Exception(f"Unknow type {type(out)}"))
            else :
                raise(Exception(f"error in Dom.calculate {varname} : please add the online calculation of {varname} or calculate before and save in df (see Dom.compute_GW_carac_from_2DH)"))
        elif varname in ["GWMASK"] : 
            new_kwargs = {
                "time_slice" : kwargs["time_slice"],
                "hinterp" : {
                    "levels" : 300,
                },
                "saved" : {},
            }
            W = self.get_data("W", **new_kwargs)
            CDI, DY, DX = self.get_data(["COASTDIST", "DY", "DX"])
            return manage_GW.get_GWzone(W, CDI, DY, DX, display=False)
        elif varname in ["GWM2R", "GWM2L"] :
            X, Y, GWM2D, GWM2X1, GWM2Y1 = self.get_data(["X3", "Y3", "GWM2D", "GWM2X1", "GWM2Y1"], **kwargs)
            angle = np.deg2rad(GWM2D) # this angle indicates the direction of Xr (see paper 1) in meteorological convention
            s = np.sin(angle)
            c = np.cos(angle)
            if varname == "GWM2R" :
                r1 = -s * GWM2X1 -c * GWM2Y1
                return -s * X -c * Y - r1
            elif varname == "GWM2L" :
                l1 = c*GWM2X1 -s*GWM2Y1
                return c*X -s*Y - l1
        elif varname in ["GWM2U", "GWM2V"] :
            U, V, GWM2D = self.get_data(["U", "V", "GWM2D"], **kwargs)
            angle = np.deg2rad(GWM2D)
            # if type(angle) in [np.array, np.ndarray] : #several timestep
            #     angle = np.expand_dims(angle, axis=(-3, -2, -1))
            s = np.sin(angle)
            c = np.cos(angle)
            if varname == "GWM2U" :
                return -s*U -c*V
            elif varname == "GWM2V" :
                return c*U -s*V
        elif varname in ["GWMASK2", "GWMASK3", "GWMASK4"] :
            GWM2R, GWM2L, CDI, BDI = self.get_data(["GWM2R", "GWM2L", "COASTDIST", "BDY_DIST"], **kwargs)
            GWM2LMAX = self.get_data("GWM2LMAX", **kwargs)
            GWM2RMAX = self.get_data("GWM2RMAX", **kwargs) if varname=="GWMASK2" else 5000
            RMIN = 2000 if varname == "GWMASK4" else 0
            mr = np.logical_and(GWM2R >= RMIN, GWM2R <= GWM2RMAX)
            ml = np.logical_and(GWM2L >= 0, GWM2L <= GWM2LMAX)
            mDI = np.logical_and(BDI > 2000, CDI < -2000)
            while mDI.ndim < mr.ndim :
                mDI = np.expand_dims(mDI, axis=0)
            return 1* np.logical_and(np.logical_and(mr, ml), mDI)
        elif varname in ["GWA"] :
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["DX_smooth"] = 1
            W = self.get_data("W", **new_kwargs)
            W2 = self.get_data("SQUARED_W", **new_kwargs)
            return np.sqrt(2*(W2 - W**2))
        elif varname.startswith("GWAVG") : #ex : GWAVGU, GWAVGV
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["DX_smooth"] = 1
            return self.get_data(varname[5:], **new_kwargs)
        elif varname.startswith("GWA") : #ex : GWAU, GWAV
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["DX_smooth"] = 1
            var = self.get_data(varname[3:], **new_kwargs)
            var2 = self.get_data("SQUARED_"+varname[3:], **new_kwargs)
            return np.sqrt(2*(var2 - var**2))
        elif varname in ["GWR", "GWR_INST"] :
            # Gravity wave radial coordinate on the 17th May 2020
            # This variable is specific to this event
            # Based on the direction GWM4D, calculated by the gradient method within the zone GWMASK4
            # GWR is used to calculate the TKE correction, and to study the differences within the GW (in paper 1 Mathieu Landreau et al.)
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["crop"] = (15, kwargs["crop"][1], kwargs["crop"][2])
            
            GWLAM = self.get_data("GWM4LAMM", **new_kwargs)
            GWD = self.get_data("GWM4D", **new_kwargs)
            angle_rad = np.deg2rad(GWD)
            W, dWdx, dWdy = self.get_data(["W"+varname[3:], "DXC_W"+varname[3:], "DYC_W"+varname[3:]], **new_kwargs)
            dWdr = -dWdx*np.sin(angle_rad) - dWdy*np.cos(angle_rad)
            dWdr_norm = dWdr  * GWLAM / (2*np.pi)
            GWR = np.arctan2(W, dWdr_norm) / (2*np.pi)
            GWR[GWR < 0] += 1
            return GWR
        elif varname in ["DGWRDT"] :
            DT = manage_time.timedelta_to_seconds(self.get_data("DT_HIST"))
            time_slice = kwargs["time_slice"]
            it_list = np.arange(5000, dtype="int")[time_slice]
            if type(it_list) in [int, np.int64] :
                it_list = np.array([it_list])
            NT = len(it_list)
            it_list2 = []
            for it in it_list :
                if it < 2 :
                    raise(Exception("error : cannot compute DGWRDT at time it = " +str(it)))
                it_list2.append(it-1)
                it_list2.append(it)
            new_kwargs = self.copy_kw_get(kwargs)
            new_kwargs["time_slice"] = it_list2
            var = self.get_data("GWR_INST", **new_kwargs)
            old_shape = var.shape
            new_shape = (NT, 2) + old_shape[1:]
            DGWR = np.diff(np.reshape(var, new_shape), axis=1)
            if NT == 1 :
                DGWR = DGWR[0]
            DGWR[DGWR > 0.5] -= 1
            DGWR[DGWR < -0.5] += 1
            return DGWR/DT
        elif varname.startswith("GWIM2") :
            # Variance induced by gravity waves displacement
            varname2 = varname[5:]
            GWSM, GWLAM, GWR, GWAvar = self.get_data(["GWM4SM", "GWM4LAMM", "GWR", "GWA"+varname2], **kwargs)
            DT = 600
            OMEGA = 2*np.pi * GWSM/GWLAM + 1e-10
            # PHI = ((GWR+0.25)%1) * 2*np.pi if varname2 == "W" else ((GWR+0.5)%1) * 2*np.pi
            PHI = GWR * 2*np.pi if varname2 == "W" else ((GWR-0.25)%1) * 2*np.pi
            return manage_GW.GW_variance(GWAvar, OMEGA, PHI, DT=600)
        elif varname.startswith("GWCM2") :
            # Variance corrected by deducing GW variance
            varname2 = varname[5:]
            GWIM2var, M2var = self.get_data(["GWIM2"+varname2, "M2"+varname2], **kwargs)
            return M2var - GWIM2var
        elif varname.startswith("GWCSTD") :
            # Standard deviation corrected by deducing GW variance
            varname2 = varname[6:]
            GWCM2var = self.get_data("GWCM2"+varname2, **kwargs)
            return np.sqrt(GWCM2var)
        elif varname in ["GWITKE"] :
            M2U, M2V, M2W = self.get_data(["GWIM2U", "GWIM2V", "GWIM2W"], **kwargs)
            return 0.5*(M2U + M2V + M2W)
        elif varname in ["GWCTKE", "GWCTKE_RES"] :
            TKE, GWITKE = self.get_data(["TKE"+varname[6:], "GWITKE"], **kwargs)
            return TKE - GWITKE
        else :
            raise(Exception(f"Unknown varname {varname} in Dom.calculate_GW"))
    
    def compute_GW_carac_from_2DH(self, itime=("2020-05-17-11", "2020-05-17-17"), levels=300, display=False, savepath="", mask="GWMASK2") :
        kw_get = {
            "itime" : itime,
            "hinterp" : {
                "levels" : levels,
            },
            "saved" : {},
        }
        W, TIME = self.get_data(["W", "TIME"], **kw_get)
        CDI, DY, DX = self.get_data(["COASTDIST", "DY", "DX"])
        NT = len(TIME)
        pref = "GWM2" if mask == "GWMASK2" else "GWM3" if mask == "GWMASK3" else "GW"
        GWMASK = self.get_data(mask, itime=itime)
        print(f"GWMASK.shape : {GWMASK.shape}")
        GWD, _ = manage_GW.compute_wave_orientation_gradient(W, DY, DX, GWMASK)
        print(f"GWD.shape : {GWD.shape}")
        GWLAM = manage_GW.compute_wavelength(W, DY, DX, expected_lambda=1700, GWD=GWD, mask=GWMASK, display=display, method="fit")
        print(f"GWLAM.shape : {np.array(GWLAM).shape}")
        GWLAMM = manage_GW.compute_wavelength(W, DY, DX, expected_lambda=1700, GWD=GWD, mask=GWMASK, display=display, method="max")
        print(f"GWLAMM.shape : {np.array(GWLAMM).shape}")
        GWS, TIME2 = manage_GW.compute_wavespeed(W, DY, DX, TIME=TIME, dit=1, lam=GWLAM, GWD=GWD, mask=GWMASK, display=display, 
                                                 fac=1, method="fit")
        print(f"GWS.shape : {np.array(GWS).shape}")
        GWSM, TIME2 = manage_GW.compute_wavespeed(W, DY, DX, TIME=TIME, dit=1, lam=GWLAMM, GWD=GWD, mask=GWMASK, display=display, 
                                                  fac=1, method="max")
        print(f"GWSM.shape : {np.array(GWSM).shape}")
        
        TIMEout = self.get_data("TIME", itime="ALL_TIMES")
        NTout = len(TIMEout)
        it0 = manage_time.get_time_slice(TIME[0], TIMEout)
        it1 = manage_time.get_time_slice(TIME[-1], TIMEout)+1
        GWDout = np.zeros((NTout))*np.nan
        GWLAMout = np.zeros((NTout))*np.nan
        GWLAMMout = np.zeros((NTout))*np.nan
        GWSout = np.zeros((NTout))*np.nan
        GWSMout = np.zeros((NTout))*np.nan
        GWDout[it0:it1] = GWD
        GWLAMout[it0:it1] = GWLAM
        GWLAMMout[it0:it1] = GWLAMM
        GWSout[it0] = GWS[0]; GWSout[it1] = GWS[-1]; GWSout[it0+1:it1-1] = 0.5*(GWS[:-1] + GWS[1:])
        GWSMout[it0] = GWSM[0]; GWSMout[it1] = GWSM[-1]; GWSMout[it0+1:it1-1] = 0.5*(GWSM[:-1] + GWSM[1:])
        if savepath == ""  :
            savepath = self.postprocdir+self.name+"_df_TIME.pkl"
        if os.path.exists(savepath):
            dfout = pd.read_pickle(savepath)
        else :
            dfout = pd.DataFrame({
                "TIME" : TIMEout,
            })
        dfout[pref+"DIR"] = GWDout
        dfout[pref+"LAM"] = GWLAMout
        dfout[pref+"LAMM"] = GWLAMMout
        dfout[pref+"S"] = GWSout
        dfout[pref+"SM"] = GWSMout
        if savepath is not None :
            dfout.to_pickle(savepath)
        return dfout
    
    def compute_GWLAMM(self, itime=("2020-05-17-11", "2020-05-17-17"), levels=300, display=False, savepath="") :
        kw_get = {
            "itime" : itime,
            "hinterp" : {
                "levels" : levels,
            },
            "saved" : {},
        }
        W, TIME = self.get_data(["W", "TIME"], **kw_get)
        CDI, DY, DX = self.get_data(["COASTDIST", "DY", "DX"])
        NT = len(TIME)
        
        GWMASK = manage_GW.get_GWzone(W, CDI, DY, DX, display=False)
        print(f"GWMASK.shape : {GWMASK.shape}")
        GWD = self.get_data("GWD", **kw_get)
        print(f"GWD.shape : {GWD.shape}")
        GWLAMM = manage_GW.compute_wavelength(W, DY, DX, expected_lambda=1700, GWD=GWD, mask=GWMASK, display=display, method="max")
        TIMEout = self.get_data("TIME", itime="ALL_TIMES")
        NTout = len(TIMEout)
        it0 = manage_time.get_time_slice(TIME[0], TIMEout)
        it1 = manage_time.get_time_slice(TIME[-1], TIMEout)+1
        GWLAMMout = np.zeros((NTout))*np.nan
        GWLAMMout[it0:it1] = GWLAMM
        if savepath == ""  :
            savepath = self.postprocdir+self.name+"_df_TIME.pkl"
        if os.path.exists(savepath):
            dfout = pd.read_pickle(savepath)
            dfout["GWLAMM"] = GWLAMMout
        else :
            dfout = pd.DataFrame({
                "TIME" : TIMEout,
                "GWLAMM" : GWLAMMout,
            })
        if savepath is not None :
            dfout.to_pickle(savepath)
        return dfout
    
    def save_GWMASK2_carac(self, savepath="") :
        """
        Description
            Generate a DataFrame with values for GWM2D, GWM2X1, GWM2Y1, GWM2RMAX, GWM2LMAX
        Parameters
            self (Dom)
        Optional
            savepath (str): path of the saved DataFrame, default: self.postprocdir+self.name+"_df_TIME.pkl"
        Returns 
            GWM2D: direction normal to the front 
            GWM2X1, GWM2Y1: coordinate in the front
            GWM2RMAX: Distance from the front where GW are still present
            GWM2LMAX: Max width of the mask area
        """
        GWM2_dict = {
            "2020-05-17_11-00-00" : [[47.217, -2.6],  20e3, np.deg2rad(62), -15, 5,  15],
            "2020-05-17_11-10-00" : [[47.217, -2.6],  20e3, np.deg2rad(61), -15, 5,  16],
            "2020-05-17_11-20-00" : [[47.217, -2.6],  20e3, np.deg2rad(60), -15, 6,  17],
            "2020-05-17_11-30-00" : [[47.217, -2.6],  20e3, np.deg2rad(59), -15, 6,  18],
            "2020-05-17_11-40-00" : [[47.217, -2.61], 20e3, np.deg2rad(58), -15, 7,  19],
            "2020-05-17_11-50-00" : [[47.217, -2.62], 20e3, np.deg2rad(58), -15, 7,  20],
            "2020-05-17_12-00-00" : [[47.217, -2.63], 20e3, np.deg2rad(57), -18, 8,  20],
            "2020-05-17_12-10-00" : [[47.22,  -2.64], 20e3, np.deg2rad(57), -19, 8,  20],
            "2020-05-17_12-20-00" : [[47.224, -2.65], 20e3, np.deg2rad(58), -19, 8,  20],
            "2020-05-17_12-30-00" : [[47.229, -2.65], 20e3, np.deg2rad(58), -19, 9,  20],
            "2020-05-17_12-40-00" : [[47.237, -2.65], 20e3, np.deg2rad(59), -19, 9,  20],
            "2020-05-17_12-50-00" : [[47.243, -2.65], 20e3, np.deg2rad(60), -19, 9,  20],
            "2020-05-17_13-00-00" : [[47.25,  -2.65], 20e3, np.deg2rad(60), -19, 10, 20],
            "2020-05-17_13-10-00" : [[47.255, -2.65], 20e3, np.deg2rad(64), -19, 10, 20],
            "2020-05-17_13-20-00" : [[47.265, -2.65], 20e3, np.deg2rad(68), -19, 11, 20],
            "2020-05-17_13-30-00" : [[47.265, -2.65], 20e3, np.deg2rad(68), -19, 11, 20],
            "2020-05-17_13-40-00" : [[47.267, -2.65], 20e3, np.deg2rad(68), -19, 12, 20],
            "2020-05-17_13-50-00" : [[47.267, -2.65], 20e3, np.deg2rad(68), -19, 12, 20],
            "2020-05-17_14-00-00" : [[47.268, -2.65], 20e3, np.deg2rad(68), -19, 13, 20],
            "2020-05-17_14-10-00" : [[47.268, -2.65], 20e3, np.deg2rad(68), -19, 13, 20],
            "2020-05-17_14-20-00" : [[47.268, -2.65], 20e3, np.deg2rad(68), -19, 14, 20],
            "2020-05-17_14-30-00" : [[47.268, -2.65], 20e3, np.deg2rad(68), -19, 14, 20],
            "2020-05-17_14-40-00" : [[47.269, -2.65], 20e3, np.deg2rad(67), -19, 15, 20],
            "2020-05-17_14-50-00" : [[47.270, -2.65], 20e3, np.deg2rad(65), -19, 15, 20],
            "2020-05-17_15-00-00" : [[47.271, -2.65], 20e3, np.deg2rad(65), -19, 16, 20],
            "2020-05-17_15-10-00" : [[47.271, -2.65], 20e3, np.deg2rad(64), -19, 20, 20],
            "2020-05-17_15-20-00" : [[47.273, -2.65], 20e3, np.deg2rad(65), -19, 20, 21],
            "2020-05-17_15-30-00" : [[47.275, -2.65], 20e3, np.deg2rad(68), -19, 20, 22],
            "2020-05-17_15-40-00" : [[47.275, -2.65], 20e3, np.deg2rad(70), -19, 20, 23],
            "2020-05-17_15-50-00" : [[47.275, -2.65], 20e3, np.deg2rad(73), -19, 20, 24],
            "2020-05-17_16-00-00" : [[47.272, -2.65], 20e3, np.deg2rad(77), -19, 20, 25],
            "2020-05-17_16-10-00" : [[47.272, -2.65], 20e3, np.deg2rad(85), -19, 20, 25],
            "2020-05-17_16-20-00" : [[47.277, -2.65], 20e3, np.deg2rad(85), -19, 20, 25],
            "2020-05-17_16-30-00" : [[47.279, -2.65], 20e3, np.deg2rad(85), -19, 20, 25],
            "2020-05-17_16-40-00" : [[47.285, -2.65], 20e3, np.deg2rad(82), -19, 20, 25],
            "2020-05-17_16-50-00" : [[47.285, -2.65], 20e3, np.deg2rad(82), -19, 20, 25],
            "2020-05-17_17-00-00" : [[47.285, -2.65], 20e3, np.deg2rad(82), -19, 20, 25],
        }

        TIMEout = self.get_data("TIME", itime="ALL_TIMES") #vecteur temps des sorties de WRF
        NTout = len(TIMEout)
        NT = len(GWM2_dict)
        #GW=gravity waves, M2 = mask2
        GWM2D = np.zeros((NTout))*np.nan #orientation du front
        GWM2X1 = np.zeros((NTout))*np.nan #position X du coin en haut à gauche
        GWM2Y1 = np.zeros((NTout))*np.nan #position Y du coin en haut à gauche
        GWM2RMAX = np.zeros((NTout))*np.nan #longueur radiale de la zone
        GWM2LMAX = np.zeros((NTout))*np.nan #longueur tangentielle de la zone
        for it, tstr in enumerate(GWM2_dict) :
            t = manage_time.to_datetime(tstr)
            itout = manage_time.get_time_slice(t, TIMEout) #indice de l'instant dans TIMEout
            p1, _, angle, GWM2X1[itout], GWM2RMAX[itout], GWM2LMAX[itout] = GWM2_dict[tstr]
            GWM2X1[itout] = GWM2X1[itout]*1000
            x0, y0 = manage_projection.ll_to_xy(p1[1], p1[0])  #position du point central de la ligne utilisée pour tracer le front
            GWM2Y1[itout] = (GWM2X1[itout]-x0)/np.tan(angle) + y0
            GWM2D[itout] = np.rad2deg(angle) + 270
        GWM2RMAX = GWM2RMAX*1000
        GWM2LMAX = GWM2LMAX*1000
        #sauvegarde dans un dataframe
        savepath = ""
        if savepath == ""  :
            savepath = self.postprocdir+self.name+"_df_TIME.pkl"
        if os.path.exists(savepath):
            dfout = pd.read_pickle(savepath)
        else :
            dfout = pd.DataFrame({
                "TIME" : TIMEout,
            })
        dfout["GWM2D"] = GWM2D
        dfout["GWM2X1"] = GWM2X1
        dfout["GWM2Y1"] = GWM2Y1
        dfout["GWM2RMAX"] = GWM2RMAX
        dfout["GWM2LMAX"] = GWM2LMAX
        if savepath is not None :
            dfout.to_pickle(savepath)
        return dfout
    
    def compute_GW_carac_from_MASK4(self, itime=("2020-05-17-11", "2020-05-17-17"), levels=300, display=False, savepath="") :
        """
        Description
            Calculate the direction, the wavelength, the wave speed within GWMASK4 (which must be already defined)
        Parameters
            self (Dom)
        Optional
            itime (see Dom.get_data): time period on which we compute it
            levels (float): height at which the carac are computed
            display (boolean): True to print figures
            savepath (str): path of the saved DataFrame, default: self.postprocdir+self.name+"_df_TIME.pkl"
        Returns 
            GWM2D: direction normal to the front 
            GWM2X1, GWM2Y1: coordinate in the front
            GWM2RMAX: Distance from the front where GW are still present
            GWM2LMAX: Max width of the mask area
        """
        kw_get = {
            "itime" : itime,
            "hinterp" : {
                "levels" : levels,
            },
            "saved" : {},
        }
        W, TIME = self.get_data(["W", "TIME"], **kw_get)
        CDI, DY, DX = self.get_data(["COASTDIST", "DY", "DX"])
        NT = len(TIME)
        pref = "GWM4"
        GWMASK = self.get_data("GWMASK4", itime=itime)
        print(f"GWMASK.shape : {GWMASK.shape}")
        GWD, _ = manage_GW.compute_wave_orientation_gradient(W, DY, DX, GWMASK)
        GWD += 180 #Arbitrary chosen to be between 180 and 360 for this case because it moves toward NW
        print(f"GWD.shape : {GWD.shape}")
        # GWLAM = manage_GW.compute_wavelength(W, DY, DX, expected_lambda=1700, GWD=GWD, mask=GWMASK, display=display, method="fit")
        # print(f"GWLAM.shape : {np.array(GWLAM).shape}")
        GWLAMM = manage_GW.compute_wavelength(W, DY, DX, expected_lambda=1700, GWD=GWD, mask=GWMASK, display=display, method="max")
        print(f"GWLAMM.shape : {np.array(GWLAMM).shape}")
        # GWS, TIME2 = manage_GW.compute_wavespeed(W, DY, DX, TIME=TIME, dit=1, lam=GWLAM, GWD=GWD, mask=GWMASK, display=display, 
        #                                          fac=1, method="fit")
        # print(f"GWS.shape : {np.array(GWS).shape}")
        GWSM, TIME2 = manage_GW.compute_wavespeed(W, DY, DX, TIME=TIME, dit=1, lam=GWLAMM, GWD=GWD, mask=GWMASK, display=display, 
                                                  fac=1, method="max")
        print(f"GWSM.shape : {np.array(GWSM).shape}")
        
        TIMEout = self.get_data("TIME", itime="ALL_TIMES")
        NTout = len(TIMEout)
        it0 = manage_time.get_time_slice(TIME[0], TIMEout)
        it1 = manage_time.get_time_slice(TIME[-1], TIMEout)+1
        GWDout = np.zeros((NTout))*np.nan
        # GWLAMout = np.zeros((NTout))*np.nan
        GWLAMMout = np.zeros((NTout))*np.nan
        # GWSout = np.zeros((NTout))*np.nan
        GWSMout = np.zeros((NTout))*np.nan
        GWDout[it0:it1] = GWD
        # GWLAMout[it0:it1] = GWLAM
        GWLAMMout[it0:it1] = GWLAMM
        # GWSout[it0] = GWS[0]; GWSout[it1] = GWS[-1]; GWSout[it0+1:it1-1] = 0.5*(GWS[:-1] + GWS[1:])
        GWSMout[it0] = GWSM[0]; GWSMout[it1] = GWSM[-1]; GWSMout[it0+1:it1-1] = 0.5*(GWSM[:-1] + GWSM[1:])
        if savepath == ""  :
            savepath = self.postprocdir+self.name+"_df_TIME.pkl"
        if os.path.exists(savepath):
            dfout = pd.read_pickle(savepath)
        else :
            dfout = pd.DataFrame({
                "TIME" : TIMEout,
            })
        dfout[pref+"D"] = GWDout
        # dfout[pref+"LAM"] = GWLAMout
        dfout[pref+"LAMM"] = GWLAMMout
        # dfout[pref+"S"] = GWSout
        dfout[pref+"SM"] = GWSMout
        if savepath is not None :
            dfout.to_pickle(savepath)
        return dfout
    
    def GWMASK4_IR_IL(self) :
        """
        Description
            Generate GWMASK4 from GWMASK2
        Parameters
            self (Dom)
        Optional
            itime (see Dom.get_data): time period on which we compute it
            levels (float): height at which the carac are computed
            display (boolean): True to print figures
            savepath (str): path of the saved DataFrame, default: self.postprocdir+self.name+"_df_TIME.pkl"
        Returns 
            GWM2D: direction normal to the front 
            GWM2X1, GWM2Y1: coordinate in the front
            GWM2RMAX: Distance from the front where GW are still present
            GWM2LMAX: Max width of the mask area
        """
        #load data
        p = {}
        kw_get = {
            "itime" : ("2020-05-17-11", "2020-05-17-17"),
            "crop" : (0, "ALL", "ALL"),
            "saved" : p,
        }
        varnames = ["GWM2R", "GWM2L", "COASTDIST", "BDY_DIST"]
        _ = self.get_data(varnames, **kw_get)
        p2 = {}
        TIME = self.get_data("TIME", **kw_get)
        NT, NY, NX = np.squeeze(p["GWM2R"]).shape
        p["it"] = np.arange(len(TIME))
        p["it"] = np.expand_dims(p["it"], axis=-1)
        p["it"] = np.repeat(p["it"], NY, axis=-1)
        p["it"] = np.expand_dims(p["it"], axis=-1)
        p["it"] = np.repeat(p["it"], NX, axis=-1)
        for v in varnames :
            p2[v] = p[v].flatten()
        p2["it"] = p["it"].flatten()
        #generate a dataframe
        df0 = pd.DataFrame(p2)
        #compute ir
        df0["ir"] = pd.cut(df0["GWM2R"], bins=np.arange(int(np.nanmin(df0["GWM2R"])/1e3-1)*1e3, int(np.nanmax(df0["GWM2R"])/1e3+2)*1e3, 1e3), 
                           labels=False).astype(int)
        df0["il"] = pd.cut(df0["GWM2L"], bins=np.arange(int(np.nanmin(df0["GWM2L"])/1e3-1)*1e3, int(np.nanmax(df0["GWM2L"])/1e3+2)*1e3, 1e3), 
                           labels=False).astype(int)
        GWIR = np.array(df0["ir"]).reshape(NT, NY, NX)
        GWIL = np.array(df0["il"]).reshape(NT, NY, NX)
        print(GWIL.shape)
        indices=["it", "il", "ir"]
        df00 = df0.set_index(indices)
        # We want to keep only points that are between 2 and 5 km from the front i.e. 2000 < GWM2R < 5000
        rmin = 2000
        rmax = 5000
        CDmin = 0
        BDmin = 1000
        delta = 300

        # conditions : mask1 for zone B (gravity wave and gravity current side of the front), mask 0 for zone C
        df00["rmask1"] = np.logical_and(df00["GWM2R"] >= rmin-delta, df00["GWM2R"] <= rmax+delta) 
        df00["rmask0"] = np.logical_and(df00["GWM2R"] <= -rmin+delta, df00["GWM2R"] >= -rmax-delta) 
        # We keep only the cells for which every points satisfy the condition i.e. the minimum value of mask is True
        irok1 = df00.groupby(["it", "ir"])["rmask1"].min()
        irok0 = df00.groupby(["it", "ir"])["rmask0"].min()
        irok01 = irok1[np.logical_or(irok1, irok0)].index # indice ir valides dans les deux zones (change en fonction de it)
        irok0 = irok0[irok0].index # indices ir valides dans la zone 0
        irok1 = irok1[irok1].index # indices ir valides dans la zone 1
        #On garde uniquement les indices ir valides dans les deux zones
        maskir = np.isin(df00.index.droplevel('il'), irok01)
        maskir0 = np.isin(df00.index.droplevel('il'), irok0)
        maskir1 = np.isin(df00.index.droplevel('il'), irok1)
        # Mask dans l'autre direction
        # On ne veut pas de points à moins de 2 km de la côte, ni à moins de 2 km de la limite du domaine
        df00["lmask"] = np.logical_and(df00["COASTDIST"] <= -CDmin, df00["BDY_DIST"] >= BDmin) 
        # We keep only the cells for which every points satisfy the condition i.e. the minimum value of mask is True
        # On groupe par "it" et "il" mais pas "ir" donc un couple (it, il) n'est valide que si toutes les cellules ir sont valides
        ilok = df00[maskir].groupby(["it", "il"])["lmask"].min()
        ilok = ilok[ilok].index
        maskil = np.isin(df00.index.droplevel('ir'), ilok)
        maskir = maskir.reshape(NT, NY, NX)
        maskir0 = maskir0.reshape(NT, NY, NX)
        maskir1 = maskir1.reshape(NT, NY, NX)
        maskil = maskil.reshape(NT, NY, NX)
        GWMASK4 = np.logical_and(maskir1, maskil)
        GWMASK5 = np.logical_and(maskir0, maskil)
        TIMEout = self.get_data("TIME", itime="ALL_TIMES")
        NTout = len(TIMEout)
        GWIRout = np.zeros((NTout, NY, NX)) * np.nan
        GWILout = np.zeros((NTout, NY, NX)) * np.nan
        GWMASK4out = np.zeros((NTout, NY, NX)) * np.nan
        GWMASK5out = np.zeros((NTout, NY, NX)) * np.nan
        time_slice = manage_time.get_time_slice(TIME, TIMEout)
        GWIRout[time_slice, :, :] = GWIR
        GWILout[time_slice, :, :] = GWIL
        GWMASK4out[time_slice, :, :] = GWMASK4
        GWMASK5out[time_slice, :, :] = GWMASK5
        for itime in range(NTout) :
            self.write_postproc("GWMASK5", GWMASK5out[itime], ('y', 'x'), itime=itime, long_name="zone 0", standard_name="GWMASK5", units="", 
                                latex_units="", typ=np.float32)
            self.write_postproc("GWMASK4", GWMASK4out[itime], ('y', 'x'), itime=itime, long_name="zone 1", standard_name="GWMASK4", units="", 
                                latex_units="", typ=np.float32)
            self.write_postproc("GWIL", GWILout[itime], ('y', 'x'), itime=itime, long_name="GWIL", standard_name="GWIL", units="", 
                                latex_units="", typ=np.float32)
            self.write_postproc("GWIR", GWIRout[itime], ('y', 'x'), itime=itime, long_name="GWIR", standard_name="GWIR", units="", 
                                latex_units="", typ=np.float32)

#################################################################################################################################
######  POST_PROCESS_LOW_LEVEL_JETS
#################################################################################################################################
  
    def calculate_LLJ(self, varname, **kwargs):
        if varname in ["LLJ_MIN"]:
            return self.get_data("LLJ_MH", **kwargs) - self.get_data("LLJ_PROM", **kwargs)
        elif varname.startswith("LLJ_"):
            varname1 = varname[4:]
            if self.get_dim(varname1) == 2:
                print(f"warning in Dom.calculate_LLJ, return {varname}={varname1}")
                return self.get_data(varname1, **kwargs)
            LLJ_IZ, LLJ = self.get_data(["LLJ_IZ", "LLJ"], **kwargs)
            new_kwargs = self.copy_kw_get(kwargs)
            izmax = self.nearest_z_index(1.2*500) #supposing manage_LLJ.max_heigth = 500 (LLJ core cannot be higher than 500 m) 
            new_kwargs["crop"] =  ([0, izmax], kwargs["crop"][1], kwargs["crop"][2])
            var1 = self.get_data(varname1, **new_kwargs)
            zaxis = self.find_axis("z", varname=varname1, **new_kwargs)
            LLJ = np.expand_dims(LLJ, axis=zaxis).astype(int)
            LLJ_IZ = np.expand_dims(LLJ_IZ, axis=zaxis).astype(int)
            LLJ_IZ[LLJ!=1] = 0
            print(var1.shape, LLJ.shape, LLJ_IZ.shape)
            var = np.take_along_axis(var1, LLJ_IZ, axis=zaxis)
            var[LLJ!=1] = np.nan
            return var
        else :
            raise(Exception(f"Unknown varname {varname} in Dom.calculate_GW"))
            
    
    def detect_LLJ1_1time(self, itime) :
        if debug : print("LLJ1(", itime, "), ", end="")
        saved = {}
        MH = self.get_data("MH", itime=itime, saved=saved)
        WD = self.get_data("WD", itime=itime, saved=saved)
        Z = self.get_data("Z", itime=itime, saved=saved)
        zaxis = 0
        NZ, NY, NX = MH.shape

        MH[np.where(Z>500)] = np.nan

        k = np.nanargmax(MH, axis=zaxis)
        k = np.expand_dims(k, axis=zaxis)
        MH_LLJ = np.nanmax(MH, axis=zaxis)
        Z_LLJ  = np.squeeze(np.take_along_axis(Z,  k, axis=zaxis), axis=zaxis)
        WD_LLJ = np.squeeze(np.take_along_axis(WD, k, axis=zaxis), axis=zaxis)

        Z_LLJ_dim = np.repeat(Z_LLJ[np.newaxis, :, :], NZ, axis=zaxis)
        above = Z > Z_LLJ_dim
        below = Z < Z_LLJ_dim
        MH_above = copy.copy(MH)
        MH_below = copy.copy(MH)
        MH_above[below] = np.nan
        MH_below[above] = np.nan

        MH_min1 = np.nanmin(MH_below, axis=zaxis)
        MH_min2 = np.nanmin(MH_above, axis=zaxis)
        MH_min = np.maximum(MH_min1, MH_min2)   
        LLJ = np.logical_and(MH_LLJ- MH_min > 0.2, MH_LLJ/MH_min2 > 0.8)        
            
        not_LLJ = np.logical_not(LLJ)
        Z_LLJ[not_LLJ] = np.nan
        MH_LLJ[not_LLJ] = np.nan
        WD_LLJ[not_LLJ] = np.nan
        self.write_postproc("LLJ1",    LLJ,    ('y', 'x'), itime=itime, long_name="LLJ1 mask",           standard_name="LLJ1",    units=" ",     latex_units=" ", typ='i1')
        self.write_postproc("MH_LLJ1", MH_LLJ, ('y', 'x'), itime=itime, long_name="LLJ1 core speed",     standard_name="MH_LLJ1", units="m.s-1", latex_units="m.s^{-1}")
        self.write_postproc("Z_LLJ1",  Z_LLJ,  ('y', 'x'), itime=itime, long_name="LLJ1 core height",    standard_name="Z_LLJ1",  units="m",     latex_units="m")
        self.write_postproc("WD_LLJ1", WD_LLJ, ('y', 'x'), itime=itime, long_name="LLJ1 core direction", standard_name="WD_LLJ1", units="°",     latex_units="°")
        
    def detect_LLJ1(self, nprocs=1) :
        NT = len(self.date_list)
        if nprocs > 1 :
            inputs = [(itime) for itime in range(NT)]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.detect_LLJ1_1time, inputs)
        else :
            for itime in range(NT) :
                self.detect_LLJ1_1time(itime)
    """        
    def detect_LLJ1(self) :
        #get_data(self, varname, itime = None, time_slice = None, crop = None, zoom=None, i_unstag = None, hinterp=None, vinterp=None, saved=None)
        Nt = len(self.date_list)
        for itime in range(len(self.date_list)):
            saved = {}
            MH = self.get_data("MH", itime=itime, saved=saved)
            WD = self.get_data("WD", itime=itime, saved=saved)
            Z = self.get_data("Z", itime=itime, saved=saved)
            
            NZ, NY, NX = MH.shape
            shape = (NY, NX)
            MH_LLJ = np.zeros(shape) * np.nan
            LLJ = np.zeros(shape) * np.nan
            Z_LLJ = np.zeros(shape) * np.nan
            WD_LLJ = np.zeros(shape) * np.nan
            k_max = np.argmax(MH, axis=0)
            MH[np.where(Z>500)] = np.nan
            k = np.nanargmax(MH, axis=0)
            MH_max = np.nanmax(MH, axis=0)
            #return MH, WD, Z, NY, NX, k, MH_max
            MH_above = copy.copy(MH)
            MH_below = copy.copy(MH)
            k = np.expand_dims(k, axis=0)
            Z_LLJ  = np.squeeze(np.take_along_axis(Z,  k, axis=0), axis=0)
            WD_LLJ = np.squeeze(np.take_along_axis(WD, k, axis=0), axis=0)
            MH_LLJ = np.squeeze(np.take_along_axis(MH, k, axis=0), axis=0)
            
            Z_LLJ_dim3 = np.tensordot(np.ones(NZ), Z_LLJ, 0) #there exist probably another numpy funciton to do this
            above = Z > Z_LLJ_dim3
            below = Z < Z_LLJ_dim3
            MH_above[below] = np.nan
            MH_below[above] = np.nan
            
            MH_min1 = np.nanmin(MH_below, axis=0)
            MH_min2 = np.nanmin(MH_above, axis=0)
            MH_min = np.maximum(MH_min1, MH_min2)   
            LLJ = np.logical_and(MH_max - MH_min > 0.2, MH_max/MH_min2 > 0.8)
            not_LLJ = np.logical_not(LLJ)
            Z_LLJ[not_LLJ] = np.nan
            MH_LLJ[not_LLJ] = np.nan
            WD_LLJ[not_LLJ] = np.nan
            
            return LLJ, Z_LLJ, WD_LLJ, MH_LLJ
    """
        
#################################################################################################################################
######  POST_PROCESS
#################################################################################################################################
     
    
    def write_postproc(self, varname, var, dims, itime=0, long_name="", standard_name="", units="", latex_units="", typ=np.float32) :
        """
        Description
            Write a variable in the postproc datafile
        Parameters
            varname : str : name of the variable, ex : "X", "LAT", "U_XSTAG"
            var : np.array : data
            dims : tuple of str : dimensions of the variable ; ex : ("x"), ("y", "x")
        Optional
            itime : int : index of the date in self.date_list ; ex : 0, 2, -1 ; default : 0, if None, it is a static file
            long_name : str : long name of the variable ; ex : "South-West coordinate" ; default : ""
            standard_name : str : standard name of the variable ; ex : "X coordinate" ; default : ""
            units : str : units name ; ex : "m", "Pa.s-1" ; default : ""
            latex_units : str : units in latex format ; ex : "m", "Pa.s^{-1}" ; default : ""
        Returns 
            Dom 
        """
        if itime is None :
            filename = self.postprocdir + self.name + "_post_static.nc"
            key = "post_static"
        else :
            date = self.date_list[itime]
            filename = self.postprocdir + self.name + "_post_" + self.date2str(date, self.software) + ".nc"
            key = "post"
        if standard_name == "" :
            standard_name = long_name 
        if long_name == "" :
            long_name = standard_name
        if self.software in ["WRF", "WPS", "WRFinput"] :
            dims_temp = ()
            for d in dims :
                if 'x' in d :
                    dims_temp += ('west_east'+d[1:],)
                elif 'y' in d:
                    dims_temp += ('north_south'+d[1:],)
                elif 'z' in d:
                    dims_temp += ('bottom_top'+d[1:],)
                else :
                    dims_temp += (d,)
            dims = dims_temp
            
        open_at_the_end = False
        if not os.path.exists(filename):
            # print("creating postproc file : ", filename)
            init = True
            ncfout = netCDF4.Dataset(filename, mode="w", format='NETCDF4_CLASSIC')
        else :
            # print("opening file : ", filename)
            init = False
            if filename in self.output_filenames[key] and self.keep_open:
                ncfout = self.output_filenames[key][filename]
                ncfout.close()
                ncfout = netCDF4.Dataset(filename, mode="a" , format='NETCDF4_CLASSIC')
            else :
                ncfout = netCDF4.Dataset(filename, mode="a" , format='NETCDF4_CLASSIC')
        if init :
            NX = self.get_data("NX")
            NY = self.get_data("NY")
            NZ = self.get_data("NZ")
            NZ_SOIL = self.get_data("NZ_SOIL")
            NX_XSTAG = self.get_data("NX_XSTAG")
            NY_YSTAG = self.get_data("NY_YSTAG")
            NZ_ZSTAG = self.get_data("NZ_ZSTAG")
            NZ_SOIL_ZSTAG = self.get_data("NZ_SOIL_ZSTAG")

            ncfout.Title        = "py_wrf_arps postproc"
            ncfout.Institution  = "LHEEA-DAUC"
            ncfout.DX           = self.get_data("DX")
            ncfout.DY           = self.get_data("DY")
            ncfout.MAPPROJ      = self.get_data("MAPPROJ")
            ncfout.TRUELAT1     = self.get_data("TRUELAT1")
            ncfout.TRUELAT2     = self.get_data("TRUELAT2")
            ncfout.TRUELON      = self.get_data("TRUELON")
            ncfout.CTRLAT       = self.get_data("CTRLAT")
            ncfout.CTRLON       = self.get_data("CTRLON")
            ncfout.FMTVER       = "NetCDF 4.0 Classic"
            ncfout.Source       = "py_wrf_arps/WRF_ARPS/class_dom.py"
            ncfout.References   = "See class_dom.py"
            ncfout.Comment      = "Postproc data saved in file"
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            ncfout.History      = "Creation date: {date}".format(date=dt_string)

            if self.software == "ARPS":
                ncfout.createDimension('x', NX)
                ncfout.createDimension('y', NY)
                ncfout.createDimension('z', NZ) 
                ncfout.createDimension('x_stag', NX_XSTAG)
                ncfout.createDimension('y_stag', NY_YSTAG)
                ncfout.createDimension('z_stag', NZ_ZSTAG) 
                ncfout.createDimension('z_soil', NZ_SOIL + 1) 
                ncfout.createDimension('Time', 1) 
                ncfout.SCLFCT       = self.get_data("SCLFCT")
                ncfout.INITIAL_TIME = self.date2str(self.get_data("INITIAL_TIME"), "WRF")
            else :
                ncfout.createDimension('west_east', NX)
                ncfout.createDimension('north_south', NY) 
                ncfout.createDimension('bottom_top', NZ)
                ncfout.createDimension('west_east_stag', NX_XSTAG)
                ncfout.createDimension('north_south_stag', NY_YSTAG) 
                ncfout.createDimension('bottom_top_stag', NZ_ZSTAG) 
                ncfout.createDimension('soil_layers', NZ_SOIL)
                ncfout.createDimension('soil_layers_stag', NZ_SOIL_ZSTAG)
                ncfout.createDimension('Time', 1) 
                ncfout.START_DATE = self.date2str(self.get_data("INITIAL_TIME"), "WRF")
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
        
        if self.keep_open :
            # we close the Dataset in "a" (append) mode and repoen it in "r" (read) mode
            self.output_filenames[key][filename] = netCDF4.Dataset(filename, mode="r", format='NETCDF4_CLASSIC')
        
#################################################################################################################################
######  Geography and location
#################################################################################################################################
               
    def nearest_index_from_self_grid(self, points, return_dist=False, **kwargs):
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
        res = (iy, ix)
        if return_dist :
            dist = I[0]
            res += (dist,)
        return res
    
    def nearest_index_to_self_grid(self, LAT_ext_in, LON_ext_in, return_dist=False):
        """for self grid points, it return the index of the nearest external grid points
        Parameters
            LAT_ext, LON_ext (np.arrays): arrays of Latitude and Longitude points, can have any dimensions, will be flattened
        Optional
            return_dist (boolean): if True, returns also the distance of the query
        Return 
            i (np.array of shape(self.NY, self.NX)): the index of the nearest point in external dataset (without NaNs) from each self grid point
            mask (np.array) : scipy.spatial.query doesn't manage NaNs, so we apply a mask to remove NaNs, and returns the mask
            dist (np.array of shape(self.NY, self.NX)): the distance between each self grid points and the nearest external point
        """
        LAT_ext, LON_ext = LAT_ext_in.flatten(), LON_ext_in.flatten()
        mask = np.logical_not(np.logical_or(np.isnan(LAT_ext), np.isnan(LON_ext)))
        LAT_ext, LON_ext = LAT_ext[mask], LON_ext[mask]
        if len(LAT_ext) > 0 :
            llgrid = np.array([LAT_ext, LON_ext]).T
            tree = spatial.cKDTree(llgrid)
            LAT, LON = self.get_data("LAT").flatten(), self.get_data("LON").flatten()
            points = np.array([LAT, LON.flatten()]).T
            dist, i = tree.query(points)
            NX, NY = self.get_data(["NX", "NY"])
            res = (i.reshape(NY, NX), mask)
            if return_dist :
                res += (dist.reshape(NY, NX),)
        else :
            return None
        return res
    
    def interpolate_to_self_grid(self, LAT_ext, LON_ext, VAL_ext, interp="nearest_neighbor", max_dist_km=None):
        if interp in ["nearest_neighbor", "nn"]:
            NX, NY = self.get_data(["NX", "NY"])
            i_ext, mask, dist_deg = self.nearest_index_to_self_grid(LAT_ext, LON_ext, return_dist=True)
            res = VAL_ext.flatten()[mask][i_ext]
            if max_dist_km is not None and max_dist_km > 0 :
                LAT = self.get_data("LAT")
                # Approximative conversion from a distance in degrees to a distance in kilometers
                # But 1 longitude degree is not equal to 1 latitude degree, so we suppose an average
                dist_km = dist_deg * np.pi/180 * constants.EARTH_RADIUS/1000 * np.sqrt(0.5+0.5*np.sin(np.deg2rad(LAT))**2)
                res[dist_km > max_dist_km] = np.nan
            return res
        elif interp in ["linear"] :
            LAT = self.get_data("LAT")
            LON = self.get_data("LON")
            f = scipy.interpolate.LinearNDInterpolator(list(zip(LAT_ext.flatten(), LON_ext.flatten())), VAL_ext.flatten())
            return f(LAT, LON)
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html#scipy.interpolate.LinearNDInterpolator
        else :
            raise(Exception("Wrong interpolation method ( interp=", interp, "), Cubic may be resolved by scipy for structured grid"))
        
    def get_zoom_index(self, points):
        """
        Description
            return the indices iy1, iy2, ix1, ix2 to zoom horizontally in the domain
            These indices are then used to crop the data and read only the data contained in the zoom domain
        Parameters
            points : can be 
                list of 4 integers : already the index [iy1, iy2, ix1, ix2]
                list of 2 lists of 2 floats : the upper-left and the lower_right [[lat1, lon1], [lat2, lon2]]
                list of (list, float, float) : the coordinates of the center and the size [[lat_c, lon_c], LX, LY]
        Returns 
            int, int, int, int : iy1, iy2, ix1, ix2 : index of the nearest grid point
        """
        if type(points) is list :
            if len(points) == 4 : 
                iy1, iy2, ix1, ix2 = points
                iy1 = min(iy1, iy2, self.get_data("NY"))
                iy2 = max(iy1, iy2, 0)
                ix1 = min(ix1, ix2, self.get_data("NX"))
                ix2 = max(ix1, ix2, 0)
                iy1 = max(iy1-1, 0)
                iy2 = min(iy2+1, self.get_data("NY"))
                ix1 = max(ix1-1, 0)
                ix2 = min(ix2+1, self.get_data("NX"))
                return [iy1, iy2, ix1, ix2]
            elif len(points) == 3 or ( len(points)==2 and type(points[1]) is not list ): 
                center = points[0]
                LX = points[1]
                if len(points) == 2:
                    LY = LX
                else :
                    LY = points[2]
                NX, NY, DX, DY = self.get_data("NX"), self.get_data("NY"), self.get_data("DX"), self.get_data("DY") #HERE
                #get index of the center point
                iyc, ixc = self.nearest_index_from_self_grid([center])
                #convert list to integer
                iyc = iyc[0]
                ixc = ixc[0]
                #calculate the number of cells between center and corners
                diy = (0.5*LY)//DY 
                dix = (0.5*LX)//DX
                #limit the size of diy to stay inside the domain and keep the center point at the center
                diy = min(diy, NY - iyc, iyc)
                dix = min(dix, NX - ixc, ixc)
                #calculate iy1, iy2, ix1, ix2
                return int(iyc-diy), int(iyc+diy), int(ixc-dix), int(ixc+dix)
            elif len(points) == 2 :
                iy, ix = self.nearest_index_from_self_grid(points)
                return min(iy), max(iy), min(ix), max(ix)
        elif type(points) is tuple and len(points) == 4 :
            return points
        else :
            raise(Exception("error in dom.get_zoom_index, the key (" + str(points) + ") is not a list"))
            
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
            return y, x
        elif type(point) is tuple :
            return point
        else :
            raise(Exception("error in dom.get_point, the key " + str(point) + " is not a list or a tuple"))
        
    def get_points_for_vcross(self, points):
        """
        return the coordinates [[lat1, lon1], [lat2, lon2]] to use vertcross
        dom : the domain on which we want to compute vertcross
            
        points : something to deduce the coordinates, can be
            - already two coordinates pairs : the upper-left and the lower_right (a list of 2 lists of 2 reals) [[lat1, lon1], [lat2, lon2]]
            - The points index (a list of 4 integers) [iy1, iy2, ix1, ix2]
            - the coordinates of the center, the length and direction [[lat_c, lon_c], L, beta]
        """
        if len(points) == 2 :
            return points
        elif len(points) == 3 :
            center = points[0]
            print(self.prefix, center)
            c_lat, c_lon = center
            distance = points[1]
            direction = points[2]
            lat1, lon1 = manage_projection.inverse_haversine([c_lat, c_lon], distance/2, direction+math.pi)
            lat2, lon2 = manage_projection.inverse_haversine([c_lat, c_lon], distance/2, direction)
            return [[lat1, lon1], [lat2, lon2]]
        elif len(points) == 4 :
            LAT = self.get_data("LAT")
            LON = self.get_data("LON")
            [iy1, iy2, ix1, ix2] = points
            lat1 = LAT[iy1, ix1]
            lon1 = LON[iy1, ix1]
            lat2 = LAT[iy2, ix2]
            lon2 = LON[iy2, ix2]
            return [[lat1, lon1], [lat2, lon2]]
        
#################################################################################################################################
######  Dates
#################################################################################################################################
    
    def date2str(self, date, fmt):
        """
        Convert a datetime object to a formatted string 
        date : datetime : date to convert
        fmt : format to convert or name of the software
        01/02/2023 : Mathieu LANDREAU
        """
        if type(date) is datetime.datetime :
            if fmt == "ARPS" :
                first_date = self.get_data('INITIAL_TIME')
                delta = date - first_date
                date_str = str(int(delta.total_seconds())).zfill(6)
                if(len(date_str) > 6) : 
                    raise(Exception("error dom.date2itime : delta out of range : " + str(delta) + ", date_str = " + str(date_str)))
                return date_str
            elif fmt in ["WRF", "WRFinput"] :
                return date.strftime("%Y-%m-%d_%H:%M:%S")
            elif fmt == "WPS" :
                return date.strftime("%Y-%m-%d_%H:%M:%S") + ".nc" #need to add .nc extension
            else :
                return date.strftime(fmt)
        elif type(date) in [list, np.array, np.ndarray] :
            temp = []
            for date_i in date:
                temp.append(self.date2str(date_i, fmt))
            return temp
        elif type(date) is np.datetime64 :
            return self.date2str(manage_time.to_datetime(date_str), fmt)
        else : 
            raise(Exception("unknown type ("+  str(type(date)) + ") of date (" + str(date) + "), cannot convert to string"))
    
    def str2date(self, date_str, fmt):
        """
        Convert a formatted string to a datetime object
        date_str : string : to convert to datetime object 
        fmt : format to convert or name of the software
        01/02/2023 : Mathieu LANDREAU
        """
        if type(date_str) is datetime.datetime:
            return date_str
        elif type(date_str) is str :
            if fmt == "ARPS" :
                first_date = self.get_data('INITIAL_TIME')
                delta = datetime.timedelta(seconds=int(date_str))
                return first_date + delta
            elif fmt in ["WRF", "WRFinput"] :
                return datetime.datetime.strptime(date_str,"%Y-%m-%d_%H:%M:%S")
            elif fmt == "WPS" :
                return datetime.datetime.strptime(date_str[:-3],"%Y-%m-%d_%H:%M:%S") #need to remove .nc extension
            elif fmt == "AROME" :
                return datetime.datetime.strptime(date_str[:-4],"%Y%m%d%H%M%S") #need to remove .grb extension
            else :
                return manage_time.to_datetime(date_str,fmt)  
        elif type(date_str) in [list, np.array, np.ndarray] :
            temp = []
            for date_str_i in date_str:
                temp.append(self.str2date(date_str_i, fmt))
            return temp
        elif type(date_str) is np.datetime64 :
            return manage_time.to_datetime(date_str)
        else : 
            raise(Exception("unknown type (" + str(type(date_str)) + ") of date_str (" + str(date_str) + "), cannot convert to datetime"))
    
                   

#################################################################################################################################
######  Display informations about the domain
#################################################################################################################################
    
    def plot_mesh(self, Z=None, DZ_cell=None):
        if Z is None or DZ_cell is None : #using self domain
            HT = self.get_data("HT")
            pos = np.where(HT == np.min(HT))
            iy, ix = pos[0][0], pos[1][0]
            saved = {}
            crop = ("ALL", iy, ix)
            kw_get = {
                "crop" : crop, 
                "itime" : 0,
            }
            Z_ZSTAG = self.get_data("Z_ZSTAG", **kw_get)
            Z = self.get_data("Z", **kw_get)
            DZ_cell = np.diff(Z_ZSTAG) 
        n = len(Z)
        fig = plt.figure(figsize=[12, 8])
        plt.semilogy(range(1, n+1), Z, '+', label="Z_cell")
        plt.semilogy(range(1, n+1), DZ_cell, 'x', label="DZ_cell")
        plt.grid(which="both")
        plt.legend()
        plt.xlabel("$i_z$")
        plt.ylabel("$Z$ or $\Delta Z$ (m)")
        plt.xlim([0, plt.xlim()[1]])
        plt.title("Vertical mesh "+self.name)
        return Z, DZ_cell, fig

    def generate_mesh(self, dzbot, dzstretch_s, dzstretch_u, max_dz=1000, z_lim1=2000, eta_c=0.2, Ptop=5000, P0=1e5):
        if Ptop != self.get_data("P_TOP")[0] :
            raise(Exception("error : your top dry hydrostatic pressure (" + str(Ptop) + ") is not equal to this domain top dry hydrostatic pressure (" + str(self.get_data("P_TOP")[0]) + ") : cannot afford the modification, please adapt the script"))
        
        #---Polynome coefficient of B(eta) : see Skamarock 2019, A Description of the Advanced Research WRF Model Version 4 (eq. 2.5)
        denom = (1-eta_c)**3
        c1 = 2*eta_c**2/denom
        c2 = -eta_c*(4+eta_c+eta_c**2)/denom
        c3 = 2*(1+eta_c+eta_c**2)/denom
        c4 = -(1+eta_c)/denom

        #---Find a point above the sea and get the vertical mesh to create the new one
        HT = self.get_data("HT")
        pos = np.where(HT == np.min(HT))
        iy, ix = pos[0][0], pos[1][0]
        kw_get = {
            "itime" : 0,
            "crop" : ("ALL", iy, ix),
        }
        PIDs_m_PIDtop_in = self.get_data("MUT", **kw_get) #MU_d = Surface hydrostatic pressure - top hydrostatic pressure = PID_surface - PID_top
        Z_ZSTAG_in = self.get_data("Z_ZSTAG", **kw_get)
        ETA_ZSTAG_in = self.get_data("ETA_ZSTAG")
        PID_ZSTAG_in = self.get_data("PID_ZSTAG", **kw_get)
        z_top = Z_ZSTAG_in[-1]
        
        #---Creating the new expected Z vector
        #1- Initial surface values
        Z_ZSTAG_out = [0, dzbot]
        Z_out = [dzbot/2]
        DZ_out = [dzbot]
        
        #2- expend dz with a constant stretch (dzstretch_s) between z=0 and z=zlim1 
        while Z_ZSTAG_out[-1] < z_lim1:
            DZ_out.append(DZ_out[-1]*dzstretch_s)
            Z_out.append(Z_ZSTAG_out[-1] + 0.5*DZ_out[-1])
            Z_ZSTAG_out.append(Z_ZSTAG_out[-1] + DZ_out[-1])
        
        #3-Adjusting dzstretch_u to stop exactly at z_top if ever max_dz is not reached
        DZ_rest = z_top - Z_ZSTAG_out[-1]
        S = DZ_rest/DZ_out[-1] #Result of the quadratic sum 1+dzstretch_u+dzstretch_u**2+....+dzstretch_u**(n_rest)
        #S = (1-dzstretch_u**(nrest+1)) / (1-dzstretch_u)
        n_rest = int((np.log(1 + S*(dzstretch_u-1))/np.log(dzstretch_u))) - 1
        if DZ_out[-1] * dzstretch_u**(n_rest-1) < max_dz : 
            pol = np.zeros(n_rest+2)
            pol[0] = 1
            pol[-2] = - S
            pol[-1] = S-1
            root = np.roots(pol)
            print(n_rest)
            root_list = []
            for r in root :
                if np.abs(np.imag(r)) > 1e-10 :
                    continue
                if np.real(r) < 1.00001 :
                    continue
                root_list.append(np.real(r))
            if len(root_list) == 0 :
                print("error : no root has been found in : ", root)
            elif len(root_list) == 1 :
                dzstretch_rest = root_list[0]
            else :
                i = np.argmin(np.abs(np.array(root_list)-dzstretch_u))
                dzstretch_rest = root_list[i]
            print("dzstretch_u = ", dzstretch_rest)
        else :
            dzstretch_rest = dzstretch_u
            
        #4-expend dz with a constant stretch (dzstretch_u) above z=zlim1 until z=z_top or dz = max_dz    
        while Z_ZSTAG_out[-1] < z_top-1 and DZ_out[-1]*dzstretch_rest < max_dz:
            DZ_out.append(DZ_out[-1]*dzstretch_rest)
            Z_out.append(Z_ZSTAG_out[-1] + 0.5*DZ_out[-1])
            Z_ZSTAG_out.append(Z_ZSTAG_out[-1] + DZ_out[-1])
        
        #5-if z_top is not reached, continue with a constant dz near max_dz
        DZ_rest = z_top - Z_ZSTAG_out[-1]
        nz_rest = DZ_rest//max_dz
        dz_top = DZ_rest/nz_rest
        print("dz_top = ", dz_top)
        while Z_ZSTAG_out[-1] < z_top-1 : 
            DZ_out.append(dz_top)
            Z_out.append(Z_ZSTAG_out[-1] + 0.5*DZ_out[-1])
            Z_ZSTAG_out.append(Z_ZSTAG_out[-1] + DZ_out[-1])

        #-- Finding ETA_ZSTAG levels from expected Z_ZSTAG using the (Z_ZSTAG, ETA_ZSTAG) of the current domain 
        #interpolation log(P) from Z since the equation is near linear
        logPID_ZSTAG_out = np.interp(Z_ZSTAG_out, Z_ZSTAG_in, np.log10(PID_ZSTAG_in))
        PID_ZSTAG_out = np.power(10, logPID_ZSTAG_out)
        nz_out = len(Z_ZSTAG_out)
        ETA_ZSTAG_out = np.zeros(nz_out)
        #Calculating ETA_ZSTAG from PID_ZSTAG for each level with eq. 2.2, 2.3 from Skamarock 2017
        top_part = False # top_part is True when ETA < eta_c
        for i in range(nz_out):
            Pd = PID_ZSTAG_out[i]
            k1 = (Pd - Ptop)/PIDs_m_PIDtop_in
            k2 = (P0 - Ptop)/PIDs_m_PIDtop_in
            if not top_part :
                pol = [c4*(1-k2), c3*(1-k2), c2*(1-k2) + k2, c1*(1-k2)-k1]
                roots = np.roots(pol)
                ETA_ZSTAG_out[i] = 2
                for r in roots :
                    if r >= 0 and r <=1 :
                        ETA_ZSTAG_out[i] = r
                if ETA_ZSTAG_out[i] < eta_c: #for ETA < eta_c, B(ETA)=0, then ETA = (PID-Ptop)/(P0-Ptop)
                    ETA_ZSTAG_out[i] = k1/k2
                    top_part = True
                if ETA_ZSTAG_out[i] > 1 :
                    print("error for i = ", i, ", roots = ", roots, ", Pd = ", Pd)
            else :
                ETA_ZSTAG_out[i] = k1/k2
                if ETA_ZSTAG_out[i] > 1 or ETA_ZSTAG_out[i] < 0 :
                    print("error for i = ", i, ", ETA_ZSTAG_out[i] = ", ETA_ZSTAG_out[i], ", Pd = ", Pd)

        fig = plt.figure(figsize=[24, 8])
        plt.subplot(121)
        plt.plot(ETA_ZSTAG_in,'+', label="old Eta")
        plt.plot(ETA_ZSTAG_out,'x', label="new Eta")
        plt.grid(which="both")
        ETA_ZSTAG_out[0] = 1.000
        ETA_ZSTAG_out[-1] = 0.000
        print(nz_out)
        manage_display.print_arr(ETA_ZSTAG_out)
        return Z_ZSTAG_out, Z_out, DZ_out, ETA_ZSTAG_out, fig
 
    def display(self, pref = ""):
        """
        Display the domain variables
        pref : str : prefix printed before the each line
        02/02/2023 : Mathieu LANDREAU 21/12/22
        """
        print("domaine", self.i_str, " (", self.software, ") :")
        pref = pref + "  "
        for k in self.VARIABLES :
            self.VARIABLES[k].display(pref+k.ljust(20)+':')
        print("") 
 

    
    
    
    
  
