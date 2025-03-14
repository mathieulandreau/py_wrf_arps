#!/usr/bin/env python3

from .class_variable import Variable
from ..lib import manage_list
from netCDF4 import Dataset
from multiprocessing import Pool, cpu_count
import numpy as np
import pygrib
import time as python_time

debug = False

class VariableRead(Variable):
    
    print_read = False
    
    def __init__(self, legend_short, legend_long, latex_units, units, dim, dataset_dict, is_time_file, var_key, file=None, cmap=0, keep_open=False, fmt="NETCDF", shape=None, index=None, i_sweep=None):
        """
        Description :
            Object that describe a variable in netcdf or grib file and can extract the data from it
        Parameters :
            self: VariableRead
            legend_short : str : a short name (see ARPS_WRF_VARNAME_DICT)
            legend_long : str : a long name (see ARPS_WRF_VARNAME_DICT)
            latex_units : str : a units written with LaTex notation (m^2, ...) (see ARPS_WRF_VARNAME_DICT)
            units : str : a units written without LaTex notation (m2, ...) (see ARPS_WRF_VARNAME_DICT)
            dim : integer : the number of dimension of the data (see ARPS_WRF_VARNAME_DICT)
            dataset_dict : dictionary : contains all the filenames where the variable can be read (one per timestep for the moment) (see Dom.get_output_filenames)
            is_time_file : logical : False for static data and True for dynamic data
            var_key : str : key of the variable in the file (see ARPS_WRF_VARNAME_DICT) : it is different than the name of the variable in order to compare the same variable in different codes (WRF-ARPS-AROME-...)
        Optional :
            file : opened file (netcdf or grib) : to avoid reopening it to find supplementary informations (see Dom.init_variables, Dom.init_raw_variables)
            cmap : int : a code for the colormap used in 2D plot for this variable (see ARPS_WRF_VARNAME_DICT)
            keep_open : logical : True if the files stay opened, False if they are opened each time we want to read them
            fmt : str : for the moment either "NETCDF" (for WRF, ARPS, WPS, WRFinput) or "GRIB" (for AROME)
            shape : tuple : the spatial shape of the data (NZ, NY, NX) for a 3D data, the time dimension is not taken into account here
        Author(s)
            06/01/2023 : Mathieu Landreau
        """  
        super().__init__(legend_short, legend_long, latex_units, units, dim, cmap=cmap)
        self.dataset_dict = dataset_dict
        self.is_time_file = is_time_file
        self.var_key = var_key
        self.cut_first_dim = False
        self.fmt = fmt
        self.keep_open = keep_open
        self.shape = shape
        self.index = index
        self.i_sweep = i_sweep
        
        if self.fmt == "NETCDF" : 
            keep_open_this_time = False
            if file is None :
                filename = next(iter(self.dataset_dict.items()))[0]
                if self.keep_open :
                    file = self.dataset_dict[filename]
                    keep_open_this_time = True
                else :
                    file = Dataset(filename, "r")
            else :
                keep_open_this_time = True
            if self.i_sweep is not None :
                sweep_name = list(file.groups.keys())[self.i_sweep]
                file_var = file[sweep_name][self.var_key]
            else :
                file_var = file[self.var_key]
            self.shape = file_var.shape
            self.dim = len(self.shape)
            if self.dim is None : #it means this is a raw variable directly read in file
                try :
                    self.units = file_var.units
                except:
                    self.units = " "    

            self.latex_units = self.units
            if len(self.shape) > 1 and self.shape[0] == 1:
                self.shape = self.shape[1:]
                self.dim = self.dim - 1
                self.cut_first_dim = True
            #search if a dimension is stagged
            self.stag_dim = 0
            if not keep_open_this_time :
                file.close()
        elif self.fmt == "GRIB" : # AROME
            if self.shape is None :
                close_after = False
                if file is None : 
                    filename = next(iter(self.dataset_dict.items()))[0]
                    file = pygrib.open(file)
                    close_after = True
                self.shape = file[1].values.shape
                if close_after :
                    file.close()
            if self.dim is None : # voir domAROME.AROME_VARNAMES
                if type(self.index) is int : 
                    self.dim = 2
                elif type(self.index) is list : 
                    nz = self.index[1] + 1 - self.index[0]
                    self.shape = (nz,) + self.shape
                else :
                    print("error in VariableRead.__init__ : for GRIB files, index must be of type int or list, not : ", type(self.index))
                    raise
        else :
            print("error in class_variableRead.__init__ : cannot manage file format : ", self.fmt)
            raise
    
    
    def get_data(self, time_slice = None, crop=None, i_unstag=None, n_procs=10):
        """
        Description :
            routine to read the data from netcdf or grib file
            it is called in Dom.get_data (more informations there)
        Parameters :
            time_slice : slice, int or list : which history timesteps do we want to read ? see Dom.get_data and Variable.prepare_crop_unstag for more informations
            crop : see Dom.get_data : which spatial area do we want to read ?
            i_unstag : integer : if ever we want to unstag a specific dimension (in WRF the velocity fields are staggered in Arakawa C-grid, with unstag, we move them in center of the cells)
        Author(s)
            06/01/2023 : Mathieu Landreau
            06/03/2024 : adding GRIB to read AROME
        """  
        if self.cut_first_dim :
            #in WRF even if there a file per history timestep, the shape of the dynamic variables are (1, NZ, NY, NX)
            #We want to cut this first dimension
            tab_slice = (0,)
            count_dim = 1
        else :
            tab_slice = ()
            count_dim = 1
        #tab_slice contains the slice to crop data in space
        tab_slice, dim_unstag, squeeze_dim_unstag, shape = self.prepare_crop_unstag(crop, count_dim, tab_slice, i_unstag, self.shape)
        if debug : print(self.var_key , ", tab_slice : ", tab_slice)
        
        if type(time_slice) in [int, np.int64] or time_slice is None  :
            nt = 1
        elif type(time_slice) is slice :
            nt = len(range(*time_slice.indices(10000)))
        else :
            nt = len(time_slice)
            #print("tab_slice : ", tab_slice)
        #if several history timestep, loop over the files
        data = np.zeros((nt,)+ shape)
        if self.is_time_file and time_slice is not None:
            filename_list = np.array(list(self.dataset_dict.keys()))[time_slice]
            if type(filename_list) in [str, np.str_] :
                filename_list = [filename_list]
            #loop over the files
            n_procs = min(int(nt//5) + 1, n_procs)
            if n_procs == 1 :
                for it in range(nt) :
                    data[it] = self.get_single_file_data(filename_list[it], tab_slice)
            elif n_procs > 1 :
                try :
                    t1 = python_time.time()
                    arg_list = [(filename_list[it], tab_slice) for it in range(nt)]
                    with Pool(processes=n_procs) as pool:
                        data[:] = pool.starmap(self.get_single_file_data, arg_list)
                    t2 = python_time.time()-t1
                    if int(t2)>1 : 
                        print("get with Pool : ", t2, ", n_procs = ", n_procs)
                except :
                    t1 = python_time.time()
                    print("warning in VariableRead.get_data : no parallel")
                    for arg in arg_list :
                        self.get_single_file_data(arg[0], arg[1], arg[2], arg[3])
                    print("get without Pool : ", python_time.time()-t1)
            else :
                print("invalid n_procs value : ", n_procs)
        #if static data or only the first history timestep :
        else :
            #open file
            filename = next(iter(self.dataset_dict.items()))[0]
            data[:] = self.get_single_file_data(filename, tab_slice)
        #unstag along a dimension if required, should not happen with AROME
        if dim_unstag is not None :
            data = manage_list.unstag(data, dim_unstag)
            #squeeze the dimension if needed
            if squeeze_dim_unstag :
                data = np.squeeze(data, axis = dim_unstag)
        #squeeze time dimension if only one time
        if nt == 1 :
            data = data[0]
        #duplicate the data along time dimension if it is static but many timestep are asked (necessary for doing calculation with static and dynamic data)
        if nt > 1 and not self.is_time_file: #should not happen with AROME
            for it in range(1, nt):
                data[it] = data[0]
        #The missing data value I used in grib files is -1e10, so I convert it to nan
        if self.fmt == "GRIB" : 
            data[data < -1e9] = np.nan
        return data
    
    
    def get_single_file_data(self, filename, tab_slice):
        if self.keep_open : file = self.dataset_dict[filename]
        elif self.fmt == "NETCDF" : file = Dataset(filename, "r")
        elif self.fmt == "GRIB" : file = pygrib.open(filename)
        #read in the file    
        if self.fmt == "NETCDF" : 
            if self.i_sweep is not None :
                sweep_name = list(file.groups.keys())[self.i_sweep]
                file_var = file[sweep_name][self.var_key]
                out = file[sweep_name].variables[self.var_key][tab_slice]
            else :
                out = file.variables[self.var_key][tab_slice]
        elif self.fmt == "GRIB"  :
            if type(self.index) is int :
                data_grib = file[self.index].values
            elif type(self.index) is list : 
                i1 = self.index[0]
                i2 = self.index[1] + 1
                data_grib = []
                for iz in range(i1, i2) :
                    data_grib.append(file[iz].values)
                data_grib = np.array(data_grib)
            out = data_grib[tab_slice]
        else :
            print("error in class_variableRead.get_single_file_data : cannot manage file format : ", self.fmt)
            raise
        #close the file
        if not self.keep_open :
            file.close()
        return np.array(out)
    
    
    def display(self, pref=""):
        print(pref + 'read ' + self.legend_long.ljust(40) + ' ' + str(self.dim).ljust(1) + ' ' + str(self.units) + ' ' + next(iter(self.dataset_dict.items()))[0] + ' ' + str(self.is_time_file))

    
