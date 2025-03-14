from . import *
from ..WRF_ARPS.ARPS_WRF_VARNAMES import ARPS_WRF_VARNAMES_DICT
from ..lib import manage_time, manage_display

import numpy as np

class Expe():
    def __init__(self):
        self.get_params_from_dict()
        self.get_output_filenames()
        self.get_other_params()
    
    def get_params_from_dict(self):
        temp = DICT_EXPE_DATE[self.code]
        self.index = temp[0]
        self.name = self.code
        self.longname = temp[1]
        self.position = temp[2]
        self.lat = temp[2][0]
        self.lon = temp[2][1]
        self.variables_list = temp[3]
        self.i_str = self.code
        
    def nearest_z_index(self, z_in, itime=0, crop=("ALL", 0, 0), return_Z=False, return_diff=False):
        Z_vec = self.Z_vec
        diff = np.abs(z_in - Z_vec)
        iz = np.argmin(diff)
        result = (int(iz),)
        if return_Z :
            result += (Z_vec[iz],)
        if return_diff :
            result += (diff[iz],)
        return result
    
    def get_other_params(self):
        pass
    
    def get_output_filenames(self):
        pass
    
    def get_data(self, varname, itime = None, time_slice = None, crop = None, **kwargs):
        pass
    
    def get_legend(self, varname, long=True, latex=True):
        if varname in ARPS_WRF_VARNAMES_DICT :
            temp = ARPS_WRF_VARNAMES_DICT[varname]
            legend = temp[0] #legend_short
            units = temp[3]
            return manage_display.get_legend(legend, units, latex=True)
        else :
            return varname
    
    def get_cmap(self, varname, *args, **kwargs) :
        """
        Get colormap of a variable
        varname : str : name of the variable
        return : int : category of the variable
        01/04/2023 : Mathieu LANDREAU
        """
        if varname in ARPS_WRF_VARNAMES_DICT :
            return ARPS_WRF_VARNAMES_DICT[varname][5]
        else :
            return 0
    
    def get_locInfo(self) :
        return LocationInfo(self.name, "France", "Europe/London", self.lat, self.lon)
    
    def get_label(self) :
        return self.name
    
    def get_ZT(self, itime="ALL_TIMES", crop=None):
        pass
    
    def date2str(self, date, fmt="%Y-%m-%d_%H:%M:%S") :
        if fmt == "WRF" : 
            fmt = "%Y-%m-%d_%H:%M:%S"
        return manage_time.date_to_str(date, fmt)
        
