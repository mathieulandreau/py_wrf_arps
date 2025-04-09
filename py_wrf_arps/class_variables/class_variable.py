#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
from ..lib import manage_display


class Variable():

    legend_is_in_latex = True
    legend_is_short = True

    def __init__(self, legend_short, legend_long, latex_units, units, dim, cmap = 0):
        """Describe a variable
        Parameters
            self (Variable)
            legend_short (str): short variable name
            legend_long (str): long variable name
            latex_units (str): latex units
            units (str): units
            dim (int): number of dimensions
        Optional
            cmap (int): colormap value, default=0
        Author(s)
            06/01/2023 : Mathieu Landreau
        """  
        self.legend_short = legend_short
        self.legend_long = legend_long
        self.units = units
        self.latex_units = latex_units
        self.dim = dim
        self.is_time_file = False
        self.cmap = cmap
        
    
    def prepare_crop_unstag(self, crop, count_dim, tab_slice, i_unstag, old_shape) :
        """
        crop can be
            int, np.int64 : index in the axis
            list : first and last index in the axis
            "ALL"
        """
        error_dims = False
        if type(crop) is tuple and len(crop) != len(old_shape) :
            if len(old_shape) == 4 : #Assuming it is (?, z, y, x)
                crop_temp = crop
                if len(crop_temp) > 4 :
                    crop = (crop_temp[-4], crop_temp[-3], crop_temp[-2], crop_temp[-1])
                elif len(crop_temp) == 3:
                    crop = ("ALL", crop_temp[-3], crop_temp[-2], crop_temp[-1])
                else :
                    error_dims = True
                
            elif len(old_shape) == 3 : #Assuming it is (z, y, x)
                crop_temp = crop
                if len(crop_temp) > 3 :
                    crop = (crop_temp[-3], crop_temp[-2], crop_temp[-1])
                else :
                    error_dims = True
            elif len(old_shape) == 2 : #Assuming it is (y, x)
                crop_temp = crop
                crop = (crop_temp[-2], crop_temp[-1])
            elif len(old_shape) == 1 : #Assuming it is (z)
                crop_temp = crop
                crop = (crop_temp[-3],)
            elif len(old_shape) == 0 :
                crop = ()
            else :
                error_dims = True
                
            if error_dims :
                print("error in Variable.prepare_crop_unstag : crop = ", crop, ", while old_shape = ", old_shape)
                raise
            
        shape = ()         
        squeeze_dim_unstag = False
        dim_unstag = None
        if crop is not None :
            for itemp, temp in enumerate(crop) :
                ## Unstag this axis : we need to get one more cell in the specified axis because 1 cell "disappear" when calling np.diff to unstag
                if itemp == i_unstag :
                    dim_unstag = count_dim
                    if type(temp) in [int, np.int64] :
                        shape += (2,)
                        tab_slice += (slice(temp,temp+2),)
                        count_dim+=1
                        squeeze_dim_unstag = True
                    elif type(temp) is list :
                        shape += (temp[1]+1 - temp[0],)
                        tab_slice += (slice(temp[0], temp[1]+1),)
                        count_dim+=1
                    elif type(temp) is str and temp == "ALL" :
                        shape += (old_shape[itemp],)
                        tab_slice += (slice(old_shape[itemp]+1),)
                        count_dim+=1
                    else :
                        print("error in VariableRead.get_data : crop must be a list of string or a list of list, temp = ", temp, type(temp), ", crop = ", crop)
                        raise
                ## Normal
                else :
                    if type(temp) in [int, np.int64]:
                        #shape lose a dimension
                        tab_slice += (temp,)
                    elif type(temp) is list :
                        shape += (temp[1] - temp[0],)
                        tab_slice += (slice(temp[0], temp[1]),)
                        count_dim+=1
                    elif type(temp) is str and temp == "ALL" :
                        shape += (old_shape[itemp],)
                        tab_slice += (slice(old_shape[itemp]+1),)
                        count_dim+=1
                    else :
                        print("error in VariableRead.get_data : crop must be a list of string or a list of list, temp = ", temp, type(temp), ", crop = ", crop)
                        raise
        else :
            for itemp, temp in enumerate(old_shape) :
                shape += (temp,)
                tab_slice += (slice(temp+1),)
                if itemp == i_unstag :
                    dim_unstag = count_dim
                count_dim+=1
        return tab_slice, dim_unstag, squeeze_dim_unstag, shape
    
    def get_data(self, *args, **kwargs):
        return None
        
    def get_units(self, latex=True) :
        return manage_display.get_units(self.units, latex=latex)
    
    def get_legend(self, units=None, long=False, latex=True):
        if units is None :
            units = self.units
        if not long :
            legend = self.legend_short
        else:
            legend = self.legend_long
        return manage_display.get_legend(legend, units, latex=latex)
    
    def get_cmap(self) : 
        return self.cmap
    
    def display(self, pref=""):
        print(pref +'calc ' + self.legend_long.ljust(40) + ' ' + str(self.dim).ljust(1) + ' ' + str(self.units))

    