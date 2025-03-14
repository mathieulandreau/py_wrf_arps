#!/usr/bin/env python3

from netCDF4 import Dataset
from .class_variable import Variable
from ..lib import manage_list
import numpy as np

debug = False
class VariableSave(Variable):

    def __init__(self, legend_short, legend_long, latex_units, units, dim, value = None, cmap=0):
        """Describe a variable
        Parameters
            self (Variable)
            legend_short (str): short variable name
            legend_long (str): long variable name
            latex_units (str): latex units
            units (str): units
            dim (int): number of dimensions
        Optional
            value: the saved value of the variable, default=None
            cmap (int): colormap value, default=0
        Author(s)
            06/01/2023 : Mathieu Landreau
        """  
        super().__init__(legend_short, legend_long, latex_units, units, dim, cmap=cmap)
        self.value = value
    
    
    def get_data(self, time_slice=None, crop=None, i_unstag=None, **kwargs):
        if type(time_slice) in [int, np.int64] or time_slice is None  :
            nt = 1
        elif type(time_slice) is slice :
            nt = len(range(*time_slice.indices(10000)))
        else :
            nt = len(time_slice)
            
        if type(self.value) in [np.array, np.ma.core.MaskedArray, np.ndarray] :
            tab_slice = ()
            count_dim = 0
            tab_slice, dim_unstag, squeeze_dim_unstag, shape = self.prepare_crop_unstag(crop, count_dim, tab_slice, i_unstag, self.value.shape)
            if debug : print('tab_slice : ', tab_slice)
            data = self.value[tab_slice]
            if dim_unstag is not None :
                data = manage_list.unstag(data, dim_unstag)
                if squeeze_dim_unstag :
                    data = np.squeeze(data, axis = dim_unstag)
            if nt > 1 :
                data_temp = np.copy(data)
                shape = (nt, ) + data.shape
                data = np.zeros(shape)
                for it in range(nt):
                    data[it] = data_temp
        else :
            data = self.value
        if type(data) in [np.array, np.ndarray] :
            return np.copy(data)
        else :
            return data
    
    def display(self, pref=""):
        print(pref + 'save ' + self.legend_long.ljust(40) + ' ' + str(self.dim).ljust(1) + ' ' + str(self.units))

    
