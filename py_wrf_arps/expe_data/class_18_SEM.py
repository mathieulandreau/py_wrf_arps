import sys

from .class_expe import Expe
from ..lib import manage_time, manage_path

import numpy as np
import pandas as pd
import os

class SEM(Expe):
    """
    Dates are written in different formats
    """
    
    code = "SEM"
    def __init__(self):
        super().__init__()
        
    def get_output_filenames(self):
        folder_list = manage_path.SEM_folder_list
        for folder in folder_list :
            if os.path.exists(folder + 'Temp_surf_houlographe_SEMREV.csv') :
                self.folder = folder
        self.filename = self.folder + 'Temp_surf_houlographe_SEMREV.csv'
        # self.postprocdir = self.folder + "post/"
        # if not os.path.exists(self.postprocdir):
        #     print('Create postproc directory in {0}'.format(self.postprocdir))
        #     os.makedirs(self.postprocdir, exist_ok=True)
        # self.postproc_filename = self.postprocdir + 'SEM_post.nc'
        
    def get_other_params(self):
        df = pd.read_csv(self.filename, header=None)
        date_list = df[0].values
        m = ["/" in s for s in date_list]
        date_list[m] = pd.to_datetime(date_list[m], format="%d/%m/%Y %H:%M")
        date_list[np.logical_not(m)] = pd.to_datetime(date_list[np.logical_not(m)])
        date_list = pd.Series(date_list).dt.to_pydatetime()
        T = np.array(df[1].replace(',', '.', regex=True), dtype=float)
        self.date_list = np.array(date_list)
        self.NT = len(self.date_list)
        self.DT = self.date_list[1] - self.date_list[0]
        self.max_time_correction = manage_time.to_datetime64(self.date_list[1]) - manage_time.to_datetime64(self.date_list[0])
        self.T = T
    
    def get_data(self, varname, itime="ALL_TIMES", time_slice=None,  **kwargs):
        if time_slice is None :
            time_slice = manage_time.get_time_slice(itime, self.date_list)
        if varname == "T":
            return self.T[time_slice]
        elif varname == "TIME":
            return self.date_list[time_slice]
        else :
            return self.calculate(varname, itime=itime, time_slice=time_slice, **kwargs)
       
    def calculate(self, varname, **kwargs) :
        raise(Exception("error in class_18_SEM.get_data, the variable name : " + str(varname) + " doesn't exist"))
    
    def get_label(self) :
        return "SEM-REV buoy"