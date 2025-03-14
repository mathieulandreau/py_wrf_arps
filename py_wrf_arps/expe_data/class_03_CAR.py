import sys

from .class_expe import Expe
from .expe_dict import SONIC_VARNAMES
from ..lib import manage_time, manage_angle, manage_path

import numpy as np
import netCDF4
import os
import netCDF4

rejected_files = []
debug = False

class CAR(Expe):
    code = "CAR"
    name = "CAR"
    frequence = "10min"
    fmt = "%Y-%m-%d_%H-%M-%S"
    def __init__(self):
        super().__init__()
        
    def get_output_filenames(self):
        folder_list = manage_path.CAR_folder_list 
        for folder in folder_list :
            if os.path.exists(folder) :
                self.folder = folder
        self.postprocdir = self.folder + "post/"
        if not os.path.exists(self.postprocdir):
            print('Create postproc directory in {0}'.format(self.postprocdir))
            os.makedirs(self.postprocdir, exist_ok=True)
                
                
        filename_list = []
        for filename in os.listdir(self.folder):
            if filename not in rejected_files and filename.startswith("cardinaux"):
                filename_list.append(self.folder+filename)
        filedate_list = []
        for f in filename_list :
            filedate_str = f[84:-3]
            if len(filedate_str) == 6 :
                filedate_str = filedate_str[:5]+"0"+filedate_str[5:]
            filedate_list.append(filedate_str)
        ags = np.argsort(filedate_list)
        self.filename_list = np.array(filename_list)[ags]
        
        # filename_list = []
        # for filename in os.listdir(self.postprocdir):
        #     if filename not in rejected_files and filename.startswith(self.name+"_post"):
        #         filename_list.append(self.postprocdir+filename)
        # self.postproc_filename_list = np.sort(filename_list)
            
    def get_other_params(self):
        date1970 = np.array(manage_time.to_datetime("1970-01-01"))
        self.date_list = []
        self.file_for_date_list = []
        self.date_in_file_list = []
        for i, fn in enumerate(self.filename_list) :
            with netCDF4.Dataset(fn) as f :
                temp = np.array(f[self.frequence]["TIMESTAMP"][:])
                temp2 = manage_time.to_timedelta(temp, "s")
                date_list_i = date1970 + temp2
                file_for_date_list_i = np.ones(date_list_i.shape, dtype=int) * i
                date_in_file_list_i = np.arange(len(date_list_i), dtype=int)
                self.date_list = np.concatenate((self.date_list, date_list_i))
                self.file_for_date_list = np.concatenate((self.file_for_date_list, file_for_date_list_i)).astype(int)
                self.date_in_file_list = np.concatenate((self.date_in_file_list, date_in_file_list_i)).astype(int)
        self.NT = len(self.date_list)
        self.max_time_correction = np.min(np.diff(self.date_list))
        self.prefix = ""
                
        
    def init_variables(self):
        self.VARIABLES = {}
    
    def get_data(self, varname, itime=None, time_slice=None, n_procs=1, saved=None, **kwargs):
        if saved is None : 
            saved = {}
        if varname in saved : 
            return saved[varname]
        self.prefix += "|"
        if time_slice is None :
            time_slice = manage_time.get_time_slice(itime, self.date_list, self.max_time_correction)
            time_slice = np.arange(self.NT)[time_slice]
        if varname in SONIC_VARNAMES :
            if debug : print(self.prefix, "read", varname)
            file_for_date = self.file_for_date_list[time_slice]
            date_in_file = self.date_in_file_list[time_slice]
            data = self.read(varname, time_slice, file_for_date, date_in_file)
        elif varname == "TIME":
            data = np.array(self.date_list[time_slice])
        else :
            data = self.calculate(varname, itime=itime, time_slice=time_slice, n_procs=n_procs, saved=saved, **kwargs)
        saved[varname] = data
        self.prefix = self.prefix[:-1]
        return data
    
    def calculate(self, varname, **kwargs) :
        if varname in ["WD180"]:
            return manage_angle.angle180(self.get_data("WD", **kwargs))
        elif varname in ["TKE"]:
            return self.get_data("M2U", **kwargs) + self.get_data("M2V", **kwargs) + self.get_data("M2W", **kwargs)
        else :
            raise(Exception("error in class_34_LI1.get_data, the variable name : " + str(varname) + " doesn't exist"))
    
    def read(self, varname, time_slice, file_for_date, date_in_file):
        varkey = SONIC_VARNAMES[varname]
        if type(time_slice) in [int, np.int64] :
            with netCDF4.Dataset(self.filename_list[file_for_date], "r") as f :
                out = f[self.frequence][varkey][:][date_in_file]
        else :
            ifile_unique, index, counts = np.unique(file_for_date, return_index=True, return_counts=True)
            out = np.zeros(time_slice.shape)
            for i, ifile in enumerate(ifile_unique.astype(int)) :
                ind0 = index[i]
                ind1 = ind0 + counts[i]
                time_slice_i = date_in_file.astype(int)[ind0:ind1]
                with netCDF4.Dataset(self.filename_list[ifile], "r") as f :
                    out[ind0: ind1] = f[self.frequence][varkey][:][time_slice_i]
        return out
    
    def get_label(self) :
        return "Sonic Cardinaux"
        