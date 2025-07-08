import sys

from .class_expe import Expe
from ..lib import manage_time, manage_angle, manage_path

import numpy as np
import os
import pandas

class Expe_MF(Expe):
    #code Meteo France for each location
    postes_dict = {
        "NAZ" : 44103001,
        "CHE" : 44184001, 
        "TAL" : 56009001,
        "YEU" : 85113001,
        "NOI" : 85163001,
        "NOE" : 35202001,
    }
    
    def __init__(self, code):
        self.code = code
        super().__init__()
        
    def get_output_filenames(self):
        self.filename_list = []
        folder_list = manage_path.MF_folder_list
        for folder in folder_list :
            if os.path.exists(folder) :
                self.folder = folder
        self.count = 0
        for filename in os.listdir(self.folder):
            self.filename = self.folder+filename
            self.filename_list.append(self.filename)
            self.count += 1
        if self.count != 1 :
            print("warning in Expe_MF : There is not a single file in MeteoFrance folder, there is ,", self.count, "file(s)")
            
    def get_other_params(self):
        self.poste = self.postes_dict[self.code]
        print(self.poste)
        print(self.filename_list)
        for filename in self.filename_list :
            df = pandas.read_csv(filename, sep=";")
            df = df[df["POSTE"] == self.poste]
            print(len(df))
            if len(df) > 0 :
                self.filename = filename
                break
        self.df = df
        DATE = np.array(df["DATE"]).astype("str")
        self.date_list = np.array(manage_time.to_datetime(DATE, fmt="%Y%m%d%H"))
        self.NT = len(self.date_list)
        self.max_time_correction = manage_time.to_datetime64(self.date_list[1]) - manage_time.to_datetime64(self.date_list[0])
        self.PSFC = np.array(df["PSTAT"])*100 #convert from hPa to Pa
        self.T2_C = np.array(df["T"])
        self.MH10 = np.array(df["FF"])
        self.WD10 = manage_angle.anglerad(np.array(df["DD"]))
    
    def get_data(self, varname, itime="ALL_TIMES", time_slice=None, crop=None, **kwargs):
        if time_slice is None :
            time_slice = manage_time.get_time_slice(itime, self.date_list)
        
        if varname in ["TIME"]:
            return self.date_list[time_slice]
        elif varname in ["T2_C", "T2_AVG_C"]:
            return self.T2_C[time_slice]
        elif varname in ["T2"]:
            T2_C = self.get_data("T2_C", itime=itime, time_slice=time_slice)
            return Celsius2Kelvin(T2_C)
        elif varname in ["PSFC"]:
            return self.PSFC[time_slice]
        elif varname in ["MH10", "MH"]:
            return self.MH10[time_slice]
        elif varname in ["WD10", "WD"]:
            return self.WD10[time_slice]
        elif varname in ["U10", "U"]:
            return self.get_data("U_AD270", itime=itime, time_slice=time_slice)
        elif varname in ["V10", "V"]:
            return self.get_data("V_AD270", itime=itime, time_slice=time_slice)
        elif varname.startswith("U_AD") or varname.startswith("U_AVG_AD"):
            #like U_AD50, U_AD220 : rotation of U and V with an angle of angle_deg ° from North
            # NOTE : U_AD270 = U, V_AD270 = V = U_AD180
            if varname.startswith("U_AD") : angle_deg = int(varname[4:])
            if varname.startswith("U_AVG_AD") : angle_deg = int(varname[8:])
            MH = self.get_data("MH", itime=itime, time_slice=time_slice)
            WD = self.get_data("WD", itime=itime, time_slice=time_slice)
            return MH*np.cos(np.deg2rad(angle_deg - WD))
        elif varname.startswith("V_AD") or varname.startswith("V_AVG_AD") :
            #like V_AD50, V_AD220 : rotation of U and V with an angle of angle_deg ° from North
            # NOTE : U_AD270 = U, V_AD270 = V
            if varname.startswith("V_AD") : angle_deg = int(varname[4:])
            if varname.startswith("V_AVG_AD") : angle_deg = int(varname[8:])
            MH = self.get_data("MH", itime=itime, time_slice=time_slice)
            WD = self.get_data("WD", itime=itime, time_slice=time_slice)
            return MH*np.sin(np.deg2rad(angle_deg - WD))
        elif varname.endswith("_AVG") :
            return self.get_data(varname[:-4], itime=itime, time_slice=time_slice)
        else :
            print("error in class_Expe_MF.get_data, the variable name :", varname, "doesn't exist")
            raise
    
    
