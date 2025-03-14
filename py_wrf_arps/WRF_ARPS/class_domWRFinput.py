#!/usr/bin/env python3
from .class_domWRF import DomWRF

import os
from netCDF4 import Dataset
import numpy as np
from ..class_variables import VariableRead

class DomWRFinput(DomWRF):
    software = "WRFinput"
    suffix_length = 0
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Force LAI and VEGFRA to come from wrflowinp if it exists
        if self.FLAGS["sfc"] :
            if self.keep_open :
                self.VARIABLES["LAI"] = VariableRead("LAI", "LAI", None, None, None, self.output_filenames["sfc"], True, "LAI",\
                                                     ncfile = next(iter(self.output_filenames["sfc"].items()))[1]) 
                self.VARIABLES["VEGFRA"] = VariableRead("VEGFRA", "VEGFRA", None, None, None, self.output_filenames["sfc"], True, "VEGFRA",\
                                                        ncfile = next(iter(self.output_filenames["sfc"].items()))[1]) 
                self.VARIABLES["LAI_INPUT"] = VariableRead("LAI", "LAI", None, None, None, self.output_filenames["base"], True, "LAI",\
                                                           ncfile = next(iter(self.output_filenames["base"].items()))[1]) 
                self.VARIABLES["VEGFRA_INPUT"] = VariableRead("VEGFRA", "VEGFRA", None, None, None, self.output_filenames["base"], True, "VEGFRA",\
                                                              ncfile = next(iter(self.output_filenames["base"].items()))[1])
            else :
                with Dataset(next(iter(self.output_filenames["sfc"].items()))[0], "r") as ncfile:
                    self.VARIABLES["LAI"] = VariableRead("LAI", "LAI", None, None, None, self.output_filenames["sfc"], True, "LAI", ncfile=ncfile) 
                    self.VARIABLES["VEGFRA"] = VariableRead("VEGFRA", "VEGFRA", None, None, None,self.output_filenames["sfc"], True, "VEGFRA", ncfile=ncfile)
                with Dataset(next(iter(self.output_filenames["base"].items()))[0], "r") as ncfile:
                    self.VARIABLES["LAI_INPUT"] = VariableRead("LAI", "LAI", None, None, None, self.output_filenames["base"], True, "LAI", ncfile=ncfile) 
                    self.VARIABLES["VEGFRA_INPUT"] = VariableRead("VEGFRA", "VEGFRA", None, None, None, self.output_filenames["base"], True, "VEGFRA",\
                                                                  ncfile=ncfile)
                    
            
    def get_output_filenames(self):
        self.output_filenames = {
            "base" : {},
            "post" : {},
            "sfc" : {},
        }
        for filename in os.listdir(self.output_data_dir):
            if filename == self.output_prefix : #wrfinput
                if self.keep_open :
                    self.output_filenames['base'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['base'][self.output_data_dir + filename] = True
                self.FLAGS['base'] = True  
            elif filename == "wrflowinp_d"+self.i_str: #wrflowinp
                if self.keep_open :
                    self.output_filenames['sfc'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['sfc'][self.output_data_dir + filename] = True
                self.FLAGS['sfc'] = True  
        for filename in os.listdir(self.postprocdir):
            if filename.startswith(self.software + self.i_str + "_post_" ):
                if self.keep_open :
                    self.output_filenames['post'][self.postprocdir + filename] = Dataset(self.postprocdir + filename, "r")
                else :
                    self.output_filenames['post'][self.postprocdir + filename] = True
                self.FLAGS['post'] = True
                
        if not self.FLAGS['base'] :
            print("error in dom_WRFinput "+self.i_str+", file "+self.output_prefix+" not found in "+self.output_data_dir)
            raise
        
    # def init_saved_variables(self): inherit from dom_WRF
    # def calculate(self, varname, itime = None): inherit from dom_WRF
  
    def modify_LAI(self, LAI_filename, modify_file=False) :
        # On glicid : LAI_filename is in /LAB-DATA/GLiCID/projects/MesoScaleABL/DATADIR/WRF/Static_Geography_Data/LAI_VEGFRA/
        LAI_file = Dataset(LAI_filename, "r")

        LAT_wrf = self.get_data("LAT")
        LON_wrf = self.get_data("LON")
        LANDMASK = self.get_data("LANDMASK")
        LAT_min, LAT_max = np.min(LAT_wrf), np.max(LAT_wrf)
        LON_min, LON_max = np.min(LON_wrf), np.max(LON_wrf)
        dLAT = (LAT_max - LAT_min)/100
        dLON = (LON_max - LON_min)/100

        LAT_all = LAI_file["lat"][:]
        LON_all = LAI_file["lon"][:]

        assert(np.min(LAT_all) < LAT_min)
        assert(np.max(LAT_all) > LAT_max)
        assert(np.min(LON_all) < LON_min)
        assert(np.max(LON_all) > LON_max)

        pos_LAT = np.argwhere(np.logical_and(LAT_all > LAT_min-dLAT, LAT_all < LAT_max+dLAT))
        pos_LON = np.argwhere(np.logical_and(LON_all > LON_min-dLON, LON_all < LON_max+dLON))
        print("reduced the number of LAT and LON points from .. to ..")
        print(len(LAT_all), len(pos_LAT))
        print(len(LON_all), len(pos_LON))
        iLAT0 = pos_LAT[0][0]
        iLAT1 = pos_LAT[-1][0]
        iLON0 = pos_LON[0][0]
        iLON1 = pos_LON[-1][0]

        LON_lai, LAT_lai = np.meshgrid(LON_all[iLON0:iLON1], LAT_all[iLAT0:iLAT1])
        LAI_lai = LAI_file["LAI"][iLAT0:iLAT1, iLON0:iLON1]
        LAI_lai[LAI_lai<0] = 0.0
        LAI_lai[LAI_lai>10] = 0.0
        LAI_lai[np.isnan(LAI_lai)] = 0.0
        
        LAI_new = self.interpolate_to_self_grid(LAT_lai, LON_lai, LAI_lai, interp="nn")
        LAI_new[np.where(LAI_new.mask)] = 0.0
        LAI_new[LANDMASK==0] = 0.0
        
        if modify_file :
            for filename in self.VARIABLES["LAI"].dataset_dict :
                print("modifying the file : ", filename)
                if self.keep_open :
                    #first closing the dataset that is in "reading" mode
                    Dataset.close(self.VARIABLES["LAI"].dataset_dict[filename])
                #then open it in writing mode
                with Dataset(filename, "a") as file :
                    nt = file["LAI"].shape[0]
                    for it in range(nt):
                        file["LAI"][it] = LAI_new
                if self.keep_open :
                    #finally reopen dataset in "reading" mode
                    self.VARIABLES["LAI"].dataset_dict[filename] = Dataset(filename, "r")
            if "LAI_INPUT" in self.VARIABLES :
                for filename in self.VARIABLES["LAI_INPUT"].dataset_dict :
                    print("modifying the file : ", filename)
                    if self.keep_open :
                        #first closing the dataset that is in "reading" mode
                        Dataset.close(self.VARIABLES["LAI_INPUT"].dataset_dict[filename])
                    #then open it in writing mode
                    with Dataset(filename, "a") as file :
                        if file["LAI"].ndim == 3 :
                            nt = file["LAI"].shape[0]
                            for it in range(nt):
                                file["LAI"][it] = LAI_new
                        elif file["LAI"].ndim == 2 :
                            file["LAI"][:] = LAI_new
                    if self.keep_open :
                        #finally reopen dataset in "reading" mode
                        self.VARIABLES["LAI_INPUT"].dataset_dict[filename] = Dataset(filename, "r")
        return LAI_new

    def modify_VEGFRA(self, VEGFRA_filename, modify_file=False) :
        # On glicid : VEGFRA_filename is in /LAB-DATA/GLiCID/projects/MesoScaleABL/DATADIR/WRF/Static_Geography_Data/LAI_VEGFRA/
        VEGFRA_file = Dataset(VEGFRA_filename, "r")

        LAT_wrf = self.get_data("LAT")
        LON_wrf = self.get_data("LON")
        LANDMASK = self.get_data("LANDMASK")
        LAT_min, LAT_max = np.min(LAT_wrf), np.max(LAT_wrf)
        LON_min, LON_max = np.min(LON_wrf), np.max(LON_wrf)
        dLAT = (LAT_max - LAT_min)/100
        dLON = (LON_max - LON_min)/100

        LAT_all = VEGFRA_file["lat"][:]
        LON_all = VEGFRA_file["lon"][:]

        assert(np.min(LAT_all) < LAT_min)
        assert(np.max(LAT_all) > LAT_max)
        assert(np.min(LON_all) < LON_min)
        assert(np.max(LON_all) > LON_max)

        pos_LAT = np.argwhere(np.logical_and(LAT_all > LAT_min-dLAT, LAT_all < LAT_max+dLAT))
        pos_LON = np.argwhere(np.logical_and(LON_all > LON_min-dLON, LON_all < LON_max+dLON))
        print("reduced the number of LAT and LON points from .. to ..")
        print(len(LAT_all), len(pos_LAT))
        print(len(LON_all), len(pos_LON))
        iLAT0 = pos_LAT[0][0]
        iLAT1 = pos_LAT[-1][0]
        iLON0 = pos_LON[0][0]
        iLON1 = pos_LON[-1][0]

        LON_vegfra, LAT_vegfra = np.meshgrid(LON_all[iLON0:iLON1], LAT_all[iLAT0:iLAT1])
        VEGFRA_vegfra = VEGFRA_file["FCOVER"][iLAT0:iLAT1, iLON0:iLON1]*100 #note : VEGFRA is in % while FCOVER is a fraction (<1)
        VEGFRA_vegfra[VEGFRA_vegfra<0] = 0.0
        #VEGFRA_vegfra[VEGFRA_vegfra>100] = 0.0
        VEGFRA_vegfra[np.isnan(VEGFRA_vegfra)] = 0.0
        
        VEGFRA_new = self.interpolate_to_self_grid(LAT_vegfra, LON_vegfra, VEGFRA_vegfra, interp="nn")
        VEGFRA_new[np.where(VEGFRA_new.mask)] = 0.0
        VEGFRA_new[LANDMASK==0] = 0.0
        
        if modify_file :
            for filename in self.VARIABLES["VEGFRA"].dataset_dict :
                print("modifying the file : ", filename)
                if self.keep_open :
                    #first closing the dataset that is in "reading" mode
                    Dataset.close(self.VARIABLES["VEGFRA"].dataset_dict[filename])
                #then open it in writing mode
                with Dataset(filename, "a") as file :
                    nt = file["VEGFRA"].shape[0]
                    for it in range(nt):
                        file["VEGFRA"][it] = VEGFRA_new
                if self.keep_open :
                    #finally reopen dataset in "reading" mode
                    self.VARIABLES["VEGFRA"].dataset_dict[filename] = Dataset(filename, "r")
            if "VEGFRA_INPUT" in self.VARIABLES :
                for filename in self.VARIABLES["VEGFRA_INPUT"].dataset_dict :
                    print("modifying the file : ", filename)
                    if self.keep_open :
                        #first closing the dataset that is in "reading" mode
                        Dataset.close(self.VARIABLES["VEGFRA_INPUT"].dataset_dict[filename])
                    #then open it in writing mode
                    with Dataset(filename, "a") as file :
                        if file["VEGFRA"].ndim == 3 :
                            nt = file["VEGFRA"].shape[0]
                            for it in range(nt):
                                file["VEGFRA"][it] = VEGFRA_new
                        elif file["VEGFRA"].ndim == 2 :
                            file["VEGFRA"][:] = VEGFRA_new
                    if self.keep_open :
                        #finally reopen dataset in "reading" mode
                        self.VARIABLES["VEGFRA_INPUT"].dataset_dict[filename] = Dataset(filename, "r")
        return VEGFRA_new