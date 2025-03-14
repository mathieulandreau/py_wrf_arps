from ..class_proj import Proj
from ..WRF_ARPS import Dom
from ..lib import manage_time

import numpy as np
import pandas as pd
import os
        
""" 
An example of how to use the ABL class :
    #py_wrf_arps version : 7607ca6cafe292b9218359c1664f65897ac489a7 , the current date is :  2025-03-04 18:25:10.509312
    from py_wrf_arps import *
    # create sim object
    sim = Proj("/data1/data-mod/mlandreau/03_simulation", "14_20200515/", ["01", "02", "03", "04", "05"], "WRF", tab_test="30")
    # create ABL object
    zone = [260, 330, 260, 330]
    itime = ("2020-05-16-08", "2020-05-19-08")
    crop = ([0, 80], "ALL", "ALL")
    start_dates = ["2020-05-16-08-10", "2020-05-17-05-50", "2020-05-18-06", "2020-05-19-06"]
    obj = manage_ABL.ABL(sim, "04", zone, itime, crop, start_dates)
    # call ABL methods
    fig = obj.plot_region(itime = "2020-05-17-18", crop=(0, "ALL", "ALL"))
    obj.get_data()
    fig = obj.plot_data()
    obj.calculate_layers()
    fig = obj.plot_layers()
    obj.save_postproc()
""" 

class ABL(): 
    def __init__(self, sim, dom, zone, itime, crop, start_dates, varname="NBV2", seuil=1e-4):
        """ find the ABL characteristics based on selected region
        Parameters
            self (ABL)
            sim: a class_Proj object
            dom: a class_Dom object (or inherited class)
            zone: the selected region to extract the ABL characteristics, see sim.get_zoom_index
            itime: the selected time, see Dom.get_data
            crop: the vertical extent (horizontal extent is forced by "zone"), see Dom.get_data
            start_dates: date to detect the CBL growing from the surface in the morning 
        Optional
            varname : the variable used to detect the ABL limits
            seuil : the threshold value to detect the ABL limits
        04/03/2025 : Mathieu Landreau
        """  
        self.sim = sim
        self.dom = sim.get_dom(dom)
        self.zone = zone
        self.itime = itime
        self.crop = crop
        self.start_dates = start_dates
        self.varname = varname
        self.seuil = seuil
        
        self.Nday = len(start_dates)
        self.p = {}
        self.kw_get = {
            "crop" : crop,
            "itime" : itime,
            "avg_area" : zone,
            "saved" : self.p,
        }
        
    def plot_region(self, varname="MH", **kw_get):
        """ Display the region and a chosen varname
        Parameters
            self (ABL)
        Optional
            varname (str): the variable we want to plot
            kw_get (dict): see Dom.get_data, result of self.dom.get_data(varname, **kw_get) must be a 2D array
        04/03/2025 : Mathieu Landreau
        """ 
        params = [{
            "typ" : "2DH", "Z" : "LANDMASK", "dom" : self.dom, "plot_cbar" : False, "title" : "",
        },{ "typ" : "RECTANGLE", "rectangle" : self.zone, "kwargs_plt" : {"linewidth" : 3, "edgecolor" : "r",},
        },{
            "typ" : "2DH", "dom" : self.dom, "Z" : varname, "kwargs_get_data" : kw_get,
        },{ "typ" : "QUIVER", "color" : "k",
        },{ "typ" : "RECTANGLE", "rectangle" : self.zone, "kwargs_plt" : {"linewidth" : 3},
        }]
        return self.sim.plot_fig(params)
        
    def get_data(self):
        """ get the needed data with sim.get_data
            Data are stored in self.kw_get["saved"] which is self.p
        Parameters
            self (ABL)
        04/03/2025 : Mathieu Landreau
        """ 
        self.Z, self.var, self.TIME = self.sim.get_data(self.dom, ["Z", self.varname, "TIME"], **self.kw_get)
        self.NT, self.NZ = np.squeeze(self.p["Z"]).shape

    def plot_data(self):
        """ plot the time-height evolution of self.varname
        Parameters
            self (ABL)
        04/03/2025 : Mathieu Landreau
        """ 
        return self.sim.plot_fig([{
            "typ" : "ZT", "dom" : self.dom, "kwargs_get_data" : self.kw_get, "Z" : self.varname, "cmap" : "Blues", "clim" : [-self.seuil, 4*self.seuil], "discrete" : 5,"DX_subplots" : 25, "NX_subplots" : 1,
        }])
    
    def calculate_layers(self):
        """ Determine the limit of each layer (CBL and capping inversion for now)
        Parameters
            self (ABL)
        04/03/2025 : Mathieu Landreau
        """ 
        Z, var, TIME = self.Z, self.var, self.TIME
        self.mask = np.logical_and(var[:, :-1]<self.seuil, var[:, 1:]>=self.seuil)
        zCBL = np.zeros((self.Nday, self.NT))*np.nan
        izCBL = np.zeros((self.Nday, self.NT), dtype=int)-1
        zCI = np.zeros((self.NT))*np.nan
        izCI = np.zeros((self.NT), dtype=int)-1
        tstart = manage_time.to_datetime(self.start_dates) 
        itstart = np.arange(self.NT, dtype=int)[np.isin(TIME, tstart)]
        print(tstart, itstart)
        
        # For each day, find the limit which is the lowest at tstart and that increase with time
        # Stop when it does not increase anymore
        # Limitation : A slightly decreasing area would break it
        for iday in range(self.Nday-1, -1, -1):
            it = itstart[iday]
            found_limit = np.arange(self.NZ-1)[self.mask[it]]
            izCBL[iday, it] = int(np.min(found_limit)) if len(found_limit) > 0 else -1
            stop = False
            iz = izCBL[iday, it]
            while not stop :
                it += 1
                temp = np.arange(self.NZ-1)[self.mask[it]]
                found_limit = temp[np.array(temp) >= iz]
                if len(found_limit) > 0 :
                    izCBL[iday, it] = int(np.min(found_limit))
                    iz = izCBL[iday, it]
                elif len(temp) > 0 : 
                    stop = True 
                if it == self.NT-1 :
                    stop = True 
        # During the entire period, define the Capping Inversion height with the same threshold but as the highest limit
        # Limitation : If the threshold is reached aloft this might be wrong
        iz = 0
        for it in range(self.NT):
            found_limit = np.arange(self.NZ-1)[self.mask[it]]
            found_limit = found_limit[np.logical_not(np.isin(found_limit, izCBL[:, it]))]
            if len(found_limit) > 0 :
                i_iz = int(np.nanargmin(np.abs(np.array(found_limit) - iz)))
                izCI[it] = found_limit[i_iz]
                iz = izCI[it]
        # Add CBL points that have reached the CI, in the CI
        iz = 0
        for it in range(self.NT):
            if izCI[it] == -1 :
                found_limit = np.arange(self.NZ-1)[self.mask[it]]
                found_limit = found_limit[found_limit >= iz]
                if len(found_limit) > 0 :
                    i_iz = np.nanargmin(np.abs(np.array(found_limit) - iz))
                    izCI[it] = found_limit[i_iz]
                    iz = izCI[it]
            else :
                iz = izCI[it]
        # For each detected point, find the height by vertically interpolating with var   
        for it in range(self.NT) :
            for iday in range(self.Nday) :
                if izCBL[iday, it] >=0 :
                    zCBL[iday, it] = np.interp(self.seuil, var[it, izCBL[iday, it]:izCBL[iday, it]+2], Z[it, izCBL[iday, it]:izCBL[iday, it]+2])
            if izCI[it] >=0 :
                zCI[it] = np.interp(self.seuil, var[it, izCI[it]:izCI[it]+2], Z[it, izCI[it]:izCI[it]+2])
                
        # Fill the firsts and lasts NaNs of CI with a constant value
        it = 0
        while izCI[it] == -1:
            it += 1
        zCI[:it] = zCI[it]
        it = self.NT-1
        while izCI[it] == -1:
            it -= 1
        zCI[it+1:] = zCI[it]

        # Fill the intermediate NaNs of CI with a linear interpolation over time
        iz = 0
        it0 = 0
        interpolating = False
        for it in range(self.NT):
            if izCI[it] == -1 :
                if not interpolating:
                    interpolating = True
                    it0 = it
            else :
                if interpolating:
                    interpolating = False
                    zCI[it0:it] = np.interp(np.arange(it0, it), np.array([it0-1, it]), np.array([zCI[it0-1], zCI[it]]))
                    
        self.izCBL = izCBL
        self.izCI = izCI
        self.zCBL = zCBL
        self.zCI = zCI
        
    def plot_layers(self):
        """ Plot the limit of the layers on the time-height plot of varname
        Parameters
            self (ABL)
        04/03/2025 : Mathieu Landreau
        """ 
        params = [{
            "typ" : "ZT", "dom" : self.dom, "kwargs_get_data" : self.kw_get, "Z" : self.varname, "cmap" : "Blues", "clim" : [-self.seuil, 4*self.seuil], 
            "discrete" : 5,"DX_subplots" : 25, "NX_subplots" : 1,
        },{ "X" : self.TIME, "Y" : self.zCI, "kwargs_plt" : {"linewidth" : 5, "color" : [0,1,0]}, "same_ax" : True, "grid" : False,
        }]
        for iday in range(self.Nday) :
            params.append({
                "X" : self.TIME, "Y" : self.zCBL[iday], "kwargs_plt" : {"linewidth" : 3, "color" : [1,0,0]}, "same_ax" : True, "grid" : False,
            })
        return self.sim.plot_fig(params)
    
    def save_postproc(self, savepath=""):
        """ Save the results in the postproc file
        Parameters
            self (ABL)
        Optional
            savepath (str): path to save the pickle file
        04/03/2025 : Mathieu Landreau
        """ 
        import os
        zCBL = np.nanmax(self.zCBL, axis=0)
        for dom in self.sim.tab_dom :
            if not dom.software in ["AROME"] :
                TIMEout = dom.get_data("TIME", itime="ALL_TIMES")
                NTout = len(TIMEout)
                time_slice = manage_time.get_time_slice(self.TIME, TIMEout)
                zCBLout = np.zeros((NTout))*np.nan
                zCIout = np.zeros((NTout))*np.nan
                zCBLout[time_slice] = zCBL
                zCIout[time_slice] = self.zCI
                savepath_i = savepath
                if savepath_i == ""  :
                    savepath_i = dom.postprocdir+dom.name+"_df_TIME.pkl"
                else :
                    savepath_i = savepath_i[:-4]+dom.name+".pkl"
                print(f"saving in {savepath_i}")
                if os.path.exists(savepath_i):
                    dfout = pd.read_pickle(savepath_i)
                else :
                    dfout = pd.DataFrame({
                        "TIME" : TIMEout,
                    })
                dfout["ZCBL"] = zCBLout
                dfout["ZCI"] = zCIout
                dfout.to_pickle(savepath_i)
 