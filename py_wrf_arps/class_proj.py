#!/usr/bin/env python3
import sys

from .lib import manage_display, manage_plot, manage_dict, manage_time, manage_projection, manage_TaylorDiagram, manage_path
from .WRF_ARPS import *
from .expe_data import *
from .default_params import default_params

import os
import re
import math
import numpy as np
import pandas as pd
from itertools import product
import datetime
import itertools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from wrf import vertcross, CoordPair, interplevel, WrfProj
from multiprocessing import Pool, cpu_count

import copy
matplotlib.use('Agg')

print_debug = False
printDebug2 = False

class Proj():
    VARPLOT = {
        'linestyle': [(0, ()),                   # solid
                      (0, (5, 1)),               # densely dashed
                      (0, (1, 1)),               # densely dotted
                      (0, (3, 1, 1, 1)),         # densely dashdotted
                      (0, (3, 1, 1, 1, 1, 1)),   # densely dashdotted
                      (0, (5, 5)),               # loosely dashed
                      (0, (1, 5)),               # loosely dotted
                      (0, (3, 5, 1, 5)),         # loosely dashdotted
                      (0, (3, 5, 1, 5, 1, 5))],  # loosely dashdotdotted
        'color': ['black', 'red', 'green', 'blue', 'cyan',
                  'magenta', 'orange', 'gray', 'lime'],
        'colors': ['k', 'r', 'g', 'b', 'c', 'm'],
        'styles' : ['-', '--', ':', '-.'],
        'marker1' : ['+', 'x', '^', 'v', 's', 'o', 'd', '>', '<', '1', '2', '3', '4', '*'],
        'dpi': 300,
        'format': 'png',
        'fs_title': 16,  # fontsize
        'fs_label': 20,
        'fs_tick': 16,
        'fs_text': 8,
        'figsize': (6.4,4.8), # default (6.4,4.8 or 4/3)
        'coastwidth' : 1,
        'coastcolor' : 'k',
        'domainwidth' : 1
    }
    

#################################################################################################################################
######  Init
#################################################################################################################################
    def __init__(self, proj_dir, data_dir, tab_dom_str, tab_arps_wrf, tab_expe_str=[], tab_test=None, aerradii=None, JeSuis=None, keep_open=False):
        """
        Plot 2D figure

        Parameters
        ----------
        self: Proj
        
        -------------- If JeSuis == "Mathieu" or None or "Test": 
        proj_dir (str): Global path toward all the simulations directory, see manage_path.proj_dir
        data_dir (str): name of this simulation directory #"02_20200519/"
        tab_dom_str (list of str): list of domaine names #["05", "04", "03", "02"]
        tab_wrf_arps (list of str): list of softwares, strings can be for the moment : "ARPS", "WRF", "WPS", "WRFinput", "AROME" (not case sensitive)

        Optional
        ----------
        tab_expe_str (list of str): list of other dataset that are loaded, strings can be for the moment : "CRO", "LI1", "CAR", "SEM", "TEI"
        tab_test (list of str): subdivision of the data_dir simulation
        aerradii : related to aerosols (used by Benjamin with arps but many things have changed since)
        JeSuis : string : if "Mathieu" or "M", the directories containings all the data are deduced with proj_dir and data_dir.
                          add your own name and organization if needed
        keep_open : boolean : if True, keep all output files open for read. It might be a little faster (I am not really sure in fact) but it needs more disk space

        Returns
        ----------

        Author(s)
        ----------
        Benjamin LUCE
        14/12/2022 : Modification of path by Mathieu Landreau
        """   
        
        if type(tab_arps_wrf) is str : tab_arps_wrf = [tab_arps_wrf]
        if type(tab_dom_str) is str : tab_dom_str = [tab_dom_str]
        if type(tab_expe_str) is str : tab_expe_str = [tab_expe_str]
        if type(tab_test) is str : tab_test = [tab_test]
        
        self.data_dir = data_dir
        self.tab_expe = [] 
        self.tab_dom = []
        self.tab_dom_str = tab_dom_str
        self.tab_expe_str = tab_expe_str
        self.domains = len(self.tab_dom_str) #nb de domaines
        self.keep_open=keep_open
    
        if len(tab_arps_wrf) == 1 :
            self.tab_arps_wrf = tab_arps_wrf*self.domains
        else :
            self.tab_arps_wrf = tab_arps_wrf
            
        if tab_test is None :
            self.tab_test = [""]*self.domains
        elif len(tab_test) == 1 :
            self.tab_test = tab_test*self.domains
        else :
            self.tab_test = tab_test
            
        if JeSuis is not None :
            self.JeSuis = JeSuis
        else :
            self.JeSuis = "Mathieu"
        
        if print_debug : print("self.tab_dom_str :", self.tab_dom_str)
        if print_debug : print("self.tab_arps_wrf :", self.tab_arps_wrf)
        if print_debug : print("self.tab_test :", self.tab_test)
        if print_debug : print("self.tab_expe_str :", self.tab_expe_str)
        if print_debug : print("self.domains :", self.domains)
        if self.domains > 0 :
            if self.JeSuis in ["Mathieu", "M", "test"] : 
                self.figdir = proj_dir + '05_figures/' + data_dir
                self.postprocdir = proj_dir + '04_output/06_python/' + data_dir
                self.tmpdir = proj_dir + '03_temp/06_python/' + data_dir
                ARPS_output_datadir = proj_dir + '04_output/04_ARPS/' + data_dir 
                ARPS_prefix = data_dir[:2]+"<dom_str>"
                WRF_output_datadir = proj_dir + '04_output/02_WRF/' + data_dir
                WRF_prefix = "wrfout_d<dom_str>"
                WPS_output_datadir = proj_dir + '03_temp/01_WPS/' + data_dir
                WPS_prefix = "met_em.d<dom_str>"
                WRFinput_output_datadir = proj_dir + '03_temp/02_WRF/' + data_dir
                WRFinput_prefix = "wrfinput_d<dom_str>"
                AROME_output_datadir = manage_path.AROME_data_dir
                AROME_prefix = "T64200_AROME_0025_"
            else : 
                raise(Exception("error in Proj __init__ : don't know who you are (" + str(self.JeSuis) + ") and where are your data"))



            for i_dom, dom_str in enumerate(self.tab_dom_str):
                test_folder =  self.tab_test[i_dom] + "/"
                if not os.path.exists(self.figdir):
                    print('Create figure directory in {0}'.format(self.figdir))
                    os.makedirs(self.figdir, exist_ok=True)
                if not os.path.exists(self.postprocdir+test_folder):
                    print('Create postproc directory in {0}'.format(self.postprocdir+test_folder))
                    os.makedirs(self.postprocdir+test_folder, exist_ok=True)
                if not os.path.exists(self.tmpdir):
                    print('Create tmp directory in {0}'.format(self.tmpdir))
                    os.makedirs(self.tmpdir, exist_ok=True)
                kwargs = {
                    "postprocdir" : self.postprocdir+test_folder,
                    "keep_open" : self.keep_open,
                    "test" : self.tab_test[i_dom],
                }
                if(self.tab_arps_wrf[i_dom].upper() == "ARPS"):
                    output_prefix = ARPS_prefix.replace("<dom_str>", dom_str)
                    self.tab_dom.append(DomARPS(dom_str, ARPS_output_datadir+test_folder, output_prefix, aerradii, **kwargs))
                elif(self.tab_arps_wrf[i_dom].upper() == "WRF"):
                    output_prefix = WRF_prefix.replace("<dom_str>", dom_str)
                    self.tab_dom.append(DomWRF(dom_str, WRF_output_datadir+test_folder, output_prefix, aerradii, **kwargs))
                elif(self.tab_arps_wrf[i_dom].upper() == "WPS"):
                    output_prefix = WPS_prefix.replace("<dom_str>", dom_str)
                    self.tab_dom.append(DomWPS(dom_str, WPS_output_datadir+test_folder, output_prefix, aerradii, **kwargs))
                elif(self.tab_arps_wrf[i_dom].upper() == "WRFINPUT"):
                    output_prefix = WRFinput_prefix.replace("<dom_str>", dom_str)
                    self.tab_dom.append(DomWRFinput(dom_str, WRFinput_output_datadir+test_folder, output_prefix, aerradii, **kwargs))
                elif(self.tab_arps_wrf[i_dom].upper() == "AROME"):
                    output_prefix = AROME_prefix + self.tab_test[i_dom] #self.tab_test[i_dom] = "2020" for example or "202005"
                    self.tab_dom.append(DomAROME(dom_str, AROME_output_datadir, output_prefix, aerradii, **kwargs))
                else : 
                    raise(Exception("error in Proj.__init__ : unknown string in tab_arps_wrf : " + dom_str))
        else :
            self.figdir = proj_dir + '05_figures/00/'
            self.postprocdir = proj_dir + '04_output/06_python/00/'
            self.tmpdir = proj_dir + '03_temp/06_python/00/'
            if not os.path.exists(self.figdir):
                print('Create figure directory in {0}'.format(self.figdir))
                os.makedirs(self.figdir, exist_ok=True)
            if not os.path.exists(self.postprocdir):
                print('Create postproc directory in {0}'.format(self.postprocdir))
                os.makedirs(self.postprocdir, exist_ok=True)
            if not os.path.exists(self.tmpdir):
                print('Create tmp directory in {0}'.format(self.tmpdir))
                os.makedirs(self.tmpdir, exist_ok=True)
                
        for expe_code in self.tab_expe_str :
            if expe_code.upper() == "CAR":
                self.tab_expe.append(CAR())
            elif expe_code.upper() == "CRO":
                self.tab_expe.append(CRO())
            elif expe_code.upper() == "SEM":
                self.tab_expe.append(SEM())
            elif expe_code.upper() == "LI1":
                self.tab_expe.append(LI1())
            elif expe_code.upper() in METEO_FRANCE_LIST : #see expe_dict
                self.tab_expe.append(Expe_MF(expe_code.upper()))
            else : 
                raise(Exception("error in Proj.__init__ : unknown string in tab_expe_str : "+expe_code))
            

#################################################################################################################################
######  Getting domains and expe objects
#################################################################################################################################

    def get_dom(self, dom_input):
        """
        Description
            Get the domain object from dom_input
        Parameters
            dom_input : can be
                Dom : return itself
                int : index of the domain in self.tab_dom
                str : see Proj.get_index
        Returns 
            Dom 
        """
        if isinstance(dom_input, Dom): return dom_input
        elif isinstance(dom_input, int): return self.tab_dom[dom_input]
        elif isinstance(dom_input, str): return self.tab_dom[self.get_index(dom_input)]
        elif dom_input is None : return None
        else :
            raise(Exception("error : cannot identify the domain from the input : ", dom_input, ". Should be either of type Dom, of type int or of type str"))
        
    def get_index(self, dom_str):
        """
        Description
            Get the index of the domain object from dom_str
        Parameters
            dom_str : str : can be
                only letters ("ARPS" or "WRF" or ...)
                only 2 digits ("03" or "02" or "XX") where XX is the dom.i_str 
                4 digits ("0302" or "0901" or "XXYY") where XX is the dom.i_str and YY is the index of the test
                mix (ARPS03, WRF02, ...) : digit format is length of 2 or 4
        Returns 
            int : index of the domain in self.tab_dom
        """
        if dom_str.isalpha() : #assuming "WRF" or "ARPS", ...
                tab_temp = self.tab_arps_wrf
        elif dom_str.isnumeric() : 
            if len(dom_str) == 2 : #assuming "02" or other "XX" where XX is the dom.i_str
                tab_temp = self.tab_dom_str
            elif len(dom_str) == 4 : #assuming "0201" or other "XXYY" where XX is the dom.i_str and YY is the index of the test
                tab_temp = [self.tab_dom_str[i]+self.tab_test[i] for i in range(self.domains)]
            else :
                raise(Exception("error in Proj.get_index : the key (", dom_str, ") cannot be recognized"))
        elif dom_str[-4:].isnumeric(): #assuming "WRF0201 or "ARPSXXYY", ... where XX is the dom.i_str and YY is the index of the test
            tab_temp = [self.tab_arps_wrf[i]+self.tab_dom_str[i]+self.tab_test[i] for i in range(self.domains)] 
        else : #assuming "WRF02 or "ARPSXX", ... where XX is the dom.i_str
            tab_temp = [self.tab_arps_wrf[i]+self.tab_dom_str[i] for i in range(self.domains)] 
        if dom_str not in tab_temp :
            raise(Exception("error : no domain corresponds to : ", dom_str, ", in ", tab_temp))
        else :
            count = tab_temp.count(dom_str)
            if count > 1 :
                raise(Exception("error : too many (", count, ") domains correspond to : ", dom_str, ", in ", tab_temp, " cannot choose one"))
            else :
                return tab_temp.index(dom_str)
     
    def get_inner_domains(self, dom):
        """
        Description
            Find the nested domains of dom
        Parameters
            dom : can be any type managed by Proj.get_dom
        Returns 
            list of Dom : list of the inner domains of dom
        """
        dom = self.get_dom(dom)
        i_dom = int(dom.i_str)
        dom_list = []
        for dom_i in self.tab_dom :
            if dom_i.software in ["ARPS", "WRF"] and int(dom_i.i_str) > i_dom :
                dom_list.append(dom_i)
        return dom_list
            
    def get_expe(self, expe_input):
        """
        Description
            Get the Expe object from expe_input
        Parameters
            expe_input : can be 
                Expe : return itself
                int : index of Expe object in self.tab_expe
                str : name of the Expe in self.tab_expe_str
        Returns 
            Expe
        """
        if isinstance(expe_input, Expe):
            return expe_input
        elif isinstance(expe_input, int):
            return self.tab_expe[expe_input]
        elif isinstance(expe_input, str):
            return self.tab_expe[self.tab_expe_str.index(expe_input.upper())]
        else :
            raise(Exception("error : cannot identify the expe object from the input : ", expe_input, ". Should be either of type Expe, int or str"))
       
    def get_dom_expe(self, dom_expe_input):
        try : 
            return self.get_dom(dom_expe_input)
        except : 
            try : 
                return self.get_expe(dom_expe_input)
            except :
                raise(Exception("error : cannot identify the dom or expe object from the input : ", dom_expe_input, ". Should be either of type Dom, Expe, int or str"))
        
#################################################################################################################################
######  Relay to Dom methods
#################################################################################################################################
    def get_data(self, dom_expe, varname, **kwargs):
        """
        Description
            Get data from dom by calling dom.get_data
        Parameters
            dom : can be any type managed by Proj.get_dom
            varname : str : name of the variable to get (see ARPS_WRF_VARNAME_DICT)
        Optional
            kwargs : dict : see Dom.get_data parameters
        Returns 
            any type the data is (probably np.array, float or int)
        """
        dom_expe = self.get_dom_expe(dom_expe)
        if "vinterp" in kwargs and "points" in kwargs["vinterp"] :
            kwargs["vinterp"]["points"] = self.get_points_for_vcross(dom_expe, kwargs["vinterp"]["points"])
        if "zoom" in kwargs :
            kwargs["zoom"] = self.get_zoom_index(dom_expe, kwargs["zoom"])
        if "avg_area" in kwargs :
            kwargs["avg_area"] = self.get_zoom_index(dom_expe, kwargs["avg_area"])
        # If one variable call dom_expe.get_data
        if type(varname) == str :
            return dom_expe.get_data(varname, **kwargs)
        # If several variables, call get data for each
        elif type(varname) in (list, tuple, np.array, np.ndarray) :
            out = ()
            for varname_i in varname :
                out = out + (self.get_data(dom_expe, varname_i, **kwargs),)
            return out
        else :
            raise(Exception("error, unknown type for varname in Dom.get_data : " + str(type(varname)) + ", varname = " + varname))
    
    def get_ts(self, dom, *args, **kwargs):
        """
        Description
            Get tslist from domWRF by calling domWRF.get_ts
        Parameters
            dom : can be any type managed by Proj.get_dom and must be a WRF domain
        *args :
            pfx : str : prefix of the ts location
            varname : str : name of the variable to get (see domWRF.get_ts)
        Returns 
            any type the data is (probably np.array, float or int)
        """
        dom = self.get_dom(dom)
        if not type(dom) is DomWRF :
            raise(Exception("error : cannot get a time series data from another type than DomWRF"))
        return dom.get_ts(*args, **kwargs)
    
    def get_lat_lon(self, point) :
        """
        Description
            Get the latitude and longitude of a point
        Parameters
            point : can be 
                str : name of an Expe object
                list of 2 floats : [lat, lon]
        Returns 
            list of 2 floats : [lat, lon]
        """
        if isinstance(point, str) or isinstance(point, Expe) :
            if point in DICT_EXPE_DATE :
                return DICT_EXPE_DATE[point][2]
            else :
                expe = self.get_expe(point)
                return [expe.lat, expe.lon]
        elif type(point) in [list, tuple, np.array] and len(point) == 2 :
            return point
        else : 
            raise(Exception("error : cannot get lat and lon from :", point))
            
    def nearest_index(self, dom, point, **kwargs) :
        """
        Description
            Get the nearest dom grid position from point
        Parameters
            dom : can be any type managed by Proj.get_dom
            point : can be 
                str : name of an Expe object
                list of 2 floats : [lat, lon]
        Returns 
            int, int : iy, ix : index of the nearest grid point
        """
        dom = self.get_dom_expe(dom)
        return dom.nearest_index_from_self_grid(self.get_lat_lon(point), **kwargs)
                
    
    def get_zoom_index(self, dom, points):
        """
        Description
            return the indices iy1, iy2, ix1, ix2 to zoom horizontally in a domain
            These indices are then use to crop the data and read only the data contained in the zoom domain
        Parameters
            dom : can be any type managed by Proj.get_dom
            points : can be 
                list of 4 integers : already the index [iy1, iy2, ix1, ix2]
                list of 2 lists of 2 floats : the upper-left and the lower_right [[lat1, lon1], [lat2, lon2]]
                list of (list, float, float) : the coordinates of the center (expe name) and the size [[lat_c, lon_c], LX, LY], ["CRO", LX, LY]
                anything managed by Proj.get_dom() : the name of a domain, it will return the limit of the domain
        Returns 
            int, int, int, int : iy1, iy2, ix1, ix2 : index of the nearest grid point
        """
        if printDebug2 : print("zoom, points = ", points)
        if type(points) is list :
            if printDebug2 : print("type(points) is list")
            dom = self.get_dom(dom)
            if len(points) == 3 :
                points[0] = self.get_lat_lon(points[0])
            return dom.get_zoom_index(points)
        elif type(points) is tuple :
            if printDebug2 : print("type(points) is tuple")
            return points
        else :
            if printDebug2 : print("type(points) is dom")
            dom2 = self.get_dom(points)
            UPPER_LEFT = dom2.get_data("UPPER_LEFT_LAT_LON")
            LOWER_RIGHT = dom2.get_data("LOWER_RIGHT_LAT_LON")
            #calling again the same function but with the points
            return self.get_zoom_index(dom, [[UPPER_LEFT.lat, UPPER_LEFT.lon], [LOWER_RIGHT.lat, LOWER_RIGHT.lon]]) 
        
    def get_line(self, dom, line):
        """
        Description
            return the coordinates y1, y2, x1, x2 to draw a line on a 2D horizontal plot
        Parameters
            dom : can be any type managed by Proj.get_dom
            points : can be 
                list of 4 integers : already the coordinates [y1, y2, x1, x2]
                list of 2 lists of 2 floats : the upper-left and the lower_right [[lat1, lon1], [lat2, lon2]]
                list of (list, float, float) : the coordinates of the center and the length and the direction [[lat_c, lon_c], L, beta]
        Returns 
            float, float, float, float : y1, y2, x1, x2 : coordinates of the two points (note : these are not the indixes)
        """
        dom = self.get_dom(dom)
        if len(line) == 3 :
            line[0] = self.get_lat_lon(line[0])
        return dom.get_line(line)
            
    def get_point(self, dom, point):
        """
        Description
            return the coordinates y, x, to plot a point on a 2D horizontal plot
        Parameters
            dom : can be any type managed by Proj.get_dom
            point : something to deduce the index, can be
                - already the location (a tuple of 2 reals) : (y, x)
                - a coordinates pair : (a list of 2 reals) [lat, lon]
                - a location name (str) : "CRO" for example
        Returns 
            float, float : y, x : coordinates of the two points (note : these are not the indices)
        """
        dom = self.get_dom_expe(dom)
        if isinstance(point, str) or isinstance(point, Expe) :
            point = self.get_lat_lon(point)
        return dom.get_point(point)
    
    def get_points_for_vcross(self, dom, points):
        """
        Description
            return the coordinates [[lat1, lon1], [lat2, lon2]] to use vertcross
        Parameters
            dom : the domain on which we want to compute vertcross
            points : something to deduce the coordinates, can be
                - already two coordinates pairs : the upper-left and the lower_right (a list of 2 lists of 2 reals) [[lat1, lon1], [lat2, lon2]]
                - The points index (a list of 4 integers) [iy1, iy2, ix1, ix2]
                - the coordinates of the center, the length and direction [[lat_c, lon_c], L, beta]
                - the name of the center, the length and direction ["CRO", L, beta]
        Returns
            points : a list of 2 lists of 2 reals : the upper-left and the lower_right coordinates (a list of 2 lists of 2 reals) [[lat1, lon1], [lat2, lon2]]
        """
        dom = self.get_dom(dom)
        #print(points)
        if len(points) == 2 : 
            return points
        elif len(points) == 3:
            center = self.get_lat_lon(points[0])
            distance = points[1]
            direction = points[2]
            lat1, lon1 = manage_projection.inverse_haversine(center, distance/2, direction+math.pi)
            lat2, lon2 = manage_projection.inverse_haversine(center, distance/2, direction)
            return [[lat1, lon1], [lat2, lon2]]
        else :
            return dom.get_points_for_vcross(points)
    
    
    def get_rectangle_params(self, dom, rec_in, Xname="X_KM", Yname="Y_KM"):
        """
        return the index [(x1,x2), Dx, Dy] to plot a rectangle on a 2D horizontal plot
        dom : the domain on which we want to zoom
        rec_in : anything that can be read by get_zoom_index 
        """
        if type(rec_in) is list and len(rec_in) == 3 and type(rec_in[0]) is tuple :
            return rec_in
        else :
            dom = self.get_dom(dom)
            X = self.get_data(dom, Xname)
            Y = self.get_data(dom, Yname)
            iy1, iy2, ix1, ix2 = self.get_zoom_index(dom, rec_in)
            X1 = X[iy1, ix1]
            X2 = X[iy2, ix2]
            Y1 = Y[iy1, ix1]
            Y2 = Y[iy2, ix2]
            return [(X1, Y1), X2-X1, Y2-Y1]

#################################################################################################################################
######  Managing figure plots
################################################################################################################################# 
    
    def plot_fig(self, params_list, **kwargs):
        """
        Description
            Plot the figure from parameters
            1- Prepare the parameters for plot
            2- Get the data in case it is not already done
            3- Generate a title and a path for save
            4- call manage_plot.plot_fig
        Parameters
            params_list : list of dict : see all_options.txt (if I wrote it) or look at tests programs, each dict corresponds to a plot
                IMPORTANT NOTE : params_list is also a kind of output since its modifications inside the function are conserved
                                 Indeed, list and dictionnaries arguments are passed by reference (not by value)
                OTHER NOTE : in Proj.prepare_params_for_get, some dict can be added to the list. These dict are present in the internal 
                             variable of plot_fig "params_list" but they are not present for the moment in the EXTERNAL argument params_list 
                             of the function plot_fig. Only the dict already contained in the input params_list are updated in the external
                             program that calls plot_fig
                             This will probably be updated in later versions
        Returns 
            mpl.figures : the figure that have been plotted, in case you want to modify it
        """
        if type(params_list) is not list : params_list = [params_list]
        params_list = self.prepare_params_for_get(params_list, **kwargs)
        self.get_params_for_plot(params_list, **kwargs)
        self.get_title_savepath(params_list, **kwargs)
        fig = manage_plot.plot_fig(params_list, **kwargs)
        return fig
    
    def prepare_params_for_get(self, params_list, **kwargs):
        """
        Description
            1st step of params preparation for plot before getting the data
            if plot_LAT_LON and plot_LANDMASK in 2DH plot : 
                add some CONTOUR dictionnary to the list of params with same_ax = True to plot on the same figure
        Parameters
            params_list : list of dict : see Proj.plot_fig
        Returns 
            list of dict : the updated params_list. Note that the dictionnaries already contained in the external params_list are externally updated
        """
        for i_params, params in enumerate(params_list) :
            #if same==True, copy all the parameters of the last plot
            #if same==-2, copy all the parameters of the second last plot, ...
            if "same" in params :
                if type(params["same"]) is int : 
                    if (params["same"] < 0): #ex : -1
                        params = manage_dict.select_params(params, params_list[i_params+params["same"]], depth=0) 
                    else:
                        params = manage_dict.select_params(params, params_list[params["same"]], depth=0) 
                elif params["same"] == True and i_params > 0 : 
                    params = manage_dict.select_params(params, params_list[i_params-1], depth=0) 
                        
        
        for i_params, params in enumerate(params_list) :       
            #Default type is 1D
            if not "typ" in params : 
                if "Z" in params :
                    params["typ"] = "SCATTER"
                else :    
                    params["typ"] = "1D"
            typename = params["typ"].upper() 
            params["typ"] = typename
            
            #deleting None in params
            to_del = []
            for k in params:
                if params[k] is None :
                    to_del.append(k)
            if len(to_del) >0 :
                for k in to_del :
                    del(params[k])
                    
            #Complete the chosen options with default options. Chosen options have higher priority degree of course
            #default_params are in .default_params.py
            params = manage_dict.select_params(params, default_params[typename]) 
            if not "same_fig" in params :
                params["same_fig"] = True
            if not params["same_fig"] :
                params["same_ax"] = False
        
        params_list2 = []
        for i_params, params in enumerate(params_list) :
            typename = params["typ"].upper() 
            params_add = []
            #Add some default plots in the list of plots and arrange params in a standard form
            
            if typename in ["2DH", "2DV", "2D", "ZT", "SCATTER"] and "cmap" in params: 
                    params["kwargs_plt"]["cmap"] = params["cmap"]
            if typename in ["2DH"]:
                #levels, points and ZP can be defined directly in params instead of in params["kwargs_get_data"]["hinterp"]
                for varname_temp in ["levels", "points", "ZP"]:
                    if varname_temp in params and "kwargs_get_data" in params and not varname_temp in params["kwargs_get_data"]["hinterp"] :
                        params["kwargs_get_data"]["hinterp"][varname_temp] = params[varname_temp] 
                if "dom" in params :
                    dom = self.get_dom_expe(params["dom"])
                            
                    #create contour params for parallels and meridians
                    if params["plot_LAT_LON"] :
                        for name in ["LON", "LAT"] :
                            params_temp = {"typ" : name, "dom" : dom}
                            params_temp = manage_dict.select_params(params_temp, default_params[name])
                            if "kwargs_LON_LAT" in params : 
                                params_temp = manage_dict.select_params(copy.deepcopy(params["kwargs_LON_LAT"]), params_temp)
                            params_add.append(params_temp)
                            
                    #create contour params for coastline
                    if params["plot_LANDMASK"] :
                        params_temp = {"typ" : "LANDMASK", "dom" : dom}
                        params_temp = manage_dict.select_params(params_temp, default_params["LANDMASK"])
                        if "kwargs_LANDMASK" in params : 
                            params_temp = manage_dict.select_params(params["kwargs_LANDMASK"], params_temp)
                        if "LANDMASK_colors" in params :
                            params_temp["kwargs_plt"]["colors"] = params["LANDMASK_colors"]
                        params_add.append(params_temp)
                        
                    #create rectangle params for inner domains
                    if params["plot_inner_doms"] :
                        dom_list = self.get_inner_domains(dom)
                        #create 1 rectangle params per inner domain
                        for dom_i in dom_list : 
                            if "Xname" in params and "Yname" in params :
                                xy, width, height = self.get_rectangle_params(dom, dom_i, Xname=params["Xname"], Yname=params["Yname"])
                            else :
                                xy, width, height = self.get_rectangle_params(dom, dom_i)
                            params_temp = {
                                "same_ax" : True,
                                "typ" : "RECTANGLE",
                                #"rlabel" : dom_i.software + dom_i.i_str,
                                "xy" : xy,
                                "width" : width,
                                "height" : height,
                            }
                            params_temp = manage_dict.select_params(params_temp, default_params["RECTANGLE"])
                            if "kwargs_inner_doms" in params : 
                                params_temp = manage_dict.select_params(params["kwargs_inner_doms"], params_temp)
                            params_add.append(params_temp)
                            
                    #create LiDAR contour
                    if params["LiDAR"] :  
                        lat_lidar, lon_lidar = [47.26047, 47.2857, 47.26047], [-2.50283, -2.51706, -2.53128]  
                        X, Y = manage_projection.ll_to_xy(lon_lidar, lat_lidar, dom.get_data("CRS"))
                        params_temp = {"X":np.array(X)/1000, "Y":np.array(Y)/1000, "dom":dom, }
                        params_temp = manage_dict.select_params(params_temp, default_params["LiDAR"])
                        params_add.append(params_temp)
                        
                    #create PARC contour
                    if params["PARC"] :
                        lat_parc, lon_parc = np.transpose(PARC_ST_NAZAIRE)  #defined in expe_data/expe_dict
                        X, Y = manage_projection.ll_to_xy(lon_parc, lat_parc, dom.get_data("CRS"))
                        params_temp = {"X":np.array(X)/1000, "Y":np.array(Y)/1000, "dom":dom, }
                        if "kwargs_PARC" in params : 
                            params_temp = manage_dict.select_params(params["kwargs_PARC"], params_temp)
                        params_temp = manage_dict.select_params(params_temp, default_params["PARC"])
                        params_add.append(params_temp)
                        
            elif typename in ["2DV"]:
                if "dom" in params :
                    dom = self.get_dom(params["dom"])
                else : 
                    dom = None
                vinterp = params["kwargs_get_data"]["vinterp"] 
                for varname_temp in ["levels", "points", "ZP"]:
                    if varname_temp in params and not varname_temp in vinterp :
                        vinterp[varname_temp] = params[varname_temp] 
                vinterp["points"] = self.get_points_for_vcross(dom, vinterp["points"])
                params["kwargs_get_data"]["vinterp"] = vinterp
                if not "Y" in params :
                    params["Y"] = manage_dict.getp("ZP", vinterp, "ZP") 
            elif typename in ["QUIVER", "BARBS"] :
                if "color" in params :
                    params["kwargs_plt"]["color"] = params["color"]
            
            if typename == "2DV" :
                params["plot_landsea"] = manage_dict.getp("plot_landsea", params, default=True)
            else :
                params["plot_landsea"] = manage_dict.getp("plot_landsea", params, default=False)
            if params["plot_landsea"] :
                params_LANDSEA = copy.deepcopy(default_params["LANDSEA"])
                if "kwargs_LANDSEA" in params :
                    params_LANDSEA = manage_dict.select_params(copy.deepcopy(params["kwargs_LANDSEA"]), params_LANDSEA)
                    params["kwargs_LANDSEA"] = None
                params_add.append(params_LANDSEA)
                
            if typename == "ZT" or ("X" in params and type(params["X"]) is str and "TIME" in params["X"]) or ("X" in params and type(params["X"]) in [list, np.array, np.ndarray] and type(np.array(params["X"]).item(0)) is datetime.datetime) :
                params["plot_nighttime"] = manage_dict.getp("plot_nighttime", params, default=True)
            else :
                params["plot_nighttime"] = manage_dict.getp("plot_nighttime", params, default=False) 
            if params["plot_nighttime"] :
                if "kwargs_NIGHTTIME" in params :
                    params_add.append(manage_dict.select_params(copy.copy(params["kwargs_NIGHTTIME"]), default_params["NIGHTTIME"]))
                    params["kwargs_NIGHTTIME"] = None
                #else :
                #    params_add.append(copy.copy(default_params["NIGHTTIME"]))
            params_list2.append(params)
            for params_supp in params_add :
                params_list2.append(params_supp)
        params_list = copy.copy(params_list2)
        #manage_dict.print_dict(params_list, "before kwargs_nighttime")
        
        #The parameters "video" must be extended to the whole plot
        ip0 = 0
        is_video = False
        is_contour = False
        for i_params, params in enumerate(params_list) :
            if not params["same_fig"] and i_params > 0 :
                ip1 = i_params
                for p in params_list[ip0:ip1] :
                    p["video"] = is_video
                    if is_contour and p["typ"] in ["LANDMASK", "LAT", "LON"] :
                        p["animate"] = is_video != False
                #reinitialize for next fig
                ip0 = i_params
                is_video = False
                is_contour = False
            if "video" in params and params["video"] != False :
                is_video = params["video"]
            if params["typ"] in ["CONTOUR", "MASK"]:
                is_contour = True
        ip1 = len(params_list)
        for p in params_list[ip0:ip1] :
            p["video"] = is_video
            if is_contour and p["typ"] in ["LANDMASK", "LAT", "LON"] :
                p["animate"] = is_video
        
        #The parameters "plot_nighttime" must be extended to the whole ax   
        ip0 = 0
        plot_nighttime = False
        for i_params, params in enumerate(params_list) :
            if not params["same_ax"] :
                ip1 = i_params
                kwargs_NIGHTTIME = copy.copy(default_params["NIGHTTIME"])
                for p in params_list[ip0:ip1] :
                    p["plot_nighttime"] = plot_nighttime
                    if "kwargs_NIGHTTIME" in p:
                        kwargs_NIGHTTIME = manage_dict.select_params(p["kwargs_NIGHTTIME"], kwargs_NIGHTTIME)
                if plot_nighttime :
                    params_list[ip1-1]["kwargs_NIGHTTIME"] = kwargs_NIGHTTIME
                #reinitialize for next fig
                ip0 = i_params
                plot_nighttime = False
            if "plot_nighttime" in params and params["plot_nighttime"] :
                plot_nighttime = True
        ip1 = len(params_list)
        kwargs_NIGHTTIME = copy.copy(default_params["NIGHTTIME"])
        for p in params_list[ip0:ip1] :
            p["plot_nighttime"] = plot_nighttime
            if "kwargs_NIGHTTIME" in p:
                kwargs_NIGHTTIME = manage_dict.select_params(p["kwargs_NIGHTTIME"], kwargs_NIGHTTIME)
        if plot_nighttime :
            params_list[ip1-1]["kwargs_NIGHTTIME"] = kwargs_NIGHTTIME 
        
        #NIGHTTIME, DATE must be at the end to be placed in the ax on the good position
        params_list2 = []
        for i_params, params in enumerate(params_list) :
            if params["typ"] not in ["NIGHTTIME"] or i_params == len(params_list)-1 or not params_list[i_params+1]["same_ax"] : #nighttime must be at the end
                params_list2.append(params)
            if (i_params==len(params_list)-1 or not params_list[i_params+1]["same_ax"]) :
                if params["plot_nighttime"] :
                    params_list2.append( manage_dict.select_params(params["kwargs_NIGHTTIME"], copy.copy(default_params["NIGHTTIME"])) )
                    if params["video"] != False :
                        params_list2[-1]["video"] = params["video"]
            if (i_params==len(params_list)-1 or not params_list[i_params+1]["same_fig"]) :
                if params["video"] != False and not params["typ"] == "DATE":
                    DATE = copy.copy(default_params["DATE"])
                    DATE["video"] = params["video"]
                    params_list2.append(DATE)
        params_list = copy.copy(params_list2)
        #manage_dict.print_dict(params_list, "end of prepare_params_for_get")
        return params_list
    
    def get_params_for_plot(self, params_list, **kwargs):
        for i_params, params in enumerate(params_list) :
            typename = params["typ"].upper() 
            #Get the missing data with dom.get_data or in the former figure if "same_ax" = True
            if "same_ax" in params and params["same_ax"] :
                i_before = i_params - 1
                while not params_list[i_before]["typ"] in ["2DH", "2DV"] and "same_ax" in params_list[i_before] and params_list[i_before]["same_ax"] and i_before > 0 : 
                    i_before -= 1
                params_before = params_list[i_before]
                params["compute_title"] = manage_dict.getp("compute_title", params, False)
            else : 
                params_before = {}
                params["compute_title"] = manage_dict.getp("compute_title", params, True)
            dom = self.get_dom_expe(params["dom"]) if "dom" in params else self.get_dom_expe(params_before["dom"]) if "dom" in params_before else None
            params["dom"] = dom
            kwargs_get_data = manage_dict.getp("kwargs_get_data", [params, params_before], default={} )
            if "zoom" in kwargs_get_data :
                if printDebug2 : print("zoom in kw_get")
                zoom_temp = kwargs_get_data["zoom"]
                if printDebug2 : print("zoom = ", zoom_temp)
                kwargs_get_data["zoom"] = self.get_zoom_index(dom, zoom_temp)
                if printDebug2 : print("then zoom = ", kwargs_get_data["zoom"])
            params["kwargs_get_data"] = kwargs_get_data
            params = self.get_params_from_same_or_dom(params, dom, params_before, kwargs_get_data)
            
            if "kwargs_get_data" in params and "vinterp" in params["kwargs_get_data"] :
                points = self.get_points_for_vcross(dom, params["kwargs_get_data"]["vinterp"]["points"])
                params["kwargs_get_data"]["vinterp"]["points"] = points
                
            if "X" in params and type(params["X"]) is str :
                # print(params["X"], i_params, params["typ"], dom.name)
                params["Xname"] = params["X"]
                params["X"] = self.get_data(dom, params["Xname"], **kwargs_get_data)
            if "Xname" in params :
                params["xlabel"] = manage_dict.getp("xlabel", params, default=self.get_legend(dom, params["Xname"]))
                if params["Xname"] == "WD" : 
                    params["xlim"] = [0, 360]
                    params["xticks"] = np.arange(0, 361, 45)
                    params["xticklabels"] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
                if params["Xname"] == "WD180" : 
                    params["xlim"] = [-180, 180]
                    params["xticks"] = np.arange(-180, 181, 45)
                    params["xticklabels"] = ["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"]
            if "Y" in params and type(params["Y"]) is str :
                params["Yname"] = params["Y"] 
                params["Y"] = self.get_data(dom, params["Yname"], **kwargs_get_data)
            if "Yname" in params :
                params["ylabel"] = manage_dict.getp("ylabel", params, default=self.get_legend(dom, params["Yname"]))
                if params["Yname"] == "WD" : 
                    params["ylim"] = [0, 360]
                    params["yticks"] = np.arange(0, 361, 45)
                    params["yticklabels"] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
                if params["Yname"] == "WD180" : 
                    params["ylim"] = [-180, 180]
                    params["yticks"] = np.arange(-180, 181, 45)
                    params["yticklabels"] = ["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"]
            if "Z" in params and type(params["Z"]) is str and typename not in ["TEXT"] :
                params["Zname"] = params["Z"] 
                params["Z"] = self.get_data(dom, params["Zname"], **kwargs_get_data) 
                if not params["typ"] in ["DATE", "TEXT"] and params["video"] == False and params["Z"].ndim > 2 and not 1 in params["Z"].shape:
                    raise(Exception("error in get_params_for_plot : dimension of Z > 2 , shapeZ = ", params["Z"].shape, ", please adapt the kwargs_get_data"))
                if not params["typ"] in ["DATE", "TEXT"] and params["video"] != False and params["Z"].ndim > 3 and not 1 in params["Z"].shape:
                    raise(Exception("error in get_params_for_plot : dimension of Z > 3 , shapeZ = ", params["Z"].shape, ", please adapt the kwargs_get_data"))
            if "Zname" in params :
                params["clabel"] = manage_dict.getp("clabel", params, default=self.get_legend(dom, params["Zname"]))
                if typename in ["2D_HORIZONTAL" , "2DH", "2D", "ZT", "VCROSS", "VERTCROSS", "2DV", "2D_VERTICAL", "SCATTER"] :
                    params["kwargs_plt"]["cmap"]= manage_dict.getp("cmap", params["kwargs_plt"], default= self.get_cmap(dom, params["Zname"]))
                if params["Zname"] == "WD" : 
                    params["clim"] = [0, 360]
                    params["ticks"] = np.arange(0, 361, 45)
                    params["ticklabels"] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
                if params["Zname"] == "WD180" : 
                    params["clim"] = [-180, 180]
                    params["ticks"] = np.arange(-180, 181, 45)
                    params["ticklabels"] = ["S", "SW", "W", "NW", "N", "NE", "E", "SE", "S"]
                
            for temp in ["U", "V"]:
                if temp in params and type(params[temp]) is str :
                    params[temp+"name"] = params[temp] 
                    params[temp] = self.get_data(dom, params[temp+"name"], **kwargs_get_data)
            
            #Get the other data
            if typename in ["LON"] : 
                LON = params["Z"]
                if LON.ndim == 3 :
                    LON = LON[0]
                    params["Z"] = LON
                lon_max = np.nanmax(LON)
                lon_min = np.nanmin(LON)
                lon_min, lon_max, dlon = manage_plot.nice_bounds(lon_min, lon_max, 5)
                params["kwargs_plt"]["levels"] = np.round(np.arange(lon_min, lon_max, dlon), 5)
            elif typename in ["LAT"] : 
                LAT = params["Z"]
                if LAT.ndim == 3 :
                    LAT = LAT[0]
                    params["Z"] = LAT
                lat_max = np.nanmax(LAT)
                lat_min = np.nanmin(LAT)
                lat_min, lat_max, dlat = manage_plot.nice_bounds(lat_min, lat_max, 5)
                params["kwargs_plt"]["levels"] = np.round(np.arange(lat_min, lat_max, dlat), 5)
            elif typename in ["2DV"] :
                #X is still undefined
                #Y must be averaged (I don't exactly remember why but it is probably a matter of NaNs)
                Z = params["Z"]
                X_vec = self.get_X_for_vinterp(params["kwargs_get_data"], Z=Z, KM=True)
                Y_vec = np.nanmean(params["Y"], axis = -1)
                if params["video"] != False :
                    Y_vec = Y_vec[0]
                params["X"], params["Y"] = np.meshgrid(X_vec, Y_vec)
            elif typename == "ZT" :
                if not "X" in params :
                    Z, t, _, _ = dom.get_ZT(**kwargs_get_data)
                    params["X"] = t
                    params["Y"] = Z
                    params["Xname"] = "TIME"
                    params["Yname"] = "Z"
            elif typename == "RECTANGLE":
                if not "xy" in params :
                    if "Xname" in params and "Yname" in params :
                        xy, width, height = self.get_rectangle_params(dom, params["rectangle"], Xname=params["Xname"], Yname=params["Yname"])
                    else :
                        xy, width, height = self.get_rectangle_params(dom, params["rectangle"])
                    params["xy"] = xy
                    params["width"] = width
                    params["height"] = height
            elif typename == "LINE":
                if not "ix1" in params :
                    params["y1"], params["y2"], params["x1"], params["x2"] = self.get_line(dom, params["points"])
                    if params["Xname"] == "X_KM":
                        params["x1"] = params["x1"]/1000 
                        params["x2"] = params["x2"]/1000 
                    if params["Yname"] == "Y_KM":
                        params["y1"] = params["y1"]/1000 
                        params["y2"] = params["y2"]/1000 
                if "color" in params :
                    params["kwargs_plt"]["color"] = params["color"]
                    params["kwargs_1"]["color"] = params["color"]
            elif typename == "POINT":
                if not "ix" in params :
                    y, x = self.get_point(dom, params["point"])
                    params["Y"] = np.array([y])
                    params["X"] = np.array([x])
                    if params["Xname"] == "X_KM":
                        params["X"] = params["X"]/1000 
                    if params["Yname"] == "Y_KM":
                        params["Y"] = params["Y"]/1000
                else :
                    X = dom.get_data(params["Xname"])
                    Y = dom.get_data(params["Yname"])
                    ix = params["ix"]
                    iy = params["iy"]
                    params["X"] = X[iy, ix]
                    params["Y"] = Y[iy, ix]
                if "color" in params :
                    params["kwargs_plt"]["color"] = params["color"]
            elif typename == "NIGHTTIME":
                if params["X"].ndim == 2 :
                    params["X"] = params["X"][:,0]
                if not "where" in params :
                    params["where"] = manage_time.is_nighttime(params["X"], params["locInfo"])
            elif typename == "LANDSEA":
                if not "where" in params :
                    params["where"] = self.get_data(dom, "LANDMASK", **kwargs_get_data)
                if params["X"].ndim == 2 :
                    if "axis" in params : 
                        params["X"] = params["X"][params["axis"]]
                    else :
                        params["X"] = params["X"][0]
            if params["video"] != False :
                if not "NT" in params :
                    if i_params>0 and params["same_ax"] and "NT" in params_list[i_params-1]:
                        params["NT"] = params_list[i_params-1]["NT"]
                    elif "TIME" in params:
                        params["NT"] = len(params["TIME"])
                    elif "dom" in params and "kwargs_get_data" in params :
                        kwargs_get_data = params["kwargs_get_data"]
                        dom = self.get_dom_expe(params["dom"])
                        TIME = self.get_data(dom, "TIME", **kwargs_get_data)
                        params["NT"] = len(TIME)
                    else :
                        if typename in ["ZT", "2D", "2DH", "2DV", "CONTOUR", "MASK", "SCATTER"] :
                            params["NT"] = len(params["Z"])
                        elif typename in ["QUIVER", "BARBS"] : 
                            params["NT"] = len(params["U"])
                        else :
                            params["NT"] = len(params["Y"])
                if i_params == 0 or not params["same_ax"] :
                    params["savefig"] = True
            if "savepath" in params :
                if not params["savepath"].startswith(self.figdir) :
                    params["savepath"] = self.figdir+params["savepath"]
        #manage_dict.print_dict(params_list, "get")
        
    def get_params_from_same_or_dom(self, params, dom, params_before, kwargs_get_data) :
        for params_key, params_i in params.items() :
            if type(params_i) is dict and not params_key.startswith("kwargs"):
                if "get" in params_i :
                    get = params_i["get"]
                    loop = True
                    j = 0
                    while loop :
                        if type(get[j]) is str :
                            if get[j].startswith("same"):
                                varname = get[j][5:]
                                if varname in params_before :
                                    # print("same : ", params["typ"], params_key, params_before["typ"], varname)
                                    params[params_key] = params_before[varname]
                                    loop = False
                            elif get[j].startswith("dom") and dom is not None :
                                varname = get[j][4:]
                                # print("get : ", dom.name, varname, kwargs_get_data["crop"])
                                params[params_key] = self.get_data(dom, varname, **kwargs_get_data)
                                loop = False
                        else : 
                            params[params_key] = get[j]
                            loop=False
                        j+=1
                        if j == len(get) :
                            loop = False
                else : 
                    params_before_i = manage_dict.getp(params_key, params_before, default={})
                    params[params_key] = self.get_params_from_same_or_dom(params_i, dom, params_before_i, kwargs_get_data)
        return params
    
    def get_title_savepath(self, params_list, **kwargs):
        dom_list = []
        typename_list = []
        varname_s_list = []
        vloc_s_list = []
        hloc_s_list = []
        time_s_list = []
        same_ax_list = []
        for params in params_list :
            same_ax_list.append(params["same_ax"])
            typename = params["typ"].upper()
            typename_list.append(typename)
            dom = params["dom"] if "dom" in params else params["expe"] if "expe" in params else None
            if dom is not None : 
                dom_list.append(dom.name)
            else :
                dom_list.append("")
            
            varname = varname_s = ""
            if dom is not None :
                if typename in ["2DH", "2DV", "2D", "CONTOUR", "MASK", "SCATTER"] and "Zname" in params:
                    varname_s = params["Zname"]
                    varname = self.get_legend(dom, params["Zname"], long=False, latex=None)
                elif typename in ["1D"] and "Yname" in params:
                    varname_s = params["Yname"]
                    varname = self.get_legend(dom, params["Yname"], long=False, latex=None)
            varname_s_list.append(varname_s)
            
            kw_get = params["kwargs_get_data"]
            vloc = vloc_s = ""
            if "hinterp" in kw_get and typename in ["2DH"] :
                hinterp = kw_get["hinterp"]
                level = str(int(hinterp["levels"]))
                ZP = hinterp["ZP"] if "ZP" in hinterp else "ZP"
                vloc_s = "_"+ZP+level
                vloc = ", at " + ZP + "=" + level + "m"
            elif "crop" in kw_get:
                crop_z = kw_get["crop"][0]
                if type(crop_z) is int :
                    vloc_s = "_iz"+str(crop_z)
                    vloc = ", iz="+str(crop_z)
            vloc_s_list.append(vloc_s)        
                    
            hloc = hloc_s = ""
            if "vinterp" in kw_get and typename in ["2DV"]:
                vinterp = kw_get["vinterp"]
                points = vinterp["points"]
                lat1 = str(round(points[0][0], 3))
                lon1 = str(round(points[0][1], 3))
                lat2 = str(round(points[1][0], 3))
                lon2 = str(round(points[1][1], 3))
                hloc_s = "_" + lat1 + "N" + lon1 + "E_" + lat2 + "N" + lon2 + "E" 
                hloc = ", from " + lat1 + "°N," + lon1 + "°E to " + lat2 + "°N," + lon2 + "°E" 
            hloc_s_list.append(hloc_s)   
                
            time_slice = None
            if "time_slice" in kw_get :
                time_slice = kw_get["time_slice"]
            elif "itime" in kw_get :
                time_slice = manage_time.get_time_slice(kw_get["itime"], dom.date_list, dom.max_time_correction)
            time = time_s = ""
            if time_slice is not None :
                date_list = dom.date_list[time_slice]
                if type(date_list) not in [list, np.array, np.ndarray] :
                    time_s = "_t" + dom.date2str(date_list, "WRF")
                    time = ",\n t=" + dom.date2str(date_list, "WRF")
                else :
                    time_s = "_t" + dom.date2str(date_list[0], "WRF") + "_"+ dom.date2str(date_list[-1], "WRF")
                    time = ",\n from " + dom.date2str(date_list[0], "WRF") + " to "+ dom.date2str(date_list[-1], "WRF")
            time_s_list.append(time_s)  
            
            if not "title" in params and params["compute_title"] and not params["same_ax"]: 
                params["title"] = varname + vloc + hloc + time
                
        new_ax_list = np.logical_not(same_ax_list)
        n_fig = np.sum(new_ax_list)
        i_params = 0
        for i_fig in range(n_fig) :
            params = params_list[i_params]
            ind_list = []
            if params["typ"] == "1D" :
                concate_name = True
            else :
                concate_name = False
            savefig = False
            savepath_exist = False
            i_params_0 = i_params
            while i_params < len(params_list)-1 and params_list[i_params+1]["same_ax"] :
                savefig = savefig or params_list[i_params]["savefig"]
                if "savepath" in params_list[i_params] :
                    savepath_exist = True
                if params_list[i_params]["typ"] == "1D" and concate_name :
                    ind_list.append(i_params)
                i_params += 1
            if not savepath_exist and savefig : 
                if not concate_name :
                    ind = i_params_0
                    params["savepath"] = dom_list[ind] + "_" + typename_list[ind] + "_" +  varname_s_list[ind] + vloc_s_list[ind] + hloc_s_list[ind] + time_s_list[ind]
                    print(params["savepath"])
                else :
                    n_diff = 0
                    i_diff = []
                    for i_k, k in enumerate([dom_list, typename_list, varname_s_list, vloc_s_list, hloc_s_list, time_s_list]) :
                        temp_list = k[ind]
                        if not np.all(temp_list == temp_list[0]) :
                            n_diff += 1
                            i_diff.append(i_k)
                    if n_diff == 0 :
                        params["savepath"] = dom_list[ind] + "_" + typename_list[ind] + "_" + varname_s_list[ind] + vloc_s_list[ind] + hloc_s_list[ind] + time_s_list[ind]
                    else :
                        savepath = ""
                        for i_k, k in enumerate([dom_list, typename_list, varname_s_list, vloc_s_list, hloc_s_list, time_s_list]) :
                            if k in i_diff :
                                for k_dom in k[ind] :
                                    savepath += k_dom
                                    if i_diff in [0, 1, 2] :
                                        savepath += "_"
                            else :
                                savepath += k[0]
                        params["savepath"] = savepath
                if "video" in params and params["video"] == True :
                    params["savepath"] = "video/" + params["savepath"]
                params["savepath"] = self.figdir + params["savepath"]
            i_params+=1 
    
    def get_X_for_vinterp(self, kw_get, Z=None, dom=None, KM=False) :
        if not "vinterp" in kw_get :
            raise(Exception("error in Proj.get_X_for_vinterp : no vinterp in kwargs_get_data"))
        else :
            points = self.get_points_for_vcross(dom, kw_get["vinterp"]["points"])
            kw_get["vinterp"]["points"] = points
        if Z is None :
            if dom is None :
                raise(Exception("error in Proj.get_X_for_vinterp : need either Z or dom to determine the X vector"))
            else :
                return dom.get_X_for_vinterp(kw_get, KM=KM)
        else :
            NX = np.shape(Z)[-1]
            distance, direction = manage_projection.haversine(points[0], points[1])
            if KM :
                distance = distance/1000
            return np.linspace(-distance/2, distance/2, NX)[:,0]
    
    def get_name(self, dom) :
        """
        Get name of dom or expe
        note  : dom can also be Expe if get_cmap is defined
        """
        try :
            dom = self.get_dom_expe(dom)
            return dom.name
        except :
            return dom
    
    def get_cmap(self, dom, varname) :
        """
        wrapper of dom.get_cmap
        note  : dom can also be Expe if get_cmap is defined
        """
        return dom.get_cmap(varname)
    
    def get_legend(self, dom, varname, **kwargs) :
        """
        wrapper of dom.get_legend
        note : dom can also be Expe if get_legend is defined
        """
        try : 
            return dom.get_legend(varname, **kwargs)
        except : 
            if varname in ARPS_WRF_VARNAMES_DICT :
                temp = ARPS_WRF_VARNAMES_DICT[varname]
                legend = temp[0] #legend_short
                units = temp[3]
                return manage_display.get_legend(legend, units, latex=True)
            else :
                return varname
    
    def get_NT(self, dom=None, Z=None, **kwargs_get_data):
        if dom is not None :
            dom = self.get_dom_expe(dom)
            return len(self.get_data(dom, "TIME", **kwargs_get_data))
        elif Z is not None :
            return len(Z)
        else :
            raise(Exception("error in Proj.get_NT : cannot deduce NT from dom or Z, please update this function or add NT in params"))
        

#################################################################################################################################
######  Some default plots
################################################################################################################################# 
    def plot_domain(self, dom, Zname="HT", cmap="Spectral_r"):
        dom = self.get_dom(dom)
        Zname = "HT"
        params = [{
            "typ" : "2DH",
            "Z" : Zname,
            "cmap" : cmap,
            "dom" : dom,
            "title" : "d"+dom.i_str,
            "plot_inner_doms" : True,
            "savepath" : self.figdir+"domains/domain"+dom.name+"_"+Zname
        }]
        for dom in self.tab_dom :
            params.append({
                "typ" : "RECTANGLE",
                "rectangle" : dom,
            })
        return self.plot_fig(params)

    def plot_dispo_expe(self):
        fig = plt.figure(figsize = [24, 8])
        ax = plt.subplot(111)
        count = 0
        for expe in DICT_EXPE_DATE : 
            date_list = DICT_EXPE_DATE[expe][4]
            if date_list is None :
                continue
            count += 1
            for i_date in range(len(date_list)//2) :
                date1 = manage_time.to_datetime(date_list[i_date*2])  
                date2 = manage_time.to_datetime(date_list[i_date*2+1])
                plt.plot([date1, date2], [count, count], "k-")
                ax.add_patch(matplotlib.patches.Rectangle([date1, count], date2-date1, 1, fill=True, color="k"))
                plt.text(date1, count+0.05, expe)
        return fig

    def plot_position_expe(self, dom_list, expe_list, varname="HT", NX_subplots=2) :
        if type(dom_list) is not list : dom_list = [dom_list]
        if type(expe_list) is not list : expe_list = [expe_list]
        params = []
        for dom in dom_list :
            dom = self.get_dom(dom)
            params.append({
                "typ" : "2DH", "dom" : dom, "Z" : varname, "title" : "",
            })
            for expe in expe_list :
                params.append({
                    "typ" : "POINT", "point" : expe, "label" : self.get_name(expe),
                })
        return self.plot_fig(params), params

    def plot_profile(self, varname_list, time_list, dom_list=[], expe=None, coord=None, ylim="data", legend_long=False, **kwargs_plot):
        if type(varname_list) is not list : varname_list = [varname_list]
        if type(time_list) is not list : time_list = [time_list]
        if type(dom_list) is str and dom_list == "ALL" :
            dom_list = self.tab_dom
        elif type(dom_list) is not list : 
            dom_list = [dom_list]
        if expe is None :
            lat = coord[0]
            lon = coord[1]
        else :
            expe = self.get_expe(expe)
            lat = expe.lat
            lon = expe.lon
        title = ""
        params = []
        for i_plot, t_prof in enumerate(time_list):
            for varname in varname_list :
                if expe is not None :
                    title = "vertical profile of " + self.get_legend(expe, varname, long=legend_long) + ", "+expe.name+" "+manage_projection.get_str_location(lat, lon) + ", date="+manage_time.to_datetime(t_prof).strftime("%Y-%m-%d_%H:%M:%S")
                    prof = expe.get_data(varname, itime=t_prof)
                    Z = expe.Z_vec
                    if ylim == "data" :
                        ylim = [0, 1.1*np.nanmax(Z)]
                    params.append({
                        "X" : prof,
                        "Y" : Z,
                        "style" : "k.-",
                        "kwargs_plt" : {
                              "linewidth" : 2,
                        },
                        "label" : expe.get_label(),
                        "xlabel" : self.get_legend(expe, varname, long=legend_long),
                        "ylim" : ylim,
                        "title" : title,
                    })
                else :
                    if ylim == "data" :
                        ylim = None
                    params.append({
                        "X" : [],
                        "Y" : [],
                        "ylim" : ylim,
                    })
                for i_dom, dom in enumerate(dom_list) :
                    dom = self.get_dom(dom)
                    iy, ix = self.nearest_index(dom, [lat, lon])
                    crop = ("ALL", iy, ix)
                    prof = self.get_data(dom, varname, itime=t_prof, crop=crop)
                    Z = self.get_data(dom, "ZP", itime=t_prof, crop=crop)
                    if title == "" : title = "vertical profile of " + self.get_legend(dom, varname) + ", "+manage_projection.get_str_location(lat, lon) + ", date="+manage_time.to_datetime(t_prof).strftime("%Y-%m-%d_%H:%M:%S")
                    params.append({
                        "same_ax" : True,
                        "X" : prof,
                        "Y" : Z,
                        "style" : self.get_style(i_dom),
                        "label" : dom.software+", d"+dom.i_str,
                        "xlabel" : self.get_legend(dom, varname, long=legend_long),
                        "ylabel" : self.get_legend(dom, "Z", long=legend_long),
                        "ylim" : ylim,
                        "title" : title,
                    })
        return self.plot_fig(params, **kwargs_plot), params
      

    def plot_ZT(self, varname_list, dom_list=[], expe=None, itime=None, coord=None, ylim="data", clim="data", legend_long=False, cropz="ALL", **kwargs_plot):
        if clim == "data" : clim_data = True
        else : clim_data = False
        if type(varname_list) is not list : varname_list = [varname_list]
        if type(dom_list) is str and dom_list == "ALL" :
            dom_list = self.tab_dom
        elif type(dom_list) is not list : 
            dom_list = [dom_list]
        if itime is None :
            start_list = []
            end_list = []
        if expe is None :
            lat = coord[0]
            lon = coord[1]
        else :
            expe = self.get_expe(expe)
            lat = expe.lat
            lon = expe.lon
            if itime is None :
                start_list.append(expe.date_list[0])
                end_list.append(expe.date_list[-1])
        if itime is None :
            for dom in dom_list :
                dom = self.get_dom(dom)
                start_list.append(dom.date_list[0])
                end_list.append(dom.date_list[-1])
            itime = (max(start_list), min(end_list))
        title = ""
        params = []
        kw_get = {
            "saved" : {},
            "itime" : itime,
        }
        for varname in varname_list :
            if expe is not None :
                title = expe.get_label() + " ," + self.get_legend(expe, varname, long=legend_long) + ", "+expe.name+" "+manage_projection.get_str_location(lat, lon)
                var = expe.get_data(varname, itime=itime)
                Z, t, _, _ = expe.get_ZT(itime=itime)
                if ylim == "data" :
                    ylim = [0, 1.1*np.nanmax(Z)]
                if clim_data :
                    clim = [np.nanmin(var), np.nanmax(var)]
                params.append({
                    "typ" : "ZT",
                    "X" : t,
                    "Y" : Z,
                    "Z" : var,
                    "kwargs_plt" : {
                        "cmap" : self.get_cmap(expe, varname),
                    },
                    "ylabel" : self.get_legend(expe, "Z", long=legend_long),
                    "ylim" : ylim,
                    "clim" : clim,
                    "title" : title,
                    "clabel" : self.get_legend(expe, varname, long=True),
                })
            else :
                if ylim == "data" :
                    ylim = None
                if clim_data : 
                    clim = None
                params.append({
                    "X" : [],
                    "Y" : [],
                    "ylim" : ylim,
                    "clim" : clim,
                })
            for i_dom, dom in enumerate(dom_list) :
                dom = self.get_dom(dom)
                iy, ix = self.nearest_index(dom, [lat, lon])
                crop = (cropz, iy, ix)
                var = self.get_data(dom, varname, crop=crop, **kw_get)
                Z, t, _, _ = dom.get_ZT(itime=itime, crop=crop)
                title = dom.software + ", d" + dom.i_str +" ,"+ self.get_legend(dom, varname) + ", "+manage_projection.get_str_location(lat, lon)
                params.append({
                    "typ" : "ZT",
                    "X" : t,
                    "Y" : Z,
                    "Z" : var,
                    "kwargs_plt" : {
                        "cmap" : self.get_cmap(dom, varname),
                    },
                    "ylabel" : self.get_legend(dom, "Z", long=legend_long),
                    "ylim" : ylim,
                    "clim" : clim,
                    "title" : title,
                    "clabel" : self.get_legend(dom, varname, long=True),
                })
        return self.plot_fig(params, **kwargs_plot), params

    def plot_Taylor_diagram(self, reference, variable, itime=None, timestep=None, dom_list=None, location=None, levels=None, legend=None, ref_legend=None, marker_list=None, color_list=None, saved_list=None, title=True, unit=None, **kw_get) :
        #preparing parameters
        reference = self.get_dom_expe(reference)
        ref_legend = reference.name if ref_legend is None else ref_legend
        if dom_list is None :
            dom_list = copy.copy(self.tab_dom)
        dom_list2 = []
        for dom in dom_list :
            dom = self.get_dom_expe(dom)
            if dom is not reference :
                dom_list2.append(dom)
        dom_list = dom_list2
        if itime is None :
            start_date = reference.date_list[0]
            end_date = reference.date_list[-1]
            if timestep is None : 
                timestep = reference.date_list[1] - reference.date_list[0]
            for dom in dom_list :
                if dom.date_list[0] > start_date :
                    start_date = dom.date_list[0]
                if dom.date_list[-1] > end_date :
                    end_date = dom.date_list[-1]
                if timestep < dom.date_list[1] - dom.date_list[0] :
                    timestep = dom.date_list[1] - dom.date_list[0]
            if end_date - start_date < 2*timestep : 
                raise(Exception("error in Proj.plot_Taylor_diagram : start_date =" + str(start_date) + ", end_date =" + str(end_date) + ", timestep =" + str(timestep)))
            itime = (start_date, end_date, timestep)
        if levels is None :
            if isinstance(reference, Expe) :
                levels = reference.Z_vec
            else :
                levels = np.linspace(50, 500, 10)
        hinterp = {
            "levels":levels,
        }
        
        #getting reference data
        if isinstance(reference, Expe) :
            crop_ref = None
            if type(levels) in [float, int] :
                if levels in reference.Z_vec :
                    crop_ref = int(np.argwhere(reference.Z_vec == levels))
                else : 
                    raise(Exception("error levels :" + str(levels) + "is not in reference.Z_vec :" + str(reference.Z_vec)))
            data_ref = self.get_data(reference, variable, itime=itime, crop=crop_ref, **kw_get)
        else :
            if location is not None :
                iy, ix = self.nearest_index(dom, location)
                crop = ("ALL", iy, ix)
            else :
                crop = ("ALL", "ALL", "ALL")
            data_ref = self.get_data(reference, variable, itime=itime, crop=crop, hinterp=hinterp, **kw_get)
        std_ref = np.nanstd(data_ref)
        pos = np.logical_not(np.isnan(data_ref.flatten()))
        
        #creating figure
        fig = plt.figure(figsize=(20,20), dpi=120)
        dia = manage_TaylorDiagram.TaylorDiagram(std_ref, fig=fig, rect=221, label=ref_legend, unit=unit)
        
        corrcoef_min = 1
        corrcoef_max = 0
        std_max = 1.5*std_ref
        for i_dom, dom in enumerate(dom_list) :
            marker = "o" if marker_list is None else marker_list[i_dom]
            color = None if color_list is None else color_list[i_dom]
            saved = None if saved_list is None else saved_list[i_dom]
            #getting element data
            if location is not None :
                iy, ix = self.nearest_index(dom, location)
                crop = ("ALL", iy, ix)
            elif isinstance(reference, Expe) :
                iy, ix = self.nearest_index(dom, [reference.lat, reference.lon])
                crop = ("ALL", iy, ix)
            else :
                crop = ("ALL", "ALL", "ALL")
            data_i = self.get_data(dom, variable, itime=itime, crop=crop, hinterp=hinterp, saved=saved)
            #calculating Pearson coefficient and std
            corrcoef_i = np.corrcoef(data_ref.flatten()[pos], data_i.flatten()[pos])[0, 1]
            corrcoef_min = min(corrcoef_min, corrcoef_i)
            corrcoef_max = max(corrcoef_max, corrcoef_i)
            std_i = np.nanstd(data_i)
            #adding point to the figure
            if legend is None :
                label = dom.name
            else :
                label = legend[i_dom]
            dia.add_sample(std_i, corrcoef_i, marker=marker, color=color, ms=10, ls='',label=label)

        # Add RMS contours, and label them
        contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
        dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
        if title :
            if levels is not None and type(levels) in [int, float, np.int64, np.float] :
                dia._ax.set_title("Taylor Diagram for "+variable+" at "+str(levels)+" m")
            else :
                dia._ax.set_title("Taylor Diagram for "+variable)
        plt.plot(np.array([0, std_max*corrcoef_min]), [0, std_max*np.sqrt(1-corrcoef_min**2)], "k--")
        plt.plot(np.array([0, std_max*corrcoef_max]), [0, std_max*np.sqrt(1-corrcoef_max**2)], "k--")
        plt.legend(loc="lower left")
        plt.show()
        savepath = self.figdir+"comp_"+reference.name+"/"+dom_list[0].name+"_Taylor1_"+variable
        if type(levels) in [int, float] :
            savepath += "_ZP"+str(int(levels))+"m"
        savepath += ".png"
        if not os.path.exists(os.path.dirname(savepath)):
            print('Create figure directory in {0}'.format(os.path.dirname(savepath)))
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, format="png", dpi=120)
        return fig
        
    def plot_mesh(self, dom, savepath=None):
        dom = self.get_dom(dom)
        HT = dom.get_data("HT")
        pos = np.where(HT == np.min(HT))
        iy, ix = pos[0][0], pos[1][0]
        crop = ("ALL", iy, ix)
        kw_get = {
            "crop" : crop, 
            "itime" : 0,
        }
        Z_ZSTAG = dom.get_data("Z_ZSTAG", **kw_get)
        Z = dom.get_data("Z", **kw_get)
        DZ_cell = np.diff(Z_ZSTAG) 
        n = len(Z)
        params = [{
            "X" : range(1, n+1), "Y" : Z, "style" : "+", "label" : "$Z$", "dom" : dom, "grid" : "both", "grid_which":"both",
            "yscale" : "log", "xlabel":  "$i_z$", "ylabel" : "Z or $\Delta Z$ (m)", "xlim" : [0, n+1], "savepath" : savepath,
            "kwargs_plt" : { "markersize" : 10, "markeredgewidth" : 2}, "dpi": 120,
        },{
            "same" : -1, "Y" : DZ_cell, "style" : "x", "label" : "$\Delta Z$", "same_ax" : True,
        }]
        fig = self.plot_fig(params)
        return Z, DZ_cell, fig

#################################################################################################################################
######  Old forgotten methods
################################################################################################################################# 
    def rename_postproc_variable(self, old_varname, new_varname):
        """ Rename a variable in the netCDF postproc files of the domains
        Parameters
            self (Proj)
            old_varname (string)
            new_varname (string)
        Mathieu LANDREAU 21/03/25
        """
        import netCDF4 as nc
        for dom in self.tab_dom :
            print("")
            print(dom.name)
            path_list = list(dom.output_filenames["post"].items())
            for it in range(len(path_list)) :
                print(it, end="")
                path = path_list[it][0]
                with nc.Dataset(path, mode="a") as file :
                    if old_varname in file.variables :
                        file.renameVariable(old_varname,new_varname)
                        print("o", end=" ")
                    else :
                        print(".", end=" ")

    def display(self, pref = ""):
        """ Display itself
        Parameters
            self (Proj)
        Optional
            pref (str): prefix printed before the each line
        Mathieu LANDREAU 21/12/22
        """
        print("####################################################")
        print("################ SIMULATION n°" + self.data_dir + " ###################")
        print("####################################################")
        print("domains : ", self.domains)
        manage_display.display_var(self.postprocdir, 'postprocdir', pref)
        manage_display.display_var(self.tmpdir, 'tmpdir', pref)
        manage_display.display_var(self.figdir, 'figdir', pref)
        print("")
        for dom in self.tab_dom:
            dom.display(pref) 
    
    
    
    
    
  
