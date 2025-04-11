from ..class_proj import Proj
from ..WRF_ARPS import Dom
from ..lib import manage_list
from ..post import manage_LLJ

import numpy as np
import scipy
import os
# I don't understand why it didn't work with multiprocessing.Pool, I found the solution at:
# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
from multiprocessing.pool import ThreadPool as Pool


class LLJ(): 
    def __init__(self, sim):
        """ compute LLJ characteristics and save in post proc files
        Parameters
            self (LLJ)
            sim: a class_Proj object
        20/03/2025 : Mathieu Landreau
        """  
        self.sim = sim
        
    def detect_LLJ(self, dom, max_height=500, prom_abs=2, prom_rel=0.2, width=50, DX_smooth_rel=10, DX_smooth=None, **kw_get):
        """ Get data and call detect_LLJ (defined outside the class)
        Parameters
            self (LLJ)
            dom: see classProj.Proj.get_dom
        Optional
            max_height, prom_abs, prom_rel, width: see detect_LLJ
            DX_smooth_rel (float): if DX_smooth is None, it takes the value of DX_smooth_rel*DX_KM
            kw_get (dict): see Dom.get_data
        20/03/2025 : Mathieu Landreau
        """ 
        if DX_smooth is None and DX_smooth_rel is not None:
            DX_KM = self.sim.get_data(dom, "DX")/1000
            DX_smooth = DX_smooth_rel*DX_KM
        MH, Z, IZ = self.sim.get_data(dom, ["MH", "Z", "iz"], DX_smooth=DX_smooth, **kw_get)
        zaxis = self.sim.get_dom(dom).find_axis("z", varname="MH", **kw_get)
        return manage_LLJ.detect_LLJ(MH, Z, IZ, zaxis, max_height, prom_abs, prom_rel, width)
    
    def write_postproc_1time(self, it, dom, dry, crop, kw):
        print(it, end=" ")
        if not dry :
            LLJ, LLJ_IZ, LLJ_Z, LLJ_MH, LLJ_PROM, LLJ_WIDTH = self.detect_LLJ(dom, itime=it, crop=crop, **kw)
            dom.write_postproc("LLJ", LLJ, ('y', 'x'), itime=it, long_name="LLJ detection", standard_name="LLJ", units="", latex_units="", typ=np.int64)  
            dom.write_postproc("LLJ_IZ", LLJ_IZ, ('y', 'x'), itime=it, long_name="LLJ core index", standard_name="LLJ_IZ", units="", latex_units="", typ=np.int64) 
            dom.write_postproc("LLJ_Z", LLJ_Z, ('y', 'x'), itime=it, long_name="LLJ core height", standard_name="LLJ_Z", units="m", latex_units="m", typ=np.float32) 
            dom.write_postproc("LLJ_MH", LLJ_MH, ('y', 'x'), itime=it, long_name="LLJ core speed", standard_name="LLJ_MH", units="m.s-1", latex_units="m.s^{-1}", typ=np.float32) 
            dom.write_postproc("LLJ_PROM", LLJ_PROM, ('y', 'x'), itime=it, long_name="LLJ peak prominence", standard_name="LLJ_PROM", units="m.s-1", latex_units="m.s^{-1}", typ=np.float32) 
            dom.write_postproc("LLJ_WIDTH", LLJ_WIDTH, ('y', 'x'), itime=it, long_name="LLJ peak width", standard_name="LLJ_WIDTH", units="m", latex_units="m", typ=np.float32) 
                
    
    def write_postproc(self, dom, itime="ALL_TIMES", nprocs=1, crop=None, max_height=500, dry=False, **kw):
        """ Save the results in the postproc file
        Parameters
            self (LLJ)
        Optional
            savepath (str): path to save the pickle file
        04/03/2025 : Mathieu Landreau
        """ 
        dom = self.sim.get_dom(dom)
        if dom.software in ["AROME"]: raise(Exception("Cannot write postproc with this kind of domain: ", dom.software))
        if crop is None:
            izmax = dom.nearest_z_index(1.2*max_height)
            crop = ([0, izmax], "ALL", "ALL")
        IT, TIME = dom.get_data(["it", "TIME"], itime=itime)
        
        # self.temp_args = (dom, dry, crop, kw)
        if not manage_list.is_iterable(IT):
            IT = [IT]
        NT = len(IT)
        if nprocs > 1 :
            inputs = [(it, dom, dry, crop, kw) for it in IT]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.write_postproc_1time, inputs)
        else :
            for it in IT :
                self.write_postproc_1time(it, dom, dry, crop, kw)
    