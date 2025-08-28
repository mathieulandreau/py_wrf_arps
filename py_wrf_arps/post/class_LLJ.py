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
        print(it, end=" ", flush=True)
        if not dry :
            LLJ, LLJ_IZ, LLJ_Z, LLJ_MH, LLJ_PROM, LLJ_WIDTH = self.detect_LLJ(dom, itime=it, crop=crop, **kw)
            dom.write_postproc("LLJ_Z", LLJ_Z, ('y', 'x'), itime=it, long_name="LLJ core height", standard_name="LLJ_Z", units="m", latex_units="m", typ='f4') 
            dom.write_postproc("LLJ", LLJ, ('y', 'x'), itime=it, long_name="LLJ detection", standard_name="LLJ", units="", latex_units="", typ='i2')  
            dom.write_postproc("LLJ_IZ", LLJ_IZ, ('y', 'x'), itime=it, long_name="LLJ core index", standard_name="LLJ_IZ", units="", latex_units="", typ='i2') 
            dom.write_postproc("LLJ_MH", LLJ_MH, ('y', 'x'), itime=it, long_name="LLJ core speed", standard_name="LLJ_MH", units="m.s-1", latex_units="m.s^{-1}", typ='f4') 
            dom.write_postproc("LLJ_PROM", LLJ_PROM, ('y', 'x'), itime=it, long_name="LLJ peak prominence", standard_name="LLJ_PROM", units="m.s-1", latex_units="m.s^{-1}", typ='f4') 
            dom.write_postproc("LLJ_WIDTH", LLJ_WIDTH, ('y', 'x'), itime=it, long_name="LLJ peak width", standard_name="LLJ_WIDTH", units="m", latex_units="m", typ='f4') 
                
    
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
        print("NT = ", NT)
        if nprocs > 1 :
            inputs = [(it, dom, dry, crop, kw) for it in IT]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.write_postproc_1time, inputs)
        else :
            for it in IT :
                self.write_postproc_1time(it, dom, dry, crop, kw)
    
    def calculate_LLJ2_Z(self, dom, itime, DX_smooth_rel=10, DX_smooth=None):
        """ Get data and get LLJ2_Z by interpolating the gradient DZ_MH just above and below the jet
        Parameters
            self (LLJ)
            dom: see classProj.Proj.get_dom
            itime: see classDom.get_data
        Optional
            DX_smooth_rel (float): if DX_smooth is None, it takes the value of DX_smooth_rel*DX_KM, use the same as previously in detect_LLJ
        28/08/2025 : Mathieu Landreau
        """ 
        dom = self.sim.get_dom(dom)
        LLJ_IZ, LLJ_Z = dom.get_data(["LLJ_IZ", "LLJ_Z"], itime=itime)
        if np.all(LLJ_IZ < 0):
            return LLJ_Z
        mask = LLJ_IZ < 0
        LLJ_IZ = np.expand_dims(LLJ_IZ.astype(int), axis=0)
        izmin = max(0, np.min(LLJ_IZ[LLJ_IZ>=0])-1)
        izmax = min(dom.get_data("NZ"), np.max(LLJ_IZ)+2)
        LLJ_IZ[LLJ_IZ < 0] = izmin+1
        if DX_smooth is None and DX_smooth_rel is not None:
            DX_KM = dom.get_data("DX")/1000
            DX_smooth = DX_smooth_rel*DX_KM
        MH, Z = dom.get_data(["MH", "Z"], DX_smooth=DX_smooth, itime=itime, crop=([izmin, izmax], "ALL", "ALL"))
        zaxis = 0
        MHim1 = np.take_along_axis(MH, LLJ_IZ-izmin-1, axis=zaxis)
        MHi = np.take_along_axis(MH, LLJ_IZ-izmin, axis=zaxis)
        MHip1 = np.take_along_axis(MH, LLJ_IZ-izmin+1, axis=zaxis)
        Zim1 = np.take_along_axis(Z, LLJ_IZ-izmin-1, axis=zaxis)
        Zi = np.take_along_axis(Z, LLJ_IZ-izmin, axis=zaxis)
        Zip1 = np.take_along_axis(Z, LLJ_IZ-izmin+1, axis=zaxis)
        if np.any(LLJ_IZ == 0):
            MHim1[LLJ_IZ==0] = 0
            Zim1[LLJ_IZ==0] = 0
        y1 = .5 * (Zim1 + Zi) #Z(i-1/2)
        y2 = .5 * (Zi + Zip1) #Z(i+1/2)
        x1 = (MHi - MHim1)/(Zi - Zim1) #DZ_MH(i-1/2)
        x2 = (MHip1 - MHi)/(Zip1 - Zi) #DZ_MH(i+1/2)
        # We define LLJ2_Z as the height at which DZ_MH == 0
        # Thus we search b in the linear function y(x) = ax+b where y is Z and x is DZ_MH
        # we can proove that :
        LLJ2_Z = np.squeeze((-y2*x1 + y1*x2)/(x2-x1))
        #where LLJ is undefined we apply the LLJ_Z default value
        LLJ2_Z[mask] = LLJ_Z[mask]
        return LLJ2_Z

    def write_postproc2_1time(self, it, dom, dry, ):
        print(it, end=" ", flush=True)
        if not dry :
            LLJ2_Z = self.calculate_LLJ2_Z(dom, itime=it)
            dom.write_postproc("LLJ2_Z", LLJ2_Z, ('y', 'x'), itime=it, long_name="LLJ better core height", standard_name="LLJ2_Z", units="m", latex_units="m", typ='f4') 

    def write_postproc2(self, dom, itime="ALL_TIMES", nprocs=1, dry=False):
        """ Save the results in the postproc file
        Parameters
            self (LLJ)
        Optional
            savepath (str): path to save the pickle file
        04/03/2025 : Mathieu Landreau
        """ 
        dom = self.sim.get_dom(dom)
        if dom.software in ["AROME"]: raise(Exception("Cannot write postproc with this kind of domain: ", dom.software))
        IT, TIME = dom.get_data(["it", "TIME"], itime=itime)
        if not manage_list.is_iterable(IT):
            IT = [IT]
        NT = len(IT)
        print("NT = ", NT)
        if nprocs > 1 :
            inputs = [(it, dom, dry) for it in IT]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.write_postproc2_1time, inputs)
        else :
            for it in IT :
                self.write_postproc2_1time(it, dom, dry)