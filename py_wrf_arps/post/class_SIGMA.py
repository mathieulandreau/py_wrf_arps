from ..class_proj import Proj
from ..WRF_ARPS import Dom

import numpy as np
import matplotlib.pyplot as plt


class SIGMA(): 
    def __init__(self, sim):
        """ Use several domains to compute the coast orientation with different values of sigma (scale)
        Parameters
            self (SIGMA)
            sim: a class_Proj object
        20/03/2025 : Mathieu Landreau
        """  
        self.sim = sim
        self.prepare_sigma_vec()
        self.p = {}
        
    def prepare_sigma_vec(self):
        """ Prepare the sigma_vec values
        Parameters
            self (SIGMA)
        03/07/2025 : Mathieu Landreau
        """ 
        lin = np.arange(1, 3, 2/10)
        sigma_vec = []
        for dom in self.sim.tab_dom[::-1] :
            sigma_vec = sigma_vec + list(np.array(lin)*dom.get_data("DX")*4)
        sigma_vec = sigma_vec + list(np.array(lin)*dom.get_data("DX")*12)
        sigma_vec = sigma_vec + list(np.array(lin)*dom.get_data("DX")*36)
        self.sigma_vec = np.array(sigma_vec)
        self.Nsigma = len(self.sigma_vec)
        # print(sigma_vec)
        plt.loglog(sigma_vec, sigma_vec, ".")
        
    def compute_COR_on_individual_domains(self) :
        """ Compute COR(sigma) using each domain separately
        Parameters
            self (SIGMA)
        03/07/2025 : Mathieu Landreau
        """ 
        plt.figure()
        for domstr in self.sim.tab_dom_str :
            self.p[domstr] = {}
            dom = self.sim.get_dom(domstr)
            NY, NX, DY, DX, BDY_DIST = dom.get_data(["NY", "NX", "DY", "DX", "BDY_DIST"])
            LX = NX*DX
            LY = NY*DY
            L = min(LX, LY)
            COR = np.zeros((self.Nsigma, NY, NX))
            for isig, sigma in enumerate(self.sigma_vec):
                if sigma > 4*dom.get_data("DX")-1e-5 and sigma < L/6 + 1e-5 :
                    CORi = dom.get_data("COR", sigma=sigma/1e3)
                    pos = sigma > BDY_DIST/3
                    CORi[pos] = np.nan
                    COR[isig] = CORi
                else :
                    COR[isig] = np.nan
            self.p[domstr]["COR"] = COR
            self.p[domstr]["BDY_DIST"] = BDY_DIST
            plt.semilogx(self.sigma_vec, self.p[domstr]["COR"][:, NY//2, NX//2], ".")
    
    def compute_COR_with_other_domains(self) :
        """ Compute COR(sigma) using other domains 
        Parameters
            self (SIGMA)
        03/07/2025 : Mathieu Landreau
        """ 
        for idom, domstr in enumerate(self.sim.tab_dom_str[::-1]) :
            for idom2, domstr2 in enumerate(self.sim.tab_dom_str[::-1]) :
                print(domstr, domstr2)
                if idom2 != idom :
                    dom = self.sim.get_dom(domstr)
                    dom2 = self.sim.get_dom(domstr2)
                    LAT2, LON2, DX2 = dom2.get_data(["LAT", "LON", "DX"])
                    for isig, sigma in enumerate(self.sigma_vec):
                        if np.any(np.isnan(self.p[domstr]["COR"][isig])) and not np.all(np.isnan(self.p[domstr2]["COR"][isig])):
                            CORi = self.p[domstr]["COR"][isig]
                            CORi2 = dom.interpolate_to_self_grid(LAT2, LON2, self.p[domstr2]["COR"][isig], interp="nearest_neighbor", max_dist_km=DX2/1e3)
                            pos = np.isnan(CORi)
                            CORi[pos] = CORi2[pos]
                            self.p[domstr]["COR"][isig] = CORi
        plt.figure()
        domstr = self.sim.tab_dom_str[-1]
        NY, NX = self.sim.get_data(domstr, ["NY", "NX"])
        plt.semilogx(self.sigma_vec, self.p[domstr]["COR"][:, NY//2, NX//2], ".")
        
    def write_postproc_COR(self):
        """ write COR(sigma) to postproc files
        Parameters
            self (SIGMA)
        03/07/2025 : Mathieu Landreau
        """ 
        for domstr in self.sim.tab_dom_str :
            dom = sim.get_dom(domstr)
            for isig,sigma in enumerate(self.sigma_vec):
                sigma_str = str(int(sigma))
                dom.write_postproc("COR"+sigma_str, self.p[domstr]["COR"][isig], ("y", "x"), itime=None, long_name="Coast orientation sigma="+sigma_str+"m", standard_name="COR"+sigma_str, units="°", latex_units="°", typ=np.float32)
                
    def complete_procedure(self):
        self.compute_COR_on_individual_domains()
        self.compute_COR_with_other_domains()
        self.write_postproc_COR()