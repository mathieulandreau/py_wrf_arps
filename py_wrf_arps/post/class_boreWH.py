import numpy as np
import scipy
from matplotlib import pyplot as plt
from ..lib import constants

# Equation 3.7
def system(x, y, Cb, D0, bore):
    eta, dz_eta = y 
    N2 = np.interp(x-eta, bore.Z1, bore.NBV2)
    dz2_eta = - N2/Cb**2 * (eta - (bore.eps*D0)/(bore.rho0*bore.gp))
    return [dz_eta, dz2_eta]

class BoreWH():
    """
    This class is based on the following paper :
        White, Brian L., et Karl R. Helfrich. 2014. « A Model for Internal Bores in Continuous Stratification ». 
        Journal of Fluid Mechanics 761 (December):282‑304. 
        https://doi.org/10.1017/jfm.2014.599.
        
    From given ambient and bore density profiles we compute the algorithm described in section 3.6 to determine Cb, the expected eta profile

    """
    def __init__(self, Z1, PRHO1, Z2, PRHO2, H, eps):
        #epsilon from eq 3.5
        self.eps = eps
        #limit height
        self.H = H
        self.Zmin = np.min(Z1)
        # ambient
        keep = Z1 <= H
        self.Z1 = Z1[keep]
        self.PRHO1 = PRHO1[keep]
        self.Z3 = self.Z1
        #bore
        keep = Z2 <= H
        self.Z2 = Z2[keep]
        self.PRHO2 = PRHO2[keep]
        #densities to define b
        self.r_0 = self.PRHO1[0] #bottom density
        self.r_H = np.interp(self.H, self.Z1, self.PRHO1) #top density
        self.rho0 = 0.5*(self.r_0 + self.r_H)
        self.gp = (self.r_0 - self.r_H)*constants.G/self.rho0 #g', defined after eq.3.6
        #b profiles
        self.b1 = self.compute_b(self.PRHO1)
        self.b2 = self.compute_b(self.PRHO2)
        #hb
        self.hb1 = self.compute_hb(self.b1, self.Z1)
        self.hb2 = self.compute_hb(self.b2, self.Z2)
        #Brunt-Vaisala frequency
        DZ_PRHO = np.gradient(self.PRHO1) / np.gradient(self.Z1)
        self.NBV2 = -constants.G/self.rho0 * DZ_PRHO
        self.NBV2[self.NBV2 < 0] = 0
        DZ_PRHO = np.gradient(self.PRHO2) / np.gradient(self.Z2)
        self.NBV22 = -constants.G/self.rho0 * DZ_PRHO
        self.NBV22[self.NBV22 < 0] = 0
        #init solution and guess
        self.eta = 0*self.Z1
        self.D0 = 0
        self.compute_profile3()
        
    ##################################################################
    #### Initial guess
    ##################################################################
    
    def define_guess(self, Cb_guess, dz_eta_0_guess, D0_guess):
        self.Cb_guess = Cb_guess
        self.dz_eta_0_guess = dz_eta_0_guess
        self.D0_guess = D0_guess
        self.Cb = Cb_guess
        self.dz_eta_0 = dz_eta_0_guess
        self.D0 = D0_guess
    
    def solve_ivp(self, eval_H=False):
        t_eval = [self.H] if eval_H else self.Z1
        y0 = [0, self.dz_eta_0]
        # Resolve ODE in [Zmin, H]
        sol = scipy.integrate.solve_ivp(system, [self.Zmin, self.H], y0, t_eval=t_eval, args=(self.Cb, self.D0, self))
        if eval_H :
            # to solve Cb we want eta(H) = 0 so we return eta(H)
            return sol.y[0, -1]
        else :
            self.eta = sol.y[0]
            self.compute_profile3()
        
    ##################################################################
    #### Step 1 in section 3.6
    ##################################################################
    
    def shooting_method_Cb(self, Cb_guess, eval_H=True):
        # Initial conditions : eta(0) = 0, eta'(0) = dz_eta_0
        self.Cb = Cb_guess
        # to solve Cb we want eta(H) = 0 so we return eta(H)
        return self.solve_ivp(eval_H=eval_H)
    
    def solve_Cb(self) :
        # solve
        sol_Cb = scipy.optimize.root(self.shooting_method_Cb, x0=self.Cb_guess) 
        # call again shooting method with the solution to save in self
        self.shooting_method_Cb(sol_Cb.x[0], eval_H=False)
    
    ##################################################################
    #### Step 2 in section 3.6
    ################################################################## 
    
    def shooting_method_dz_eta_0(self, dz_eta_0_guess):
        self.dz_eta_0 = dz_eta_0_guess
        # Solve Cb first
        self.solve_Cb()
        self.solve_ivp(eval_H=False)
        # to solve dz_eta_0 we want hb3 = hb2 so we return hb3 - hb2
        return self.hb3 - self.hb2
    
    def solve_dz_eta_0(self) :
        sol_dz_eta_0 = scipy.optimize.root(self.shooting_method_dz_eta_0, x0=self.dz_eta_0_guess) 
        self.shooting_method_dz_eta_0(sol_dz_eta_0.x[0])
        
    ##################################################################
    #### Step 3 in section 3.6
    ################################################################## 
    
    def shooting_method_D0(self, D0_guess):
        self.D0 = D0_guess
        # Solve dz_eta_0 first
        self.solve_dz_eta_0()
        #eq.3.4
        f = 0.25*self.rho0*self.Cb**2*self.dz_eta**3 - (1-0.5*self.dz_eta)*self.Delta
        # to solve dz_eta_0 we want hb3 = hb2 so we return hb3 - hb2
        res = np.trapz(f, x=self.Z1)
        print(res, self.D0)
        return res
    
    def solve_D0(self) :
        sol_D0 = scipy.optimize.root(self.shooting_method_D0, x0=self.D0_guess) 
        self.shooting_method_D0(sol_D0.x[0])
    
    ##################################################################
    #### Equation function
    ################################################################## 
    
    def compute_profile3(self) :
        # Compute the expected bore profile with eta
        self.PRHO3 = np.interp(self.Z1-self.eta, self.Z1, self.PRHO1)
        self.b3 = self.compute_b(self.PRHO3)
        self.hb3 = self.compute_hb(self.b3, self.Z1)
        self.dz_eta = np.gradient(self.eta)/np.gradient(self.Z1)
        self.Delta = self.D0*(0.5+self.eps*(self.b3-0.5))
        
    def compute_b(self, PRHO) :
        # buoyancy, defined in section 3.2, before eq.3.5
        return (PRHO - self.r_H)/(self.r_0 - self.r_H)
    
    def compute_hb(self, b, Z) :
        # bore amplitude, defined in section 3.6
        return np.trapz(b, x=Z)
        
    
    ##################################################################
    #### Display
    ################################################################## 
    
    def print(self):
        print(f"----------------------INIT")
        print(f"H : {self.H}")
        print(f"eps : {self.eps}")
        print(f"r_0 : {self.r_0}")
        print(f"r_H : {self.r_H}")
        print(f"rho0 : {self.rho0}")
        print(f"gp : {self.gp}")
        print(f"hb1 : {self.hb1}")
        print(f"hb2 : {self.hb2}")
        print(f"----------------------SOLUTION")
        print(f"Cb_guess : {self.Cb_guess}")
        print(f"Cb : {self.Cb}")
        print(f"dz_eta_0_guess : {self.dz_eta_0_guess}")
        print(f"dz_eta_0 : {self.dz_eta_0}")
        print(f"dz_eta_0_guess : {self.D0_guess}")
        print(f"dz_eta_0 : {self.D0}")
        print(f"hb3 : {self.hb3}")
        print(f"Amplitude max : {np.max(self.eta)}")
        
        
    def plot_solution(self):
        fig = plt.figure(figsize=[24, 6])
        
        plt.subplot(131)
        plt.plot(self.PRHO1, self.Z1, "b", label="ambient")
        plt.plot(self.PRHO2, self.Z2, "orange", label="bore")
        plt.plot(self.PRHO3, self.Z3, "r", label="model")
        plt.axhline(y=self.hb1, color="b")
        plt.axhline(y=self.hb2, color="orange")
        plt.axhline(y=self.hb3, color="r")
        plt.legend()
        plt.xlabel("Potential density $(kg.m^{-3})$")
        plt.ylabel("Z $(m)$")
        plt.grid()
        
        plt.subplot(132)
        plt.plot(self.eta, self.Z1, "k")
        plt.axhline(y=self.hb1, color="b")
        plt.axhline(y=self.hb2, color="orange")
        plt.axhline(y=self.hb3, color="r")
        plt.xlabel("Displacement $\eta$ $(m)$")
        plt.ylabel("Z $(m)$")
        plt.grid()
        
        plt.subplot(133)
        plt.plot(self.Delta, self.Z1, "k")
        plt.axhline(y=self.hb1, color="b")
        plt.axhline(y=self.hb2, color="orange")
        plt.axhline(y=self.hb3, color="r")
        plt.xlabel("Dissipation $\Delta$ $(?)$")
        plt.ylabel("Z $(m)$")
        plt.grid()
        
        if False :
            fig = plt.figure(figsize=[12, 8])
            plt.plot(self.b1, self.Z1, "b", label="ambient")
            plt.plot(self.b2, self.Z2, "orange", label="bore")
            plt.plot(self.b3, self.Z3, "r", label="model")
            plt.axhline(y=self.hb1, color="b")
            plt.axhline(y=self.hb2, color="orange")
            plt.axhline(y=self.hb3, color="r")
            plt.legend()
            plt.xlabel("Buoyancy number $()$")
            plt.ylabel("Z $(m)$")
            plt.grid()
        
    def plot_init(self):
        fig = plt.figure(figsize=[12, 8])
        plt.plot(self.PRHO1, self.Z1, "b", label="ambient")
        plt.plot(self.PRHO2, self.Z2, "orange", label="bore")
        plt.axhline(y=self.hb1, color="b")
        plt.axhline(y=self.hb2, color="orange")
        plt.legend()
        plt.xlabel("Potential density $(kg.m^{-3})$")
        plt.ylabel("Z $(m)$")
        plt.grid()
        
        if False :
            fig = plt.figure(figsize=[12, 8])
            plt.plot(self.b1, self.Z1, "b", label="ambient")
            plt.plot(self.b2, self.Z2, "orange", label="bore")
            plt.axhline(y=self.hb1, color="b")
            plt.axhline(y=self.hb2, color="orange")
            plt.legend()
            plt.xlabel("Buoyancy number $()$")
            plt.ylabel("Z $(m)$")
            plt.grid()
        
        fig = plt.figure(figsize=[12, 8])
        plt.plot(self.NBV2, self.Z1, "b", label="ambient")
        plt.plot(self.NBV22, self.Z2, "orange", label="bore")
        plt.axhline(y=self.hb1, color="b")
        plt.axhline(y=self.hb2, color="orange")
        plt.legend()
        plt.xlabel("$N_{BV}^2$ $(s^{-2})$")
        plt.ylabel("Z $(m)$")
        plt.grid()