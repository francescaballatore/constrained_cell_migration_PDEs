# Libraries {{{
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")
# }}}

# Class for turnover ordinary differential equation {{{
class Turnover(object):
    # Properties {{{
    @property
    def f_ref(self):
        return self._f_ref
    @property
    def Rw_max(self):
        return self._Rw_max
    @property
    def Rt_max(self):
        return self._Rt_max
    @property
    def Rt_freq(self):
        return self._Rt_freq
    @property
    def Rt_threshold(self):
        return self._Rt_threshold
    @property
    def Rt_phase(self):
        return self._Rt_phase
    @property
    def inhibition_threshold(self):
        return self._inhibition_threshold
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        # Parameters for inhibition of covalent-link recovery
        self._f_ref = kwargs["f_ref"]
        self._Rw_max = kwargs["Rw_max"]
        self._inhibition_threshold = kwargs.get("inhibition_threshold", 0.9)
        # Parameters for turnover dynamics
        self._Rt_max = kwargs.get("Rt_max", 5.4472)
        Rt_period = kwargs["Rt_period"]
        self._Rt_freq = 2.0*np.pi/Rt_period
        Rt_length = kwargs.get("Rt_length", 1.0)
        self._Rt_threshold = np.cos(self.Rt_freq*Rt_length)
        self._Rt_phase = kwargs.get("Rt_phase", 0.0)
        return
    # }}}
    # Rw of time {{{
    def Rw_of_time(self, t, f):
        """
        Function to compute the rate of covalent-link-recovery inhibition
        due to the application of an external load.
        """
        f_ref = self.f_ref
        Rw_max = self.Rw_max
        return Rw_max*f**2.0/(f_ref**2.0 + f**2.0)
    # }}}
    # Rt of time {{{
    def Rt_of_time(self, t):
        """
        Function to compute the rate of covalent-link turnover
        """
        Rt_max = self.Rt_max
        Rt_freq = self.Rt_freq
        Rt_threshold = self.Rt_threshold
        Rt_phase = self.Rt_phase
        rw = np.where(np.sin(Rt_freq*t + Rt_phase) > Rt_threshold, Rt_max, 0.0)
        return rw
    # }}}
    # ODE function {{{
    def ode_function(self, t, u, f):
        # Compute recovery term (add 1.0e-6 to perturb ueq = 0)
        recovery = (1.0 - (u + 1.0e-6))*(u + 1.0e-6)
        # Compute inhibition only when u < inhibition_threshold (so 1.0 is remains stable)
        if u > self.inhibition_threshold:
            inhibition = 0.0
        else:
            Rw = self.Rw_of_time(t, f)
            inhibition = Rw*(1.0 - u)*u
        # Compute turnover
        Rt = self.Rt_of_time(t)
        turnover = Rt*u
        return recovery - inhibition - turnover
    # }}}
    # __call__ {{{
    def __call__(self, t0, tf, u0, f):
        # Define t_span
        t_span = [t0, tf]
        # Define ode function to give to solve_ivp
        ode_f = lambda t, u : self.ode_function(t, u, f)
        # Solve system
        solution = solve_ivp(ode_f, t_span, [u0],
                             method = "LSODA")
        return solution.y[0][-1]
    # }}}
# }}}