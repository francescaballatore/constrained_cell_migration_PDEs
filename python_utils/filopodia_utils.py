# Libraries {{{
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx.fem import Constant
from petsc4py import PETSc
from pdb import set_trace
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
# }}}

# General ea_class {{{
class ea_class(object):
    # Time function {{{
    def eaTimeFunc(self, t):
        # one and zero values
        zero = Constant(t.ufl_domain(), PETSc.ScalarType(0.0))
        one  = Constant(t.ufl_domain(), PETSc.ScalarType(1.0))
        # Compute value
        if t.value >= self.t0 and t.value < self.t0 + self.period:
            return one
        else:
            return zero
    # }}}
    # __call__ {{{
    def __call__(self, phi, t):
        eaSpace = self.eaSpaceFunc(phi)
        eaTime = self.eaTimeFunc(t)
        return eaSpace*eaTime
    # }}}
# }}}
# Class to define ea Gaussian {{{
class ea_Gaussian(ea_class):
    # Properties {{{
    @property
    def height(self):
        return self._height
    @property
    def width(self):
        return self._width
    @property
    def period(self):
        return self._period
    @property
    def theta(self):
        return self._theta
    @property
    def t0(self):
        return self._t0
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._theta = kwargs.get("theta", 0.0)
        self._t0 = kwargs.get("t0", 0.0)
        self._width = kwargs["width"]
        self._height = kwargs["height"]
        self._period = kwargs["period"]
        return
    # }}}
    # Space and time functions {{{
    def eaSpaceFunc(self, phi):
        return self.height*ufl.exp(-(phi - self.theta)**2.0/(2.0*self.width**2.0))
    def eaTimeFunc(self, t):
        ome = 2.0*np.pi/self.period
        return (1.0 + ufl.sin(ome*(t - self.t0) - np.pi/2.0))/2.0
    # }}}
# }}}
# Class to define ea from cosine {{{
class ea_Cos(ea_Gaussian):
    # Space and time functions {{{
    def eaSpaceFunc(self, phi):
        return self.height*ufl.cos((phi - self.theta)/2.0)**(int(self.width))
    def eaTimeFunc(self, t):
        # one and zero values
        zero = Constant(t.ufl_domain(), PETSc.ScalarType(0.0))
        one  = Constant(t.ufl_domain(), PETSc.ScalarType(1.0))
        # Compute value
        ome = 2.0*np.pi/self.period
        sinval = np.sin(ome*(t.value - self.t0))
        if sinval > 0.0:
            return one
        else:
            return zero
    # }}}
# }}}
# Class to sum up eas defined for expressions {{{
class eaSum(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            eas[k1] = ForceClass(theta = thetas[k1],
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  height = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, dt, domain):
        sum_eas = 0.0
        for ea_k1 in self.eas:
            sum_eas += ea_k1(phi, t)
        return sum_eas
    # }}}
# }}}
# Class to get maximum eas defined for expressions {{{
class eaMax(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            eas[k1] = ForceClass(theta = thetas[k1],
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  height = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, dt, domain):
        max_eas = 0.0
        for ea_k1 in self.eas:
            max_eas = ufl.max_value(max_eas, ea_k1(phi, t))
        return max_eas
    # }}}
# }}}
# Class to define ea by distance to a point {{{
class ea_Distance(ea_class):
    # Properties {{{
    @property
    def magnitude(self):
        return self._magnitude
    @property
    def width(self):
        return self._width
    @property
    def source(self):
        return self._source
    @property
    def t0(self):
        return self._t0
    @property
    def period(self):
        return self._period
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._magnitude = kwargs["magnitude"]
        self._width = kwargs["width"]
        self._source = kwargs["source"]
        self._t0 = kwargs["t0"]
        self._period = kwargs["period"]
        return
    # }}}
    # Space functions {{{
    def eaSpaceFunc(self, xArray, h):
        # Create KDTree
        tree = KDTree(xArray)
        # Define the number of points and effective force
        eqPressure = self.magnitude/self.width
        numForcePoints = np.ceil(self.width/h)
        dumpPressure = self.magnitude/(h*numForcePoints)
        effForce = eqPressure/dumpPressure*self.magnitude
        # Find nearest points
        distance, indices = tree.query(self.source, k = numForcePoints)
        # Find direction
        direction = self.source - xArray[indices]
        if len(direction.shape) == 1:
            direction = np.array([(direction)])
        dirMag = np.linalg.norm(direction, axis = 1)
        direction[:, 0] = direction[:, 0]/dirMag
        direction[:, 1] = direction[:, 1]/dirMag
        return indices, effForce, direction
    # }}}
    # Time function {{{
    def eaTimeFunc(self, t):
        # Compute value
        if t.value >= self.t0 and t.value < self.t0 + self.period:
            return 1.0
        else:
            return 0.0
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, **kwargs):
        h = kwargs["h"]
        indices, effForce, direction = self.eaSpaceFunc(phi, h)
        eaTime = self.eaTimeFunc(t)
        return indices, effForce*eaTime, direction
    # }}}
# }}}
# Class to sum eas defined by nodes {{{
class eaSumNodal(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        sourceDia = kwargs.get("sourceDia", 1.0e6)
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            theta = thetas[k1]
            source = np.array([np.cos(theta), np.sin(theta)])*sourceDia
            eas[k1] = ForceClass(source = source,
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  magnitude = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, **kwargs):
        indices = []
        forces = []
        directions = []
        for ea_k1 in self.eas:
            indices_k1, force_k1, direction_k1 = ea_k1(phi, t, **kwargs)
            indices.append(indices_k1)
            forces.append(np.ones(indices_k1.size)*force_k1)
            directions.append(direction_k1)
        return indices, forces, directions
    # }}}

# }}}