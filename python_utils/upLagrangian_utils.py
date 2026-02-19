# Libraries {{{
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Constant, Function, dirichletbc, Expression
from dolfinx.io import XDMFFile, VTKFile

import ufl
from ufl import (TestFunction, TrialFunction, Identity, grad, inner, det,
                 inv, tr, as_vector, outer, derivative, dev, sqrt)

import basix
from basix.ufl import element, quadrature_element

import os, sys
import numpy as np
from datetime import datetime
from pdb import set_trace

# In-house modules
from .misc_utils import *
# }}}

# Class to solve a displacement-base updated Lagrangian formulation {{{
class UpdatedLagrangian(object):
    # Properties {{{
    @property
    def comm(self):
        return self._comm
    @property
    def dt(self):
        return self._dt
    @property
    def domain(self):
        return self._domain
    @property
    def dx(self):
        return self._dx
    @property
    def ds(self):
        return self._ds
    @property
    def feSpaces(self):
        return self._feSpaces
    @property
    def mainFunctions(self):
        return self._mainFunctions
    @property
    def stateVariables(self):
        return self._stateVariables
    @property
    def expressions(self):
        return self._expressions
    @property
    def constants(self):
        return self._constants
    @property
    def weakForm(self):
        return self._weakForm
    @property
    def bcs(self):
        return self._bcs
    @property
    def problem(self):
        return self._problem
    @property
    def solver(self):
        return self._solver
    @property
    def out(self):
        return self._out
    @property
    def projections(self):
        return self._projections
    @property
    def UpdateVariables(self):
        try:
            self._UpdateVariables
        except AttributeError:
            message = "_UpdateVariables has not been yet defined. It should be defined in _SetUpWeakForm."
            raise AttributeError(message)
        return self._UpdateVariables
    # }}}
    # __init__ function {{{
    def __init__(self, **kwargs):
        # Get comm
        self._comm = kwargs["comm"]
        # Get time step
        self._dt = kwargs["dt"]
        # Get domain and integration measures
        self._domain = kwargs["domain"]
        try:
            self._dx = kwargs["dx"]
        except KeyError:
            self._dx = ufl.Measure("dx", domain = self.domain)
        try:
            self._ds = kwargs["ds"]
        except KeyError:
            self._ds = ufl.Measure("ds", domain = self.domain)
        # Create function spaces
        self._SetUpFiniteElementSpaces(kwargs["order"])
        self._expressions = {}
        self._projections = {}
        # Create main and state variable functions and constants
        self._SetUpMainFunctions()
        self._SetUpVariables(**kwargs)
        # Set up weak form
        self._SetUpWeakForm(**kwargs)
        # Set up boundary conditions
        self._SetUpBoundaryConditions(**kwargs)
        # Set up nonlinear problem
        self._SetUpNonlinearProblem(**kwargs)
        # Set up writer
        self._SetUpWriter(**kwargs)
        return
    # }}}
    # Set up finite element spaces {{{
    def _SetUpFiniteElementSpaces(self, order):
        domain = self.domain
        dim = domain.topology.dim
        # Element types: constant scalar by element
        V_dg0 = fem.functionspace(domain, ("DG", 0))
        # Element types: quadratic scalar
        ele_scalar = element("Lagrange", domain.basix_cell(), order)
        V_scalar = fem.functionspace(domain, ele_scalar)
        # Element types: quadratic displacement
        ele_u = element("Lagrange", domain.basix_cell(), order, shape = (dim, ))
        V_u = fem.functionspace(domain, ele_u)
        # Element types: quadratic stress
        ele_gauss_stress = quadrature_element(domain.basix_cell(), degree = order, scheme = "default")
        ele_gauss_stress = basix.ufl.blocked_element(ele_gauss_stress, shape = (dim, dim))
        V_gauss_T = fem.functionspace(domain, ele_gauss_stress)
        ele_stress = element("Lagrange", domain.basix_cell(), order, shape = (dim, dim))
        V_T = fem.functionspace(domain, ele_stress)
        #
        self._feSpaces = {"scalar" : V_scalar,
                          "scalar_DG" : V_dg0,
                          "vector" : V_u,
                          "tensor" : V_T,
                          "tensor_gaussian" : V_gauss_T}
        return
    # }}}
    # Set up main functions {{{
    def _SetUpMainFunctions(self):
        V = self.feSpaces["vector"]
        self._mainFunctions = {"u" : Function(V),
                               "v" : TestFunction(V),
                               "du" : TrialFunction(V)}
        return
    # }}}
    # Set up state variables {{{
    def _SetUpVariables(self, **kwargs):
        raise NotImplementedError("Subclasses must implement _SetUpVariables")
    # }}}
    # Set up weak form {{{
    def _SetUpWeakForm(self, **kwargs):
        raise NotImplementedError("Subclasses must implement _SetUpWeakForm")
    # }}}
    # Set up boundary conditions {{{
    def _SetUpBoundaryConditions(self, **kwargs):
        raise NotImplementedError("Subclasses must implement _SetUpBoundaryConditions")
    # }}}
    # Set up Writer {{{
    def _SetUpWriter(self, **kwargs):
        raise NotImplementedError("Subclasses must implement _SetUpWriter")
    # }}}
    # Set up Nonlinear variational problem {{{
    def _SetUpNonlinearProblem(self, **kwargs):
        Res = self.weakForm["residual"]
        a = self.weakForm["tangent"]
        u = self.mainFunctions["u"]
        SetSolverOpt = kwargs["SetSolverOpt"]
        bcs = self.bcs
        self._problem = NonlinearProblem(Res, u, bcs, a)
        self._solver = NewtonSolver(self.comm, self.problem)
        SetSolverOpt(self.solver)
        return
    # }}}
    # Solve {{{
    def Solve(self):
        solver = self.solver
        u = self.mainFunctions["u"]
        (iters, converged) = solver.solve(u)
        return iters, converged
    # }}}
    # Interpolate {{{
    def Interpolate(self, oldCase, scalarList, vectorList, tensorList,
                    padding = 1.0e-6):
        # Create interpolation data
        inter_index_map = self.domain.topology.index_map(self.domain.topology.dim)
        inter_cells = np.arange(inter_index_map.size_local + inter_index_map.num_ghosts,
                                dtype = np.int32)
        sca_inter_data = fem.create_interpolation_data(self.feSpaces["scalar"],
                                                       oldCase.feSpaces["scalar"],
                                                       inter_cells, padding)
        vec_inter_data = fem.create_interpolation_data(self.feSpaces["vector"],
                                                       oldCase.feSpaces["vector"],
                                                       inter_cells, padding)
        ten_inter_data = fem.create_interpolation_data(self.feSpaces["tensor"],
                                                       oldCase.feSpaces["tensor"],
                                                       inter_cells, padding)
        gTe_inter_data = fem.create_interpolation_data(self.feSpaces["tensor_gaussian"],
                                                       oldCase.feSpaces["tensor_gaussian"],
                                                       inter_cells, padding)
        # Interpolation of scalar functions
        names = scalarList
        for name in names:
            new_func = self.stateVariables[name]
            old_func = oldCase.stateVariables[name]
            new_func.interpolate_nonmatching(old_func, inter_cells, sca_inter_data)
        # Interpolation of vector functions
        new_u = self.mainFunctions["u"]
        old_u = oldCase.mainFunctions["u"]
        new_u.interpolate_nonmatching(old_u, inter_cells, vec_inter_data)
        names = vectorList
        for name in names:
            new_func = self.stateVariables[name]
            old_func = oldCase.stateVariables[name]
            new_func.interpolate_nonmatching(old_func, inter_cells, vec_inter_data)
        # Interpolation of tensor functions
        V_T_old = oldCase.feSpaces["tensor"]
        V_T_gauss_new = self.feSpaces["tensor_gaussian"]
        V_T_new = self.feSpaces["tensor"]
        names = tensorList
        # for name in names:
        #     new_func = self.stateVariables[name]
        #     old_func = oldCase.stateVariables[name]
        #     new_func.interpolate_nonmatching(old_func, inter_cells, gTe_inter_data)
        for name in names:
            # Recover function
            # new_func = Function(V_T_gauss_new)
            new_func = self._stateVariables[name]
            old_func = oldCase.stateVariables[name]
            # Project old function into lagrangian nodes
            proj_problem_toLag = setup_projection(old_func, V_T_old, oldCase.dx)
            old_lag = proj_problem_toLag.solve()
            # Interpolate using lagrangian elements
            new_lag = Function(V_T_new)
            new_lag.interpolate_nonmatching(old_lag, inter_cells, ten_inter_data)
            # Project into gaussian elements
            proj_problem_toGauss = setup_projection(new_lag, V_T_gauss_new, self.dx)
            new_func = proj_problem_toGauss.solve()
            self._stateVariables[name].x.array[:] = new_func.x.array[:]
        return
    # }}}
# }}}
