# Libraries {{{
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix, create_matrix
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Constant, Function, dirichletbc, Expression
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.la import create_petsc_vector
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, inner, det, div, dot, inv, tr, as_vector, outer, derivative, dev, sqrt, eq, max_value)

import basix
from basix.ufl import element, quadrature_element

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
import pyvista
import gmsh
from shapely.geometry import Polygon
from pdb import set_trace

# In-house modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from output_utils import *
from mesh_utils import *
from misc_utils import *
from filopodia_utils import *
from forces_utils import *
from curvature_utils import *
# }}}

def grad_Gamma(f, normal):
    grad_f = grad(f)
    normal_component = dot(grad_f, normal)
    proj_normal = as_vector([normal_component * normal[0],
                             normal_component * normal[1]])
    return grad_f - proj_normal

# Class GSPDE {{{
class GSPDE(object):
    # __init__ {{{
    def __init__(self, other_gspde=None, **kwargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.numRanks = self.comm.Get_size()
        self.kwargs = kwargs
        self.other_gspde = other_gspde
        # Set domain
        self.SetDomain()
        # Set measures
        self.SetMeasures(**kwargs)
        # Set finite element spaces
        self.SetFESpaces()
        # Set functions and constants
        self.SetVariables()
        # Set expressions
        self.SetExpressions(**kwargs)
        # Set initialisation
        self.SetInitialisation()
        # Set weak forms
        self.SetWeakForm(**kwargs)
        self.SetWeakFormChem(**kwargs)
        # Set nonlinear problems
        self.SetNonlinearProblem()
        if self.role == "nucleus":
            self.SetNonlinearProblemChem()
        return
    # }}}
    # Set domain {{{
    def SetDomain(self):
        model = self.kwargs["model"]
        # Mesh
        self.domain, self.cellTags, self.facetTags = io.gmshio.model_to_mesh(model, self.comm, 0, gdim = 2)
        # Get dimensions
        self.dimSur = self.domain.topology.dim
        self.dimSpa = self.dimSur + 1
        # Data distribution
        self.imap = self.domain.geometry.index_map()
        self.global_node_ids = self.imap.local_to_global(np.arange(self.domain.geometry.x.shape[0], dtype=np.int32))        
        self.gather_global_node_ids = self.comm.allgather(self.global_node_ids)
        local_node_ids = np.arange(self.imap.size_local + self.imap.num_ghosts, dtype=np.int32)
        self.node_ids_arg = np.argwhere(local_node_ids >= 0).flatten()
        self.node_ids = local_node_ids[self.node_ids_arg]
        # Get number of nodes
        self.numNods, _ = self.domain.geometry.x.shape
        # Get ordered list of node ids
        connectivities = self.domain.geometry.dofmap
        connectivities = np.array([self.imap.local_to_global(local_ele) for local_ele in connectivities])
        connectivities = self.comm.allgather(connectivities)
        connectivities = np.concatenate([array for array in connectivities if array.size > 0])
        self.numEles, nnod = connectivities.shape
        if nnod == 2:
            connectivities = np.column_stack([connectivities,
                                              -np.ones([self.numEles, 1], dtype = int)])
        self.global_orderedNodeIds = OrderNodeList(connectivities[0, 0],
                                                   connectivities[0, 0],
                                                   connectivities, self.numEles)
        all_local_indices = np.arange(self.imap.size_local + self.imap.num_ghosts, dtype=np.int32)
        all_globals = self.imap.local_to_global(all_local_indices)
        sort_idx = np.argsort(all_globals)
        pos = np.searchsorted(all_globals[sort_idx], self.global_orderedNodeIds)
        local_orederedNodeIds = all_local_indices[sort_idx[pos]]
        self.orderedNodeArg = np.argwhere(local_orederedNodeIds >= 0).flatten()
        self.orderedNodeIds = local_orederedNodeIds[self.orderedNodeArg]
        return
    # }}}
    # Set measures {{{
    def SetMeasures(self, **kwargs):
        self.normalDirection = kwargs.get("normalDirection", 1.0)
        quadrature_degree = self.kwargs["quadrature_degree"]
        self.x = ufl.SpatialCoordinate(self.domain)
        self.n = ufl.CellNormal(self.domain)
        self.dx = ufl.Measure("dx", domain = self.domain,
                              metadata = {"quadrature_degree" : quadrature_degree,
                                          "quadrature_rule" : "default"})
        return
    # }}}
    # Set finite element spaces {{{
    def SetFESpaces(self):
            import ufl
            import basix.ufl
            meshOrder = self.kwargs["meshOrder"]
            cell_type = self.domain.ufl_cell()

            # Element types: scalar
            ele_scalar = ufl.FiniteElement("Lagrange", cell_type, meshOrder)
            self.V_scalar = fem.functionspace(self.domain, ele_scalar)

            # Element types: vector
            ele_u = ufl.VectorElement("Lagrange", cell_type, meshOrder, dim=self.dimSpa)
            self.V_u = fem.functionspace(self.domain, ele_u)

            # Element types: tensor
            ele_T = ufl.TensorElement("Lagrange", cell_type, meshOrder, shape=(self.dimSpa, self.dimSpa))
            self.V_tensor = fem.functionspace(self.domain, ele_T)

            # Mixed element
            ele_mixed = ufl.MixedElement([ele_u, ele_scalar])
            self.V_mixed = fem.functionspace(self.domain, ele_mixed)
            return
    # }}}
    # Set variables {{{
    def SetVariables(self):
        aRef = self.kwargs["aRef"]
        Dia = self.kwargs["Dia"]
        periRef = self.kwargs["periRef"]
        dt = self.kwargs["dt"]
        t = self.kwargs.get("t", 0.0)
        H_ref = self.kwargs["Href"]
        peri_stiffness = self.kwargs["peri_stiffness"]
        area_stiffness = self.kwargs["area_stiffness"]
        peri_max_factor = self.kwargs["peri_max_factor"]
        width = self.kwargs["width"]
        length = self.kwargs["length"]
        height = self.kwargs["height"]
        x_left = self.kwargs["x_left"]
        k_bar = self.kwargs["k_bar"]
        beta = self.kwargs["beta"]
        k_pr = self.kwargs["k_pr"]
        k_el = self.kwargs["k_el"] 
        factor = self.kwargs["factor"] 
        alpha = self.kwargs["alpha"] 
        k_rep = self.kwargs["k_rep"] 
        role = self.kwargs["role"]
        tensionStiffness_Ctrl = self.kwargs["surfacetension_Ctrl"]
        tensionStiffness_KO = self.kwargs["surfacetension_KO"]
        bendingStiffness_Ctrl = self.kwargs["bendingstiffness_Ctrl"]
        bendingStiffness_KO = self.kwargs["bendingstiffness_KO"]
        omega_value = self.kwargs["omega"]
        k_sr = self.kwargs["k_sr"]
        delta_d = self.kwargs["delta_d"]
        kappa_d  = self.kwargs["kappa_d"]
        delta_n = self.kwargs["delta_n"]
        kappa_n  = self.kwargs["kappa_n"]
        if role == "nucleus":
            N_chem = self.kwargs["N_chem"]
            D_chem = self.kwargs["D_chem"]
            init = self.kwargs["init"]
            coseno = self.kwargs["coseno"]
        # Floats
        self.dt = dt
        self.aRef = aRef
        self.periRef = periRef
        self.area = aRef
        self.Dia = Dia
        self.perimeter = periRef
        self.Href = H_ref
        self.peri_stiffness = peri_stiffness
        self.area_stiffness = area_stiffness
        self.peri_max_factor = peri_max_factor
        self.width = width
        self.length = length
        self.height = height
        self.x_left = x_left
        self.beta = beta
        self.k_bar = k_bar
        self.k_pr = k_pr
        self.k_el = k_el
        self.factor = factor
        self.alpha = alpha
        self.k_rep = k_rep
        self.role = role
        self.x_front = Dia/2
        self.x_front_p = 1
        self.x_rear = -Dia/2
        self.tensionStiffness_Ctrl = tensionStiffness_Ctrl
        self.tensionStiffness_KO = tensionStiffness_KO
        self.bendingStiffness_Ctrl = bendingStiffness_Ctrl
        self.bendingStiffness_KO = bendingStiffness_KO
        self.omega_value = omega_value
        self.k_sr = k_sr
        self.delta_d = delta_d
        self.kappa_d = kappa_d
        self.delta_n = delta_n
        self.kappa_n = kappa_n
        if self.role == "nucleus":
            self.N_chem = N_chem
            self.D_chem = D_chem
            self.init = init
            self.coseno = coseno
        # Constants
        self.dk = Constant(self.domain, PETSc.ScalarType(dt))
        self.t_constant = Constant(self.domain, PETSc.ScalarType(t))
        self.opre_area = Constant(self.domain, PETSc.ScalarType(0.0))
        self.opre_peri = Constant(self.domain, PETSc.ScalarType(0.0))
        # Scalar functions
        self.opre_total = Function(self.V_scalar)
        self.smoothed_H = Function(self.V_scalar)
        # Main functions
        self.w = Function(self.V_mixed)
        self.u, self.H = ufl.split(self.w)
        self.u_test, self.H_test = TestFunctions(self.V_mixed)
        self.dw = TrialFunction(self.V_mixed)
        # Scalar functions
        if self.role == "nucleus":
            self.a_chem      = [fem.Function(self.V_scalar) for _ in range(N_chem)]
            self.a_chem_old  = [fem.Function(self.V_scalar) for _ in range(N_chem)]
            self.a_chem_test = [ufl.TestFunction(self.V_scalar) for _ in range(N_chem)]
            self.a_chem_trial = [ufl.TrialFunction(self.V_scalar) for _ in range(N_chem)]
            self.Res_a = [None] * N_chem
            self.tangent_chem = [None] * N_chem
            self.problemChem = [None] * self.N_chem
            self.solverChem  = [None] * self.N_chem

        self.H_old = Function(self.V_scalar)
        self.tensionStiffness = Function(self.V_scalar)
        self.bendingStiffness = Function(self.V_scalar)
        self.omega = Constant(self.domain, PETSc.ScalarType(omega_value))
        self.selfRepuForce = Function(self.V_scalar)
        self.barrierForce = Function(self.V_scalar)
        self.movForce = Function(self.V_scalar)
        self.elasticForce = Function(self.V_scalar)
        self.repulsiveForce = Function(self.V_scalar)
        self.phi = Function(self.V_scalar)
        self.totalForce = Function(self.V_scalar)
        # Vector functions
        self.x_old = Function(self.V_u)
        self.x0 = Function(self.V_u)
        self.disp = Function(self.V_u)
        self.normal = Function(self.V_u)
        self.filoDir = Function(self.V_u)
        # PDEs
        self.a = 10.0
        self.b = 50.0
        self.c = 20.0
        self.k1 = 40
        self.k2 = 250
        self.k3 = 50
        self.k4 = 25
        return
    # }}}
    # Set expressions {{{
    def SetExpressions(self, **kwargs):
        op = self.opre_peri
        sigmoid = lambda x: 1.0 / (1.0 + ufl.exp(-x))
        opre_H = sigmoid(-10.0 * op) * self.smoothed_H * op
        opre_total_math = opre_H + self.opre_area

        totalForce_math = (opre_total_math
                        + self.barrierForce
                        + self.selfRepuForce
                        + self.movForce
                        + self.elasticForce
                        + self.repulsiveForce)

        self.normal_expr = Expression(self.n, self.V_u.element.interpolation_points())
        self.x_expr = Expression(self.w.sub(0), self.V_u.element.interpolation_points())
        self.H_expr = Expression(self.w.sub(1), self.V_scalar.element.interpolation_points())
        self.disp_expr = Expression(self.x_old - self.x0, self.V_u.element.interpolation_points())
        self.opre_expr = Expression(opre_total_math, self.V_scalar.element.interpolation_points())
        self.totalForce_expr = Expression(totalForce_math, self.V_scalar.element.interpolation_points())
        
        if self.role == "nucleus":
            self.a_chem_expr = [Expression(i, self.V_scalar.element.interpolation_points()) for i in self.a_chem]
        return
    # }}}
    # Set initialisation {{{
    def SetInitialisation(self):
        # Initial normal vector
        self.normal.interpolate(self.normal_expr)
        # Initial x (identity map)
        self.x_expr_id = Expression(self.x, self.V_u.element.interpolation_points())
        self.x_old.interpolate(self.x_expr_id)
        self.x0.interpolate(self.x_expr_id)
        # Initial curvature
        InitialCurvature(self.H_old, self.normal, self.x_old, self.dx)
        # Initialisation of phi
        orderedPhi = np.linspace(0.0, 2.0*np.pi, self.global_orderedNodeIds.size)
        orderedPhi = orderedPhi[::-1]
        self.phi.x.array[self.orderedNodeIds] = orderedPhi[self.orderedNodeArg]
        # Tension and bending stiffness
        self.tensionStiffness.x.array[:] = self.tensionStiffness_Ctrl
        self.bendingStiffness.x.array[:] = self.bendingStiffness_Ctrl
        # Chemical concentration
        if self.role == "nucleus":
            self.a_chem_old[0].x.array[:] = self.init + self.coseno*0.2*np.cos(10*self.phi.x.array)
            self.a_chem_old[1].x.array[:] = 0.2 + self.coseno*0.2*np.cos(10*self.phi.x.array)
            self.a_chem_old[2].x.array[:] = 0.0
    # }}}
    # Set weak form {{{
    def SetWeakForm(self, **kwargs):
        #Href = self.kwargs["Href"]
        Mu = (self.omega/self.dk)*inner(inner(self.u - self.x_old, self.normal), self.H_test)*self.dx
        Su = self.bendingStiffness*inner(grad_Gamma(self.H,self.normal),grad_Gamma(self.H_test,self.normal))*self.dx
        #Hpow2 = self.H_old**2.0
        Hpow2 = self.H**2 - self.Href**2.0
        Qu = -0.5*self.bendingStiffness*inner(Hpow2*self.H, self.H_test)*self.dx
        Tu = inner(self.tensionStiffness*self.H, self.H_test)*self.dx
        Fu = inner(self.totalForce, self.H_test)*self.dx
        Res_u = Mu + Su + Qu - Fu + Tu
        MH = inner(self.H*self.normal, self.u_test)*self.dx
        SH = inner(grad_Gamma(self.u,self.normal), grad_Gamma(self.u_test,self.normal))*self.dx
        Res_H = MH - SH # H = div(x)

        self.Res = Res_u + Res_H
        self.tangent = derivative(self.Res, self.w, self.dw)
        return
    def SetWeakFormChem(self, **kwargs):
        if self.role == "nucleus":
            fa = [self.k1*self.a_chem_old[0]*(1-self.a_chem_old[0]) - self.a*self.a_chem_old[0],  
                  self.k2*self.a_chem_old[1]*(1-self.a_chem_old[1]) - self.k3*self.a_chem_old[0] - self.b*self.a_chem_old[1], 
                  self.k4*self.a_chem_old[1] - self.c*self.a_chem_old[2]]
            if self.role == "nucleus":
                for i in range(self.N_chem):
                    Lu = (1.0/self.dk)*inner(self.a_chem[i], self.a_chem_test[i])*self.dx + self.D_chem[i] * inner(grad_Gamma(self.a_chem[i], self.normal),grad_Gamma(self.a_chem_test[i], self.normal)) * self.dx
                    au = (1.0/self.dk)*inner(self.a_chem_old[i], self.a_chem_test[i])*self.dx + inner(fa[i], self.a_chem_test[i]) * self.dx
                    self.Res_a[i] = Lu - au
                    self.tangent_chem[i] = ufl.derivative(self.Res_a[i],self.a_chem[i],self.a_chem_trial[i])

    # }}}
    # Update variables {{{
    def UpdateVariables(self):
        equidistribute = self.kwargs.get("equidistribute", True)
        # Update current position and curvature
        self.x_old.interpolate(self.x_expr)
        self.H_old.interpolate(self.H_expr)
        # Update chemical concentration
        if self.role == "nucleus":
            for i in range(self.N_chem):
                self.a_chem_old[i].interpolate(self.a_chem_expr[i])
            # Update bending stiffness and surface tension based on a_chem[2]
            self.bendingStiffness.interpolate(fem.Expression(ComputeBendingStiffness(self.a_chem_old[2], self.bendingStiffness_Ctrl, self.bendingStiffness_KO, self.a, self.b, self.c, self.k1, self.k2, self.k3, self.k4), self.V_scalar.element.interpolation_points()))
            self.tensionStiffness.interpolate(fem.Expression(ComputeTensionStiffness(self.a_chem_old[2], self.tensionStiffness_Ctrl, self.tensionStiffness_KO, self.a, self.b, self.c, self.k1, self.k2, self.k3, self.k4), self.V_scalar.element.interpolation_points()))
            self.omega.value = PETSc.ScalarType(max(self.omega_value * self.a_chem_old[2].x.array.mean(), self.omega_value/2))
            print(self.omega.value)
        # Update mesh
        uMat = FromVectorToMatrix(self.x_old.x.array, self.dimSpa)
        self.domain.geometry.x[:, :self.dimSpa] = uMat
        # Update area and osmotic pressure
        global_uMat = self.GetGlobalArray(uMat)
        global_orderedNodes = global_uMat[self.global_orderedNodeIds]
        xCoor = global_orderedNodes[:-1, 0]
        yCoor = global_orderedNodes[:-1, 1]
        poly = Polygon(zip(xCoor, yCoor))
        self.area = poly.area
        self.perimeter = poly.length
        # Mesh tangential movement for equidistribution
        if equidistribute:
            global_newOrderedNodes = EquidistributeMesh(global_orderedNodes, optimal = False)
            self.domain.geometry.x[self.orderedNodeIds, :self.dimSpa] = global_newOrderedNodes[self.orderedNodeArg]
            self.x_old.interpolate(self.x_expr_id)
        # Update total displacement
        self.disp.interpolate(self.disp_expr)
        # Update normal
        self.normal.interpolate(self.normal_expr)
        # Interpolate phi if tangential movement is applied
        if equidistribute:
            phi = self.phi.x.array[:]
            global_phi = self.GetGlobalArray(phi)
            global_orderedPhi = global_phi[self.global_orderedNodeIds]
            global_newOrderedPhi = CurveInterpolation(global_orderedNodes,
                                                      global_orderedPhi,
                                                      global_newOrderedNodes)
            self.phi.x.array[self.orderedNodeIds] = global_orderedPhi[self.orderedNodeArg]
        #compute x_front and x_rear
        x_array = self.x_old.x.array.reshape(-1, self.dimSpa)
        x_coords = x_array[:, 0]  # componente x
        self.x_front = np.max(x_coords)
        self.x_front_p = np.argmax(x_coords)
        self.x_rear = np.min(x_coords)
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        # Osmotic pressure
        self.OsmoticPressure()
        self.opre_total.interpolate(self.opre_expr)
        # Self-repulsive force
        self.SelfRepulsiveForce()
        # Barrier force
        self.BarrierForce()
        # Movement force
        self.MovForce()
        # Nucleus to cytoplasm force
        self.ElasticForce()
        # Repulsive force
        self.RepulsiveForce()
        # Update total force
        self.totalForce.interpolate(self.totalForce_expr)
        return
    # }}}

    # Osmotic pressure {{{
    def OsmoticPressure(self):
        self.SmoothCurvature()
        area_stiffness = self.area_stiffness
        peri_stiffness = self.peri_stiffness
        max_peri = self.peri_max_factor*self.periRef
        # lambda_peri = (-1.0 - peri_stiffness*(gspde.perimeter - gspde.periRef))
        lambda_peri = peri_stiffness*(max_peri - self.perimeter)
        lambda_area = area_stiffness*(self.aRef - self.area)
        self.opre_peri.value = lambda_peri
        self.opre_area.value = lambda_area
        return
    # }}}
    
    # Self-repulsive force {{{
    def SelfRepulsiveForce(self):
        normalArray = FromVectorToMatrix(self.normal.x.array, self.dimSpa)
        global_normalArray = self.GetGlobalArray(normalArray)
        xArray = self.domain.geometry.x[:, :self.dimSpa]
        global_xArray = self.GetGlobalArray(xArray)
        repuForce = SelfRepulsiveForce_kdtree(xArray, normalArray,
                                              global_xArray, global_normalArray,
                                              self.delta_n, self.kappa_n, self.delta_d,
                                              self.k_sr, self.kappa_d)
        self.selfRepuForce.x.array[:] = repuForce
        return
    # }}}

    ### External forces
    # Barrier force (both cell and nucleus)
    def BarrierForce(self):
        y_top_inner = +self.width / 2
        y_bot_inner = -self.width / 2

        # Forza totale
        total_vector_barrier_force = ufl.as_vector([0.0] * self.dimSpa)
        
        # Versori diretti lungo y
        y_unit_up = ufl.as_vector([0.0, 1.0])
        y_unit_down = ufl.as_vector([0.0, -1.0])

        in_x_range = ufl.And(ufl.ge(self.x[0], self.x_left), ufl.le(self.x[0], self.x_left + self.length))

        # Penetrazione nella parete superiore: distanza da y_top_inner
        penetration_top = self.x[1] - y_top_inner

        # Penetrazione nella parete inferiore: distanza da y_bot_inner
        penetration_bot = y_bot_inner - self.x[1]

        # Forza nella parete superiore (verso il basso, se penetra e dentro il rettangolo)
        force_mag_top = ufl.conditional(
            in_x_range,
            self.k_bar * ufl.exp(self.beta * penetration_top),
            0.0
        )
        total_vector_barrier_force += force_mag_top * y_unit_down

        # Forza nella parete inferiore (verso l’alto, se penetra e dentro il rettangolo)
        force_mag_bot = ufl.conditional(
            in_x_range,
            self.k_bar * ufl.exp(self.beta * penetration_bot),
            0.0
        )
        total_vector_barrier_force += force_mag_bot * y_unit_up

        # Centri dei semicerchi: (sinistri e destri)
        centres = [
            ufl.as_vector([self.x_left, +self.width/2 + self.height/2]),  # top-left
            ufl.as_vector([self.x_left, -self.width/2 - self.height/2]),  # bot-left
            ufl.as_vector([self.x_left+self.length, +self.width/2 + self.height/2]),  # top-right
            ufl.as_vector([self.x_left+self.length, -self.width/2 - self.height/2])   # bot-right
        ]

        in_x_range = ufl.Or(ufl.le(self.x[0], self.x_left), ufl.ge(self.x[0], self.x_left + self.length))

        for c_vec in centres:
            delta = self.x - c_vec
            dist = sqrt(dot(delta, delta) + 1e-6)
            f_dir = delta / dist
            penetration = self.height/2 - dist
            force_mag = ufl.conditional(
                in_x_range,
                self.k_bar * ufl.exp(self.beta * penetration),
                0.0)
            total_vector_barrier_force += force_mag * f_dir

        # Proiezione sulla normale
        scalar_normal_component = ufl.dot(total_vector_barrier_force, self.normal)

        # Risoluzione del problema FEM
        V_b = self.barrierForce.function_space
        q_b = ufl.TestFunction(V_b)
        p_b = ufl.TrialFunction(V_b)
        a_b = inner(p_b, q_b) * self.dx
        L_b = inner(scalar_normal_component, q_b) * self.dx

        problem_b = fem.petsc.LinearProblem(a_b, L_b, bcs=[], u=self.barrierForce,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem_b.solve()

        return

    # Pressure to push cell (cell only)
    def MovForce(self):
        if self.dimSpa == 2:
            f_dir_vec = [1.0, 0.0]
        else:  # self.dimSpa == 3
            f_dir_vec = [1.0, 0.0, 0.0]

        f_dir = ufl.as_vector(f_dir_vec)
        # x_c = compute_center(self.domain)
        # centre_dir = ufl.as_vector(self.x - x_c)
        # norm_centre_dir = ufl.sqrt(ufl.dot(centre_dir, centre_dir))
        # centre_dir /= norm_centre_dir
        f_cyto = self.k_pr * f_dir
        scalar_normal_component = ufl.dot(f_cyto, self.normal)

        V_m = self.movForce.function_space
        q_m = ufl.TestFunction(V_m)
        p_m = ufl.TrialFunction(V_m)

        a_m = inner(p_m, q_m) * self.dx
        L_m = inner(scalar_normal_component, q_m) * self.dx

        problem_m = fem.petsc.LinearProblem(a_m, L_m, bcs=[], u=self.movForce,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem_m.solve()
        return
    
        # Force that avoid the contact and the intersection between cell and nucleus (cell and nucleus)
    def RepulsiveForce(self):
        if self.role == "nucleus":
            V_c = self.repulsiveForce.function_space
            delta = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_c)
            phi_delta = ufl.exp(-self.alpha * delta)
            f_rep = -self.k_rep * phi_delta 
            #f_rep = 1e-10
        elif self.role == "cell":
            V_c = self.repulsiveForce.function_space
            delta = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_c)
            phi_delta = ufl.exp(-self.alpha * delta)
            f_rep = self.k_rep * phi_delta 
            #f_rep = 1e-10
        else:
            f_rep = 1e-10 #ufl.zero()

        # Variational form
        V_c = self.repulsiveForce.function_space
        q_c = ufl.TestFunction(V_c)
        p_c = ufl.TrialFunction(V_c)

        a_c = inner(p_c, q_c) * self.dx
        L_c = inner(f_rep, q_c) * self.dx

        # Solve and store in self.repulsiveForce
        problem = fem.petsc.LinearProblem(
            a_c, L_c, u=self.repulsiveForce, bcs=[],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        problem.solve()
    

    # Elastic force that link nucleus to cell 
    def ElasticForce(self):
        if self.role == "nucleus":
            # Calcola i centroidi del nucleo e della cellula
            x_n = compute_center(self.domain)
            x_c = compute_center(self.other_gspde.domain)
            delta = x_c - x_n
            delta = ufl.as_vector(delta[:2].tolist())
            #norm_delta = ufl.sqrt(ufl.dot(delta, delta)) + 1e-8  # evita divisione per zero
            #delta_normalized = (delta / norm_delta)
            #projection = ufl.dot(delta_normalized, self.normal) 
            projection = ufl.dot(delta, self.normal)

            #H_at_front = self.H_old.x.array[self.x_front_p]
            #H_at_rear = self.H_old.x.array[self.x_rear_p]

            f_int = self.k_el * projection
            #f_int = projection
        else:
            x_c = compute_center(self.domain)
            x_n = compute_center(self.other_gspde.domain)
            delta = x_c - x_n
            delta = ufl.as_vector(delta[:2].tolist())
            #norm_delta = ufl.sqrt(ufl.dot(delta, delta)) + 1e-8  # evita divisione per zero
            #delta_normalized = (delta / norm_delta)
            #projection = ufl.dot(delta_normalized, self.normal) 
            projection = ufl.dot(delta, self.normal) 

            #H_at_front = self.other_gspde.H_old.x.array[self.other_gspde.x_front_p]
            #H_at_rear = self.other_gspde.H_old.x.array[self.other_gspde.x_rear_p]

            #if np.isclose(H_at_front, H_at_rear, atol=3e-2): 
            #     coeff = self.k_el
            #else:
            #     coeff = self.k_el*self.factor

            #f_int = - coeff * projection
            f_int = - self.k_el * projection

        V_n = self.elasticForce.function_space
        q_n = ufl.TestFunction(V_n)
        p_n = ufl.TrialFunction(V_n)

        a_n = inner(p_n, q_n) * self.dx
        L_n = inner(1e-8 + f_int, q_n) * self.dx  # (aggiunta costante per evitare zero identico)

        problem = fem.petsc.LinearProblem(a_n, L_n, bcs=[], u=self.elasticForce,
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()
        return

    # Set nonlinear problem {{{
    def SetNonlinearProblem(self):
        SetSolverOpt = self.kwargs["SetSolverOpt"]
        self.problem = NonlinearProblem(self.Res, self.w, [], self.tangent)
        self.solver = NewtonSolver(self.comm, self.problem)
        SetSolverOpt(self.solver)
        return
    def SetNonlinearProblemChem(self):
        for i in range(self.N_chem):
            SetSolverOpt = self.kwargs["SetSolverOpt"]
            self.problemChem[i] = NonlinearProblem(self.Res_a[i], self.a_chem[i], [], self.tangent_chem[i])
            self.solverChem[i] = NewtonSolver(self.comm, self.problemChem[i])
            SetSolverOpt(self.solverChem[i])
        return
    # }}}
    # Solve {{{
    def Solve(self):
        iters, converged = self.solver.solve(self.w)
        return iters, converged
    def SolveChem(self):
        for i in range(self.N_chem):
            iters, converged = self.solverChem[i].solve(self.a_chem[i])
        return iters, converged

    # }}}
    # Get global array {{{
    def GetGlobalArray(self, array):
        # Gather array
        gather_array = self.comm.allgather(array)
        # Initialisation of global array
        if len(array.shape) == 1:
            size = [self.imap.size_global]
        elif len(array.shape) == 2:
            _, cols = array.shape
            size = [self.imap.size_global, cols]
        else:
            raise("Not yet available for arrays of len(shape) > 2")
        global_array = np.zeros(size)
        # Fill array
        for k1 in range(self.numRanks):
            global_array[self.gather_global_node_ids[k1]] = gather_array[k1]
        return global_array
    # }}}
    def SmoothCurvature(self):
        # Smoothing parameters
        window_length = 5  # Must be odd
        polyorder = 2       # Order of polynomial
        orderedArray = self.H_old.x.array[self.orderedNodeIds]
        # smoothed_array = savgol_filter(orderedArray, window_length, polyorder)
        smoothed_array = gaussian_filter1d(orderedArray, 10)
        self.smoothed_H.x.array[self.orderedNodeIds] = smoothed_array
        return
# }}}

# Solve iteration {{{
def SolveIteration(t, gspdes):
    # Update time
    for gspde_i in gspdes:
        gspde_i.t_constant.value = t
    # Update loads
    for gspde_i in gspdes:
        gspde_i.UpdateLoads()
    # Solve problem
    for gspde_i in gspdes:
        gspde_i.Solve()
        # Collect results form MPI ghost processes
        gspde_i.w.x.scatter_forward()
    if gspde_i.role == "nucleus":
            gspde_i.SolveChem()
            for i in range(gspde_i.N_chem):
                gspde_i.a_chem[i].x.scatter_forward()
    # Update variables
    for gspde_i in gspdes:
        gspde_i.UpdateVariables()
    return
# }}}



def ComputeBendingStiffness(a_chem_2, bendingStiffness_Ctrl, bendingStiffness_KO, a, b, c, k1, k2, k3, k4):
    """
    Compute bending stiffness as a function of a_chem[2].
    """
    W2 = k4*(k2-b)/(c*k2)
    delta = (k2-b)**2-4*k2*k3/k1*(k1-a)
    U4 = ((k2-b)+sqrt(delta))/(2*k2)
    W4 = k4*U4/c
    return max_value(0.0, (bendingStiffness_KO-bendingStiffness_Ctrl)/(W2-W4)*(a_chem_2-W4))+bendingStiffness_Ctrl

def ComputeTensionStiffness(a_chem_2, tensionStiffness_Ctrl, tensionStiffness_KO, a, b, c, k1, k2, k3, k4):
    """
    Compute tension stiffness as a function of a_chem[2].
    """
    W2 = k4*(k2-b)/(c*k2)
    delta = (k2-b)**2-4*k2*k3/k1*(k1-a)
    U4 = ((k2-b)+sqrt(delta))/(2*k2)
    W4 = k4*U4/c
    return max_value(0.0, (tensionStiffness_KO-tensionStiffness_Ctrl)/(W2-W4)*(a_chem_2-W4))+tensionStiffness_Ctrl