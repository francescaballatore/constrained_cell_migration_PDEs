# Description {{{
"""
Units:
    Basic:
        Length: µm
        Mass: g
        Time: s
    Derived:
        Mass density: g/(µm)³
        Force: nN
        Pressure: kPa
        Energy: 10^-15 J
"""
# }}}

# Libraries {{{
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix, create_matrix
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Constant, Function, dirichletbc, Expression, form, assemble_scalar
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.la import create_petsc_vector

import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, inner, det, div, dot, inv, tr, as_vector, outer, derivative, dev, sqrt)
import csv
import basix
from basix.ufl import element, quadrature_element

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")
from datetime import datetime
import pyvista
import gmsh
from shapely.geometry import Polygon
import copy
from pdb import set_trace

# In-house modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from python_utils.output_utils import *
from python_utils.mesh_utils import *
from python_utils.misc_utils import *
from python_utils.upLagrangian_utils import *
from python_utils.mecha_utils import *
from python_utils.solver_utils import *
from python_utils.gspde_utils_project import *
from python_utils.turnover_utils import *
#from python_utils.forces_utils import *
# }}}

# Setting {{{
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
log.set_log_level(log.LogLevel.WARNING)

visualisation = 0
np.random.seed(16)
# }}}

# Parameters {{{
# Geometry-mesh
nucleusDia = 10.0
cellDia = 20.0
lc_ce = 5.0e-2
lc_n = 5.0e-2
meshOrder = 2
# Material
Aref_ce = (cellDia/2.0)**2.0*np.pi
periRef_ce = (cellDia/2.0)*2.0*np.pi
Aref_n = (nucleusDia/2.0)**2.0*np.pi
periRef_n = (nucleusDia/2.0)*2.0*np.pi
# Membrane tension is given here in nN/µm and bending stiffness in 10^-15 J
tensionStiffness_cell = 1.0
bendingStiffness_cell = 1.0e-3
tensionStiffness_n = tensionStiffness_cell*10
bendingStiffness_n = bendingStiffness_cell*10
# Initial conditions
opre0_ce = 0.0
opre0_n = 0.0
# Results name
results_name = "../results/Actin/" 
# Time scheme
Ttot = 2.0
dt = 1.0e-3 
print_each = 1
#Viscous force
omega = 1e-1 
# Repulsive force
k_sr = 1.0e0
delta_d = 1.0e-1
kappa_d  = 5.0e0
delta_n = -0.7
kappa_n  = 5.0e0
# Pressure force
k_pr = 16 
# Microchannel geometry
length = 100
height = 20
width = 6
x_left = 20.0
# Barrier force
k_bar = 20.0
beta = 3.5
# Elastic force
k_el = 6.0
factor = 2
# Repulsive force
alpha = 5.0
k_rep = 12.0
# PDEs chemical 
N_chem = 3  # number of species
D_chem = [1.0, 0.1, 1.0]
# }}}

# Solver
quadrature_degree = 8
def SetSolverOpt(solver):
    # Newton solver
    solver.convergence_criterion = "incremental"
    solver.rtol = 1.0e-8
    solver.atol = 1.0e-8
    solver.max_it = 25
    solver.report = True
    solver.relaxation_parameter = 1.0
    # Krylov solver
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"]   = "preonly"
    opts[f"{option_prefix}pc_type"]    = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    opts[f"{option_prefix}ksp_max_it"] = 1000
    ksp.setFromOptions()
    return
# }}}

# Make mesh {{{
def MakeCircle(radius : float, lc : float, meshOrder = 1):
    model = gmsh.model
    geo = model.occ
    model.add("cell")
    # Create points
    cc = geo.addPoint(0.0, 0.0, 0.0, lc)
    rp = geo.addPoint(radius, 0.0, 0.0, lc)
    lp = geo.addPoint(-radius, 0.0, 0.0, lc)
    # Create lines
    topLine = geo.addCircleArc(rp, cc, lp)
    botLine = geo.addCircleArc(lp, cc, rp)
    # Create loop
    loop = geo.addCurveLoop([topLine, botLine])
    # Synchronise
    geo.synchronize()
    # Groups
    gr_gamma = model.addPhysicalGroup(1, [topLine, botLine])
    # Mesh
    model.mesh.generate(dim = 1)
    model.mesh.setOrder(meshOrder)
    # gmsh.fltk.run()
    return model
# }}}
# Initialisation
gmsh.initialize()
# Set cell {{{
model_ce = MakeCircle(cellDia/2.0, lc_ce, meshOrder = meshOrder)
cell_params = {
        "model" : model_ce,
        "quadrature_degree" : quadrature_degree,
        "meshOrder" : meshOrder,
        "dt" : dt,
        "opre0" : opre0_ce,
        "SetSolverOpt" : SetSolverOpt,
        "Dia" : cellDia,
        "aRef" : Aref_ce,
        "periRef" : periRef_ce,
        "Href" : 2.0/cellDia,
        "k_sr" : k_sr,
        "delta_d" : delta_d,
        "kappa_d" : kappa_d,
        "delta_n" : delta_n,
        "kappa_n" : kappa_n,
        "length" : length,
        "height" : height,
        "width" : width,
        "x_left" : x_left,
        "beta" : beta, 
        "omega" : omega, 
        "k_bar" : k_bar, 
        "k_pr" : k_pr, 
        "k_el" : k_el,
        "factor" : factor,
        "alpha" : alpha,
        "k_rep" : k_rep,
        "surfacetension" : tensionStiffness_cell,
        "bendingstiffness" : bendingStiffness_cell,
        "typeOpressure" : "area",
        "equidistribute" : True,
        "role" : "cell",
        }
cellGS = GSPDE(**cell_params)
# }}}

# Set basement membrane {{{
n_params = copy.deepcopy(cell_params)
model_n = MakeCircle(nucleusDia/2.0, lc_n, meshOrder = meshOrder)
n_params["model"] = model_n
n_params["aRef"] = Aref_n
n_params["periRef"] = periRef_n
n_params["Dia"] = nucleusDia
n_params["Href"] = 2.0/nucleusDia
n_params["opre0"] = opre0_n
n_params["k_pr"] = 1e-8
n_params["typeOpressure"] = "area"
n_params["equidistribute"] = True
n_params["role"] = "nucleus"
n_params["surfacetension"] = tensionStiffness_n
n_params["bendingstiffness"] = bendingStiffness_n   
n_params["N_chem"] = N_chem
n_params["D_chem"] = D_chem

nGS = GSPDE(other_gspde=cellGS, **n_params)
cellGS.other_gspde = nGS
# }}}

# Set up output {{{
timeList = [0.0]
areaList_ce = [cellGS.area]
periList_ce = [cellGS.perimeter]
velocityList_ce = [0.0]
stressList_ce = [0.0]
x_front = [cellDia/2]
x_rear = -cellDia/2
# Plasma membrane
functions_list = [cellGS.disp, cellGS.H_old, cellGS.normal,
                  cellGS.selfRepuForce, cellGS.barrierForce,
                  cellGS.movForce, cellGS.repulsiveForce, cellGS.elasticForce,
                  cellGS.phi, cellGS.bendingStiffness, cellGS.tensionStiffness]

names_list = ["u", "H", "n", "Fsr", "Fbar", "Fpr", "Frep", "Fel",
              "phi", "Fb", "Fs"]

out_ce = Output(cellGS.domain, functions_list, names_list, "/Actin/resu_ce", comm)
# Nuclear envelope
functions_list = [nGS.disp, nGS.H_old, nGS.normal,
                  nGS.selfRepuForce, nGS.barrierForce,
                  nGS.elasticForce, nGS.repulsiveForce, nGS.phi,
                  nGS.bendingStiffness, nGS.tensionStiffness] + nGS.a_chem
names_list = ["u", "H", "n", "Fsr", "Fbar", "Fel", "Frep",
              "phi", "Fb", "Fs"] + [f"a_chem_{i}" for i in range(len(nGS.a_chem))]
out_n = Output(nGS.domain, functions_list, names_list, "/Actin/resu_n", comm)

areaList_n = [nGS.area]
periList_n = [nGS.perimeter]
x_center_old = compute_center(cellGS.domain)
SurfaceEnergyList = [assemble_scalar(form(cellGS.tensionStiffness * cellGS.dx))]
BendingEnergyList = [assemble_scalar(form(cellGS.bendingStiffness/2 * cellGS.H_old**2 * cellGS.dx))]
# }}}

PlotMicrochannel(x_left, length, width, height, "../results/Actin/vtk/barrier_plot.vtk")
#PlotCircles(bmCenters, bmDia, "results/Actin/vtk/barrier_plot.vtk")
     
csv_filename = "../results/Actin/Actin.csv"
data_written = False
touch = False
cell_touch = 0.0

# Calculation loop {{{
# Initialisation
t = 0.0
mprint("------------------------------------", rank = rank)
mprint("Simulation Start", rank = rank)
mprint("------------------------------------", rank = rank)
startTime = datetime.now()
printTime0 = datetime.now()
# To solve variables
gspdes_list = [cellGS, nGS]
# Time stepping solution procedure loop
k1 = 0
cellGS.SetInitialisation()
nGS.SetInitialisation()
out_ce.WriteResults(t = t)
out_n.WriteResults(t = t)

while (round(t + dt, 9) <= Ttot):
    # Update iteration
    k1 += 1
    # Solution
    t += dt
    SolveIteration(t, gspdes_list)

    # Report area and perimeter
    timeList.append(t)
    areaList_ce.append(cellGS.area)
    periList_ce.append(cellGS.perimeter)
    areaList_n.append(nGS.area)
    periList_n.append(nGS.perimeter)
     # Save velocity 
    x_center = compute_center(cellGS.domain)
    vel_norm = np.linalg.norm((x_center - x_center_old) / dt)
    velocityList_ce.append(vel_norm)
    # Save stress
    #stressList_ce.append(cellGS.avg_sigma_n)
    x_center_old = x_center
    # Save front and rear
    x_front.append(cellGS.x_front)
    x_rear = cellGS.x_rear
    # Save surface energy
    surface_energy_form = form(cellGS.tensionStiffness * cellGS.dx) 
    surface_energy = assemble_scalar(surface_energy_form)
    SurfaceEnergyList.append(surface_energy)
    # Save bending energy
    bending_energy_form = form(cellGS.bendingStiffness/2 * cellGS.H_old**2 * cellGS.dx) 
    bending_energy = assemble_scalar(bending_energy_form)
    BendingEnergyList.append(bending_energy)
    # Print progress
    printTime1 = datetime.now()
    cpu_time = printTime1 - printTime0
    printTime0 = printTime1
    mprint("------------------------------------", rank = rank)
    mprint("Increment: {} | CPU time: {}".format(k1, cpu_time), rank = rank)
    mprint("dt: {} s | Simulation time {} s of {} s".format(round(dt, 4), round(t, 4), Ttot), rank = rank)
    mprint("", rank = rank)
    mprint("------------------------------------", rank = rank)
    # Write output results
    if k1%print_each == 0:
        out_ce.WriteResults(t)
        out_n.WriteResults(t)
        # Save data file
        data = {
            "Time" : np.array(timeList),
            "Area_ce" : np.array(areaList_ce),
            "Area_n" : np.array(areaList_n),
            "Peri_ce" : np.array(periList_ce),
            "Peri_n" : np.array(periList_n),
            "Vel_ce" : np.array(velocityList_ce),  
            #"Stress_ce" : np.array(stressList_ce),
            "x_front" : np.array(x_front),
            "SurfaceEnergy_ce" : np.array(SurfaceEnergyList),
            "BendingEnergy_ce" : np.array(BendingEnergyList),
        }
        data = pd.DataFrame(data)
        data.to_csv("../results/Actin/resu.csv")
    if (not touch) and (np.min(cellGS.barrierForce.x.array) < -1):
            cell_touch = t
            touch = True 
    if (not data_written) and (x_rear >= x_left):
        with open(csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Entry time", "Cell diameter", "Surface tension"])
            writer.writerow([round(t-cell_touch,3), cellDia, tensionStiffness_cell])

        data_written = True   
    
# Close files
out_ce.Close()
out_n.Close()

mprint("-----------------------------------------", rank = rank)
mprint("End computation", rank = rank)
# Report elapsed real time for the analysis
endTime = datetime.now()
elapseTime = endTime - startTime
mprint("------------------------------------------", rank = rank)
mprint("Elapsed real time:  {}".format(elapseTime))
mprint("------------------------------------------", rank = rank)

#}}}