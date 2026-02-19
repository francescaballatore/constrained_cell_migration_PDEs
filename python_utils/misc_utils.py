# Libraries {{{
import dolfinx
from dolfinx import plot, fem
import pyvista

import ufl
from ufl import (TestFunction, TrialFunction, Identity, grad, inner, det,
                 inv, tr, as_vector, outer, derivative, dev, sqrt)

import numpy as np
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")
# }}}

# Visualisation {{{
def Visu(domain, vdim, tagObject, tags):
    # Set up plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Mesh", font_size = 14, color = "black",
                     position = "upper_edge")
    # Add whole mesh
    unGrid = pyvista.UnstructuredGrid
    plotter.add_mesh(unGrid(*plot.vtk_mesh(domain, domain.topology.dim)),
                     show_edges = True, show_scalar_bar = False)
    # Add sub meshes: cells
    if vdim == 1:
        for tag_i in tags:
            plotter.add_mesh(unGrid(*plot.vtk_mesh(domain,
                                                   entities = tagObject.find(tag_i),
                                                   dim = vdim)),
                             show_edges = True,
                             edge_color = "black",
                             line_width = 10)
    else:
        for tag_i in tags:
            plotter.add_mesh(unGrid(*plot.vtk_mesh(domain,
                                                   entities = tagObject.find(tag_i),
                                                   dim = vdim)),
                             show_edges = True,
                             edge_color = "red")
    plotter.view_xy()
    plotter.show()
    return
# }}}
# Projection problem {{{
# From: https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FV01_VHB_uniaxial_tension_eq.ipynb
def setup_projection(u, V, dx):

    trial = ufl.TrialFunction(V)
    test  = ufl.TestFunction(V)

    a = ufl.inner(trial, test)*dx
    L = ufl.inner(u, test)*dx

    projection_problem = dolfinx.fem.petsc.LinearProblem(a, L, [], \
        petsc_options={"ksp_type": "cg",
                       "ksp_rtol": 1e-16,
                       "ksp_atol": 1e-16,
                       "ksp_max_it": 1000})

    return projection_problem
# }}}
# mprint {{{
# From: https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FV09_NBR_bushing_shear_MPI.py
# this forces the program to still print (but only from one CPU)
# when run in parallel.
def mprint(*argv, rank = 0):
    if rank==0:
        out = ""
        for arg in argv:
            out = out + str(argv)
        print(out, flush = True)
# }}}
# L1 and L2-norm {{{
def L2norm(domain, u, dx):
    normForm = fem.form(inner(u, u)*dx)
    norm = fem.assemble_scalar(normForm)
    return np.sqrt(norm)
def L1norm(domain, u, dx):
    normForm = fem.form(sqrt(inner(u, u))*dx)
    norm = fem.assemble_scalar(normForm)
    return norm
# }}}
# From vector to matrix {{{
def FromVectorToMatrix(vector :  np.ndarray, numComp : int):
    # Check compatibility
    vSize = vector.shape[0]
    if not vSize%numComp == 0:
        raise("Incompatible vector size with number of components")
    # Initialise matrix
    numRows = int(vSize/numComp)
    matrix = np.zeros((numRows, numComp))
    # Assign values
    for k1 in range(numRows):
        for k2 in range(numComp):
            matrix[k1, k2] = vector[numComp*k1 + k2]
    return matrix
# }}}
# Plot 2D lines {{{
def Plot2DLines(df, xkey, ykeys, **kwargs):
    # Get figure size
    cm = 1.0 / 2.54
    figsize = kwargs.get("figsize", [8.0 * cm, 6.0 * cm])
    # Initialisation of figure
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    # Get x axis data
    x = df[xkey[0]] if isinstance(xkey, list) else df[xkey]
    # Plot lines
    for ykey in ykeys:
        ax.plot(x, df[ykey], label=ykey)
    # Set x label
    ax.set_xlabel(r"$t$")
    # External configuration
    figConf = kwargs.get("figConf", lambda fi, ai: (fi, ai))
    fig, ax = figConf(fig, ax)
    return fig, ax

import gmsh
import os

def PlotCircles(bmCenters, bmDia, filename):
    gmsh.initialize()
    model = gmsh.model
    geo = model.occ

    model.add("barrier")
    lc = 0.5

    surfaces = []

    for center_coords in bmCenters:
        cp = geo.addPoint(*center_coords, 0.0, lc)
        rp = geo.addPoint(center_coords[0] + bmDia/2, center_coords[1], 0.0, lc)
        lp = geo.addPoint(center_coords[0] - bmDia/2, center_coords[1], 0.0, lc)
        arc1 = geo.addCircleArc(rp, cp, lp)
        arc2 = geo.addCircleArc(lp, cp, rp)
        loop = geo.addCurveLoop([arc1, arc2])
        surface = geo.addPlaneSurface([loop])
        surfaces.append(surface)

    geo.synchronize()

    model.addPhysicalGroup(2, surfaces, tag=2)
    model.setPhysicalName(2, 2, "barrier_surfaces")

    model.mesh.generate(2)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gmsh.write(filename)
    gmsh.finalize()


def PlotMicrochannel(x_left, length, width, height, filename):
    gmsh.initialize()
    model = gmsh.model
    geo = model.occ

    model.add("barrier")
    lc = 0.5

    centres = [
        (x_left, +width/2 + height / 2.0),    
        (x_left, -width/2 - height / 2.0),    
        (x_left + length, +width/2 + height / 2.0),  
        (x_left + length, -width/2 - height / 2.0)   
    ]

    surfaces = []

    for i, (cx, cy) in enumerate(centres):
        center = geo.addPoint(cx, cy, 0, lc)

        if i < 2:  
            pt_top = geo.addPoint(cx, cy + height / 2.0, 0, lc)
            pt_mid = geo.addPoint(cx - height / 2.0, cy, 0, lc)
            pt_bot = geo.addPoint(cx, cy - height / 2.0, 0, lc)
            arc1 = geo.addCircleArc(pt_top, center, pt_mid)
            arc2 = geo.addCircleArc(pt_mid, center, pt_bot)
            line = geo.addLine(pt_bot, pt_top)
        else:  
            pt_bot = geo.addPoint(cx, cy - height / 2.0, 0, lc)
            pt_mid = geo.addPoint(cx + height / 2.0, cy, 0, lc)
            pt_top = geo.addPoint(cx, cy + height / 2.0, 0, lc)
            arc1 = geo.addCircleArc(pt_bot, center, pt_mid)
            arc2 = geo.addCircleArc(pt_mid, center, pt_top)
            line = geo.addLine(pt_top, pt_bot)

        loop = geo.addCurveLoop([arc1, arc2, line])
        surf = geo.addPlaneSurface([loop])
        surfaces.append(surf)

    # Parete superiore
    x0, x1 = x_left, x_left + length
    y_top, y_bot = width / 2, -width / 2
    p1 = geo.addPoint(x0, y_top, 0, lc)
    p2 = geo.addPoint(x1, y_top, 0, lc)
    p3 = geo.addPoint(x1, y_top + height, 0, lc)
    p4 = geo.addPoint(x0, y_top + height, 0, lc)
    l1 = geo.addLine(p1, p2)
    l2 = geo.addLine(p2, p3)
    l3 = geo.addLine(p3, p4)
    l4 = geo.addLine(p4, p1)
    loop_top = geo.addCurveLoop([l1, l2, l3, l4])
    surfaces.append(geo.addPlaneSurface([loop_top]))

    # Parete inferiore
    p5 = geo.addPoint(x0, y_bot, 0, lc)
    p6 = geo.addPoint(x1, y_bot, 0, lc)
    p7 = geo.addPoint(x1, y_bot - height, 0, lc)
    p8 = geo.addPoint(x0, y_bot - height, 0, lc)
    l5 = geo.addLine(p5, p6)
    l6 = geo.addLine(p6, p7)
    l7 = geo.addLine(p7, p8)
    l8 = geo.addLine(p8, p5)
    loop_bot = geo.addCurveLoop([l5, l6, l7, l8])
    surfaces.append(geo.addPlaneSurface([loop_bot]))

    geo.synchronize()
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.model.setPhysicalName(2, 1, "barrier_surfaces")

    gmsh.model.mesh.generate(2)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gmsh.write(filename)
    gmsh.finalize()

def compute_center(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    coords = mesh.geometry.x
    center3d = np.mean(coords, axis=0)
    center2d = center3d[:2]  # prendi solo x e y
    return center2d

def compute_distance_to_cortex(nucleus_mesh: dolfinx.mesh.Mesh, cortex_mesh: dolfinx.mesh.Mesh, V: fem.FunctionSpace) -> fem.Function:
    """
    For each point on the nucleus mesh, compute the minimal Euclidean distance to the cortex mesh and return it as a fem.Function.
    """
    # Coordinates of points on each surface
    x_n = nucleus_mesh.geometry.x  # coordinates of the nucleus mesh
    x_c = cortex_mesh.geometry.x   # coordinates of the cell (cortex)

    # Build a KDTree on the cortex points
    cortex_tree = cKDTree(x_c)

    # Query minimal distance for each point of the nucleus
    distances, _ = cortex_tree.query(x_n)

    # Create a Function and interpolate distances
    delta_func = fem.Function(V)
    delta_func.x.array[:] = distances
    delta_func.x.scatter_forward()

    return delta_func

# Adaptive time solution {{{
def AdaptiveTimeSolver(ite, tf, dt, maxForceDiff, gspdes, 
                       toSolve_list, barrierForceId_list, minStepFrac = 4.0):
    # Solve the system in a test
    run = True
    test_dt = dt
    t0 = tf - dt
    test_t = t0 + test_dt
    gspde_solve = [gspdes[k1] for k1 in toSolve_list]
    gspde_tests = [gspdes[k1] for k1 in barrierForceId_list]
    numTests = len(gspde_tests)
    while run:
        # Create a copy of the problem
        w_copy = []
        for gspde_i in gspdes:
            w_copy.append(np.copy(gspde_i.w.x.array))
        # Update time
        for gspde_i in gspdes:
            gspde_i.dk.value = test_dt
            gspde_i.t_constant.value = test_t
        # Update variables
        if ite > 1:
            for gspde_i in gspde_solve:
                gspde_i.UpdateVariables()
        # Update loads
        for gspde_i in gspdes:
            gspde_i.UpdateLoads()
        # Evaluate current force
        global_currentForces = []
        for gspde_test in gspde_tests:
            currentForce = gspde_test.barrierForce.x.array + gspde_test.selfRepuForce.x.array + gspde_test.movForce.x.array + gspde_test.mechForce.x.array + gspde_test.retainForce.x.array + gspde_test.nucleusForce.x.array + gspde_test.repulsiveForce.x.array
            global_currentForces.append(np.copy(gspde_test.GetGlobalArray(currentForce)))
        global_currentForce = np.hstack(global_currentForces)
        # Solve problems
        for gspde_i in gspde_solve:
            gspde_i.Solve()
            # Collect results form MPI ghost processes
            gspde_i.w.x.scatter_forward()
        # New ws
        new_w = []
        for gspde_i in gspdes:
            new_w.append(np.copy(gspde_i.w.x.array))
        # Check new force {{{
        for gspde_i in gspde_solve:
            gspde_i.UpdateVariables()
        # Update loads
        for gspde_i in gspdes:
            gspde_i.UpdateLoads()
        # Compute future force
        global_futureForces = []
        for gspde_test in gspde_tests:
            futureForce = gspde_test.barrierForce.x.array + gspde_test.selfRepuForce.x.array + gspde_test.movForce.x.array + gspde_test.mechForce.x.array + gspde_test.retainForce.x.array + gspde_test.nucleusForce.x.array + gspde_test.repulsiveForce.x.array 
            global_futureForces.append(np.copy(gspde_test.GetGlobalArray(futureForce)))
        global_futureForce = np.hstack(global_futureForces)
        diffForce = np.linalg.norm(np.abs(global_futureForce - global_currentForce),
                                   np.inf)
        print("--------------------")
        print("test_t ", test_t, "test_dt", test_dt)
        print("Max force difference: ", maxForceDiff, "Force difference: ", diffForce)
        # }}}
        # Check
        if ite == 1:
            run = False
        else:
            if diffForce < maxForceDiff:
                if np.isclose(test_t, tf):
                    run = False
                else:
                    test_t += test_dt
            else:
                # Check size of dt
                if np.isclose(test_dt, dt/minStepFrac):
                    # Finish while if tf is reached
                    if np.isclose(test_t, tf):
                        run = False
                    # Continue with the same dt
                    test_t += test_dt
                    print("Warning: not able to reduce the time step!")
                else:
                    # Reduce dt
                    test_dt = test_dt/2.0
                    test_t -= test_dt
                    # Go to previous solution
                    for gspde_i, w_i in zip(gspdes, w_copy):
                        gspde_i.w.x.array[:] = w_i[:]
    return new_w
# }}}

