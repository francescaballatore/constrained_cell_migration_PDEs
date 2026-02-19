# Libraries {{{
from ufl import TestFunction, TrialFunction, grad, inner, dot, derivative
from dolfinx.fem import Function, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
# }}}

# Compute curvature by finite elements {{{
def InitialCurvature(Hh, nh, xh, dx,
                     SetSolverOpt = lambda solver : solver):
    comm = MPI.COMM_WORLD
    V = nh.function_space
    Hvec = Function(V)
    H_test = TestFunction(V)
    dH = TrialFunction(V)
    Res = inner(Hvec, H_test)*dx - inner(grad(xh), grad(H_test))*dx
    a = derivative(Res, Hvec, dH)
    problem = NonlinearProblem(Res, Hvec, [], a)
    solver = NewtonSolver(comm, problem)
    SetSolverOpt(solver)
    (iters, converged) = solver.solve(Hvec)
    Hvec.x.scatter_forward()
    # Get mean curvature
    V = Hh.function_space
    H_expr = Expression(dot(Hvec, nh), V.element.interpolation_points())
    Hh.interpolate(H_expr)
    return
# }}}
