# Libraries {{{
import ufl
from ufl import (Identity, grad, inner, det, inv, tr, outer, sqrt)
# }}}

# Constitutive relations {{{
# Saint venant second Piola-Kirchhoff stress
def S_SaintVenant(C, mu, la, I):
    EE = 0.5*(C - I)
    return la*tr(EE)*I + 2.0*mu*EE
# Compressible Neo-Hookean second Piola-Kirchhoff stress
def S_CompNeoHookean(C, mu, la, I):
    """
    Strain energy function:
        (mu/2)*(I1 - 3) - mu*ln(J) + (la/2)*(ln(J))**2
    """
    return mu*(I - inv(C)) + la*ufl.ln(det(C))*inv(C)
# }}}

# Elastic relations {{{
la_func  = lambda E, nu: E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
mu_func  = lambda E, nu: E/(2.0*(1.0 + nu))
tau_func = lambda E, eta: eta/E
G_func   = lambda E, nu: E/(2.0*(1.0 + nu))
ka_func  = lambda E, nu: E/(3.0*(1.0 - 2.0*nu))
# }}}

# Newmark beta method {{{
def update_a(delta_u, v_old, a_old, dt, beta = 0.25):
    term1 = (delta_u)/(beta*dt**2.0)
    term2 = v_old/(beta*dt)
    term3 = (1.0/(2.0*beta) - 1.0)*a_old
    return term1 - term2 - term3
def update_v(a, delta_u, v_old, a_old, dt, beta = 0.25, gamma = 0.5):
    term1 = gamma*(delta_u)/(beta*dt)
    term2 = (1.0 - gamma/beta)*v_old
    term3 = dt*(1.0 - gamma/(2.0*beta))*a_old
    return term1 + term2 + term3
# }}}
