r"""Analytic problems with an exact tangent stiffness matrix

The problems wrap the callables required by :class:`pyfe3d.solver.Analysis`,
where ``kT = kC + kG`` is the exact Jacobian of ``fint``.

"""
import numpy as np
from scipy.sparse import csc_matrix

from pyfe3d.solver import Analysis


class Problem(object):
    """Wrap analytic callbacks and record every call to ``calc_fint``

    Each recorded call stores the number of increments already accepted by
    the solver, allowing the iterations of each step to be counted.

    """
    def __init__(self, fext, fint, kC, kG):
        self.fext = fext
        self.fint = fint
        self.kC = kC
        self.kG = kG
        self.an = None
        self.fint_calls = []

    def calc_fext(self, inc=1., silent=True):
        return inc*self.fext

    def calc_fint(self, c, silent=True):
        self.fint_calls.append((len(self.an.increments), c.copy()))
        return self.fint(c)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        return csc_matrix(self.kC)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csc_matrix(self.kG(c))

    def analysis(self, **kwargs):
        an = Analysis(calc_fext=self.calc_fext, calc_fint=self.calc_fint,
                      calc_kC=self.calc_kC, calc_kG=self.calc_kG)
        for k, v in kwargs.items():
            setattr(an, k, v)
        self.an = an
        return an

    def calls_per_step(self):
        """Number of ``calc_fint`` calls spent to reach each accepted step"""
        steps = [s for s, _ in self.fint_calls]
        return [steps.count(i) for i in range(len(self.an.increments))]

    def solve_reference(self, lbd=1., c0=None, tol=1e-15):
        """Plain dense Newton-Raphson to obtain a reference solution"""
        c = np.linalg.solve(self.kC, lbd*self.fext) if c0 is None else c0
        for _ in range(100):
            R = self.fint(c) - lbd*self.fext
            if np.linalg.norm(R) <= tol*np.linalg.norm(lbd*self.fext):
                return c
            c = c + np.linalg.solve(self.kC + self.kG(c), -R)
        raise RuntimeError('reference solution did not converge')


def hardening_springs(P=50.):
    """Chain of 4 springs with cubic hardening: ``fint = K u + a u**3``"""
    n = 4
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = 100.
    a = 50.*np.arange(1, n+1)
    fext = np.zeros(n)
    fext[-1] = P
    return Problem(fext, lambda u: K @ u + a*u**3, K,
                   lambda u: np.diag(3*a*u**2))


def von_mises_truss(frac=0.99, ks=10.):
    """Shallow von Mises truss (height h=1) loaded through a linear spring

    DOF 0 is the apex deflection, DOF 1 the loaded point, connected to the
    apex through a spring of stiffness ``ks``. The truss softens and reaches
    a limit point at ``w = 1 - 1/sqrt(3)``; the load is ``frac`` of the
    limit load.

    The internal force of the two bars is the "shallow truss theory" of
    Crisfield, M. A., Non-linear Finite Element Analysis of Solids and
    Structures, Vol. 1, Wiley, 1991, Sec. 1.2, Eq. (1.7), p. 4, ``W =
    EA/l**3*(z**2*w + 3/2*z*w**2 + 1/2*w**3)`` for one bar, with ``w``
    positive towards the base (``w -> -w``, ``W -> -W``), two bars,
    ``EA/l**3 = 1`` and the rise ``z = 1``: ``F(w) = w**3 - 3*w**2 + 2*w``.
    The spring in series carries the same force, so that at equilibrium
    ``F(w) = lambda*fext[1]`` for any ``ks``: the load factor has the
    maximum ``1/frac`` at ``w = 1 - 1/sqrt(3)`` and the minimum ``-1/frac``
    at ``w = 1 + 1/sqrt(3)``, ``dF/dw = 3*w**2 - 6*w + 2 = 0``, and ``F =
    0`` again at ``w = 2``, the inverted configuration.

    """
    Flim = 2/(3*np.sqrt(3))
    fext = np.array([0., frac*Flim])
    kC = np.array([[2. + ks, -ks], [-ks, ks]])

    def fint(c):
        w = c[0]
        return np.array([w**3 - 3*w**2 + 2*w + ks*(c[0] - c[1]),
                         ks*(c[1] - c[0])])

    def kG(c):
        w = c[0]
        return np.array([[3*w**2 - 6*w, 0.], [0., 0.]])
    return Problem(fext, fint, kC, kG)


def linear_problem():
    K = np.array([[4., -1., 0.], [-1., 3., -1.], [0., -1., 2.]])
    fext = np.array([1., 2., 3.])
    return Problem(fext, lambda c: K @ c, K, lambda c: np.zeros_like(K))


def convergence_orders(errors, emin=1e-13, emax=1e-2):
    """Estimate ``log(e_{k+1})/log(e_k)`` in the asymptotic range"""
    orders = []
    for ek, ek1 in zip(errors[:-1], errors[1:]):
        if ek < emax and ek1 > emin:
            orders.append(np.log(ek1)/np.log(ek))
    return orders
