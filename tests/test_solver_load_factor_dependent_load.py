r"""Load vector depending on the load factor: prescribed displacements

Three hardening springs in series, `f_i(e) = k_i e + a_i e^3`, between a
wall and the nodes 1, 2 and 3, with the displacement of node 3 prescribed,
`x_3 = d_0 + \lambda d_1`, and an optional load `\lambda F` on node 1. The
constraint is imposed exactly, by a null-space reduction:

.. math::

    \{c\} = [T]\{c_r\} + \{c_{p0}\} + \lambda \{c_{p1}\}, \qquad
    [T] = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}, \qquad
    \{c_{p0}\} = \{0, 0, d_0\}, \qquad \{c_{p1}\} = \{0, 0, d_1\}

such that the reduced problem has two degrees of freedom and the residual

.. math::

    \{R_r(c_r, \lambda)\} = [T]^T \left(\lambda \{F_{ext}\} -
    \{F_{int}(c)\}\right), \qquad
    \frac{\partial R_r}{\partial \lambda} = [T]^T \left(\{F_{ext}\} -
    [K_T(c)] \{c_{p1}\}\right)

whose derivative with respect to the load factor depends on the
configuration and on the load factor, passed to :class:`pyfe3d.solver.Analysis`
with ``calc_fext(inc=1., c=c_r, lbd=lbd)``, see :mod:`pyfe3d.solver.callbacks`.
Under the prescribed displacement alone, `F = 0`, the reference vector
``calc_fext(inc=1.)`` is null.
Ported from the tests of structsolve.

"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.sparse import csr_matrix

from pyfe3d.solver import Analysis
from pyfe3d.solver import callbacks


T = np.array([[1., 0.], [0., 1.], [0., 0.]])


class SpringsProblem(object):
    def __init__(self, k=(100., 150., 200.), a=(20., 40., 60.), d0=0.,
                 d1=1., F=0., with_lbd=True):
        self.k = np.asarray(k)
        self.a = np.asarray(a)
        self.cp0 = np.array([0., 0., d0])
        self.cp1 = np.array([0., 0., d1])
        self.Fext = np.array([F, 0., 0.])
        self.with_lbd = with_lbd
        self.an = None
        # (lbd, c_r) of each call to calc_fext with lbd, and (inc, ||R_r||,
        # number of converged increments) of each call to calc_fint
        self.fext_calls = []
        self.fint_calls = []

    def full(self, c_r, lbd):
        return T @ c_r + self.cp0 + lbd*self.cp1

    def elongations(self, c):
        return np.array([c[0], c[1] - c[0], c[2] - c[1]])

    def fint_full(self, c):
        k, a = self.k, self.a
        f = k*self.elongations(c) + a*self.elongations(c)**3
        return np.array([f[0] - f[1], f[1] - f[2], f[2]])

    def kT_full(self, c):
        ke = self.k + 3*self.a*self.elongations(c)**2
        return np.array([[ke[0] + ke[1], -ke[1], 0.],
                         [-ke[1], ke[1] + ke[2], -ke[2]],
                         [0., -ke[2], ke[2]]])

    def residual(self, c_r, lbd):
        return T.T @ (lbd*self.Fext - self.fint_full(self.full(c_r, lbd)))

    def dR_dlbd(self, c_r, lbd):
        c = self.full(c_r, lbd)
        return T.T @ (self.Fext - self.kT_full(c) @ self.cp1)

    def reaction(self, c_r, lbd):
        r"""Force of the support on node 3, `c_{p1}^T r / d_1`, with `r =
        F_{int}(c) - \lambda F_{ext}`"""
        c = self.full(c_r, lbd)
        r = self.fint_full(c) - lbd*self.Fext
        return self.cp1 @ r/self.cp1[2]

    # callables of pyfe3d.solver.Analysis
    def calc_fext(self, inc=1., silent=True, c=None, lbd=None):
        if c is None:
            return inc*(T.T @ self.Fext)
        self.fext_calls.append((lbd, c.copy()))
        return inc*self.dR_dlbd(c, lbd)

    def calc_fint(self, c, inc=1., silent=True):
        step = len(self.an.increments) if self.an is not None else 0
        self.fint_calls.append((inc, np.linalg.norm(self.residual(c, inc)),
                                step))
        return T.T @ self.fint_full(self.full(c, inc))

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=True):
        if c is None or not NLgeom:
            return csr_matrix(T.T @ self.kT_full(np.zeros(3)) @ T)
        return csr_matrix(T.T @ self.kT_full(self.full(c, inc)) @ T)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix((2, 2))

    def analysis(self, **kwargs):
        if self.with_lbd:
            calc_fext = self.calc_fext
        else:
            calc_fext = lambda inc=1., silent=True: self.calc_fext(inc)
        an = Analysis(calc_fext, self.calc_fint, self.calc_kC, self.calc_kG)
        for key, value in kwargs.items():
            setattr(an, key, value)
        self.an = an
        return an

    def exact(self, lbd):
        r"""Solution without the load on node 1: the three springs carry the
        same force `P`, found from `\sum_i e_i(P) = x_3`"""
        assert self.Fext[0] == 0.
        x3 = self.cp0[2] + lbd*self.cp1[2]

        def e(P, i):
            return brentq(lambda ei: self.k[i]*ei + self.a[i]*ei**3 - P,
                          -1e4, 1e4, xtol=1e-15, rtol=1e-15)

        def gap(P):
            return sum(e(P, i) for i in range(3)) - x3

        P = 0. if x3 == 0 else brentq(gap, -1e4, 1e4, xtol=1e-14,
                                      rtol=1e-15)
        e1, e2 = e(P, 0), e(P, 1)
        return np.array([e1, e1 + e2]), P

    def solve_reference(self, lbd, tol=1e-14):
        """Dense Newton-Raphson with continuation on the load factor"""
        c_r = np.zeros(2)
        for l in np.linspace(0, lbd, 21)[1:]:
            for _ in range(100):
                R = self.residual(c_r, l)
                if np.linalg.norm(R) <= tol*max(1., np.linalg.norm(
                        self.dR_dlbd(c_r, l))):
                    break
                kT = T.T @ self.kT_full(self.full(c_r, l)) @ T
                c_r = c_r + np.linalg.solve(kT, R)
        return c_r


def test_calc_load_vector_passes_lbd_only_when_declared():
    fext = np.array([1., 2.])
    calls = []

    class A(object):
        calc_fext = None

    def with_c(inc=1., silent=True, c=None):
        calls.append(('c', c))
        return fext

    def with_c_lbd(inc=1., silent=True, c=None, lbd=None):
        calls.append(('c, lbd', c, lbd))
        return fext

    def with_kwargs(inc=1., silent=True, c=None, **kwargs):
        calls.append(('c, **kwargs', c, kwargs))
        return fext

    def without(inc=1., silent=True):
        calls.append('without')
        return fext

    an = A()
    c = np.array([3., 4.])
    for func in (with_c, with_c_lbd, with_kwargs, without):
        an.calc_fext = func
        callbacks.calc_load_vector(an, c, fext, 0.5)
    assert calls[0] == ('c', c)
    assert calls[1] == ('c, lbd', c, 0.5)
    # a **kwargs does not count, as for inc
    assert calls[2] == ('c, **kwargs', c, {})
    # the constant reference vector is returned without calling calc_fext
    assert len(calls) == 3
    assert not callbacks.is_load_factor_dependent(A())
    an.calc_fext = with_c_lbd
    assert callbacks.is_load_factor_dependent(an)


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
@pytest.mark.parametrize('convergence', ['crisfield', 'relative'])
def test_displacement_control_relative_criterion(method, convergence):
    r"""The prescribed displacement alone, with the tolerance of the relative
    force imbalance only, ``absTOL=None``: every converged increment is the
    exact solution, and ``calc_fext`` receives the load factor of the
    configuration"""
    pb = SpringsProblem(d1=0.8)
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-10,
                     absTOL=None, convergence=convergence)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    assert len(an.increments) > 2
    for lbd, c_r in zip(an.increments, an.cs):
        ref, P = pb.exact(lbd)
        assert np.allclose(c_r, ref, rtol=1e-8, atol=1e-12), (lbd, c_r, ref)
        assert np.isclose(pb.reaction(c_r, lbd), P, rtol=1e-8)
        assert (np.linalg.norm(pb.residual(c_r, lbd))
                <= 1e-9*np.linalg.norm(lbd*pb.dR_dlbd(c_r, lbd)))
    # the arc-length solvers evaluate dR/dlbd at the configurations and load
    # factors of their iterations
    if method != 'NR':
        assert len(pb.fext_calls) > len(an.increments)
        lbds = [lbd for lbd, _ in pb.fext_calls]
        assert lbds[0] == 0.
        assert all(lbd is not None for lbd in lbds)
        assert len(set(lbds)) > 2


def test_displacement_control_without_lbd_does_not_converge():
    r"""The reason for ``lbd``: without it, the reference vector and the
    reduced internal force vector vanish at equilibrium, and the relative
    criterion cannot be satisfied"""
    pb = SpringsProblem(d1=0.8, with_lbd=False)
    an = pb.analysis(initialInc=0.2, relTOL=1e-10, absTOL=None,
                     maxNumIter=10, minInc=0.01)
    an.static(NLgeom=True, silent=True)
    assert len(an.increments) == 0


def test_newton_raphson_first_predictor_is_linear_solution():
    r"""With linear springs the predictor of the first increment,
    `[K_0]^{-1} \partial R/\partial \lambda`, is the solution: each
    increment converges at its first evaluation of the residual"""
    pb = SpringsProblem(a=(0., 0., 0.), d1=0.5)
    an = pb.analysis(initialInc=0.3, relTOL=1e-12)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    steps = [step for _, _, step in pb.fint_calls]
    assert steps.count(0) == 1, pb.fint_calls
    ref, _ = pb.exact(1.)
    assert np.allclose(an.cs[-1], ref, rtol=1e-12)


def orders(errors, floor=1e-12):
    """Convergence orders log(e2/e1)/log(e1/e0) above the round-off floor"""
    return [np.log(e2/e1)/np.log(e1/e0)
            for e0, e1, e2 in zip(errors[:-2], errors[1:-1], errors[2:])
            if e2 > floor and e0 < 0.1]


def test_riks_quadratic_with_load_factor_dependent_load_vector():
    r"""The bordered system of Riks is Newton-Raphson applied to the
    augmented system with the exact `\partial R/\partial \lambda`"""
    pb = SpringsProblem(k=(10., 15., 20.), a=(200., 400., 600.), d1=1.)
    an = pb.analysis(NL_method='arc_length_riks', initialInc=0.5,
                     relTOL=1e-13, maxNumIter=60)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    first = [e for _, e, step in pb.fint_calls if step == 0]
    ps = orders(first)
    assert len(ps) >= 1, first
    assert min(ps) > 1.7, first


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
def test_constant_and_incremented_prescriptions(method):
    r"""A constant prescribed displacement `d_0`, held during the analysis,
    with a load `\lambda F` on node 1 and an incremented displacement
    `\lambda d_1`"""
    pb = SpringsProblem(d0=-0.3, d1=0.6, F=40.)
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-11)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c_r in zip(an.increments, an.cs):
        ref = pb.solve_reference(lbd)
        assert np.allclose(c_r, ref, rtol=1e-8, atol=1e-12), (lbd, c_r, ref)
    # at lbd = 0 the constant displacement alone is not in equilibrium with
    # c_r = 0: the first increment starts from a non-null residual
    assert np.linalg.norm(pb.residual(np.zeros(2), 0.)) > 1.


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
def test_null_load_vector_at_the_origin(method):
    r"""The last spring is purely cubic, `f_3 = a_3 e^3`, such that
    `\partial R/\partial \lambda = 0` at `c = 0`, `\lambda = 0`: the
    reference of the relative criterion is `\lambda \partial R/\partial
    \lambda` at the current iterate, not at the origin"""
    pb = SpringsProblem(k=(100., 150., 0.), a=(20., 40., 600.), d1=0.5)
    assert np.linalg.norm(pb.dR_dlbd(np.zeros(2), 0.)) == 0.
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-10,
                     absTOL=None)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c_r in zip(an.increments, an.cs):
        ref, P = pb.exact(lbd)
        assert np.allclose(c_r, ref, rtol=1e-8, atol=1e-12), (lbd, c_r, ref)


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
def test_free_dofs_and_prescribed_displacement(method):
    r"""The same springs with the callables of the full model and ``bu``,
    ``uk``: the reduction of :class:`pyfe3d.solver.FreeDofs` gives the exact
    solution, the full configuration with the prescribed displacement, and
    the reaction from the internal force vector of the full model"""
    pb = SpringsProblem(d1=0.8)
    calls = []

    def calc_fint(c, silent=True):
        calls.append(c.copy())
        return pb.fint_full(c)

    def calc_kC(c=None, NLgeom=False, silent=True):
        return csr_matrix(pb.kT_full(np.zeros(3) if c is None else c))

    an = Analysis(lambda inc=1., silent=True: np.zeros(3), calc_fint, calc_kC,
                  lambda c=None, NLgeom=False, silent=True: csr_matrix((3, 3)),
                  bu=np.array([True, True, False]),
                  uk=np.array([0., 0., 0.8]))
    an.NL_method = method
    an.initialInc = 0.2
    an.relTOL = 1e-10
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c in zip(an.increments, an.cs):
        assert c.shape == (3,)
        assert c[2] == lbd*0.8
        ref, P = pb.exact(lbd)
        assert np.allclose(c[:2], ref, rtol=1e-8, atol=1e-12), (lbd, c, ref)
        assert np.isclose(pb.fint_full(c)[2], P, rtol=1e-8)
    # NOTE every configuration seen by the callables has the prescribed
    #      displacement of a load factor
    assert all(c.shape == (3,) for c in calls)
