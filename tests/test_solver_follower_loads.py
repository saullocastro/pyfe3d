r"""Configuration-dependent (follower) loads in the static analyses

The analytic problem has the internal force of hardening springs and a load
that depends on the configuration through an unsymmetric matrix ``Q``:

.. math::

    \boldsymbol{F}_{int}(\boldsymbol{c}) = \boldsymbol{K} \boldsymbol{c} + \{a c^3\}, \qquad
    \boldsymbol{F}(\boldsymbol{c}, \lambda) = \lambda (\boldsymbol{q}_0 + \boldsymbol{Q} \boldsymbol{c})

passed to :class:`pyfe3d.solver.Analysis` as described in
:mod:`pyfe3d.solver.callbacks`: ``calc_fint(c, inc)`` contains ``-inc Q c``,
``calc_kC(c, NLgeom, inc)`` contains ``-inc Q``, and ``calc_fext(inc, c)`` is
the load of the configuration ``c``. A geometrically linear structure under
a load affine in the configuration is solved by the load correction
iterations of the linear static analysis.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyfe3d.solver import Analysis
from pyfe3d.solver import callbacks


class FollowerProblem(object):
    def __init__(self, K, a, q0, Q, with_c=True):
        self.K = K
        self.a = a
        self.q0 = q0
        self.Q = Q
        self.with_c = with_c
        self.an = None
        # (inc, ||R||/||lbd*q||, number of converged increments) of each
        # calc_fint call
        self.calls = []

    def fint_el(self, c):
        return self.K @ c + self.a*c**3

    def residual(self, c, lbd):
        return lbd*(self.q0 + self.Q @ c) - self.fint_el(c)

    def calc_fext(self, inc=1., silent=True, c=None):
        if c is None:
            return inc*self.q0
        return inc*(self.q0 + self.Q @ c)

    def calc_fint(self, c, inc=1., silent=True):
        R = self.residual(c, inc)
        ref = max(np.linalg.norm(inc*(self.q0 + self.Q @ c)), 1e-300)
        step = len(self.an.increments) if self.an is not None else 0
        self.calls.append((inc, np.linalg.norm(R)/ref, step))
        return self.fint_el(c) - inc*(self.Q @ c)

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=True):
        if not NLgeom:
            return csr_matrix(self.K)
        return csr_matrix(self.K + np.diag(3*self.a*c**2) - inc*self.Q)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix(self.K.shape)

    def analysis(self, **kwargs):
        if self.with_c:
            calc_fext = self.calc_fext
        else:
            calc_fext = lambda inc=1., silent=True: self.calc_fext(inc)
        an = Analysis(calc_fext, self.calc_fint, self.calc_kC, self.calc_kG)
        for k, v in kwargs.items():
            setattr(an, k, v)
        self.an = an
        return an


def make_problem(with_c=True):
    n = 3
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[-1, -1] = 100.
    a = np.array([20., 40., 60.])
    q0 = np.array([0., 0., 30.])
    # unsymmetric load stiffness, of the order of K
    Q = np.array([[0., 30., -20.],
                  [-30., 10., 40.],
                  [10., -40., 20.]])
    return FollowerProblem(K, a, q0, Q, with_c=with_c)


def test_accepts_kwarg():
    def f(c, inc=1., silent=True):
        pass
    def g(c, silent=True, **kwargs):
        pass
    class A(object):
        def m(self, c, inc=1.):
            pass
    assert callbacks.accepts_kwarg(f, 'inc')
    assert not callbacks.accepts_kwarg(f, 'c_cte')
    # **kwargs does not count, such that old callables are never surprised
    assert not callbacks.accepts_kwarg(g, 'inc')
    assert callbacks.accepts_kwarg(A().m, 'inc')
    assert not callbacks.accepts_kwarg(len, 'inc')


def test_callables_without_inc_are_called_as_before():
    calls = []
    def calc_fint(c, silent=True):
        calls.append('fint')
        return 2*c
    def calc_kC(c=None, NLgeom=False, silent=True):
        calls.append('kC')
        return csr_matrix(2*np.eye(2))
    def calc_kG(c=None, NLgeom=False, silent=True):
        calls.append('kG')
        return csr_matrix((2, 2))
    def calc_fext(inc=1., silent=True):
        return inc*np.array([1., 2.])
    an = Analysis(calc_fext, calc_fint, calc_kC, calc_kG)
    an.static(NLgeom=True, silent=True)
    assert np.allclose(an.cs[-1], [0.5, 1.])
    an.NL_method = 'arc_length_riks'
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    assert np.allclose(an.cs[-1], [0.5, 1.])


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
def test_equilibrium_with_follower_load(method):
    pb = make_problem()
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-10)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c in zip(an.increments, an.cs):
        R = pb.residual(c, lbd)
        assert np.linalg.norm(R) <= 1e-8*np.linalg.norm(lbd*pb.q0), (lbd, R)
    # the follower terms matter: the dead-load solution is different
    pb_dead = make_problem()
    pb_dead.Q = 0*pb.Q
    an_dead = pb_dead.analysis(relTOL=1e-10)
    an_dead.static(NLgeom=True, silent=True)
    assert np.linalg.norm(an_dead.cs[-1] - an.cs[-1]) > 0.1*np.linalg.norm(an.cs[-1])


def orders(errors, floor=1e-12):
    """Convergence orders log(e2/e1)/log(e1/e0) above the round-off floor"""
    return [np.log(e2/e1)/np.log(e1/e0)
            for e0, e1, e2 in zip(errors[:-2], errors[1:-1], errors[2:])
            if e2 > floor and e0 < 5e-2]


def test_newton_raphson_quadratic_with_follower_load():
    pb = make_problem()
    an = pb.analysis(initialInc=1., relTOL=1e-14)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    errors = [e for inc, e, _ in pb.calls if inc == 1.]
    ps = orders(errors)
    assert len(ps) >= 1, errors
    assert min(ps) > 1.7, errors


def test_riks_quadratic_only_with_configuration_load_vector():
    r"""The bordered system of Riks is the Newton-Raphson method of the
    augmented system only when dR/dlbd = q0 + Q c, i.e. with ``calc_fext``
    accepting ``c``; with the constant reference vector the corrections
    converge linearly"""
    worst = {}
    for with_c in (True, False):
        pb = make_problem(with_c=with_c)
        an = pb.analysis(NL_method='arc_length_riks', initialInc=0.5,
                         relTOL=1e-13, maxNumIter=60)
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        # residuals of the iterations of the first step, whose first value
        # is the one of the tangent predictor
        first = [e for inc, e, step in pb.calls if step == 0]
        worst[with_c] = (min(orders(first)) if orders(first) else None,
                         first)
    p_c, errors_c = worst[True]
    assert p_c is not None and p_c > 1.7, errors_c
    p_const, errors_const = worst[False]
    assert len(errors_const) > len(errors_c), (errors_c, errors_const)


class LinearFollower(object):
    r"""Geometrically linear structure under a configuration-dependent load
    ``F(c) = F0 + Q c``"""
    def __init__(self, K0, Q, F0, with_c=True):
        self.K0 = K0
        self.Q = Q
        self.F0 = F0
        self.with_c = with_c

    def calc_fext(self, inc=1., silent=True, c=None):
        if c is None:
            return inc*self.F0
        return inc*(self.F0 + self.Q @ c)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        return csr_matrix(self.K0)

    def analysis(self):
        if self.with_c:
            return Analysis(calc_fext=self.calc_fext, calc_kC=self.calc_kC)
        return Analysis(calc_fext=lambda inc=1., silent=True:
                        self.calc_fext(inc), calc_kC=self.calc_kC)


def linear_follower_problem(with_c=True, factor=0.3):
    rng = np.random.default_rng(4)
    n = 6
    X = rng.standard_normal((n, n))
    K0 = X @ X.T + n*np.eye(n)
    W = rng.standard_normal((n, n))
    # unsymmetric load stiffness, scaled to a fraction of the critical one
    Q = W - 0.5*W.T
    rho = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(K0, Q))))
    Q *= factor/rho
    F0 = rng.standard_normal(n)
    return LinearFollower(K0, Q, F0, with_c=with_c)


def test_analysis_linear_static_with_configuration_dependent_load():
    pb = linear_follower_problem()
    an = pb.analysis()
    an.relTOL = 1e-12
    increments, cs = an.static(NLgeom=False, silent=True)
    ref = np.linalg.solve(pb.K0 - pb.Q, pb.F0)
    assert np.allclose(cs[0], ref, rtol=1e-9)
    # without c the load is the reference one, as before
    pb_old = linear_follower_problem(with_c=False)
    increments, cs = pb_old.analysis().static(NLgeom=False, silent=True)
    assert np.allclose(cs[0], np.linalg.solve(pb.K0, pb.F0))


def test_analysis_linear_static_beyond_the_critical_load():
    r"""Beyond the critical load of the linearized problem the load
    correction iterations diverge and a warning is issued, the analysis
    returns the last iterate"""
    pb = linear_follower_problem(factor=1.5)
    an = pb.analysis()
    an.maxNumIter = 20
    increments, cs = an.static(NLgeom=False, silent=True)
    ref = np.linalg.solve(pb.K0 - pb.Q, pb.F0)
    assert not np.allclose(cs[0], ref, rtol=1e-6)
