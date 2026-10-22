r"""Reuse of evaluations and factorizations by the solvers

- With a ``calc_fext`` that declares ``lbd``, `\partial R/\partial \lambda`
  is evaluated only together with the tangent stiffness matrix, right after
  ``calc_kC`` and ``calc_kG`` at the same configuration, see
  :mod:`pyfe3d.solver.callbacks`; the solutions are those of the tests of
  ``test_solver_load_factor_dependent_load.py``.
- Newton-Raphson factorizes the tangent stiffness matrix once per update.
- The linear static analysis with a configuration-dependent load factorizes
  ``K0`` once.

Ported from the tests of structsolve.

"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyfe3d.solver import Analysis
from pyfe3d.solver import newton_raphson, analysis
from test_solver_load_factor_dependent_load import SpringsProblem


class Recorder(SpringsProblem):
    r"""The springs, recording the sequence of the calls with ``c``"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events = []

    def calc_fext(self, inc=1., silent=True, c=None, lbd=None):
        if c is not None:
            self.events.append(('fext', c.copy(), lbd))
        return super().calc_fext(inc=inc, silent=silent, c=c, lbd=lbd)

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=True):
        if c is not None and NLgeom:
            self.events.append(('kC', c.copy(), inc))
        return super().calc_kC(c=c, NLgeom=NLgeom, inc=inc, silent=silent)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        if c is not None and NLgeom:
            self.events.append(('kG', c.copy(), None))
        return super().calc_kG(c=c, NLgeom=NLgeom, silent=silent)


def check_events(pb, origin_calls=1):
    r"""Every `\partial R/\partial \lambda` but those at the origin follows
    the tangent stiffness matrix at the same configuration and load factor,
    and there is at most one per tangent"""
    ev = pb.events
    fext = [i for i, e in enumerate(ev) if e[0] == 'fext']
    tangents = [i for i, e in enumerate(ev) if e[0] == 'kC']
    assert len(fext) <= len(tangents) + origin_calls
    for i in fext:
        _, c, lbd = ev[i]
        if i < 2 or ev[i - 1][0] != 'kG':
            assert not np.any(c) and lbd == 0., ev[i]
            continue
        assert np.array_equal(ev[i - 1][1], c)
        assert ev[i - 2][0] == 'kC' and np.array_equal(ev[i - 2][1], c)
        assert ev[i - 2][2] == lbd
    return len(fext), len(tangents)


def count(monkeypatch, module, name):
    calls = []
    func = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return func(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
@pytest.mark.parametrize('modified_NR', [False, True])
def test_load_vector_evaluated_with_the_tangent(method, modified_NR):
    pb = Recorder(d0=-0.3, d1=0.6, F=40.)
    an = pb.analysis(NL_method=method, initialInc=0.2, relTOL=1e-11,
                     modified_NR=modified_NR)
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    for lbd, c_r in zip(an.increments, an.cs):
        ref = pb.solve_reference(lbd)
        assert np.allclose(c_r, ref, rtol=1e-8, atol=1e-12), (lbd, c_r, ref)
    n_fext, n_tangents = check_events(pb)
    # dR/dlbd at the origin, then at most one per tangent
    assert n_fext >= 2
    if method == 'NR':
        # one per tangent, none at the converged iterates
        assert n_fext == n_tangents + 1


def test_newton_raphson_one_factorization_per_tangent(monkeypatch):
    factorizations = count(monkeypatch, newton_raphson, '_factorize')
    for modified_NR in (False, True):
        del factorizations[:]
        pb = Recorder(k=(10., 15., 20.), a=(200., 400., 600.), d1=1.)
        an = pb.analysis(NL_method='NR', initialInc=0.5, relTOL=1e-12,
                         modified_NR=modified_NR, compute_every_n=3,
                         kT_initial_state=True)
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        n_tangents = sum(1 for e in pb.events if e[0] == 'kC')
        assert len(factorizations) == n_tangents
        corrections = sum(1 for e in pb.fint_calls) - len(an.increments)
        if modified_NR:
            # the factorization is reused between the updates
            assert len(factorizations) < corrections


def test_newton_raphson_singular_tangent_cuts_the_step():
    """A singular tangent stops the iterations of the step with a warning,
    the step being repeated with a smaller increment"""
    pb = SpringsProblem(F=10., d1=0.)
    calls = []

    def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
        if c is not None and NLgeom and not calls:
            calls.append(1)
            return csr_matrix(np.array([[1., 1.], [1., 1.]]))
        return pb.calc_kC(c=c, NLgeom=NLgeom, inc=inc, silent=silent)

    an = pb.analysis(initialInc=0.5, relTOL=1e-10)
    an.calc_kC = calc_kC
    an.static(NLgeom=True, silent=True)
    assert calls and np.isclose(an.increments[-1], 1.)


def test_linear_static_follower_factorizes_K0_once(monkeypatch):
    factorizations = count(monkeypatch, analysis, '_factorize')
    K0 = csr_matrix(np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 3.]]))
    F0 = np.array([1., 0., 2.])
    B = 0.3*np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

    def calc_fext(inc=1., silent=True, c=None):
        return inc*(F0 if c is None else F0 + B @ c)

    an = Analysis(calc_fext, None, lambda c=None, NLgeom=False, silent=True:
                  K0, None)
    an.relTOL = 1e-12
    _, cs = an.static(NLgeom=False, silent=True)
    exact = np.linalg.solve(K0.toarray() - B, F0)
    assert np.allclose(cs[-1], exact, rtol=1e-10)
    assert len(factorizations) == 1
