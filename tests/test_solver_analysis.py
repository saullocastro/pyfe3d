r"""Tests for :class:`pyfe3d.solver.Analysis`: linear static analysis,
Newton-Raphson on a linear system, invalid parameters and the convergence
criteria, relative and Crisfield's
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.sparse import csc_matrix, csr_matrix

from pyfe3d.solver import Analysis, crisfield_weights, crisfield_zeta
from pyfe3d.solver.convergence import check_convergence


def test_analysis_linear_static():
    n = 10
    K = csc_matrix(np.eye(n)*3.0)
    f = np.ones(n)*6.0
    an = Analysis(calc_fext=lambda inc=1., silent=True: f,
                  calc_kC=lambda c=None, NLgeom=False, silent=True: K)
    increments, cs = an.static(NLgeom=False, silent=True)
    assert increments == [1.0]
    np.testing.assert_allclose(cs[0], 2.0*np.ones(n))
    assert an.last_analysis == 'static'


def test_analysis_newton_raphson_linear_system():
    r"""Newton-Raphson on a linear system converges in one iteration"""
    n = 5
    K = csc_matrix(np.eye(n)*2.0)
    f = np.ones(n)*4.0
    an = Analysis(calc_fext=lambda inc=1., silent=True: f,
                  calc_kC=lambda c=None, NLgeom=False, silent=True: K,
                  calc_kG=lambda c=None, NLgeom=False, silent=True:
                  csc_matrix((n, n)),
                  calc_fint=lambda c=None, silent=True: K @ c)
    an.initialInc = 1.0
    increments, cs = an.static(NLgeom=True, silent=True)
    assert increments == [1.0]
    np.testing.assert_allclose(cs[-1], 2.0*np.ones(n), rtol=1e-12)


@pytest.mark.parametrize('convergence', ['crisfield', 'relative'])
@pytest.mark.parametrize('NLgeom, NL_method, modified_NR', [
    (False, 'NR', False),
    (True, 'NR', False),
    (True, 'NR', True),
    (True, 'arc_length_riks', False),
    (True, 'arc_length_crisfield', True)])
def test_null_row_raises(NLgeom, NL_method, modified_NR, convergence):
    r"""A null row and column of the linear stiffness matrix, a
    degree-of-freedom without stiffness, raises an error naming it in the
    linear and in every non-linear analysis, instead of being removed"""
    n = 4
    K = np.diag([2., 3., 0., 4.])
    an = Analysis(calc_fext=lambda inc=1., silent=True: inc*np.ones(n),
                  calc_kC=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix(K),
                  calc_kG=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix((n, n)),
                  calc_fint=lambda c=None, silent=True: K @ c)
    an.NL_method = NL_method
    an.modified_NR = modified_NR
    an.convergence = convergence
    with pytest.raises(LinAlgError, match=r'null row or column at 1 '
                       r'degree\(s\)-of-freedom, of indices 2\.'):
        an.static(NLgeom=NLgeom, silent=True)


def test_invalid_parameters():
    n = 3
    K = csc_matrix(np.eye(n))
    an = Analysis(calc_fext=lambda inc=1., silent=True: np.ones(n),
                  calc_kC=lambda c=None, NLgeom=False, silent=True: K,
                  calc_kG=lambda c=None, NLgeom=False, silent=True:
                  csc_matrix((n, n)),
                  calc_fint=lambda c=None, silent=True: K @ c)
    an.NL_method = 'arc_length'
    with pytest.raises(ValueError, match='invalid NL_method'):
        an.static(NLgeom=True, silent=True)
    an.NL_method = 'NR'
    an.convergence = 'euclidean'
    with pytest.raises(ValueError, match='invalid convergence'):
        an.static(NLgeom=True, silent=True)
    an.convergence = 'crisfield'
    an.relTOL = None
    with pytest.raises(ValueError, match='relTOL or absTOL'):
        an.static(NLgeom=True, silent=True)
    an.relTOL = 1.e-6
    an.uk = np.ones(n)
    with pytest.raises(ValueError, match='uk requires bu'):
        an.static(NLgeom=True, silent=True)


def test_crisfield_weights_and_zeta():
    r"""`w_j = 1/|K_{jj}|`; `\zeta = \|R\|_{\boldsymbol{d}}/
    \max(\|F_{ext}\|_{\boldsymbol{d}}, \|F_{int}\|_{\boldsymbol{d}})`; a
    null row and column raises the error of the solvers, a null diagonal
    entry in a row that is not null a ValueError"""
    K = csr_matrix(np.array([[4., -1.], [-1., -2.]]))
    w = crisfield_weights(K)
    np.testing.assert_allclose(w, [0.25, 0.5])
    fext = np.array([2., 0.])
    fint = np.array([1., 1.])
    R = fext - fint
    zeta = crisfield_zeta(R, fext, fint, w)
    ref = max(np.sqrt(4*0.25), np.sqrt(0.25 + 0.5))
    np.testing.assert_allclose(zeta, np.sqrt(0.25 + 0.5)/ref)
    # NOTE an additional reference, e.g. lbd*dR/dlbd
    assert crisfield_zeta(R, fext, fint, w, q=np.array([10., 0.])) < zeta
    assert crisfield_zeta(np.zeros(2), np.zeros(2), np.zeros(2), w) == 0.
    assert crisfield_zeta(R, np.zeros(2), np.zeros(2), w) == np.inf
    with pytest.raises(LinAlgError, match=r'K0 is singular: null row or '
                       r'column at 1 degree\(s\)-of-freedom, of indices 2\.'):
        crisfield_weights(csr_matrix(np.array([[4., -1., 0.], [-1., -2., 0.],
                                               [0., 0., 0.]])))
    with pytest.raises(ValueError, match='null diagonal'):
        crisfield_weights(csr_matrix(np.array([[0., 1.], [1., 2.]])))


def test_check_convergence_criteria():
    r"""Crisfield's `\zeta \le` ``relTOL`` and the relative criterion of the
    Euclidean norm, with ``absTOL`` an alternative of both"""
    class An(object):
        convergence = 'crisfield'
        relTOL = 1.e-3
        absTOL = None
    an = An()
    w = np.array([1., 1.e-6])
    fext = np.array([1., 0.])
    fint = np.array([1., 0.5])
    R = fext - fint
    # NOTE the second component is a moment of a stiff rotation
    conv, Rnorm, zeta = check_convergence(an, R, fint, fext, w=w)
    assert conv and zeta < 1.e-3
    an.convergence = 'relative'
    conv, Rnorm, zeta = check_convergence(an, R, fint, fext)
    assert not conv and np.isclose(zeta, 0.5/np.linalg.norm(fint))
    an.absTOL = 0.6
    assert check_convergence(an, R, fint, fext)[0]
    an.convergence = 'crisfield'
    with pytest.raises(ValueError, match='weights'):
        check_convergence(an, R, fint, fext)



def test_free_dofs_invalid_bu_and_uk():
    from pyfe3d.solver import FreeDofs

    def f(*args, **kwargs):
        return None

    for bu in [np.array([1, 0, 1]), np.ones((2, 2), dtype=bool)]:
        with pytest.raises(ValueError, match='bu must be a 1-D boolean'):
            FreeDofs(f, f, f, f, bu=bu)
    with pytest.raises(ValueError, match='uk must have the size of bu'):
        FreeDofs(f, f, f, f, bu=np.array([True, False]), uk=np.zeros(3))


def springs_chain(k1=100., k2=300.):
    """Stiffness matrix of 2 springs in series, DOFs 0-1 and 1-2"""
    return np.array([[k1, -k1, 0.], [-k1, k1 + k2, -k2], [0., -k2, k2]])


def test_linear_static_prescribed_displacement():
    """Linear static analysis with ``bu`` and ``uk``: the prescribed
    displacement of the last DOF stretches the springs, the first DOF being
    fixed"""
    K = springs_chain()
    an = Analysis(calc_fext=lambda inc=1., silent=True: np.zeros(3),
                  calc_kC=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix(K),
                  bu=np.array([False, True, False]),
                  uk=np.array([0., 0., 0.3]))
    increments, cs = an.static(NLgeom=False, silent=True)
    assert increments == [1.]
    np.testing.assert_allclose(cs[0], [0., 300.*0.3/400., 0.3], rtol=1e-12)


def test_free_dofs_configuration_and_load_factor_dependent_load():
    """A follower load declaring ``c`` and ``lbd``, through ``bu``: the
    callables of the full model receive the full configuration, and the
    solution is the one of the reduced model"""
    K = springs_chain()
    P = 20.
    seen = []

    def fext_full(inc=1., silent=True, c=None, lbd=None):
        seen.append(None if c is None else c.size)
        u2 = 0. if c is None else c[2]
        return inc*np.array([0., 0., P*(1. + 2.*u2)])

    def fext_reduced(inc=1., silent=True, c=None, lbd=None):
        u2 = 0. if c is None else c[1]
        return inc*np.array([0., P*(1. + 2.*u2)])

    def model(fext, Km, **kwargs):
        an = Analysis(calc_fext=fext,
                      calc_fint=lambda c, silent=True: Km @ c,
                      calc_kC=lambda c=None, NLgeom=False, silent=True:
                      csr_matrix(Km),
                      calc_kG=lambda c=None, NLgeom=False, silent=True:
                      csr_matrix(Km.shape), **kwargs)
        an.initialInc = 0.5
        an.relTOL = 1e-12
        an.static(NLgeom=True, silent=True)
        return an

    full = model(fext_full, K, bu=np.array([False, True, True]))
    reduced = model(fext_reduced, K[1:, 1:])
    assert full.increments == reduced.increments == [0.5, 1.]
    for c, cr in zip(full.cs, reduced.cs):
        assert c[0] == 0.
        np.testing.assert_allclose(c[1:], cr, rtol=1e-12)
    # NOTE the residual uses the reference load, calc_fext(c=c, lbd=lbd)
    #      being dR/dlbd, see pyfe3d.solver.callbacks
    np.testing.assert_allclose(K[1:, :] @ full.cs[-1], [0., P],
                               rtol=1e-10, atol=1e-10)
    assert 3 in seen


def test_linear_static_follower_not_finite(capsys):
    """Load correction iterations of the linear static analysis: a load that
    is not finite in the deformed configuration stops them with a warning,
    returning the solution of the reference load"""
    K = springs_chain()[1:, 1:]
    F0 = np.array([0., 10.])

    def calc_fext(inc=1., silent=True, c=None):
        return inc*(F0 if c is None else np.full(2, np.nan))

    an = Analysis(calc_fext=calc_fext,
                  calc_kC=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix(K))
    increments, cs = an.static(NLgeom=False, silent=False)
    out = capsys.readouterr().out
    np.testing.assert_allclose(cs[0], np.linalg.solve(K, F0), rtol=1e-12)
    assert 'WARNING: The load correction iterations did not converge' in out


def test_linear_static_follower_factorization_failure(monkeypatch, capsys):
    """A failed factorization of the linear stiffness matrix stops the load
    correction iterations with a warning, returning the last iterate. The
    factorization is the one of the matrix that has just solved the linear
    problem, therefore the failure is forced here"""
    from pyfe3d.solver import analysis

    def fails(k0):
        raise LinAlgError('kT is exactly singular')

    monkeypatch.setattr(analysis, '_factorize', fails)
    K = springs_chain()[1:, 1:]
    F0 = np.array([0., 10.])

    def calc_fext(inc=1., silent=True, c=None):
        u2 = 0. if c is None else c[1]
        return inc*F0*(1. + u2)

    an = Analysis(calc_fext=calc_fext,
                  calc_kC=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix(K))
    increments, cs = an.static(NLgeom=False, silent=False)
    out = capsys.readouterr().out
    np.testing.assert_allclose(cs[0], np.linalg.solve(K, F0), rtol=1e-12)
    assert 'WARNING: The load correction iterations did not converge' in out
