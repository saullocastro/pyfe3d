r"""Tests for :class:`pyfe3d.solver.Analysis`: linear static analysis,
Newton-Raphson on a linear system, invalid parameters and the convergence
criteria, ported from the tests of structsolve and completed for
Crisfield's criterion
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
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
    r"""`w_j = 1/|K_{jj}|` at the free degrees-of-freedom, zero at the null
    rows and columns; `\zeta = \|R\|_D/\max(\|F_{ext}\|_D,
    \|F_{int}\|_D)`"""
    K = csr_matrix(np.array([[4., -1., 0.], [-1., -2., 0.], [0., 0., 0.]]))
    w = crisfield_weights(K)
    np.testing.assert_allclose(w, [0.25, 0.5, 0.])
    fext = np.array([2., 0., 7.])
    fint = np.array([1., 1., 0.])
    R = fext - fint
    zeta = crisfield_zeta(R, fext, fint, w)
    ref = max(np.sqrt(4*0.25), np.sqrt(0.25 + 0.5))
    np.testing.assert_allclose(zeta, np.sqrt(0.25 + 0.5)/ref)
    # NOTE an additional reference, e.g. lbd*dR/dlbd
    assert crisfield_zeta(R, fext, fint, w, q=np.array([10., 0., 0.])) < zeta
    assert crisfield_zeta(np.zeros(3), np.zeros(3), np.zeros(3), w) == 0.
    assert crisfield_zeta(R, np.zeros(3), np.zeros(3), w) == np.inf
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
