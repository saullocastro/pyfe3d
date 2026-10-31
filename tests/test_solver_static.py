r"""Tests for :mod:`pyfe3d.solver.static` and :mod:`pyfe3d.solver.sparseutils`

:func:`pyfe3d.solver.solve` raises an error for a singular matrix, naming
the degrees-of-freedom of its null rows and columns, see
:func:`pyfe3d.solver.check_null_rows_cols`, and
:func:`pyfe3d.solver.linear_static` is based on it, tested with a Ritz plate
whose deflection is compared with the Navier solution.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse import random as sprandom
from scipy.special import roots_legendre

from pyfe3d.solver import (solve, linear_static, check_null_rows_cols,
                           is_symmetric)
from pyfe3d.solver.static import _factorize

from ritz import isotropic_plate, vecf, vecf_xi, vecf_xixi


def test_solve_diagonal_system():
    n = 10
    diag = np.arange(1, n + 1, dtype=float)
    u = solve(csc_matrix(np.diag(diag)), np.ones(n))
    np.testing.assert_allclose(u, 1./diag)


def test_solve_with_null_columns():
    r"""Constrained degrees-of-freedom left as null rows and columns raise
    an error naming them; partitioned out, the system is solved"""
    n = 6
    K_dense = np.zeros((n, n))
    active = np.array([1, 2, 3, 4])
    for i in active:
        K_dense[i, i] = float(i + 1)
    f = np.zeros(n)
    f[active] = 1.0
    with pytest.raises(LinAlgError, match=r'A is singular: null row or '
                       r'column at 2 degree\(s\)-of-freedom, of indices 0, 5\.'):
        solve(csc_matrix(K_dense), f)
    bu = np.zeros(n, dtype=bool)
    bu[active] = True
    u = solve(csc_matrix(K_dense)[bu, :][:, bu], f[bu])
    np.testing.assert_allclose(u, 1.0/(active + 1))


def test_solve_symmetric_system():
    A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    f = np.array([1., 2., 3.])
    np.testing.assert_allclose(solve(csc_matrix(A), f),
                               np.linalg.solve(A, f), rtol=1e-10)


def test_solve_unsymmetric_with_null_row_or_column():
    r"""A null row alone or a null column alone raises the error, as an
    exactly singular matrix without null rows or columns, instead of
    returning NaN"""
    A = np.array([[4., 1., 0., 0.],
                  [2., 3., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 5.]])
    b = np.array([1., 2., 0., 10.])
    # NOTE a null row with a column that is not null, and the opposite
    B = A.copy()
    B[0, 2] = 1.
    with pytest.raises(LinAlgError, match=r'at 1 degree.*of indices 2\.'):
        solve(csr_matrix(B), b)
    with pytest.raises(LinAlgError, match=r'at 1 degree.*of indices 2\.'):
        solve(csr_matrix(B.T), b)
    # NOTE exactly singular without null rows or columns
    C = np.array([[1., 2.], [2., 4.]])
    with pytest.raises(LinAlgError, match='A is exactly singular'):
        solve(csr_matrix(C), np.ones(2))
    keep = [0, 1, 3]
    x = solve(csr_matrix(A[keep, :][:, keep]), b[keep])
    assert np.allclose(x, [0.1, 0.6, 2.])


def test_factorize():
    r"""The factorization reused by the non-linear solvers raises the same
    error as :func:`solve` for a null row or column and for an exactly
    singular matrix"""
    A = np.array([[4., 1., 0.], [2., 3., 0.], [0., 0., 0.]])
    with pytest.raises(LinAlgError, match=r'kT is singular: null row or '
                       r'column at 1 degree\(s\)-of-freedom, of indices 2\.'):
        _factorize(csr_matrix(A))
    with pytest.raises(LinAlgError, match='kT is exactly singular'):
        _factorize(csr_matrix(np.array([[1., 2.], [2., 4.]])))
    solve_kT = _factorize(csr_matrix(A[:2, :2]))
    np.testing.assert_allclose(solve_kT(np.array([1., 2.])), [0.1, 0.6])


def test_linear_static():
    n = 5
    increments, cs = linear_static(csc_matrix(np.eye(n)*2.0), 4.*np.ones(n))
    assert increments == [1.0]
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0], 2.0*np.ones(n))


def test_linear_static_unsymmetric():
    r"""``linear_static(K0 + KF, F0)``, the linear static solution of a
    follower load affine in ``u``, ``F(u) = F0 - KF u``"""
    K = np.array([[4.5, -1.], [-2., 3.]])
    F0 = np.array([0., 1.])
    increments, cs = linear_static(csr_matrix(K), F0)
    assert np.allclose(K @ cs[0], F0)


def test_static_deflection_plate_clpt():
    r"""Simply supported Ritz plate under a central point load, classical
    plate theory, against the Navier solution

    The reference is the Navier solution of Timoshenko, S., and
    Woinowsky-Krieger, S., Theory of Plates and Shells, 2nd ed.,
    McGraw-Hill, 1959, Art. 29, p. 111, Eq. (c), the deflection at the
    centre under a load at the centre,
    ``w_max = 4*P/(pi**4*a*b*D)*sum(1/((m/a)**2 + (n/b)**2)**2)`` over
    ``m, n = 1, 3, 5, ...``, here summed up to 1999, which changes it by
    less than 1e-5. For ``b/a = 7/3`` it gives ``w_max = alpha*P*a**2/D``
    with ``alpha = 0.016785``, between the values 0.01651 and 0.01690 of
    their Table 23, p. 143, for ``b/a = 2`` and 3.

    The Ritz deflection converges slowly to it, the curvature being singular
    under the point load: with 20x20 terms it is 1.09 % below the reference,
    with 30x30 terms 0.44 %, with 40x40 terms 0.24 %; the two finer
    approximations take 23 s and 160 s to assemble.
    """
    m1 = m2 = 20
    N = m1*m2
    pts1, weights1 = roots_legendre(2*m1 - 1)
    pts2, weights2 = roots_legendre(2*m2 - 1)
    E = 200.e9
    nu = 0.3
    a = 3
    b = 7
    h = 0.005
    # NOTE simply supported
    xit1, xir1, xit2, xir2 = 0, 1, 0, 1
    etat1, etar1, etat2, etar2 = 0, 1, 0, 1
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    K = np.zeros((N, N))
    detJ = a*b/4
    for xi, wxi in zip(pts1, weights1):
        Pxi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Pxi_xi = vecf_xi(m1, xi, xit1, xir1, xit2, xir2)
        Pxi_xixi = vecf_xixi(m1, xi, xit1, xir1, xit2, xir2)
        for eta, weta in zip(pts2, weights2):
            Peta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Peta_eta = vecf_xi(m2, eta, etat1, etar1, etat2, etar2)
            Peta_etaeta = vecf_xixi(m2, eta, etat1, etar1, etat2, etar2)
            Sw_xx = (Pxi_xixi[:, None]*Peta[None, :]*(2/a)**2).ravel()
            Sw_yy = (Pxi[:, None]*Peta_etaeta[None, :]*(2/b)**2).ravel()
            Sw_xy = (Pxi_xi[:, None]*Peta_eta[None, :]*(2/a)*(2/b)).ravel()
            e1xx = -Sw_xx
            e1yy = -Sw_yy
            e1xy = -2*Sw_xy
            Mxx = prop.D11*e1xx + prop.D12*e1yy + prop.D16*e1xy
            Myy = prop.D12*e1xx + prop.D22*e1yy + prop.D26*e1xy
            Mxy = prop.D16*e1xx + prop.D26*e1yy + prop.D66*e1xy
            weight = wxi*weta
            K += np.outer(detJ*weight*Mxx, e1xx)
            K += np.outer(detJ*weight*Myy, e1yy)
            K += np.outer(detJ*weight*Mxy, e1xy)
    Pxi = vecf(m1, 0., xit1, xir1, xit2, xir2)
    Peta = vecf(m2, 0., etat1, etar1, etat2, etar2)
    Sw = (Pxi[:, None]*Peta[None, :]).ravel()
    Pforce = 1.0
    # NOTE the Ritz model applies the boundary conditions by zeroing
    #      approximation functions, whose null rows and columns are
    #      partitioned out before calling the solver
    bu = np.any(K != 0, axis=0)
    assert np.count_nonzero(bu) < N
    u = solve(csc_matrix(K[bu, :][:, bu]), Pforce*Sw[bu])
    w_center = Sw[bu] @ u
    # NOTE Timoshenko and Woinowsky-Krieger (1959), Eq. (c) of Art. 29
    D = E*h**3/(12*(1 - nu**2))
    m = np.arange(1, 2000, 2)[:, None]
    n = np.arange(1, 2000, 2)[None, :]
    w_navier = 4*Pforce/(np.pi**4*a*b*D)*np.sum(
        1.0/((m/a)**2 + (n/b)**2)**2)
    print('w_center %.6e  w_navier %.6e' % (w_center, w_navier))
    np.testing.assert_allclose(w_center, w_navier, rtol=0.02)


def test_check_null_rows_cols():
    r"""The error names the number of degrees-of-freedom of the null rows
    and columns and the first ten of their indices, explicitly stored zeros
    counting as null"""
    check_null_rows_cols(csr_matrix(np.eye(4)))
    check_null_rows_cols(np.eye(4))
    A = np.zeros((5, 5))
    A[1, 1] = 1.0
    A[3, 3] = 2.0
    A[1, 3] = 0.5
    A[3, 1] = 0.5
    with pytest.raises(LinAlgError, match=r'^KC0 is singular: null row or '
                       r'column at 3 degree\(s\)-of-freedom, of indices 0, 2, '
                       r'4\. .*Partition out the constrained'):
        check_null_rows_cols(csr_matrix(A), 'KC0')
    # NOTE explicitly stored zeros
    E = coo_matrix((np.array([1., 0., 2.]), (np.array([0, 1, 2]),
                    np.array([0, 1, 2]))), shape=(3, 3)).tocsr()
    assert E.nnz == 3
    with pytest.raises(LinAlgError, match=r'of indices 1\.'):
        check_null_rows_cols(E)
    # NOTE only the first ten indices are listed
    D = np.diag(np.r_[np.zeros(12), np.ones(3)])
    with pytest.raises(LinAlgError, match=r'at 12 degree\(s\)-of-freedom, '
                       r'of indices 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, \.\.\.\.'):
        check_null_rows_cols(csr_matrix(D))


def test_is_symmetric():
    A = sprandom(30, 30, density=0.2, random_state=1)
    S = A + A.T
    assert is_symmetric(S)
    assert not is_symmetric(A)
    assert is_symmetric(S + 1e-14*A)
    assert is_symmetric(csr_matrix((4, 4)))


if __name__ == '__main__':
    test_static_deflection_plate_clpt()


def test_solve_complex_and_options():
    r"""A complex right-hand side with a real or a complex matrix, and the
    keyword arguments passed to :func:`scipy.sparse.linalg.splu`"""
    A = np.array([[4., 1., 0.], [2., 3., 1.], [0., 1., 5.]])
    b = np.array([1. + 2j, -1j, 3.])
    np.testing.assert_allclose(solve(csr_matrix(A), b),
                               np.linalg.solve(A, b), rtol=1e-12)
    Ac = A + 1j*np.eye(3)
    np.testing.assert_allclose(solve(csr_matrix(Ac), b),
                               np.linalg.solve(Ac, b), rtol=1e-12)
    np.testing.assert_allclose(solve(csr_matrix(Ac), b.real),
                               np.linalg.solve(Ac, b.real), rtol=1e-12)
    np.testing.assert_allclose(solve(A, b.real, permc_spec='NATURAL'),
                               np.linalg.solve(A, b.real), rtol=1e-12)


def test_linear_static_not_positive_definite():
    r"""A linear static analysis requires a symmetric ``K`` to be positive
    definite: the chain of unit springs without support, singular with the
    rigid translation, whose last pivot is exactly zero, and an indefinite
    ``K`` raise a ``LinAlgError`` naming the analysis and the likely causes,
    with :func:`linear_static` and :meth:`pyfe3d.solver.Analysis.static`;
    :func:`solve` itself solves symmetric indefinite systems, and a positive
    definite one is solved with the factorization of the check, also with a
    complex right-hand side"""
    from pyfe3d.solver import Analysis
    n = 6
    chain = csr_matrix(np.diag(np.r_[1., 2*np.ones(n - 2), 1.])
                       - np.eye(n, k=1) - np.eye(n, k=-1))
    f = np.ones(n)
    with pytest.raises(LinAlgError, match=r'^Linear static analysis: the '
                       r'stiffness matrix K is not positive definite, its '
                       r'factorization met a zero pivot, i.e. K is singular '
                       r'or indefinite. Likely causes: insufficient boundary '
                       r'conditions.*mechanism.*re-entrant.*preload'):
        linear_static(chain, f)
    K = csr_matrix(np.diag([1., -2., 3., 4.]))
    with pytest.raises(LinAlgError, match=r'has 1 negative pivot\(s\), i.e. '
                       r'K has 1 eigenvalue\(s\) that are negative'):
        linear_static(K, np.ones(4))
    an = Analysis(lambda inc=1., silent=True: np.ones(4), None,
                  lambda c=None, NLgeom=False, silent=True: K, None)
    with pytest.raises(LinAlgError, match='^Linear static analysis: the '
                       'stiffness matrix K is not positive definite'):
        an.static(silent=True)
    np.testing.assert_allclose(solve(K, np.ones(4)), [1., -0.5, 1/3, 0.25])
    Kpd = csr_matrix(np.array([[4., -1., 0.], [-1., 4., -1.],
                               [0., -1., 3.]]))
    b = np.array([1. + 2j, -1j, 3.])
    np.testing.assert_allclose(linear_static(Kpd, b)[1][0],
                               np.linalg.solve(Kpd.toarray(), b),
                               rtol=1e-12)
