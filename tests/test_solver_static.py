r"""Tests for :mod:`pyfe3d.solver.static` and :mod:`pyfe3d.solver.sparseutils`

:func:`pyfe3d.solver.solve` removes the null rows and columns of the matrix
and :func:`pyfe3d.solver.linear_static` is based on it. Ported from the
tests of structsolve, with a Ritz plate whose deflection is compared with
the Navier solution.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse import random as sprandom
from scipy.special import roots_legendre

from pyfe3d.solver import solve, linear_static, remove_null_cols, is_symmetric

from ritz import isotropic_plate, vecf, vecfxi, vecfxixi


def test_solve_diagonal_system():
    n = 10
    diag = np.arange(1, n + 1, dtype=float)
    u = solve(csc_matrix(np.diag(diag)), np.ones(n))
    np.testing.assert_allclose(u, 1./diag)


def test_solve_with_null_columns():
    r"""The constrained degrees-of-freedom, null rows and columns, are
    removed and their solution is zero"""
    n = 6
    K_dense = np.zeros((n, n))
    active = np.array([1, 2, 3, 4])
    for i in active:
        K_dense[i, i] = float(i + 1)
    f = np.zeros(n)
    f[active] = 1.0
    u = solve(csc_matrix(K_dense), f)
    assert u[0] == 0.0
    assert u[5] == 0.0
    for i in active:
        np.testing.assert_allclose(u[i], 1.0/(i + 1))


def test_solve_symmetric_system():
    A = np.array([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]])
    f = np.array([1., 2., 3.])
    np.testing.assert_allclose(solve(csc_matrix(A), f),
                               np.linalg.solve(A, f), rtol=1e-10)


def test_solve_unsymmetric_with_null_row_and_column():
    r"""Null rows and columns are removed only together, an equation whose
    row is not null is kept"""
    A = np.array([[4., 1., 0., 0.],
                  [2., 3., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 5.]])
    b = np.array([1., 2., 0., 10.])
    x = solve(csr_matrix(A), b)
    assert np.allclose(x, [0.1, 0.6, 0., 2.])
    # NOTE a row that is not null with a null column: singular, not dropped
    B = A.copy()
    B[2, 0] = 1.
    with pytest.warns(Warning):
        x = solve(csr_matrix(B), b)
    assert not np.all(np.isfinite(x)) or not np.allclose(B @ x, b)


def test_linear_static():
    n = 5
    increments, cs = linear_static(csc_matrix(np.eye(n)*2.0), 4.*np.ones(n))
    assert increments == [1.0]
    assert len(cs) == 1
    np.testing.assert_allclose(cs[0], 2.0*np.ones(n))


def test_linear_static_unsymmetric():
    r"""``linear_static(K0 + KL, F0)``, the linear static solution of a
    follower load affine in ``u``, ``F(u) = F0 - KL u``"""
    K = np.array([[4.5, -1.], [-2., 3.]])
    F0 = np.array([0., 1.])
    increments, cs = linear_static(csr_matrix(K), F0)
    assert np.allclose(K @ cs[0], F0)


def test_static_deflection_plate_clpt():
    r"""Simply supported Ritz plate under a central point load, classical
    plate theory, against the Navier solution"""
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
        P_xi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Px_xi = vecfxi(m1, xi, xit1, xir1, xit2, xir2)
        Pxx_xi = vecfxixi(m1, xi, xit1, xir1, xit2, xir2)
        for eta, weta in zip(pts2, weights2):
            P_eta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Px_eta = vecfxi(m2, eta, etat1, etar1, etat2, etar2)
            Pxx_eta = vecfxixi(m2, eta, etat1, etar1, etat2, etar2)
            Swxx = (Pxx_xi[:, None]*P_eta[None, :]*(2/a)**2).ravel()
            Swyy = (P_xi[:, None]*Pxx_eta[None, :]*(2/b)**2).ravel()
            Swxy = (Px_xi[:, None]*Px_eta[None, :]*(2/a)*(2/b)).ravel()
            e1xx = -Swxx
            e1yy = -Swyy
            e1xy = -2*Swxy
            Mxx = prop.D11*e1xx + prop.D12*e1yy + prop.D16*e1xy
            Myy = prop.D12*e1xx + prop.D22*e1yy + prop.D26*e1xy
            Mxy = prop.D16*e1xx + prop.D26*e1yy + prop.D66*e1xy
            weight = wxi*weta
            K += np.outer(detJ*weight*Mxx, e1xx)
            K += np.outer(detJ*weight*Myy, e1yy)
            K += np.outer(detJ*weight*Mxy, e1xy)
    P_xi = vecf(m1, 0., xit1, xir1, xit2, xir2)
    P_eta = vecf(m2, 0., etat1, etar1, etat2, etar2)
    Sw = (P_xi[:, None]*P_eta[None, :]).ravel()
    Pforce = 1.0
    u = solve(csc_matrix(K), Pforce*Sw)
    w_center = Sw @ u
    D = E*h**3/(12*(1 - nu**2))
    w_navier = 0.0
    for m in range(1, 80, 2):
        for n in range(1, 80, 2):
            w_navier += 1.0/(m**2/a**2 + n**2/b**2)**2
    w_navier *= 4*Pforce/(np.pi**4*D*a*b)
    print('w_center %.6e  w_navier %.6e' % (w_center, w_navier))
    np.testing.assert_allclose(w_center, w_navier, rtol=0.02)


def test_remove_null_cols():
    A = np.zeros((5, 5))
    A[1, 1] = 1.0
    A[3, 3] = 2.0
    A[1, 3] = 0.5
    A[3, 1] = 0.5
    K_reduced, used = remove_null_cols(csr_matrix(A))
    assert K_reduced.shape == (2, 2)
    np.testing.assert_array_equal(used, [1, 3])
    # NOTE several matrices with the null rows and columns of the first
    n = 6
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    active = [0, 2, 4]
    for i in active:
        A[i, i] = float(i + 1)
        B[i, i] = float(i + 1)*10
    A_red, B_red, used = remove_null_cols(csr_matrix(A), csr_matrix(B))
    assert A_red.shape == B_red.shape == (3, 3)
    np.testing.assert_array_equal(used, active)
    np.testing.assert_allclose(B_red.diagonal(), [10., 30., 50.])
    # NOTE nothing removed
    K_red, used = remove_null_cols(csr_matrix(np.eye(4)))
    assert K_red.shape == (4, 4) and len(used) == 4


def test_is_symmetric():
    A = sprandom(30, 30, density=0.2, random_state=1)
    S = A + A.T
    assert is_symmetric(S)
    assert not is_symmetric(A)
    assert is_symmetric(S + 1e-14*A)
    assert is_symmetric(csr_matrix((4, 4)))


if __name__ == '__main__':
    test_static_deflection_plate_clpt()
