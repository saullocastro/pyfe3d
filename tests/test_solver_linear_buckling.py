r"""Tests for :func:`pyfe3d.solver.linear_buckling`: order of the load
multipliers, verification, dense fallback and unsymmetric matrices

With analytic matrices and the Quad4 plate of ``test_solver.py`` under a
compressive and a tensile reference load.
"""
import os
import subprocess
import sys
sys.path.append('..')

import textwrap
import warnings

import numpy as np
import pytest
from scipy.linalg import eig, block_diag
from scipy.sparse import (csc_matrix, csr_matrix, diags, identity, bmat, tril,
                          random as sprandom)
from scipy.sparse.linalg import splu
from scipy.special import roots_legendre

from pyfe3d.solver import (linear_buckling, check_eigenpairs,
                           estimate_cayley_sigma, DenseFallbackWarning,
                           is_symmetric)
from pyfe3d.solver import eigen

from ritz import isotropic_plate, vecf, vecfxi
from test_solver import plate_matrices


def relative_residuals(K, KG, eigvals, eigvecs):
    Ku = K @ eigvecs
    KGu = KG @ eigvecs
    return (np.linalg.norm(Ku + KGu*eigvals, axis=0)
            / (np.linalg.norm(Ku, axis=0)
               + np.abs(eigvals)*np.linalg.norm(KGu, axis=0)))


def mixed_sign_problem(lambdas, num_null=0):
    r"""Diagonal ``K`` and ``KG`` with the load multipliers ``lambdas``,
    coupled by a sparse congruence, and ``num_null`` degrees-of-freedom where
    ``KG`` is null, e.g. in-plane ones"""
    lambdas = np.concatenate([lambdas, np.full(num_null, np.inf)])
    n = lambdas.size
    k = np.linspace(1., 3., n)
    T = (identity(n) + 0.1*tril(sprandom(n, n, density=2/n, random_state=1),
                                -1)).tocsc()
    K = (T.T @ diags(k) @ T).tocsr()
    KG = (T.T @ diags(-k/lambdas) @ T).tocsr()
    return K, KG


def spd(n, rng):
    X = rng.standard_normal((n, n))
    return X @ X.T + n*np.eye(n)


def test_simple_eigenvalues():
    r"""``K = diag(2, 4, ...)`` and ``KG = I``: all the load multipliers are
    negative, ``-K_ii``, and the ones closest to zero are returned"""
    n = 10
    k_diag = np.arange(2, 2*n + 2, 2, dtype=float)
    eigvals, eigvecs = linear_buckling(csc_matrix(np.diag(k_diag)),
                                       csc_matrix(np.eye(n)), num_eigvalues=5)
    np.testing.assert_allclose(eigvals, -k_diag[:5], rtol=1e-10)
    assert eigvecs.shape == (n, 5)
    eig_dense, _ = linear_buckling(csc_matrix(np.diag(k_diag)),
                                   csc_matrix(np.eye(n)), sparse_solver=False)
    np.testing.assert_allclose(eig_dense, -k_diag, rtol=1e-10)


def test_dense_solver():
    n = 10
    eigvals, eigvecs = linear_buckling(csc_matrix(2*np.eye(n)),
                                       csc_matrix(np.eye(n)),
                                       sparse_solver=False)
    np.testing.assert_allclose(eigvals, -2.0, rtol=1e-10)
    assert eigvecs.shape == (n, n)


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_shift_mixed_spectrum(max_dense_size):
    r"""Sparse and dense solvers agree when the load multipliers have mixed
    signs; the shift bounds the largest `|\mu|`"""
    n = 80
    k = np.linspace(1e6, 5e6, n)
    lambdas = np.concatenate([[1.0, 1.0, 1.1, 1.2],
                              np.linspace(1.5, 4., 36),
                              -np.linspace(1.5, 4., 40)])
    mu = -1./lambdas
    K = diags(k).tocsc()
    KG = diags(mu*k).tocsc()
    assert estimate_cayley_sigma(K, KG) >= np.abs(mu).max()
    eig_sparse, _ = linear_buckling(K, KG, num_eigvalues=10,
                                    max_dense_size=max_dense_size)
    eig_dense, _ = linear_buckling(K, KG, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse[:4], [1.0, 1.0, 1.1, 1.2],
                               rtol=1e-8)
    np.testing.assert_allclose(eig_sparse[:4], eig_dense[:4], rtol=1e-8)


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_sparse_sorted_as_dense(max_dense_size):
    r"""The positive load multipliers first, in increasing order, followed
    by the negative ones closest to zero, as the dense solver"""
    n = 40
    k = np.linspace(1., 3., n)
    rng = np.random.RandomState(1)
    lambdas = rng.permutation(np.concatenate([np.linspace(1., 20., 5),
                                              -np.linspace(1., 20., n - 5)]))
    K = diags(k).tocsr()
    KG = diags(-k/lambdas).tocsr()
    eig_sparse, eigvecs = linear_buckling(K, KG, num_eigvalues=10,
                                          max_dense_size=max_dense_size)
    eig_dense, _ = linear_buckling(K, KG, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse, eig_dense[:10], rtol=1e-8)
    np.testing.assert_allclose(eig_sparse[:5], np.linspace(1., 20., 5),
                               rtol=1e-8)
    assert np.all(eig_sparse[5:] < 0)
    assert np.all(np.diff(np.abs(eig_sparse[5:])) > 0)
    assert np.all(relative_residuals(K, KG, eig_sparse, eigvecs) < 1e-8)


@pytest.mark.parametrize('max_dense_size', [0, 2000])
def test_negative_multipliers_closest_to_zero(max_dense_size):
    r"""After the positive load multipliers, the negative ones closest to
    zero, i.e. the critical load multipliers of the reversed load, for the
    sparse and the dense solvers, symmetric or not"""
    positive = np.linspace(1., 20., 5)
    negative = -np.linspace(2., 40., 35)
    K, KG = mixed_sign_problem(np.concatenate([positive, negative]),
                               num_null=10)
    eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=10,
                                       max_dense_size=max_dense_size)
    np.testing.assert_allclose(eigvals[:5], positive, rtol=1e-8)
    np.testing.assert_allclose(eigvals[5:], negative[:5], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)
    eig_dense, _ = linear_buckling(K, KG, sparse_solver=False)
    np.testing.assert_allclose(eig_dense[:10], eigvals, rtol=1e-8)
    # NOTE the infinite load multipliers of the null rows of KG come last
    assert np.all(np.abs(eig_dense[-10:]) > 1e8)
    eig_gen, _ = linear_buckling(K, KG, num_eigvalues=10, symmetric=False,
                                 max_dense_size=max_dense_size)
    pos, neg = eig_gen[eig_gen > 0], eig_gen[eig_gen < 0]
    np.testing.assert_allclose(pos, positive[:pos.size], rtol=1e-8)
    np.testing.assert_allclose(neg, negative[:neg.size], rtol=1e-8)
    np.testing.assert_array_equal(eig_gen, np.concatenate((pos, neg)))


def test_tensile_reference_load():
    r"""No positive load multiplier: the sparse solver returns the negative
    load multipliers closest to zero, without the dense fallback"""
    negative = -np.linspace(2., 40., 40)
    K, KG = mixed_sign_problem(negative, num_null=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=10,
                                           max_dense_size=0)
    np.testing.assert_allclose(eigvals, negative[:10], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)
    eigvals_rev, _ = linear_buckling(K, -KG, num_eigvalues=10,
                                     max_dense_size=0)
    np.testing.assert_allclose(eigvals_rev, -negative[:10], rtol=1e-8)


def test_quad4_plate_reference_load():
    r"""The Quad4 plate of ``test_solver.py``: a reference load twice as
    large halves the load multipliers, a constant preload belongs to ``K``,
    and the tensile reference load gives the negative load multipliers
    closest to zero, without the dense fallback"""
    KC0uu, KGuu = plate_matrices(Nxx=-1.)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4)
        lambda_1 = eigvals[0]
        eigvals2, _ = linear_buckling(KC0uu, 2*KGuu, num_eigvalues=4)
        eigvals_pre, _ = linear_buckling(KC0uu + 0.4*lambda_1*KGuu, KGuu,
                                         num_eigvalues=4)
        eigvals_rev, eigvecs_rev = linear_buckling(KC0uu, -KGuu,
                                                   num_eigvalues=4)
    assert np.all(eigvals > 0)
    np.testing.assert_allclose(eigvals2, eigvals/2, rtol=1e-8)
    np.testing.assert_allclose(eigvals_pre[0], 0.6*lambda_1, rtol=1e-8)
    np.testing.assert_allclose(eigvals_rev, -eigvals, rtol=1e-8)
    assert np.all(relative_residuals(KC0uu, -KGuu, eigvals_rev, eigvecs_rev)
                  < 1e-6)


def test_null_KG_rows_and_null_cols():
    r"""Condensed solution with null rows of ``KG`` and null rows of ``K``,
    with eigenvectors of the size of the original matrices"""
    n = 60
    A = sprandom(n, n, density=0.1, random_state=2)
    K = (A @ A.T + identity(n)).tocsr()
    B = sprandom(n//3, n//3, density=0.3, random_state=3)
    KG = bmat([[-(B @ B.T + 0.1*identity(n//3)), None],
               [None, csc_matrix((n - n//3, n - n//3))]]).tocsr()
    Z = csc_matrix((2, 2))
    K2 = bmat([[K, None], [None, Z]]).tocsr()
    KG2 = bmat([[KG, None], [None, Z]]).tocsr()
    eig_sparse, eigvecs = linear_buckling(K2, KG2, num_eigvalues=10)
    eig_dense, _ = linear_buckling(K2, KG2, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse, eig_dense[:10], rtol=1e-8)
    assert eigvecs.shape == (n + 2, 10)
    assert np.all(eigvecs[n:, :] == 0)
    assert np.all(relative_residuals(K2, KG2, eig_sparse, eigvecs) < 1e-8)
    # NOTE the dense fallback alone
    mu, v = eigen._eigh_condensed(K, KG, 10)
    np.testing.assert_allclose(-1/mu, eig_dense[:10], rtol=1e-8)


def test_singular_K_shift():
    K = csc_matrix(np.diag([1., 2., 0., 4., 5.]))
    assert estimate_cayley_sigma(K, csc_matrix(np.eye(5))) == 1.


@pytest.mark.parametrize('ratio', [1., 0.1, 0.01, 0.001])
def test_sparse_mixed_spectrum_coupled(ratio):
    r"""Negative load multipliers down to ``-ratio`` coupled by a sparse
    congruence: the sparse solver, without dense fallback, returns the
    critical load multipliers 1.0, 1.0, 1.1, 1.2"""
    n = 600
    k = np.linspace(1e6, 5e6, n)
    lambdas = np.concatenate([[1.0, 1.0, 1.1, 1.2],
                              np.linspace(1.5, 4., n//2 - 4),
                              -ratio*np.linspace(1., 4., n//2)])
    T = (identity(n) + 0.1*tril(sprandom(n, n, density=2/n, random_state=1),
                                -1)).tocsc()
    K = (T.T @ diags(k) @ T).tocsr()
    KG = (T.T @ diags(-k/lambdas) @ T).tocsr()
    eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=25,
                                       max_dense_size=0)
    np.testing.assert_allclose(eigvals[:4], [1.0, 1.0, 1.1, 1.2], rtol=1e-8)
    np.testing.assert_allclose(eigvals, np.sort(lambdas[lambdas > 0])[:25],
                               rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)


def test_plate_buckling_fsdt_ritz():
    r"""Simply supported Ritz plate, first-order shear deformation theory,
    against the classical critical load `4 \pi^2 D/b^2`"""
    m1 = 20
    m2 = 10
    N = 3*m1*m2
    pts1, weights1 = roots_legendre(2*m1 - 1)
    pts2, weights2 = roots_legendre(2*m2 - 1)
    E = 200.e9
    nu = 0.3
    # NOTE w simply supported, rotations free
    wxit1, wxir1, wxit2, wxir2 = 0, 1, 0, 1
    wetat1, wetar1, wetat2, wetar2 = 0, 1, 0, 1
    xit1, xir1, xit2, xir2 = 1, 1, 1, 1
    etat1, etar1, etat2, etar2 = 1, 1, 1, 1
    a, b, h = 0.3, 0.1, 0.003
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    Nxxhat = -100.
    Swx = np.zeros(N)
    Swy = np.zeros(N)
    Sphix = np.zeros(N)
    Sphiy = np.zeros(N)
    Sphixx = np.zeros(N)
    Sphixy = np.zeros(N)
    Sphiyx = np.zeros(N)
    Sphiyy = np.zeros(N)
    K = np.zeros((N, N))
    KG = np.zeros((N, N))
    detJ = a*b/4
    for xi, wxi in zip(pts1, weights1):
        wPx_xi = vecfxi(m1, xi, wxit1, wxir1, wxit2, wxir2)
        wP_xi = vecf(m1, xi, wxit1, wxir1, wxit2, wxir2)
        P_xi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Px_xi = vecfxi(m1, xi, xit1, xir1, xit2, xir2)
        for eta, weta in zip(pts2, weights2):
            wP_eta = vecf(m2, eta, wetat1, wetar1, wetat2, wetar2)
            wPx_eta = vecfxi(m2, eta, wetat1, wetar1, wetat2, wetar2)
            P_eta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Px_eta = vecfxi(m2, eta, etat1, etar1, etat2, etar2)
            weight = wxi*weta
            Pi, Pj = np.meshgrid(P_xi, P_eta, indexing='ij')
            Sphix[m1*m2:2*m1*m2] = (Pi*Pj).flatten()
            Sphiy[2*m1*m2:] = (Pi*Pj).flatten()
            Pxi, Pj = np.meshgrid(wPx_xi, wP_eta, indexing='ij')
            Swx[:m1*m2] = (Pxi*Pj*(2/a)).flatten()
            Pxi, Pj = np.meshgrid(Px_xi, P_eta, indexing='ij')
            Sphixx[m1*m2:2*m1*m2] = (Pxi*Pj*(2/a)).flatten()
            Sphiyx[2*m1*m2:] = (Pxi*Pj*(2/a)).flatten()
            Pi, Pxj = np.meshgrid(wP_xi, wPx_eta, indexing='ij')
            Swy[:m1*m2] = (Pi*Pxj*(2/b)).flatten()
            Pi, Pxj = np.meshgrid(P_xi, Px_eta, indexing='ij')
            Sphixy[m1*m2:2*m1*m2] = (Pi*Pxj*(2/b)).flatten()
            Sphiyy[2*m1*m2:] = (Pi*Pxj*(2/b)).flatten()
            e1xx = Sphixx
            e1yy = Sphiyy
            e1xy = Sphixy + Sphiyx
            g0yz = Sphiy + Swy
            g0xz = Sphix + Swx
            Mxx = prop.D11*e1xx + prop.D12*e1yy + prop.D16*e1xy
            Myy = prop.D12*e1xx + prop.D22*e1yy + prop.D26*e1xy
            Mxy = prop.D16*e1xx + prop.D26*e1yy + prop.D66*e1xy
            Qy = prop.A44*g0yz + prop.A45*g0xz
            Qx = prop.A45*g0yz + prop.A55*g0xz
            K += np.outer(detJ*weight*Mxx, e1xx)
            K += np.outer(detJ*weight*Myy, e1yy)
            K += np.outer(detJ*weight*Mxy, e1xy)
            K += np.outer(detJ*weight*Qy, g0yz)
            K += np.outer(detJ*weight*Qx, g0xz)
            KG += np.outer(detJ*weight*Nxxhat*Swx, Swx)
    eigvals, eigvecs = linear_buckling(csc_matrix(K), csc_matrix(KG),
                                       num_eigvalues=4)
    D = E*h**3/(12*(1 - nu**2))
    Ncr_analytical = 4*np.pi**2*D/b**2
    Ncr_computed = eigvals[0]*abs(Nxxhat)
    print('Ncr %.4f  analytical %.4f' % (Ncr_computed, Ncr_analytical))
    np.testing.assert_allclose(Ncr_computed, Ncr_analytical, rtol=0.02)


def test_check_eigenpairs_wrong_and_missing():
    r"""The verification detects wrong eigenpairs and missing load
    multipliers, also below the highest one returned"""
    lambdas = np.concatenate([[1., 1., 2., 3.], np.linspace(4., 20., 26)])
    K, KG = mixed_sign_problem(lambdas, num_null=5)
    mu, eigvecs = eigen._eigh_condensed(K, KG, 10)
    eigvals = -1./mu
    assert check_eigenpairs(K, KG, eigvals, eigvecs) is None
    assert 'relative residual' in check_eigenpairs(K, KG, eigvals,
                                                   eigvecs[:, ::-1])
    assert 'relative residual' in check_eigenpairs(K, KG, eigvals*1.01,
                                                   eigvecs)
    assert 'missing' in check_eigenpairs(K, KG, eigvals[1:], eigvecs[:, 1:])
    # NOTE both equal critical load multipliers missing
    assert 'missing' in check_eigenpairs(K, KG, eigvals[2:], eigvecs[:, 2:])
    # NOTE a multiplier missing between the returned ones
    keep = [0, 1, 3, 4]
    assert 'missing' in check_eigenpairs(K, KG, eigvals[keep],
                                         eigvecs[:, keep])


def penalty_chains(n=20, kp=1e8):
    r"""Two chains of ``n`` unit springs, each fixed at its outer end,
    connected by a penalty spring ``kp``, which sets `\|K\|_F`, and ``KG =
    -I``: the load multipliers are the eigenvalues of ``K``"""
    K = 2*np.eye(2*n) - np.eye(2*n, k=1) - np.eye(2*n, k=-1)
    K[n - 1, n] = K[n, n - 1] = -kp
    K[n - 1, n - 1] = K[n, n] = 1 + kp
    return csc_matrix(K), csc_matrix(-np.eye(2*n))


def test_check_eigenpairs_penalty():
    r"""Regression test: with a penalty spring setting `\|K\|_F`, load
    multipliers 2 % too low, relative residuals of 1 %, are rejected. A
    floor of the denominator of 1e-8 times the matrix-norm scale `(\|K\|_F
    + |\lambda| \|K_G\|_F) \|u\|`, about 170 times `\|K u\| + |\lambda|
    \|K_G u\|` here, accepted them"""
    K, KG = penalty_chains()
    lam, u = np.linalg.eigh(K.toarray())
    lam, u = lam[:4], u[:, :4]
    assert check_eigenpairs(K, KG, lam, u, 1e-3) is None
    res = eigen._relative_residuals(K, KG, 0.98*lam, u)
    np.testing.assert_allclose(res, 0.02/1.98, rtol=1e-3)
    error = check_eigenpairs(K, KG, 0.98*lam, u, 1e-3)
    assert 'relative residual' in error


@pytest.mark.parametrize('precondition', [True, False])
def test_penalty_connections(precondition):
    r"""Regression test: blade-stiffened plate with penalty connections,
    whose matrices are saved in ``tests/data``, the stiff penalty springs
    setting `\|K\|_F = 5.1 \cdot 10^{12}`. The lowest load multiplier is that
    certified by the inertia of `[K] + s [K_G]`. With
    ``precondition=False``, ``eigsh()`` returns eigenpairs with relative
    residuals of 1 to 2 % and a lowest load multiplier 0.08 % too low,
    which a floor of the denominator of the relative residual, 1e-8 times
    the matrix-norm scale `(\|K\|_F + |\lambda| \|K_G\|_F) \|u\|`,
    accepted; they fail the verification, and the dense fallback is used.
    With ``precondition=True``, the default, the relative residuals of
    ``eigsh()`` are below 2.2e-4 (SciPy 1.17.1, Intel MKL 2025.0.0)

    The matrices were assembled with ``panels`` for Figure 7 (Section 3.2.2)
    of Stamatelos and Labeas (2023): a square plate, ``a = b = 0.5`` m, of
    11 ``[0_3/90_3]s`` skin strips, ply thickness 0.134 mm, with 10
    ``[0/90]s`` blades of height 2 mm, every domain a ``plate_clpt_donnell``
    with 12 x 6 (skin) and 12 x 4 (blades) terms, connected by penalty
    springs 100 times those of ``calc_kt_kr``, under a uniform compression
    of 1 N/m.

    Reference: Stamatelos, D.G. and Labeas, G.N., Buckling analysis of
    laminated stiffened plates with material anisotropy using the
    Rayleigh-Ritz approach, Computation, 11(6), 110, 2023,
    https://doi.org/10.3390/computation11060110.

    """
    from scipy.sparse import load_npz

    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    K = load_npz(os.path.join(data, 'blade_stiffened_penalty_K.npz')).tocsr()
    KG = load_npz(os.path.join(data,
                               'blade_stiffened_penalty_KG.npz')).tocsr()
    lambda_1 = 2583.916728
    # NOTE one load multiplier between lambda_1 (1 -/+ 1e-6)
    assert eigen._negative_pivots(K + lambda_1*(1 - 1e-6)*KG) == 0
    assert eigen._negative_pivots(K + lambda_1*(1 + 1e-6)*KG) == 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=4,
                                           precondition=precondition)
    np.testing.assert_allclose(eigvals[0], lambda_1, rtol=1e-6)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-3)


def test_raises_when_verification_fails(monkeypatch):
    r"""A RuntimeError instead of wrong eigenpairs, listing every attempted
    solver; the condensed dense solution is the last resort, with a
    warning"""
    K, KG = mixed_sign_problem(np.linspace(1., 20., 40), num_null=10)
    ref, _ = linear_buckling(K, KG, num_eigvalues=5)

    def wrong_eigsh(K, KG, k, tol, sigma):
        rng = np.random.RandomState(0)
        return -rng.rand(k), rng.randn(K.shape[0], k)

    monkeypatch.setattr(eigen, '_eigsh_cayley', wrong_eigsh)
    with pytest.raises(RuntimeError, match='verification') as e:
        linear_buckling(K, KG, num_eigvalues=5, max_dense_size=0)
    assert str(e.value).count('relative residual') == 2
    assert 'dense fallback disabled with max_dense_size=0' in str(e.value)
    with pytest.raises(RuntimeError, match='skipped'):
        linear_buckling(K, KG, num_eigvalues=5, max_dense_size=10)
    with pytest.warns(DenseFallbackWarning, match='relative residual'):
        eigvals, _ = linear_buckling(K, KG, num_eigvalues=5)
    np.testing.assert_allclose(eigvals, ref, rtol=1e-8)
    # NOTE check=False returns the sparse solution without verification
    eigvals_unchecked, _ = linear_buckling(K, KG, num_eigvalues=5,
                                           check=False)
    assert not np.allclose(eigvals_unchecked, ref)


def spy(monkeypatch, names):
    r"""Record the calls of the functions ``names`` of
    :mod:`pyfe3d.solver.eigen`"""
    calls = []
    for name in names:
        func = getattr(eigen, name)

        def wrapper(*args, _name=name, _func=func, **kwargs):
            calls.append(_name)
            return _func(*args, **kwargs)

        monkeypatch.setattr(eigen, name, wrapper)
    return calls


LB_SOLVERS = ['_eigsh_cayley', '_eigs_shift_search', '_eigh_condensed',
              '_eig_condensed', '_eig_complex']


@pytest.mark.parametrize('max_dense_size', [0, 2000, 10**9])
@pytest.mark.parametrize('symmetric', [True, False])
def test_sparse_solver_runs_first(monkeypatch, symmetric, max_dense_size):
    r"""The sparse solver is called first for any ``max_dense_size``, and no
    dense solver is called when it succeeds"""
    K, KG = mixed_sign_problem(np.linspace(1., 20., 40), num_null=10)
    calls = spy(monkeypatch, LB_SOLVERS)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, _ = linear_buckling(K, KG, num_eigvalues=5,
                                     symmetric=symmetric,
                                     max_dense_size=max_dense_size)
    assert set(calls) == {'_eigsh_cayley' if symmetric
                          else '_eigs_shift_search'}
    np.testing.assert_allclose(eigvals, np.linspace(1., 20., 40)[:5],
                               rtol=1e-8)


def test_estimates_once_for_all_attempts(monkeypatch):
    r"""The shift and the positive definiteness of ``K`` are computed once
    for all the attempts"""
    K, KG = mixed_sign_problem(np.linspace(1., 20., 40), num_null=10)
    ref, _ = linear_buckling(K, KG, num_eigvalues=5)
    estimates = []
    inertia = []
    estimate = eigen._estimate_mu_max
    pd = eigen.is_positive_definite
    monkeypatch.setattr(eigen, '_estimate_mu_max', lambda *a, **k:
                        estimates.append(1) or estimate(*a, **k))
    monkeypatch.setattr(eigen, 'is_positive_definite', lambda A:
                        inertia.append(1) or pd(A))
    eigsh_cayley = eigen._eigsh_cayley
    attempts = []

    def first_fails(K, KG, k, tol, sigma):
        attempts.append(sigma)
        mu, v = eigsh_cayley(K, KG, k, tol, sigma)
        if len(attempts) == 1:
            v = v[:, ::-1]
        return mu, v

    monkeypatch.setattr(eigen, '_eigsh_cayley', first_fails)
    eigvals, _ = linear_buckling(K, KG, num_eigvalues=5)
    assert len(attempts) == 2 and np.isclose(attempts[1], 10*attempts[0])
    np.testing.assert_allclose(eigvals, ref, rtol=1e-8)
    assert len(estimates) == 1
    # NOTE K is factorized once for its positive definiteness, the other
    #      factorizations are the inertia checks of K + s KG
    assert len(inertia) >= 1


# NOTE unsymmetric matrices


def dense_load_multipliers(K, KG):
    r"""Finite load multipliers of the full problem with a direct dense
    eig"""
    mu = eig(a=np.asarray(KG, dtype=complex), b=np.asarray(K, dtype=complex),
             right=False)
    mu = mu[np.isfinite(mu)]
    mu = mu[np.abs(mu) > 1e-10*np.abs(mu).max()]
    return -1./mu


def matrices_complex_KG_block():
    r"""The degrees-of-freedom 3 to 5 have null rows and columns of ``KG``,
    condensed out, and the block of ``KG`` has a complex pair of
    eigenvalues"""
    n = 6
    K = np.diag([4., 5., 6., 7., 8., 9.]) + 0.1*(np.eye(n, k=1)
                                                  + np.eye(n, k=-1))
    KG = np.zeros((n, n))
    KG[:3, :3] = [[-1., 0.5, 0.], [-0.5, -1., 0.], [0., 0., -2.]]
    return K, KG


def test_unsymmetric_complex_pair_condensed():
    r"""A complex eigenpair of the condensed problem, even when not
    returned, is back-substituted with the real factorization"""
    K, KG = matrices_complex_KG_block()
    ref = dense_load_multipliers(K, KG)
    real = np.abs(ref.imag) <= 1e-8*np.abs(ref)
    assert np.sum(real) == 1 and np.sum(~real) == 2
    lam_real = ref[real].real[0]
    lam_cplx = ref[~real]
    eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                       num_eigvalues=1, symmetric=False)
    assert eigvals.dtype == np.float64 and eigvecs.dtype == np.float64
    assert eigvecs.shape == (6, 1)
    np.testing.assert_allclose(eigvals, [lam_real], rtol=1e-10)
    eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                       num_eigvalues=2, symmetric=False)
    assert eigvals.shape == (2,) and eigvecs.shape == (6, 2)
    assert np.iscomplexobj(eigvals) and np.iscomplexobj(eigvecs)
    assert eigvals[0].imag == 0 and np.all(eigvecs[:, 0].imag == 0)
    np.testing.assert_allclose(eigvals[0].real, lam_real, rtol=1e-10)
    assert np.min(np.abs(lam_cplx - eigvals[1])) <= 1e-10*abs(eigvals[1])
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-10)


def test_general_eigenpairs_real_when_kept_real():
    mu = np.array([-0.5 + 0.2j, -0.5 - 0.2j, -0.25 + 1e-14j, -0.1, np.inf])
    v = np.random.RandomState(0).randn(4, 5)*(1 + 1j)
    m, u = eigen._general_eigenpairs(mu, v, 2)
    assert m.dtype == np.float64 and u.dtype == np.float64
    np.testing.assert_allclose(m, [-0.25, -0.1])
    r = v[:, 2].real
    np.testing.assert_allclose(u[:, 0],
                               np.sqrt(2)*r*np.sign(r[np.argmax(abs(r))]))
    m, u = eigen._general_eigenpairs(mu, v)
    assert np.iscomplexobj(m) and m.shape == (4,)
    np.testing.assert_allclose(m[:2], [-0.25, -0.1])
    assert np.all(u[:, :2].imag == 0)


def test_unsymmetric_complex_eigenpairs():
    r"""No positive real load multiplier: the shift search fails, which
    raises without the dense fallback, and the dense fallback returns the
    load multipliers of smallest modulus with a warning"""
    n, nb = 30, 12
    K = (diags(np.linspace(4., 8., n)) + diags(0.5*np.ones(n - 1), 1)
         + diags(0.5*np.ones(n - 1), -1)).toarray()
    rng = np.random.RandomState(4)
    A = rng.randn(nb, nb)
    KG = np.zeros((n, n))
    KG[:nb, :nb] = -np.eye(nb) + 0.5*(A - A.T)
    ref = dense_load_multipliers(K, KG)
    assert np.all(np.abs(ref.imag) > 1e-3*np.abs(ref))
    k = 4
    with pytest.raises(RuntimeError, match='no positive real load multiplier'):
        linear_buckling(csr_matrix(K), csr_matrix(KG), num_eigvalues=k,
                        symmetric=False, max_dense_size=0)
    with pytest.warns(DenseFallbackWarning), \
            pytest.warns(RuntimeWarning, match='no divergence load'):
        eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                           num_eigvalues=k, symmetric=False)
    assert np.iscomplexobj(eigvals) and eigvals.shape == (k,)
    np.testing.assert_allclose(np.abs(eigvals), np.sort(np.abs(ref))[:k],
                               rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-8)
    assert np.all(np.linalg.norm(eigvecs[nb:, :], axis=0) > 0)


def test_check_eigenpairs_real_and_complex():
    K, KG = matrices_complex_KG_block()
    K = csr_matrix(K)
    KG = csr_matrix(KG)
    mu, eigvecs = eigen._eig_condensed(K, KG, 3)
    assert np.iscomplexobj(mu) and np.iscomplexobj(eigvecs)
    lam = -1./mu
    assert check_eigenpairs(K, KG, lam, eigvecs, 1e-8,
                            check_inertia=False) is None
    assert 'relative residual' in check_eigenpairs(K, KG, lam,
            eigvecs[:, ::-1], 1e-8, check_inertia=False)
    # NOTE the complex load multipliers are ignored by the inertia check
    assert check_eigenpairs(K, KG, lam, eigvecs, 1e-8) is None


@pytest.mark.parametrize('symmetric, solver', [
    (False, '_eigs_shift_search'), (True, '_eigsh_cayley')])
def test_solver_type_error_falls_back(monkeypatch, symmetric, solver):
    n = 40
    k = np.linspace(1., 3., n)
    lambdas = np.linspace(1., 20., n)
    K = diags(k).tocsr()
    KG = diags(-k/lambdas).tocsr()

    def raise_type_error(*args, **kwargs):
        raise TypeError('Cannot cast array data from dtype complex128')

    monkeypatch.setattr(eigen, solver, raise_type_error)
    with pytest.warns(DenseFallbackWarning, match='TypeError: Cannot cast'):
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=5,
                                           symmetric=symmetric)
    np.testing.assert_allclose(eigvals, lambdas[:5], rtol=1e-8)
    assert eigvals.dtype == np.float64 and eigvecs.dtype == np.float64


@pytest.mark.parametrize('sparse_solver, max_dense_size',
                         [(True, 2000), (True, 0), (False, 2000)])
def test_nearly_symmetric(sparse_solver, max_dense_size):
    r"""A symmetric matrix with ``1e-12`` asymmetric noise, forced to the
    solvers of unsymmetric matrices, gives the load multipliers of the
    symmetric solver, with real eigenvectors"""
    n = 60
    A = sprandom(n, n, density=0.1, random_state=2)
    K = (A @ A.T + identity(n)).tocsr()
    B = sprandom(n//3, n//3, density=0.3, random_state=3)
    KG = np.zeros((n, n))
    KG[:n//3, :n//3] = -(B @ B.T + 0.1*identity(n//3)).toarray()
    KG = csr_matrix(KG)
    noise = sprandom(n, n, density=0.2, random_state=5).toarray()
    Kn = csr_matrix(K.toarray() + 1e-12*abs(K).max()*noise)
    ref, _ = linear_buckling(K, KG, num_eigvalues=10)
    eigvals, eigvecs = linear_buckling(Kn, KG, num_eigvalues=10,
                                       symmetric=False,
                                       sparse_solver=sparse_solver,
                                       max_dense_size=max_dense_size)
    if not sparse_solver:
        assert np.all(eigvals[:10].imag == 0)
        eigvals = eigvals[:10].real
        eigvecs = eigvecs[:, :10].real
    np.testing.assert_allclose(eigvals, ref, rtol=1e-8)
    assert np.all(relative_residuals(Kn, KG, eigvals, eigvecs) < 1e-8)


def test_shift_search_helpers():
    assert eigen._covered([(0., 1.)]) == 1.
    assert eigen._covered([(0., 1.), (2.5, 1.)]) == 1.
    assert eigen._covered([(0., 1.), (2.5, 1.), (1.5, 0.6)]) == 3.5
    assert eigen._covered([(3., 3.)]) == 6.
    assert eigen._covered([(3., 2.)]) == 0.
    assert eigen._permutation_parity([0, 1, 2]) == 1
    assert eigen._permutation_parity([1, 0, 2]) == -1
    assert eigen._permutation_parity([1, 2, 0]) == 1
    rng = np.random.RandomState(0)
    for trial in range(20):
        A = rng.randn(12, 12)
        lu = splu(csr_matrix(A).tocsc())
        assert eigen._det_sign(lu) == np.sign(np.linalg.det(A))


def test_parity_check(monkeypatch):
    r"""A failed parity check is a failed sparse solution"""
    rng = np.random.default_rng(3)
    n = 40
    K = csr_matrix(spd(n, rng))
    W = rng.standard_normal((n, n))
    KG = csr_matrix(-spd(n, rng)/n + 1e-3*(W - W.T))
    signs = iter([1, -1])
    monkeypatch.setattr(eigen, '_det_sign', lambda lu: next(signs))
    with pytest.raises(RuntimeError, match='parity check failed'):
        linear_buckling(K, KG, symmetric=False, max_dense_size=0)


@pytest.mark.parametrize('sparse_solver,max_dense_size',
                         [(True, 2000), (True, 0), (False, 2000)])
def test_unsymmetric_real(sparse_solver, max_dense_size):
    r"""Unsymmetric ``KG`` with real eigenvalues, against
    :func:`scipy.linalg.eig`"""
    rng = np.random.default_rng(3)
    n = 40
    K = spd(n, rng)
    S = -spd(n, rng)/n
    W = rng.standard_normal((n, n))
    KG = S + 1e-3*(W - W.T)
    assert not is_symmetric(KG)
    eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                       sparse_solver=sparse_solver,
                                       num_eigvalues=6,
                                       max_dense_size=max_dense_size)
    assert np.isrealobj(eigvals) and np.isrealobj(eigvecs)
    mu = eig(KG, K, right=False)
    lam = np.sort((-1/mu).real)
    lam = lam[lam > 0]
    assert np.allclose(eigvals[:5], lam[:5], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals[:5], eigvecs[:, :5])
                  <= 1e-8)


def test_unsymmetric_complex_pairs():
    r"""A strongly skew ``KG`` gives complex conjugate pairs, returned after
    the real load multipliers"""
    rng = np.random.default_rng(5)
    n = 20
    K = spd(n, rng)
    W = rng.standard_normal((n, n))
    KG = -np.eye(n) + 2*(W - W.T)
    eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                       sparse_solver=False)
    assert np.iscomplexobj(eigvals)
    ref = -1/eig(KG, K, right=False)
    assert eigvals.shape == ref.shape
    dist = np.abs(eigvals[:, None] - ref[None, :])
    assert np.all(dist.min(axis=1) <= 1e-8*np.abs(eigvals))
    assert np.all(dist.min(axis=0) <= 1e-8*np.abs(ref))
    real = np.isreal(eigvals)
    assert not np.any(real[np.argmin(real):]) or np.all(real)


def test_symmetric_path_unchanged():
    rng = np.random.default_rng(7)
    n = 30
    K = spd(n, rng)
    KG = -spd(n, rng)/n
    e1, v1 = linear_buckling(csr_matrix(K), csr_matrix(KG), num_eigvalues=5)
    e2, v2 = linear_buckling(csr_matrix(K), csr_matrix(KG), num_eigvalues=5,
                             symmetric=True)
    e3, v3 = linear_buckling(csr_matrix(K), csr_matrix(KG), num_eigvalues=5,
                             symmetric=False)
    assert np.array_equal(e1, e2)
    assert np.allclose(e1[:5], e3[:5], rtol=1e-9)


def test_ziegler_no_divergence():
    r"""Ziegler's double pendulum, see ``test_solver_natural_frequency.py``,
    with decoupled degrees-of-freedom such that the sparse shift search
    runs: ``K(P) = K0 + P KF`` is never singular, and neither the shift
    search nor the dense fallback finds a finite load multiplier, the
    defective null pair of the condensed problem, split into about +-1e-9 by
    the round-off, being null"""
    k, l = 3., 0.5
    K0 = block_diag(np.array([[2*k, -k], [-k, k]]), np.diag(np.arange(1., 9.)))
    KF = block_diag(np.array([[-l, l], [0., 0.]]), np.zeros((8, 8)))
    with pytest.raises(RuntimeError, match='Attempted: eigs'):
        linear_buckling(csr_matrix(K0), csr_matrix(KF), max_dense_size=0)
    with pytest.raises(RuntimeError, match='eig.. with condensed '
                       'degrees-of-freedom: no finite load multiplier'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            linear_buckling(csr_matrix(K0), csr_matrix(KF))
    eigvals = linear_buckling(csr_matrix(K0), csr_matrix(KF),
                              sparse_solver=False)[0]
    assert np.all(eigvals == -np.inf)


def test_null_mu_pencil():
    r"""Regression test: the round-off splits a null eigenvalue of a Jordan
    block of size m into m eigenvalues of modulus about eps^(1/m) ||a||,
    e.g. +-9.4e-10 for Ziegler's double pendulum (2.6e-17 and 0 with
    another LAPACK), which the relative test of _null_mu kept, giving
    spurious load multipliers of +-8.05e8 (64e6 times the flutter load)
    from the dense solvers of linear_buckling(), with their scaled
    matrices; their first-order perturbation bound is of the order of their
    modulus, that of genuine eigenvalues much smaller"""
    def flags(a, b):
        mu, vl, vr = eigen._eig_complex(a, b, left=True)
        return mu, eigen._null_mu_pencil(a, b, mu, vl, vr)

    k, l = 3., 0.5
    mu, null = flags(np.array([[-l, l], [0., 0.]]),
                     np.array([[2*k, -k], [-k, k]]))
    assert np.all(np.abs(mu) < 1e-8) and np.all(null)
    # Jordan blocks of size m, B^-1 A = T J T^-1
    rng = np.random.default_rng(1)
    for m in (2, 3, 4):
        J = np.eye(m, k=1)
        T = rng.standard_normal((m, m)) + m*np.eye(m)
        B = rng.standard_normal((m, m))
        B = B @ B.T + m*np.eye(m)
        mu, null = flags(B @ T @ J @ np.linalg.inv(T), B)
        assert np.all(np.abs(mu) < 1e-3) and np.all(null)
    # genuine eigenvalues down to 1e-10 times the largest are kept, also
    # beside a Jordan block in a non-normal pencil, down to 1e-6 times the
    # largest, above the disc of about 2e-8 of the split null pair, inside
    # which a genuine eigenvalue is not determined
    mu, null = flags(np.diag([1., -1e-6, 1e-10]), np.eye(3))
    assert not np.any(null)
    n = 200
    D = np.diag(np.concatenate([[0., 0.], np.logspace(0, -6, n - 2)
                                * rng.choice([-1, 1], n - 2)]))
    D[0, 1] = 1.
    T = rng.standard_normal((n, n))/np.sqrt(n) + np.eye(n)
    B = rng.standard_normal((n, n))/np.sqrt(n)
    B = B @ B.T + np.eye(n)
    mu, null = flags(B @ T @ D @ np.linalg.inv(T), B)
    assert np.sum(null) == 2 and np.all(np.abs(mu[null]) < 1e-6)
    np.testing.assert_allclose(np.sort(np.abs(mu[~null])),
                               np.sort(np.abs(np.diag(D)[2:])), rtol=1e-4)


def ziegler_decoupled(extra=8):
    r"""``K0`` and ``KF`` of Ziegler's double pendulum with ``extra``
    decoupled degrees-of-freedom: ``K0^-1 KF`` has rank 1 and is nilpotent,
    of norm 1/3"""
    from scipy.sparse import block_diag as sparse_block_diag

    k, l = 3., 0.5
    K0 = sparse_block_diag([csr_matrix([[2*k, -k], [-k, k]]),
                            diags(np.arange(1., extra + 1))], format='csc')
    KF = sparse_block_diag([csr_matrix([[-l, l], [0., 0.]]),
                            csr_matrix((extra, extra))], format='csr')
    return K0, KF


def test_eigs_shifted_roundoff_nu_are_null(monkeypatch):
    r"""Regression test: when all the nu returned by eigs are round-off, here
    those of SciPy 1.17.1 for Ziegler's double pendulum at s = 0, they are
    all null, instead of null relative to the largest one only, which left
    four spurious load multipliers of about +-1e33 that linear_buckling()
    returned unless the parity check rejected them; eigs is replaced, such
    that the test does not depend on the restarts of ARPACK"""
    K0, KF = ziegler_decoupled()
    nu = np.array([1.10e-33, -5.93e-34, 2.95e-34, -1.44e-34, 2.57e-49,
                   -8.55e-50, 6.0e-51, -2.3e-51])

    def eigs_roundoff(OP, k, **kwargs):
        return nu[:k].astype(complex), np.eye(OP.shape[0], k, dtype=complex)

    monkeypatch.setattr(eigen, 'eigs', eigs_roundoff)
    for s in (0., 1e3):
        lu = eigen._splu(K0 + s*KF)
        mu, eigvecs, radius = eigen._eigs_shifted(K0, KF, s, 6, 0, lu)
        assert mu.size == 0 and eigvecs.shape == (10, 0)
        assert radius == np.inf
    with pytest.raises(RuntimeError,
                       match='shift 0: no finite load multiplier'):
        linear_buckling(K0, KF, max_dense_size=0)


@pytest.mark.parametrize('seed', range(8))
def test_eigs_shift_search_ziegler_any_start(monkeypatch, seed):
    r"""Ziegler's double pendulum, K0^-1 KF nilpotent: whatever the starting
    vector and, with SciPy 1.17 or newer, the restart vectors of ARPACK,
    which give round-off nu of 0, about 1e-33 or 1e-17 with SciPy 1.17.1,
    the shift search finds no finite load multiplier"""
    from pyfe3d.solver import arpackutils

    def start_kwargs(n):
        kwargs = {'v0': np.random.RandomState(seed).uniform(-1., 1., n)}
        if arpackutils._arpack_takes_rng():
            kwargs['rng'] = np.random.default_rng(seed)
        return kwargs

    monkeypatch.setattr(eigen, 'start_kwargs', start_kwargs)
    K0, KF = ziegler_decoupled()
    for s in (0., 1e3, 1e8):
        lu = eigen._splu(K0 + s*KF)
        mu, eigvecs, radius = eigen._eigs_shifted(K0, KF, s, 8, 0, lu)
        assert mu.size == 0 and radius == np.inf
    with pytest.raises(RuntimeError, match='no finite load multiplier'):
        eigen._eigs_shift_search(K0, KF, 6, 0, 1e-3)


def test_norm_estimate():
    r"""_norm_estimate is between ||O v||/||v|| and ||O||_2, O = A^-1 KG,
    and exact for the rank-one O of Ziegler's double pendulum, whatever the
    number of degrees-of-freedom where KG is null"""
    from pyfe3d.solver.arpackutils import start_vector

    rng = np.random.default_rng(0)
    for trial in range(10):
        A = rng.standard_normal((15, 15)) + 15*np.eye(15)
        KG = rng.standard_normal((15, 15))
        KG[:, rng.permutation(15)[:10]] = 0.
        O = np.linalg.solve(A, KG)
        v = rng.uniform(-1., 1., 15)
        est = eigen._norm_estimate(splu(csc_matrix(A)), csr_matrix(KG), v)
        lower = np.linalg.norm(O @ v)/np.linalg.norm(v)
        assert lower*(1 - 1e-12) <= est <= np.linalg.norm(O, 2)*(1 + 1e-12)
    for extra in (8, 1000, 100000):
        K0, KF = ziegler_decoupled(extra)
        v = start_vector(extra + 2)
        est = eigen._norm_estimate(splu(K0), KF, v)
        np.testing.assert_allclose(est, 1/3, rtol=1e-12)
        O_v = splu(K0).solve(KF @ v)
        assert np.linalg.norm(O_v)/np.linalg.norm(v) < 0.2
    assert eigen._norm_estimate(splu(K0), csr_matrix(KF.shape), v) == 0.


def test_eig_complex_mkl_crash_regression(tmp_path):
    r"""Regression test: with SciPy 1.17.1 linked against Intel MKL
    2025.0.0, :func:`scipy.linalg.eig` on these real 48 x 48 matrices,
    i.e. the LAPACK driver ``dggev``, corrupts the heap and the process
    exits with the code ``0xc0000374``, whereas the complex driver
    ``zggev`` of :func:`pyfe3d.solver.eigen._eig_complex` returns. The
    dense solvers of :func:`linear_buckling` for unsymmetric matrices use
    the latter. Run in a subprocess, which a crash does not take down with
    the test session"""
    code = textwrap.dedent("""
        import numpy as np
        from scipy.sparse import csr_matrix
        from pyfe3d.solver import linear_buckling
        rng = np.random.RandomState(4)
        KG = rng.randn(48, 48)
        K = rng.randn(48, 48) + 48*np.eye(48)
        eigvals, eigvecs = linear_buckling(csr_matrix(K), csr_matrix(KG),
                                           symmetric=False,
                                           sparse_solver=False)
        Ku = K @ eigvecs
        KGu = KG @ eigvecs
        res = (np.linalg.norm(Ku + KGu*eigvals, axis=0)
               / (np.linalg.norm(Ku, axis=0)
                  + np.abs(eigvals)*np.linalg.norm(KGu, axis=0)))
        assert eigvecs.shape == (48, 48), eigvecs.shape
        assert np.all(res < 1e-8), res
        print('ok')
        """)
    # NOTE the subprocess imports the pyfe3d of this test session, compiled,
    #      and runs outside the repository, whose source tree would shadow
    #      an installed pyfe3d
    import pyfe3d
    root = os.path.dirname(os.path.dirname(os.path.abspath(pyfe3d.__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [root] + [p for p in [env.get('PYTHONPATH')] if p])
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            cwd=str(tmp_path), capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, (hex(result.returncode & 0xffffffff),
                                    result.stderr[-2000:])
    assert result.stdout.strip().endswith('ok')



def test_check_eigenpairs_inertia_skipped_or_failing():
    r"""``K = diag(1, 2)``, ``KG = -I``: load multipliers 1 and 2. The
    inertia check, which tells that the multiplier 1 is missing, is skipped
    when no positive multiplier is given or when ``K`` is not positive
    definite, and it tells when a multiplier is given twice"""
    K = diags([1., 2.]).tocsr()
    KG = -identity(2, format='csr')
    e0 = np.array([[1.], [0.]])
    e1 = np.array([[0.], [1.]])
    assert (check_eigenpairs(K, KG, [2.], e1)
            == 'at least one load multiplier lower than 2 is missing')
    assert check_eigenpairs(K, KG, [2.], e1, K_positive_definite=False) \
        is None
    # NOTE the reversed load, negative multipliers only
    assert check_eigenpairs(K, -KG, [-1.], e0) is None
    assert (check_eigenpairs(K, KG, [1., 1., 2.], np.hstack((e0, e0, e1)))
            == '2 load multipliers lower than 1.9998 were found, but only 1 '
            'exist')


def test_negative_pivots_not_finite():
    """The inertia is unknown when a pivot is not finite"""
    assert eigen._negative_pivots(csr_matrix(np.diag([1., -2.]))) == 1
    assert eigen._negative_pivots(csr_matrix(np.diag([np.inf, 1.]))) is None
    assert eigen._negative_pivots(csr_matrix(np.diag([np.nan, 1.]))) is None


def test_count_multipliers():
    r"""Load multipliers 1 and 0.5 below `10^6/|\mu|_{max}`, the count being
    unknown without ``mu_max`` or with ``K`` not positive definite"""
    K = identity(10, format='csr')
    KG = diags([-1., -2.] + [0.]*8).tocsr()
    pd = eigen.is_positive_definite
    assert eigen._count_multipliers(K, KG, 2., pd) == 2
    assert eigen._count_multipliers(K, -KG, 2., pd) == 0
    assert eigen._count_multipliers(K, KG, None, pd) is None
    assert eigen._count_multipliers(K, KG, 2., lambda K: False) is None


def test_eigsh_both_signs_without_multipliers():
    """A null ``KG`` has no finite load multiplier of either sign"""
    K = identity(10, format='csc')
    KG = csr_matrix((10, 10))
    with pytest.raises(RuntimeError, match='no finite load multiplier'):
        eigen._eigsh_both_signs(K, KG, 3, 0, 1., 1.,
                                eigen.is_positive_definite)


def test_check_both_signs():
    r"""``K = diag(1, 2)``, ``KG = I``: `\mu` = 1 and 0.5, the negative load
    multipliers -1 and -2, verified with the reversed load"""
    K = diags([1., 2.]).tocsr()
    KG = identity(2, format='csr')
    e0 = np.array([[1.], [0.]])
    e1 = np.array([[0.], [1.]])
    assert eigen._check_both_signs(K, KG, [1.], e0, 1e-8) is None
    error = eigen._check_both_signs(K, KG, [1.], e1, 1e-8)
    assert error.startswith('reversed load: relative residual')
    assert (eigen._check_both_signs(K, KG, [0., np.inf], np.hstack((e0, e1)),
                                    1e-8)
            == 'no finite load multiplier was found')


def test_condense_null_KG():
    K = identity(3, format='csr')
    with pytest.raises(ValueError, match='KG is a null matrix'):
        eigen._condense(K, csr_matrix((3, 3)), np.zeros(3, dtype=bool))


def shift_search_matrices():
    r"""``K = I``: load multipliers `\lambda = -1/d_i` of ``KG = diag(d)``,
    eight negative ones close to zero and the positive ones 1/1.2 and 1"""
    d = [10., 11., 12., 13., 14., 15., 16., 17., -1., -1.2]
    return identity(10, format='csc'), diags(d).tocsr()


def test_eigs_shift_search_errors(monkeypatch):
    """The errors of the search of shifts of the unsymmetric solver: too few
    degrees-of-freedom, singular ``K``, no positive load multiplier, too
    many shifts and a singular shifted matrix"""
    search = eigen._eigs_shift_search
    with pytest.raises(ValueError, match='at least 3 degrees-of-freedom'):
        search(identity(2), identity(2), 1, 0, 1e-6)
    K = csr_matrix(np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]]))
    with pytest.raises(RuntimeError, match='K is singular'):
        search(K, identity(3), 1, 0, 1e-6)
    with pytest.raises(RuntimeError, match='no positive real load '
                       'multiplier exists, all the finite ones were found'):
        search(identity(10, format='csc'),
               diags([1., 2.] + [0.]*8).tocsr(), 6, 0, 1e-6)

    K, KG = shift_search_matrices()
    mu, eigvecs = search(K, KG, 6, 0, 1e-6)
    assert np.isclose(-1./mu[0], 1./1.2)
    monkeypatch.setattr(eigen, 'SHIFT_SEARCH_MAX_STEPS', 1)
    with pytest.raises(RuntimeError, match='after 1 shifts'):
        search(K, KG, 6, 0, 1e-6)
    monkeypatch.undo()

    # NOTE every factorization after the one of K fails
    splu_calls = []
    splu = eigen._splu

    def singular_after_K(A):
        splu_calls.append(1)
        if len(splu_calls) > 1:
            raise RuntimeError('Factor is exactly singular')
        return splu(A)

    monkeypatch.setattr(eigen, '_splu', singular_after_K)
    with pytest.raises(RuntimeError, match=r'K \+ s KG is singular'):
        search(K, KG, 6, 0, 1e-6)
    # NOTE K, then the shift and its two perturbations
    assert len(splu_calls) == 4
