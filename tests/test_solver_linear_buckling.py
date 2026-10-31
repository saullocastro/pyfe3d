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
from numpy.linalg import LinAlgError
from scipy.linalg import eig, block_diag
from scipy.sparse import (csc_matrix, csr_matrix, diags, identity, bmat, tril,
                          random as sprandom)
from scipy.sparse.linalg import splu
from scipy.special import roots_legendre

from pyfe3d.solver import (linear_buckling, check_eigenpairs,
                           estimate_cayley_sigma, DenseFallbackWarning,
                           is_symmetric)
from pyfe3d.solver import eigen

from ritz import isotropic_plate, vecf, vecf_xi
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


def test_null_KG_rows_and_null_K_rows():
    r"""Condensed solution with null rows of ``KG``; null rows and columns of
    ``K`` raise an error naming their degrees-of-freedom, with the sparse
    and the dense solvers"""
    n = 60
    A = sprandom(n, n, density=0.1, random_state=2)
    K = (A @ A.T + identity(n)).tocsr()
    B = sprandom(n//3, n//3, density=0.3, random_state=3)
    KG = bmat([[-(B @ B.T + 0.1*identity(n//3)), None],
               [None, csc_matrix((n - n//3, n - n//3))]]).tocsr()
    eig_sparse, eigvecs = linear_buckling(K, KG, num_eigvalues=10)
    eig_dense, _ = linear_buckling(K, KG, sparse_solver=False)
    np.testing.assert_allclose(eig_sparse, eig_dense[:10], rtol=1e-8)
    assert eigvecs.shape == (n, 10)
    assert np.all(relative_residuals(K, KG, eig_sparse, eigvecs) < 1e-8)
    Z = csc_matrix((2, 2))
    K2 = bmat([[K, None], [None, Z]]).tocsr()
    KG2 = bmat([[KG, None], [None, Z]]).tocsr()
    for kwargs in [{}, {'sparse_solver': False}, {'symmetric': False}]:
        with pytest.raises(LinAlgError, match=r'K is singular: null row or '
                           r'column at 2 degree\(s\)-of-freedom, of indices '
                           r'%d, %d\.' % (n, n + 1)):
            linear_buckling(K2, KG2, num_eigvalues=10, **kwargs)
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
    r"""Hard simply supported Ritz plate, first-order shear deformation
    theory, ``a/b = 3``, ``b/h = 33``, uniform compression, against the
    Navier solution of the same theory

    The reference is given by Seide, P., and Stowell, E. Z., Elastic and
    Plastic Buckling of Simply Supported Solid-Core Sandwich Plates in
    Compression, NACA Report 967, 1950, Eq. (1), p. 2, derived in Appendix
    A, Eq. (A7), p. 7,

    .. math::
        N_{cr} = k \frac{\pi^2 D}{b^2}, \qquad
        k = \frac{(m/\beta + \beta/m)^2}{1 + r (1 + m^2/\beta^2)}, \qquad
        \beta = \frac{a}{b}, \qquad r = \frac{\pi^2 D}{b^2 D_Q}

    minimised over `m`, with the transverse shear stiffness `D_Q` of the
    sandwich core taken equal to `A_{55}`, their governing equation (A5)
    being that of a plate with transverse shear stiffness `D_Q`; the same
    load follows from Wang, C. M., Wang, C. Y., and Reddy, J. N., Exact
    Solutions for Buckling of Structural Members, CRC Press, 2005, Sec.
    4.7.2, Eq. (4.7.17), with the foundation moduli equal to zero and
    `\gamma = 0` (page not verified). It is 0.51 % below the classical load
    `4 \pi^2 D/b^2`. The rotations are zero across the edges they are
    tangent to, the hard simple support of their Navier solution, Wang et al.
    (2005), Eq. (4.7.10b). With 20x10 terms the load agrees with the
    reference to 1e-8, with 10x5 terms it is 0.025 % above it. With the
    rotations free on all edges, the soft simple support used before, the
    load is 1.28 % below the reference with 20x10 terms and 1.73 % with
    30x15, the soft support lowering the load as the terms resolve the
    boundary layer along the edges.
    """
    m1 = 20
    m2 = 10
    N = 3*m1*m2
    pts1, weights1 = roots_legendre(2*m1 - 1)
    pts2, weights2 = roots_legendre(2*m2 - 1)
    E = 200.e9
    nu = 0.3
    # NOTE w simply supported; each rotation uses the functions of w across
    #      the edges it is tangent to, and free functions along them
    wxit1, wxir1, wxit2, wxir2 = 0, 1, 0, 1
    wetat1, wetar1, wetat2, wetar2 = 0, 1, 0, 1
    xit1, xir1, xit2, xir2 = 1, 1, 1, 1
    etat1, etar1, etat2, etar2 = 1, 1, 1, 1
    a, b, h = 0.3, 0.1, 0.003
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    Nxxhat = -100.
    Sw_x = np.zeros(N)
    Sw_y = np.zeros(N)
    Sphix = np.zeros(N)
    Sphiy = np.zeros(N)
    Sphix_x = np.zeros(N)
    Sphix_y = np.zeros(N)
    Sphiy_x = np.zeros(N)
    Sphiy_y = np.zeros(N)
    K = np.zeros((N, N))
    KG = np.zeros((N, N))
    detJ = a*b/4
    for xi, wxi in zip(pts1, weights1):
        wPxi_xi = vecf_xi(m1, xi, wxit1, wxir1, wxit2, wxir2)
        wPxi = vecf(m1, xi, wxit1, wxir1, wxit2, wxir2)
        Pxi = vecf(m1, xi, xit1, xir1, xit2, xir2)
        Pxi_xi = vecf_xi(m1, xi, xit1, xir1, xit2, xir2)
        for eta, weta in zip(pts2, weights2):
            wPeta = vecf(m2, eta, wetat1, wetar1, wetat2, wetar2)
            wPeta_eta = vecf_xi(m2, eta, wetat1, wetar1, wetat2, wetar2)
            Peta = vecf(m2, eta, etat1, etar1, etat2, etar2)
            Peta_eta = vecf_xi(m2, eta, etat1, etar1, etat2, etar2)
            weight = wxi*weta
            # NOTE hard simple support: phix is zero on the edges y = 0, b
            #      and phiy on the edges x = 0, a
            Sphix[m1*m2:2*m1*m2] = np.outer(Pxi, wPeta).flatten()
            Sphix_x[m1*m2:2*m1*m2] = np.outer(Pxi_xi*(2/a), wPeta).flatten()
            Sphix_y[m1*m2:2*m1*m2] = np.outer(Pxi, wPeta_eta*(2/b)).flatten()
            Sphiy[2*m1*m2:] = np.outer(wPxi, Peta).flatten()
            Sphiy_x[2*m1*m2:] = np.outer(wPxi_xi*(2/a), Peta).flatten()
            Sphiy_y[2*m1*m2:] = np.outer(wPxi, Peta_eta*(2/b)).flatten()
            Sw_x[:m1*m2] = np.outer(wPxi_xi*(2/a), wPeta).flatten()
            Sw_y[:m1*m2] = np.outer(wPxi, wPeta_eta*(2/b)).flatten()
            e1xx = Sphix_x
            e1yy = Sphiy_y
            e1xy = Sphix_y + Sphiy_x
            g0yz = Sphiy + Sw_y
            g0xz = Sphix + Sw_x
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
            KG += np.outer(detJ*weight*Nxxhat*Sw_x, Sw_x)
    # NOTE the Ritz model applies the boundary conditions by zeroing
    #      approximation functions, whose null rows and columns are
    #      partitioned out before calling the solver
    bu = np.any(K != 0, axis=0)
    assert np.count_nonzero(bu) < N
    eigvals, eigvecs = linear_buckling(csc_matrix(K[bu, :][:, bu]),
                                       csc_matrix(KG[bu, :][:, bu]),
                                       num_eigvalues=4)
    # NOTE Seide and Stowell (1950), Eq. (1), with D_Q = A55
    D = E*h**3/(12*(1 - nu**2))
    r = np.pi**2*D/(b**2*prop.A55)
    beta = a/b
    k = min((m/beta + beta/m)**2/(1 + r*(1 + m**2/beta**2))
            for m in range(1, 21))
    Ncr_analytical = k*np.pi**2*D/b**2
    Ncr_computed = eigvals[0]*abs(Nxxhat)
    print('Ncr %.4f  analytical %.4f' % (Ncr_computed, Ncr_analytical))
    np.testing.assert_allclose(Ncr_computed, Ncr_analytical, rtol=1e-5)


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
    connected by a penalty spring ``kp``, which sets `\|\boldsymbol{K}\|_F`, and ``KG =
    -I``: the load multipliers are the eigenvalues of ``K``"""
    K = 2*np.eye(2*n) - np.eye(2*n, k=1) - np.eye(2*n, k=-1)
    K[n - 1, n] = K[n, n - 1] = -kp
    K[n - 1, n - 1] = K[n, n] = 1 + kp
    return csc_matrix(K), csc_matrix(-np.eye(2*n))


def test_check_eigenpairs_penalty():
    r"""Regression test: with a penalty spring setting `\|\boldsymbol{K}\|_F`, load
    multipliers 2 % too low, relative residuals of 1 %, are rejected. A
    floor of the denominator of 1e-8 times the matrix-norm scale `(\|\boldsymbol{K}\|_F
    + |\lambda| \|\boldsymbol{K}_G\|_F) \|\boldsymbol{u}\|`, about 170 times `\|\boldsymbol{K} \boldsymbol{u}\| + |\lambda|
    \|\boldsymbol{K}_G \boldsymbol{u}\|` here, accepted them"""
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
    setting `\|\boldsymbol{K}\|_F = 5.1 \cdot 10^{12}`. The lowest load multiplier is that
    certified by the inertia of `\boldsymbol{K} + s \boldsymbol{K}_G`. With
    ``precondition=False``, ``eigsh()`` returns eigenpairs with relative
    residuals of 1 to 2 % and a lowest load multiplier 0.08 % too low,
    which a floor of the denominator of the relative residual, 1e-8 times
    the matrix-norm scale `(\|\boldsymbol{K}\|_F + |\lambda| \|\boldsymbol{K}_G\|_F) \|\boldsymbol{u}\|`,
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
    the round-off, being null

    Reference: H. Ziegler, Die Stabilitätskriterien der Elastomechanik,
    Ingenieur-Archiv, 20(1), 49-56, 1952,
    https://doi.org/10.1007/BF00536796: the follower force does not make
    the stiffness singular, ``det K(P) = k**2`` for every ``P``, and the
    pendulum loses stability by flutter, not by divergence."""
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
    r"""``K0`` and ``KF`` of Ziegler's double pendulum (Ziegler, H., Die
    Stabilitätskriterien der Elastomechanik, Ingenieur-Archiv, 20(1), 49-56,
    1952, https://doi.org/10.1007/BF00536796, see
    ``test_solver_natural_frequency.py``) with ``extra`` decoupled
    degrees-of-freedom: ``K0^-1 KF = (l/k)*[[-1, 1], [-1, 1]]`` in the
    pendulum block, of rank 1, nilpotent (null trace and determinant) and of
    2-norm ``2*l/k = 1/3`` for ``k = 3`` and ``l = 0.5``"""
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


def test_splu_warnings_shown_once():
    r"""A warning of the caller, shown once per location with the
    ``'default'`` action, is not repeated after :func:`eigen._splu`, also
    when it raises for an exactly singular matrix, nor after the shift
    search of :func:`linear_buckling` for unsymmetric matrices, which
    factorizes with it: no :func:`warnings.catch_warnings` block resets the
    registry of the warnings already shown. Load multipliers 1 to 20 of an
    upper triangular `\boldsymbol{K}^{-1} \boldsymbol{K}_G`"""
    n = 20
    K = diags(np.arange(1., n + 1)).tocsr()
    KG = csr_matrix(-np.eye(n) + 0.05*np.triu(np.ones((n, n)), k=1))

    def emit():
        warnings.warn('caller warning', UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')
        for _ in range(3):
            emit()
            eigen._splu(K)
            with pytest.raises(RuntimeError, match='exactly singular'):
                eigen._splu(csr_matrix((n, n)) + diags([1.] + [0.]*(n - 1)))
            eigvals, _ = linear_buckling(K, KG, num_eigvalues=3,
                                         symmetric=False)
            np.testing.assert_allclose(eigvals, [1., 2., 3.], rtol=1e-10)
    assert [str(w.message) for w in caught] == ['caller warning']


# NOTE repeated load multipliers


def laplacian_3d(n):
    r"""Finite difference Laplacian of the unit cube with ``n`` inner points
    per direction and its eigenvalues

    `\boldsymbol{K} = \boldsymbol{T} \otimes \boldsymbol{I} \otimes \boldsymbol{I} + \boldsymbol{I} \otimes \boldsymbol{T} \otimes \boldsymbol{I} + \boldsymbol{I}
    \otimes \boldsymbol{I} \otimes \boldsymbol{T}`, with `\boldsymbol{T}` = tridiag(-1, 2, -1) of size `n`.
    Closed form: `\boldsymbol{T} \boldsymbol{x}_i = \mu_i \boldsymbol{x}_i` with `x_{i,j} = \sin(i j
    \theta)`, `\theta = \pi/(n + 1)` and `\mu_i = 2 - 2\cos(i \theta)`, since
    `-\sin((j - 1) i \theta) + 2\sin(j i \theta) - \sin((j + 1) i \theta) =
    (2 - 2\cos(i \theta))\sin(j i \theta)` and `\sin(0) = \sin((n + 1) i
    \theta) = 0` at the ends; the Kronecker products of the `\boldsymbol{x}_i` are
    the eigenvectors of `\boldsymbol{K}`, with the eigenvalues `\mu_i + \mu_j + \mu_l`,
    exactly repeated for the permutations of `(i, j, l)`: multiplicity 3 for
    `(1, 1, 2)` and `(1, 2, 2)`, 6 for `(1, 2, 3)`. With `\boldsymbol{K}_G = -\boldsymbol{I}` they
    are the load multipliers.
    """
    T = diags([-np.ones(n - 1), 2*np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    I = identity(n)
    from scipy.sparse import kron
    K = (kron(kron(T, I), I) + kron(kron(I, T), I)
         + kron(kron(I, I), T)).tocsr()
    mu = 2 - 2*np.cos(np.arange(1, n + 1)*np.pi/(n + 1))
    lam = np.sort((mu[:, None, None] + mu[None, :, None]
                   + mu[None, None, :]).ravel())
    return K, lam


def one_copy_each(eigsh_cayley, rtol=1e-8):
    r"""``_eigsh_cayley`` returning one copy of each repeated eigenvalue,
    the lowest ``k`` of them, as the Lanczos process does in exact
    arithmetic, see :func:`pyfe3d.solver.arpackutils.deflated_eigsh`"""
    def wrapper(K, KG, k, tol, sigma):
        mu, v = eigsh_cayley(K, KG, min(3*k, K.shape[0] - 2), tol, sigma)
        keep = np.ones(mu.size, dtype=bool)
        keep[1:] = np.abs(np.diff(mu)) > rtol*np.abs(mu[1:])
        return mu[keep][:k], v[:, keep][:, :k]
    return wrapper


@pytest.mark.parametrize('k', [4, 7, 10])
def test_repeated_load_multipliers_completed(monkeypatch, k):
    r"""All the copies of the repeated load multipliers of
    :func:`laplacian_3d`, of multiplicity 3, are returned without the dense
    fallback when the first call of ``eigsh()`` returns one copy of each,
    the missing copies being found by the deflated passes of
    :func:`pyfe3d.solver.eigen._complete_repeated`; without them,
    ``MAX_COMPLETION_PASSES = 0``, the verification reports the missing
    copies and the dense solver is used"""
    K, lam = laplacian_3d(6)
    KG = -identity(K.shape[0], format='csr')
    calls = []
    deflated = eigen.deflated_eigsh

    def spy_deflated(*args, **kwargs):
        calls.append(1)
        return deflated(*args, **kwargs)

    monkeypatch.setattr(eigen, '_eigsh_cayley',
                        one_copy_each(eigen._eigsh_cayley))
    monkeypatch.setattr(eigen, 'deflated_eigsh', spy_deflated)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=k)
    np.testing.assert_allclose(eigvals, lam[:k], rtol=1e-10)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-8)
    # NOTE the copies are independent: K-orthogonal eigenvectors
    G = eigvecs.T @ K @ eigvecs
    d = np.sqrt(np.diag(G))
    np.testing.assert_allclose(G/np.outer(d, d), np.eye(k), atol=1e-8)
    assert calls
    monkeypatch.setattr(eigen, 'MAX_COMPLETION_PASSES', 0)
    with pytest.warns(DenseFallbackWarning, match='are missing'):
        eigvals, _ = linear_buckling(K, KG, num_eigvalues=k)
    np.testing.assert_allclose(eigvals, lam[:k], rtol=1e-10)


def test_repeated_load_multipliers_completed_second_strategy(monkeypatch):
    r"""The result of the second sparse strategy, ``eigsh()`` with a 10
    times larger shift, is also completed by deflated passes when it misses
    copies of repeated load multipliers, before the dense fallback: the
    first call of ``eigsh()`` raises, such that there is nothing to complete
    after it, and the second one returns one copy of each repeated load
    multiplier of :func:`laplacian_3d`, as in
    :func:`test_repeated_load_multipliers_completed`"""
    K, lam = laplacian_3d(6)
    KG = -identity(K.shape[0], format='csr')
    k = 7
    one_copy = one_copy_each(eigen._eigsh_cayley)
    sigmas = []
    events = []

    def first_raises(K, KG, k, tol, sigma):
        sigmas.append(sigma)
        events.append('eigsh')
        if len(sigmas) == 1:
            raise RuntimeError('ARPACK failed')
        return one_copy(K, KG, k, tol, sigma)

    deflated = eigen.deflated_eigsh

    def spy_deflated(*args, **kwargs):
        events.append('deflated')
        return deflated(*args, **kwargs)

    monkeypatch.setattr(eigen, '_eigsh_cayley', first_raises)
    monkeypatch.setattr(eigen, 'deflated_eigsh', spy_deflated)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=k)
    np.testing.assert_allclose(eigvals, lam[:k], rtol=1e-10)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-8)
    assert len(sigmas) == 2 and np.isclose(sigmas[1], 10*sigmas[0])
    # NOTE the deflated passes follow the second call of eigsh()
    assert events[:2] == ['eigsh', 'eigsh'] and 'deflated' in events[2:]
    # NOTE without them the dense solver is used, the second strategy
    #      having missed copies
    sigmas.clear()
    monkeypatch.setattr(eigen, 'MAX_COMPLETION_PASSES', 0)
    with pytest.warns(DenseFallbackWarning, match='10 times larger shift: '
                      r'\d+ load multiplier\(s\) lower than .* are missing'):
        eigvals, _ = linear_buckling(K, KG, num_eigvalues=k)
    np.testing.assert_allclose(eigvals, lam[:k], rtol=1e-10)


def test_completion_not_applicable(monkeypatch):
    r"""The deflated passes complete only accurate eigenpairs, found with a
    shift of the Cayley mode larger than the estimate of the largest
    `|\mu|`: the inaccurate eigenpairs of the wrong eigenvectors of
    ``wrong_eigsh`` and the eigenpairs of too small a shift, which are not
    the lowest ones, are not completed, and the completion is then not
    listed among the attempted solvers"""
    K, lam = laplacian_3d(6)
    KG = -identity(K.shape[0], format='csr')
    eigsh_cayley = eigen._eigsh_cayley

    def wrong_eigsh(K, KG, k, tol, sigma):
        mu, v = eigsh_cayley(K, KG, k, tol, sigma)
        return mu, v[:, ::-1]

    monkeypatch.setattr(eigen, '_eigsh_cayley', wrong_eigsh)
    with pytest.raises(RuntimeError, match='verification') as e:
        linear_buckling(K, KG, num_eigvalues=4, max_dense_size=0)
    assert 'deflated' not in str(e.value)
    monkeypatch.setattr(eigen, '_eigsh_cayley', eigsh_cayley)
    with pytest.raises(RuntimeError, match='are missing') as e:
        linear_buckling(K, KG, num_eigvalues=4, sigma=1e-6,
                        max_dense_size=0)
    assert 'deflated' not in str(e.value)

    eigsh_both_signs = eigen._eigsh_both_signs

    def nan_eigsh(*args):
        mu, v = eigsh_both_signs(*args)
        return np.full_like(mu, np.nan), v

    # NOTE no finite eigenpair to complete
    monkeypatch.setattr(eigen, '_eigsh_both_signs', nan_eigsh)
    with pytest.raises(RuntimeError) as e:
        linear_buckling(K, KG, num_eigvalues=4, max_dense_size=0)
    assert 'deflated' not in str(e.value)


def test_completion_capped_ncv(monkeypatch):
    r"""With ``ncv`` of ARPACK capped, see
    :func:`pyfe3d.solver.arpackutils.arpack_max_ncv`, each deflated pass
    asks for at most ``(max_ncv - 1)//2`` eigenpairs, and the missing
    copies of :func:`test_repeated_load_multipliers_completed` are still
    found within ``MAX_COMPLETION_PASSES`` passes"""
    K, lam = laplacian_3d(6)
    KG = -identity(K.shape[0], format='csr')
    k = 4
    max_ncv = 7
    asked = []
    deflated = eigen.deflated_eigsh

    def spy_deflated(A, k_pass, *args, **kwargs):
        asked.append(k_pass)
        return deflated(A, k_pass, *args, **kwargs)

    monkeypatch.setattr(eigen, '_eigsh_cayley',
                        one_copy_each(eigen._eigsh_cayley))
    monkeypatch.setattr(eigen, 'deflated_eigsh', spy_deflated)
    monkeypatch.setattr(eigen, 'arpack_max_ncv', lambda: max_ncv)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=k)
    np.testing.assert_allclose(eigvals, lam[:k], rtol=1e-10)
    assert asked and max(asked) <= (max_ncv - 1)//2


def test_dense_solver_error_K_positive_definite(monkeypatch):
    r"""With the dense solver, an error of ``eigh()`` is raised unchanged
    when `\boldsymbol{K}` is positive definite, the diagnosis of a
    stiffness matrix that is not positive definite being then wrong"""
    K, lam = laplacian_3d(4)
    KG = -identity(K.shape[0], format='csr')

    def failing_eigh(a, b):
        raise LinAlgError('eigh failed')

    monkeypatch.setattr(eigen, 'eigh', failing_eigh)
    with pytest.raises(LinAlgError, match='^eigh failed$'):
        linear_buckling(K, KG, num_eigvalues=4, sparse_solver=False)


def torsion_cylinder_Z33(ntheta):
    r"""``K`` and ``KG`` of the cylinder Z33 under torsion of
    ``test_quad4_linear_buckling_cylinder_Nxy.py``, with ``ntheta`` elements
    around the circumference: Castro et al. (2014), Composite Structures,
    118, 303-315, Table 3 (material Geier1997), Table 4 (geometry and
    stacking sequence), Table 1 (boundary condition CC2)"""
    from scipy.sparse import coo_matrix
    from pyfe3d.shellprop_utils import laminated_plate
    from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF
    from test_quad4_linear_buckling_cylinder_Nxy import \
        _cylindrical_transformation
    L, R = 0.510, 0.250
    b = 2*np.pi*R
    nlength = int(ntheta*L/b)
    if nlength % 2 == 0:
        nlength += 1
    prop = laminated_plate(stack=[0, 0, 19, -19, 37, -37, 45, -45, 51, -51],
                           plyt=0.125e-3, laminaprop=(123.55e9, 8.708e9,
                           0.319, 5.695e9, 5.695e9, 3.400e9),
                           shear_correction='constant')
    data = Quad4Data()
    probe = Quad4Probe()
    nids_mesh = (1 + np.arange(nlength*(ntheta + 1))).reshape(nlength,
                                                              ntheta + 1)
    nids_mesh[:, -1] = nids_mesh[:, 0]
    nids = np.unique(nids_mesh)
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    zlin = np.linspace(0, L, nlength)
    thetalin = np.linspace(0, 2*np.pi - 2*np.pi/ntheta, ntheta)[::-1]
    zmesh, thetamesh = np.meshgrid(zlin, thetalin)
    ncoords = np.vstack((R*np.cos(thetamesh.T).ravel(),
                         R*np.sin(thetamesh.T).ravel(), zmesh.T.ravel())).T
    x = ncoords.ravel()
    z = ncoords[:, 2]
    n1s = nids_mesh[:-1, :-1].ravel()
    n2s = nids_mesh[1:, :-1].ravel()
    n3s = nids_mesh[1:, 1:].ravel()
    n4s = nids_mesh[:-1, 1:].ravel()
    ne = len(n1s)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=DOUBLE)
    N = DOF*nlength*ntheta
    for i, (n1, n2, n3, n4) in enumerate(zip(n1s, n2s, n3s, n4s)):
        quad = Quad4(probe)
        quad.n1, quad.n2, quad.n3, quad.n4 = n1, n2, n3, n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        quad.init_k_KG = i*data.KG_SPARSE_SIZE
        quad.update_rotation_matrix(x, 0., 0., 1.)
        quad.update_probe_xe(x)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quad.update_KG_given_stress(0, 0, -1000/b, KGr, KGc, KGv)
    T = _cylindrical_transformation(ncoords)
    KC0 = (T.T @ coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)) @ T).tocsc()
    KG = (T.T @ coo_matrix((KGv, (KGr, KGc)), shape=(N, N)) @ T).tocsc()
    bk = np.zeros(N, dtype=bool)
    edges = np.isclose(z, 0) | np.isclose(z, L)
    bk[0::DOF][edges] = True
    bk[1::DOF][edges] = True
    bk[4::DOF][edges] = True
    bk[2::DOF][np.flatnonzero(np.isclose(z, L/2))[0]] = True
    bu = ~bk
    return KC0[bu, :][:, bu], KG[bu, :][:, bu]


def test_repeated_load_multipliers_cylinder():
    r"""The torsional buckling loads of a cylinder come in pairs equal up to
    the round-off, the modes `\cos n\theta` and `\sin n\theta` of the
    rotationally symmetric mesh; both copies of the lowest pair are
    returned, without the dense fallback, and the inertia of `\boldsymbol{K} +
    s \boldsymbol{K}_G` just below and above them confirms that exactly two load
    multipliers lie there

    The cylinder Z33 of ``test_quad4_linear_buckling_cylinder_Nxy.py`` with
    32 elements around the circumference, ``n = 1919`` degrees-of-freedom:
    measured with SciPy 1.17.1 and Intel MKL 2025.0.0, ``eigsh()`` in Cayley
    mode with ``num_eigvalues=2`` returns 96.4816 and 100.4531, one copy of
    each of the two lowest pairs, the verification reports 1 load multiplier
    missing below 100.443, and the deflated pass of
    :func:`pyfe3d.solver.eigen._complete_repeated` finds the second copy of
    96.4816. Before it, with ``max_dense_size=0``, ``linear_buckling``
    raised a ``RuntimeError``; with 80 elements around the circumference,
    ``num_eigvalues`` of 2, 3 and 4 also raised for the 3680 non-null rows
    of ``KG``, more than the default ``max_dense_size=2000``. Whether the
    first call misses the copy depends on the rounding errors, so that
    other platforms may not need the deflated pass.
    """
    K, KG = torsion_cylinder_Z33(32)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=2,
                                           max_dense_size=0)
    np.testing.assert_allclose(eigvals[1], eigvals[0], rtol=1e-8)
    assert np.all(relative_residuals(K, KG, eigvals, eigvecs) < 1e-6)
    G = eigvecs.T @ K @ eigvecs
    assert abs(G[0, 1])/np.sqrt(G[0, 0]*G[1, 1]) < 1e-6
    # NOTE independent of the eigenvalue solvers, by Sylvester's law of
    #      inertia: no load multiplier below the pair, two up to it
    lam = eigvals[0]
    assert eigen._negative_pivots(K + lam*(1 - 1e-6)*KG) == 0
    assert eigen._negative_pivots(K + lam*(1 + 1e-6)*KG) == 2


def test_complete_repeated_stops():
    r"""The count of :func:`pyfe3d.solver.eigen._missing_counter` is null
    without eigenvalues or at a null shift, and the passes of
    :func:`pyfe3d.solver.eigen._complete_repeated` stop when no
    eigenpair can be added, the subspace being full, or when a pass keeps
    nothing; ``K = diag(1, ..., 6)``, ``M = I``"""
    n = 6
    K = diags(np.arange(1., n + 1)).tocsr()
    M = identity(n, format='csr')
    count = eigen._missing_counter(K, -M, lambda w: w, 3)
    assert count(np.zeros(0)) == 0
    assert count(np.array([0.])) == 0
    assert count(np.array([1., 3.])) == 1
    V = np.eye(n)[:, [0, 2]]
    w, vecs, num_added = eigen._complete_repeated(
        K, M, np.array([1., 3.]), V, -1., 0, lambda w: 1,
        lambda w: np.zeros(w.size, dtype=bool), 3)
    assert num_added == 0
    np.testing.assert_allclose(w, [1., 3.])
    V = np.eye(n)[:, :n - 2]
    w, vecs, num_added = eigen._complete_repeated(
        K, M, np.arange(1., n - 1), V, -1., 0, lambda w: 1, np.isfinite, 3)
    assert num_added == 0
    # NOTE the missing eigenvalue 2 is added
    V = np.eye(n)[:, [0, 2]]
    w, vecs, num_added = eigen._complete_repeated(
        K, M, np.array([1., 3.]), V, -1., 0, count, np.isfinite, 3)
    assert num_added == 1
    np.testing.assert_allclose(w, [1., 2., 3.], rtol=1e-12)


# NOTE stiffness matrices that are not positive definite


def test_K_not_positive_definite():
    r"""``linear_buckling`` requires a positive definite ``K``: an
    indefinite ``K``, ``diag(1, -2, 3, ...)``, and a singular one, the
    chain of unit springs without support, whose last pivot is exactly zero,
    raise a ``LinAlgError`` naming the analysis and the likely causes, with
    the sparse and the dense solvers"""
    n = 10
    KG = -identity(n, format='csr')
    K = diags(np.arange(1., n + 1)*np.where(np.arange(n) == 1, -1, 1)).tocsr()
    for sparse_solver in (True, False):
        with pytest.raises(LinAlgError, match=r'^Linear buckling analysis: '
                           r'the stiffness matrix K is not positive '
                           r'definite, its factorization has 1 negative '
                           r'pivot\(s\).*Likely causes: insufficient '
                           r'boundary conditions.*re-entrant'):
            linear_buckling(K, KG, sparse_solver=sparse_solver)
    chain = diags([-np.ones(n - 1), np.r_[1., 2*np.ones(n - 2), 1.],
                   -np.ones(n - 1)], [-1, 0, 1]).tocsr()
    with pytest.raises(LinAlgError, match='met a zero pivot, i.e. K is '
                       'singular or indefinite. Likely causes: insufficient '
                       'boundary conditions'):
        linear_buckling(chain, KG)
