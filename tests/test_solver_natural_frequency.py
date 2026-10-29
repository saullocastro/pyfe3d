r"""Tests for :func:`pyfe3d.solver.natural_frequency`: verification, dense
fallback, rigid-body modes and unsymmetric matrices

With analytic matrices and the Quad4 plate of ``test_solver.py``;
:func:`pyfe3d.solver.natural_frequency` returns `\omega_n`.

Ziegler's double pendulum: two rigid bars of length ``l``, rotational springs
``k`` at the base and at the middle hinge, masses ``2m`` at the middle hinge
and ``m`` at the tip, and a follower force ``P`` at the tip, tangential to
the upper bar. With the absolute rotations of the bars as
degrees-of-freedom:

.. math::

    \boldsymbol{M} = m l^2 \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
    \boldsymbol{K}(P) = \begin{bmatrix} 2k - P l & -k + P l \\ -k & k \end{bmatrix}

The stiffness matrix is unsymmetric and never singular, ``det K = k**2``,
and the system loses stability by flutter at ``P_f = (7/2 - sqrt(2)) k/l``,
the smallest root of the discriminant of ``det(K - omega**2 M) = 0`` in
``omega**2``, where two natural frequencies coalesce.

Reference: H. Ziegler. "Die Stabilitätskriterien der Elastomechanik".
Ingenieur-Archiv, 20(1), 49-56, 1952. https://doi.org/10.1007/BF00536796
"""
import sys
sys.path.append('..')

import warnings

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.linalg import block_diag, eigh, eig
from scipy.sparse import (csc_matrix, csr_matrix, diags, identity,
                          random as sprandom)

from pyfe3d.solver import natural_frequency, DenseFallbackWarning
from pyfe3d.solver import eigen

from test_solver import plate_matrices

K_SPRING = 3.
L_BAR = 0.5
MASS = 2.
P_FLUTTER = (3.5 - np.sqrt(2))*K_SPRING/L_BAR


def ziegler(P, extra=0):
    r"""``K(P)`` and ``M``, with ``extra`` decoupled oscillators such that
    the sparse solvers, limited to ``k < n - 1``, can be used"""
    k, l, m = K_SPRING, L_BAR, MASS
    K = np.array([[2*k - P*l, -k + P*l], [-k, k]])
    M = m*l**2*np.array([[3., 1.], [1., 1.]])
    if extra:
        rng = np.random.default_rng(0)
        K = block_diag(K, np.diag(1e3 + rng.random(extra)))
        M = block_diag(M, np.eye(extra))
    return K, M


def residuals(K, M, omegan, eigvecs):
    w2 = omegan**2
    return (np.linalg.norm(K @ eigvecs - (M @ eigvecs)*w2, axis=0)
            / np.linalg.norm(K @ eigvecs, axis=0))


def test_simple():
    n = 10
    k_vals = np.arange(1, n + 1, dtype=float)*100
    omegan, eigvecs = natural_frequency(csc_matrix(np.diag(k_vals)),
                                        csc_matrix(np.eye(n)),
                                        num_eigvalues=5)
    np.testing.assert_allclose(omegan, np.sqrt(k_vals[:5]), rtol=1e-10)
    assert eigvecs.shape == (n, 5)
    omegan, eigvecs = natural_frequency(csc_matrix(np.diag(k_vals)),
                                        csc_matrix(np.eye(n)),
                                        sparse_solver=False)
    np.testing.assert_allclose(omegan, np.sqrt(k_vals), rtol=1e-10)


def test_null_dofs():
    r"""Null rows and columns of ``K`` raise an error naming their
    degrees-of-freedom; those of ``M``, e.g. a lumped mass matrix without
    rotary inertia, give infinite frequencies, which are not returned"""
    n = 30
    null = [0, 6, 7]
    k_vals = np.arange(1, n + 1, dtype=float)*100
    m_vals = np.ones(n)
    m_vals[null] = 0.
    K = np.diag(k_vals)
    K[null, null] = 0.
    for kwargs in [{}, {'sparse_solver': False}, {'symmetric': False}]:
        with pytest.raises(LinAlgError, match=r'K is singular: null row or '
                           r'column at 3 degree\(s\)-of-freedom, of indices '
                           r'0, 6, 7\.'):
            natural_frequency(csc_matrix(K), diags(m_vals), num_eigvalues=3,
                              **kwargs)
    # NOTE stiffness at the massless degrees-of-freedom
    for sparse_solver in [True, False]:
        omegan, eigvecs = natural_frequency(csc_matrix(np.diag(k_vals)),
                                            diags(m_vals), num_eigvalues=3,
                                            sparse_solver=sparse_solver,
                                            max_dense_size=0)
        np.testing.assert_allclose(omegan[:3], np.sqrt([200., 300., 400.]),
                                   rtol=1e-10)
        assert eigvecs.shape[0] == n
        assert np.allclose(eigvecs[null, :3], 0.,
                           atol=1e-12*np.abs(eigvecs).max())


def freq_problem(symmetric, n=300, seed=0):
    rng = np.random.RandomState(seed)
    A = sprandom(n, n, density=0.02, random_state=rng)
    K = A @ A.T + diags(np.linspace(1., 100., n))
    if not symmetric:
        K = K + 1e-2*sprandom(n, n, density=0.02, random_state=rng)
    M = diags(1. + rng.rand(n))
    return csc_matrix(K), csc_matrix(M)


@pytest.mark.parametrize('num_eigvalues', [5, 25, 40])
@pytest.mark.parametrize('symmetric', [True, False])
def test_lowest_frequencies(symmetric, num_eigvalues):
    r"""The sparse solver, without dense fallback, returns the lowest
    frequencies of the full dense solution, also for many modes, which
    :func:`pyfe3d.solver.arpackutils.capped_eigsh` computes in several
    passes when ``ncv`` is capped"""
    K, M = freq_problem(symmetric)
    omegan, eigvecs = natural_frequency(K, M, num_eigvalues=num_eigvalues,
                                        max_dense_size=0)
    ref, _ = natural_frequency(K, M, sparse_solver=False,
                               symmetric=symmetric)
    assert omegan.shape == (num_eigvalues,)
    np.testing.assert_allclose(omegan, np.sort(ref.real)[:num_eigvalues],
                               rtol=1e-8)
    assert np.isrealobj(omegan)
    assert np.all(residuals(K, M, omegan, eigvecs) < 1e-8)


@pytest.mark.parametrize('sparse_solver', [True, False])
@pytest.mark.parametrize('symmetric', [True, False])
def test_sorted_with_the_eigenvectors(sparse_solver, symmetric):
    r"""The frequencies are sorted together with the eigenvectors, the
    rigid-body mode giving a zero frequency"""
    rng = np.random.RandomState(3)
    n = 12
    k = rng.permutation(np.arange(1, n + 1, dtype=float))*100
    K = np.diag(k)
    K[:2, :2] = 50.*np.array([[1., -1.], [-1., 1.]])
    M = np.eye(n)
    K, M = csc_matrix(K), csc_matrix(M)
    omegan, eigvecs = natural_frequency(K, M, num_eigvalues=6,
                                        sparse_solver=sparse_solver,
                                        symmetric=symmetric)
    assert omegan.shape[0] == eigvecs.shape[1]
    assert np.all(np.diff(omegan) >= 0)
    assert omegan[0] < 1e-6*omegan[1]
    res = np.linalg.norm(K @ eigvecs - (M @ eigvecs)*omegan**2, axis=0)
    assert np.all(res <= 1e-6*np.linalg.norm(K.toarray(), 2))


@pytest.mark.parametrize('symmetric', [True, False])
def test_rigid_body_modes(symmetric):
    r"""A free-free chain of springs has a rigid-body mode, `\omega_n^2 = 0`
    up to the round-off, which passes the residual check of the sparse
    solvers without the dense fallback, its residual being relative to the
    matrix-norm scale `(\|\boldsymbol{K}\|_F + |\omega_n^2| \|\boldsymbol{M}\|_F) \|\boldsymbol{u}\|`"""
    n = 30
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[0, 0] = K[-1, -1] = 100.
    K, M = csc_matrix(K), csc_matrix(np.eye(n))
    w2_ref = np.linalg.eigvalsh(K.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        omegan, eigvecs = natural_frequency(K, M, num_eigvalues=5,
                                            symmetric=symmetric,
                                            max_dense_size=0)
    assert omegan[0]**2 <= 1e-10*w2_ref[1]
    np.testing.assert_allclose(omegan[1:]**2, w2_ref[1:5], rtol=1e-8)
    res = eigen._relative_residuals(K, M, -omegan**2, eigvecs,
                                    rigid_rtol=eigen.RIGID_BODY_RTOL)
    assert np.all(res <= 1e-6)


def penalty_chains(n=20, kp=1e8, free=False):
    """Two chains of ``n`` unit springs, each fixed at its outer end unless
    ``free=True``, connected by a penalty spring ``kp``, which sets
    ``||K||_F``, and a unit mass matrix"""
    K = 2*np.eye(2*n) - np.eye(2*n, k=1) - np.eye(2*n, k=-1)
    K[n - 1, n] = K[n, n - 1] = -kp
    K[n - 1, n - 1] = K[n, n] = 1 + kp
    if free:
        K[0, 0] = K[-1, -1] = 1.
    return csc_matrix(K), csc_matrix(np.eye(2*n))


def test_check_freq_penalty():
    r"""Regression test: with a penalty spring setting `\|\boldsymbol{K}\|_F`, an
    eigenvalue `\omega_n^2` 0.01 % too low, relative residual 5e-5, is
    rejected with ``rtol=1e-6``. A floor of the denominator of 1e-8 times
    the matrix-norm scale `(\|\boldsymbol{K}\|_F + |\omega_n^2| \|\boldsymbol{M}\|_F) \|\boldsymbol{u}\|`, about
    170 times `\|\boldsymbol{K} \boldsymbol{u}\| + |\omega_n^2| \|\boldsymbol{M} \boldsymbol{u}\|` here, accepted it"""
    K, M = penalty_chains()
    w2, u = np.linalg.eigh(K.toarray())
    w2, u = w2[:1], u[:, :1]
    assert eigen._check_freq(K, M, w2, u, 1e-6, True) is None
    w2_low = w2*(1 - 1e-4)
    # the residual of the eigenpair of eigh(), up to 2e-3 of 5e-5 with
    # Apple's LAPACK, adds to the one of the perturbation
    np.testing.assert_allclose(
        eigen._relative_residuals(K, M, -w2_low, u,
                                  rigid_rtol=eigen.RIGID_BODY_RTOL),
        1e-4/(2 - 1e-4), rtol=1e-2)
    error = eigen._check_freq(K, M, w2_low, u, 1e-6, True)
    assert 'relative residual' in error


@pytest.mark.parametrize('symmetric', [True, False])
def test_rigid_body_mode_penalty(symmetric):
    r"""Two free-free chains of springs connected by a penalty spring: the
    rigid-body mode, whose `\|\boldsymbol{K} \boldsymbol{u}\| + |\omega_n^2| \|\boldsymbol{M} \boldsymbol{u}\|` is at the
    round-off of the matrix-norm scale, passes the residual check without
    the dense fallback, while the other eigenpairs keep the relative
    residual"""
    K, M = penalty_chains(kp=1e6, free=True)
    w2_ref = np.linalg.eigvalsh(K.toarray())
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        omegan, u = natural_frequency(K, M, num_eigvalues=5,
                                      symmetric=symmetric, max_dense_size=0)
    w2 = omegan**2
    # null up to the round-off of ||K||_F
    assert abs(w2[0]) <= eigen.RIGID_BODY_RTOL*eigen._frobenius(K)
    np.testing.assert_allclose(w2[1:], w2_ref[1:5], rtol=1e-8)
    Ku = K @ u
    Mu = M @ u
    scale = ((eigen._frobenius(K) + np.abs(w2)*eigen._frobenius(M))
             *np.linalg.norm(u, axis=0))
    d = np.linalg.norm(Ku, axis=0) + np.abs(w2)*np.linalg.norm(Mu, axis=0)
    rigid = d <= eigen.RIGID_BODY_RTOL*scale
    assert rigid.tolist() == [True, False, False, False, False]
    res = eigen._relative_residuals(K, M, -w2, u,
                                    rigid_rtol=eigen.RIGID_BODY_RTOL)
    assert np.all(res <= 1e-6)
    plain = np.linalg.norm(Ku - Mu*w2, axis=0)/d
    np.testing.assert_allclose(res[1:], plain[1:], rtol=1e-12)
    np.testing.assert_allclose(res[1:],
                               eigen._relative_residuals(K, M, -w2, u)[1:],
                               rtol=1e-12)


def beam_cantilever():
    r"""The cantilever of ``test_beamc_natural_freq_cantilever_pre_stress.py``
    saved in ``tests/data``: 49 ``BeamC`` elements of a steel beam of length
    3 m with a 0.05 x 0.05 m square cross-section, clamped at one end,
    ``KC0`` and ``M`` (``mtype=0``) of the free degrees-of-freedom, and
    ``KG`` of a unit axial compression at the tip"""
    import os
    from scipy.sparse import load_npz

    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    return [load_npz(os.path.join(data, 'beam_cantilever_%s.npz' % name)
                     ).tocsr() for name in ('K', 'KG', 'M')]


@pytest.mark.parametrize('precondition', [True, False])
@pytest.mark.parametrize('symmetric', [True, False])
def test_near_buckling(symmetric, precondition):
    r"""Regression test: the cantilever pre-stressed to 0.9999 of its
    buckling load has a nearly singular `\boldsymbol{K}`. Its two lowest eigenpairs,
    `\omega_n^2 = 0.0895`, a double eigenvalue of the square cross-section,
    have relative residuals of 2.3e-6 to 1.3e-5 with the sparse and the
    dense solvers, larger than the default ``check_rtol=1e-6``, but backward
    errors at the round-off, and pass the verification of the sparse solver
    without the dense fallback and without a larger ``check_rtol``"""
    from pyfe3d.solver import linear_buckling

    K0, KG, M = beam_cantilever()
    lambda_cr = linear_buckling(K0, KG, num_eigvalues=1)[0][0]
    K = (K0 + 0.9999*lambda_cr*KG).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        omegan, eigvecs = natural_frequency(K, M, num_eigvalues=3,
                                            symmetric=symmetric,
                                            precondition=precondition,
                                            max_dense_size=0)
    w2 = np.real(omegan)**2
    # NOTE M u = (1/omega**2) K u, accurate for the lowest omega**2
    mu = eigh(M.toarray(), K.toarray(), eigvals_only=True)
    w2_ref = np.sort(1/mu[mu > 0])[:3]
    np.testing.assert_allclose(w2, w2_ref, rtol=1e-4)
    assert np.all(w2[:2] < 1e-5*w2[2])
    # NOTE ||K u|| + omega**2 ||M u|| is about 5000 eps times the
    #      matrix-norm scale, which amplifies the relative residual
    u = np.real(eigvecs[:, :2])
    d = (np.linalg.norm(K @ u, axis=0)
         + w2[:2]*np.linalg.norm(M @ u, axis=0))
    scale = ((eigen._frobenius(K) + w2[:2]*eigen._frobenius(M))
             *np.linalg.norm(u, axis=0))
    assert np.all(d < 1e4*np.finfo(float).eps*scale)
    res = eigen._relative_residuals(K, M, -w2, np.real(eigvecs),
                                    rigid_rtol=eigen.RIGID_BODY_RTOL)
    be = eigen._backward_errors(K, M, -w2, np.real(eigvecs))
    assert np.all((res <= 1e-6) | (be <= eigen.BACKWARD_ERROR_RTOL))


def test_check_freq_backward_error():
    r"""The residual check of :func:`pyfe3d.solver.natural_frequency` on both
    sides of ``BACKWARD_ERROR_RTOL``, with eigenvalues perturbed from the
    dense eigenpairs: the lowest eigenpair of the cantilever pre-stressed to
    0.9999 of its buckling load, `\omega_n^2` 1e-5 too large, relative
    residual about 5e-6, backward error about 0.03 eps, is accepted; the
    lowest eigenpairs of the blade-stiffened plate with penalty connections
    of ``tests/data``, `\lambda` 0.1 % too low as the inaccurate eigenpairs
    of ``eigsh()``, relative residuals about 5e-4, backward errors of 16 eps
    and more, are rejected"""
    import os
    from scipy.sparse import load_npz

    K0, KG, M = beam_cantilever()
    mu, u = eigh(KG.toarray(), K0.toarray())
    lambda_cr = -1/mu[0]
    K = (K0 + 0.9999*lambda_cr*KG).tocsr()
    mu, u = eigh(M.toarray(), K.toarray())
    w2 = 1/mu[-1:]*(1 + 1e-5)
    u = u[:, -1:]
    res = eigen._relative_residuals(K, M, -w2, u,
                                    rigid_rtol=eigen.RIGID_BODY_RTOL)
    be = eigen._backward_errors(K, M, -w2, u)
    assert np.all(res > 1e-6)
    assert np.all(be <= 0.1*eigen.BACKWARD_ERROR_RTOL)
    assert eigen._check_freq(K, M, w2, u, 1e-6, True) is None

    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    K = load_npz(os.path.join(data, 'blade_stiffened_penalty_K.npz')).tocsr()
    KG = load_npz(os.path.join(data,
                               'blade_stiffened_penalty_KG.npz')).tocsr()
    mu, u = eigen._eigh_condensed(K, KG, 4)
    lam = -1/mu
    # NOTE (K + lam KG) u = 0 checked as (K - omega**2 M) u = 0
    assert eigen._check_freq(K, -KG, lam, u, 1e-6, False) is None
    lam = lam*(1 - 1e-3)
    res = eigen._relative_residuals(K, KG, lam, u)
    be = eigen._backward_errors(K, KG, lam, u)
    assert np.all(res > 1e-4)
    assert np.all(be >= 10*eigen.BACKWARD_ERROR_RTOL)
    error = eigen._check_freq(K, -KG, lam, u, 1e-6, False)
    assert 'relative residual' in error and 'backward error' in error


def test_quad4_plate_prestress():
    r"""The Quad4 plate of ``test_solver.py`` under a compressive prestress
    `\lambda \boldsymbol{K}_G`: the frequencies decrease with `\lambda`, and the
    squared lowest one vanishes linearly at the critical load multiplier of
    :func:`pyfe3d.solver.linear_buckling`; with ``symmetric=True`` the
    eigenvectors are normalized with respect to the mass matrix"""
    from pyfe3d.solver import linear_buckling
    KC0uu, KGuu, Muu = plate_matrices(Nxx=-1., with_mass=True)
    lambda_1 = linear_buckling(KC0uu, KGuu, num_eigvalues=2)[0][0]
    omega2 = []
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        for frac in [0., 0.5, 0.9, 0.99, 0.999]:
            omegan, eigvecs = natural_frequency(KC0uu + frac*lambda_1*KGuu,
                                                Muu, num_eigvalues=4,
                                                symmetric=True)
            assert np.all(np.diff(omegan) >= 0)
            np.testing.assert_allclose(eigvecs.T @ Muu @ eigvecs, np.eye(4),
                                       atol=1e-8)
            omega2.append(omegan**2)
    omega2 = np.array(omega2)
    assert np.all(np.diff(omega2, axis=0) <= 1e-12*omega2[0, 0])
    np.testing.assert_allclose(omega2[-2, 0]/omega2[-1, 0], 10., rtol=1e-2)


@pytest.mark.parametrize('sparse_solver,extra', [(False, 0), (True, 10)])
def test_ziegler_flutter(sparse_solver, extra):
    r"""The general solvers give real frequencies below the flutter load and
    complex ones above it; the flutter load found by bisection is the
    analytical one"""
    kw = dict(sparse_solver=sparse_solver, num_eigvalues=4, symmetric=False)
    below = natural_frequency(*map(csr_matrix, ziegler(0.9*P_FLUTTER, extra)),
                              **kw)[0]
    above = natural_frequency(*map(csr_matrix, ziegler(1.1*P_FLUTTER, extra)),
                              **kw)[0]
    assert np.isrealobj(below)
    assert np.iscomplexobj(above)
    assert np.max(np.abs(above.imag)) > 1e-3*np.abs(above).max()
    a, b = 0.9*P_FLUTTER, 1.1*P_FLUTTER
    for _ in range(40):
        c = 0.5*(a + b)
        omegan = natural_frequency(*map(csr_matrix, ziegler(c, extra)),
                                   **kw)[0]
        if np.iscomplexobj(omegan):
            b = c
        else:
            a = c
    assert np.isclose(0.5*(a + b), P_FLUTTER, rtol=1e-4)


def test_ziegler_frequencies():
    r"""Below the flutter load the frequencies are the roots of
    ``det(K - omega**2 M) = 0``"""
    K, M = ziegler(0.5*P_FLUTTER)
    omegan = natural_frequency(csr_matrix(K), csr_matrix(M),
                               sparse_solver=False)[0]
    a = np.linalg.det(M)
    b = -(K[0, 0]*M[1, 1] + K[1, 1]*M[0, 0] - K[0, 1]*M[1, 0]
          - K[1, 0]*M[0, 1])
    c = np.linalg.det(K)
    w2 = np.sort(np.roots([a, b, c]).real)
    assert np.allclose(omegan**2, w2, rtol=1e-10)


def test_symmetric_option():
    r"""``symmetric=True`` (eigsh, eigh) against the general solvers"""
    rng = np.random.default_rng(1)
    n = 30
    X = rng.standard_normal((n, n))
    K = csr_matrix(X @ X.T + n*np.eye(n))
    M = csr_matrix(np.diag(1 + rng.random(n)))
    for sparse_solver in (True, False):
        w_gen = natural_frequency(K, M, sparse_solver=sparse_solver,
                                  num_eigvalues=6, symmetric=False)[0]
        w_sym = natural_frequency(K, M, sparse_solver=sparse_solver,
                                  num_eigvalues=6, symmetric=True)[0]
        assert np.allclose(w_gen[:5], w_sym[:5], rtol=1e-8)


def test_unsymmetric_against_dense_eig():
    K, M = freq_problem(False, n=60, seed=2)
    omegan, eigvecs = natural_frequency(K, M, num_eigvalues=6)
    w2 = eig(K.toarray(), M.toarray(), right=False)
    w2 = np.sort(w2.real)
    np.testing.assert_allclose(omegan**2, w2[:6], rtol=1e-8)


def spy(monkeypatch, names):
    calls = []
    for name in names:
        func = getattr(eigen, name)

        def wrapper(*args, _name=name, _func=func, **kwargs):
            calls.append(_name)
            return _func(*args, **kwargs)

        monkeypatch.setattr(eigen, name, wrapper)
    return calls


@pytest.mark.parametrize('max_dense_size', [0, 2000, 10**9])
@pytest.mark.parametrize('symmetric', [True, False])
def test_sparse_solver_runs_first(monkeypatch, symmetric, max_dense_size):
    K, M = freq_problem(symmetric, n=120)
    calls = spy(monkeypatch, ['capped_eigsh', 'eigs', '_dense_freq'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        natural_frequency(K, M, symmetric=symmetric,
                          max_dense_size=max_dense_size)
    assert calls == ['capped_eigsh' if symmetric else 'eigs']


@pytest.mark.parametrize('symmetric', [True, False])
def test_explicit_dense(monkeypatch, symmetric):
    K, M = freq_problem(symmetric, n=120)
    calls = spy(monkeypatch, ['capped_eigsh', 'eigs', '_dense_freq'])
    natural_frequency(K, M, symmetric=symmetric, sparse_solver=False)
    assert calls == ['_dense_freq']


@pytest.mark.parametrize('symmetric', [True, False])
def test_dense_fallback(monkeypatch, symmetric):
    r"""A sparse solver returning wrong eigenpairs: the dense fallback
    returns the lowest frequencies with a warning, a RuntimeError is raised
    without it"""
    K, M = freq_problem(symmetric, n=120)
    ref, _ = natural_frequency(K, M, symmetric=symmetric, num_eigvalues=10)
    name = 'capped_eigsh' if symmetric else 'eigs'

    def wrong(A, k, *args, **kwargs):
        rng = np.random.RandomState(0)
        return rng.rand(k) + 1., rng.randn(A.shape[0], k)

    monkeypatch.setattr(eigen, name, wrong)
    with pytest.warns(DenseFallbackWarning, match='relative residual'):
        omegan, _ = natural_frequency(K, M, symmetric=symmetric,
                                      num_eigvalues=10)
    np.testing.assert_allclose(omegan, ref, rtol=1e-8)
    for max_dense_size in (0, 50):
        with pytest.raises(RuntimeError, match='Attempted:'):
            natural_frequency(K, M, symmetric=symmetric, num_eigvalues=10,
                              max_dense_size=max_dense_size)


def test_missing_mode_detected(monkeypatch):
    r"""The inertia of ``K - s M`` detects a missing lowest mode, which a
    deflated pass completes without the dense fallback; without the
    deflated passes, ``MAX_COMPLETION_PASSES = 0``, the dense fallback is
    used, or a ``RuntimeError`` raised"""
    K, M = freq_problem(True, n=120)
    sparse = eigen.capped_eigsh

    def missing_lowest(A, k, *args, **kwargs):
        w2, eigvecs = sparse(A=A, k=k + 1, *args, **kwargs)
        order = np.argsort(w2)
        return w2[order[1:]], eigvecs[:, order[1:]]

    monkeypatch.setattr(eigen, 'capped_eigsh', missing_lowest)
    w_ref = eigh(K.toarray(), M.toarray(), eigvals_only=True)[:10]
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        omegan, eigvecs = natural_frequency(K, M, symmetric=True,
                                            num_eigvalues=10)
    np.testing.assert_allclose(omegan**2, w_ref, rtol=1e-10)
    assert np.all(residuals(K, M, omegan, eigvecs) < 1e-8)
    monkeypatch.setattr(eigen, 'MAX_COMPLETION_PASSES', 0)
    with pytest.raises(RuntimeError, match='eigenvalues omega'):
        natural_frequency(K, M, symmetric=True, num_eigvalues=10,
                          max_dense_size=0)
    with pytest.warns(DenseFallbackWarning, match='eigenvalues omega'):
        omegan, _ = natural_frequency(K, M, symmetric=True, num_eigvalues=10)
    np.testing.assert_allclose(omegan**2, w_ref, rtol=1e-10)


def test_dense_symmetric_lowest_frequencies():
    r"""The dense solver of symmetric matrices solves `\boldsymbol{M} \boldsymbol{u} =
    (1/\omega_n^2) \boldsymbol{K} \boldsymbol{u}`: for the Quad4 plate of ``test_solver.py``
    (726 dofs) with a lumped, positive definite mass matrix, the 10 lowest
    frequencies have relative residuals of about 1.6e-10, against about
    1.4e-8 for :func:`scipy.linalg.eigh` on `\boldsymbol{K} \boldsymbol{u} = \omega_n^2
    \boldsymbol{M} \boldsymbol{u}` (SciPy 1.17.1, Intel MKL 2025.0.0); the frequencies agree"""
    K, _, _ = plate_matrices(Nxx=-1., with_mass=True)
    rng = np.random.RandomState(3)
    M = diags(1. + rng.rand(K.shape[0])).tocsr()
    omegan, eigvecs = natural_frequency(K, M, sparse_solver=False,
                                        symmetric=True)
    assert omegan.shape == (K.shape[0],)
    w2, eigvecs = omegan[:10]**2, eigvecs[:, :10]
    assert np.all(residuals(K, M, np.sqrt(w2), eigvecs) < 1e-9)
    w2_ref = eigh(K.toarray(), M.toarray(), eigvals_only=True)[:10]
    np.testing.assert_allclose(w2, w2_ref, rtol=1e-8)



def test_check_freq_without_eigenpairs():
    K = diags([1., 2., 3.]).tocsr()
    M = diags([1., 1., 1.]).tocsr()
    assert (eigen._check_freq(K, M, np.zeros(0), np.zeros((3, 0)), 1e-8,
                              True) == 'no eigenpair was found')
    assert (eigen._check_freq(K, M, np.array([np.inf]),
                              np.array([[1.], [0.], [0.]]), 1e-8, True)
            == 'no finite eigenvalue was found')
    assert eigen._check_freq(K, M, np.array([1.]),
                             np.array([[1.], [0.], [0.]]), 1e-8, True) is None


def test_dense_freq_indefinite_K():
    r"""With an indefinite ``K`` the symmetric dense solver solves
    `\boldsymbol{K} \boldsymbol{u} = \omega^2 \boldsymbol{M} \boldsymbol{u}` directly, the lowest ``k`` eigenvalues
    first, a negative one included"""
    K = diags([3., -1., 2.]).tocsr()
    M = diags([1., 1., 2.]).tocsr()
    w2, eigvecs = eigen._dense_freq(K, M, 2, 0., True)
    np.testing.assert_allclose(w2, [-1., 1.])
    np.testing.assert_allclose(np.abs(eigvecs), [[0., 0.], [1., 0.],
                                                 [0., 1/np.sqrt(2)]],
                               atol=1e-14)
    w2, eigvecs = eigen._dense_freq(K, M, None, 0., True)
    np.testing.assert_allclose(w2, [-1., 1., 3.])


def test_K_not_positive_definite_message():
    r"""When the solvers fail and ``K`` is not positive definite, the
    ``RuntimeError`` starts with that diagnosis and the likely causes; an
    indefinite ``K`` is otherwise accepted, e.g. by the dense fallback,
    since a preload beyond the buckling load gives negative `\omega_n^2`
    and null frequencies

    ``K = diag(1, ..., 60)`` with ``-1e6`` as its sixth entry and ``M = I``:
    the shift-invert mode about `\omega_n^2 = -1` finds the eigenvalues 1 to
    4, nearer the shift than `-10^6`, which the inertia check finds
    missing."""
    d = np.arange(1., 61.)
    d[5] = -1e6
    K = diags(d).tocsr()
    M = identity(60, format='csr')
    with pytest.raises(RuntimeError, match=r'^Frequency analysis: the '
                       r'stiffness matrix K is not positive definite, its '
                       r'factorization has 1 negative pivot\(s\).*Likely '
                       r'causes: insufficient boundary conditions.*'
                       r're-entrant.*preload beyond the buckling load; '
                       r'negative eigenvalues omega\*\*2 are expected only '
                       r'in the latter case\. Frequency analysis failed.*'
                       r'Attempted: eigsh\(\) in shift-invert mode: '):
        natural_frequency(K, M, num_eigvalues=4, max_dense_size=0)
    with pytest.warns(DenseFallbackWarning):
        omegan, _ = natural_frequency(K, M, num_eigvalues=4)
    np.testing.assert_allclose(omegan, [0., 1., np.sqrt(2.), np.sqrt(3.)])


def reentrant_plate():
    r"""``KC0uu``, ``Muu`` and ``KGuu`` of the hard simply supported plate of
    ``test_quad4r_natural_freq_distorted.py``, 40x50 ``Quad4R`` elements,
    with the former distortion of that test, the same random fraction, up
    to 0.4 of the element size, along `x` and `y`, which makes elements
    re-entrant, and the number of re-entrant elements"""
    from scipy.sparse import coo_matrix
    from pyfe3d.shellprop_utils import isotropic_plate
    from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
    nx, ny = 41, 51
    a, b, h = 0.3, 0.5, 0.01
    data = Quad4RData()
    probe = Quad4RProbe()
    xmesh, ymesh = np.meshgrid(np.linspace(0, a, nx), np.linspace(0, b, ny))
    ncoords = np.vstack((xmesh.T.ravel(), ymesh.T.ravel(),
                         np.zeros(nx*ny))).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    edge = (np.isclose(x, 0) | np.isclose(x, a) | np.isclose(y, 0)
            | np.isclose(y, b))
    np.random.seed(20)
    rdm = -1 + 2*np.random.rand(np.count_nonzero(~edge))
    x[~edge] += a/(nx - 1)*rdm*0.4
    y[~edge] += b/(ny - 1)*rdm*0.4
    xflat = ncoords.ravel()
    nids_mesh = np.arange(nx*ny).reshape(nx, ny)
    n1s = nids_mesh[:-1, :-1].ravel()
    n2s = nids_mesh[1:, :-1].ravel()
    n3s = nids_mesh[1:, 1:].ravel()
    n4s = nids_mesh[:-1, 1:].ravel()
    ne = n1s.size
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*ne, dtype=DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*ne, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*ne, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*ne, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*ne, dtype=DOUBLE)
    prop = isotropic_plate(thickness=h, E=203.e9, nu=0.33, rho=7.83e3)
    num_reentrant = 0
    for i, nodes in enumerate(zip(n1s, n2s, n3s, n4s)):
        r = ncoords[list(nodes)]
        if any(np.cross(r[(j + 1) % 4] - r[j],
                        r[(j + 2) % 4] - r[(j + 1) % 4])[2] <= 0
               for j in range(4)):
            num_reentrant += 1
        quad = Quad4R(probe)
        quad.n1, quad.n2, quad.n3, quad.n4 = nodes
        quad.c1, quad.c2, quad.c3, quad.c4 = [DOF*n for n in nodes]
        quad.init_k_KC0 = i*data.KC0_SPARSE_SIZE
        quad.init_k_M = i*data.M_SPARSE_SIZE
        quad.init_k_KG = i*data.KG_SPARSE_SIZE
        quad.K6ROT = 100.
        quad.update_rotation_matrix(xflat)
        quad.update_probe_xe(xflat)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quad.update_M(Mr, Mc, Mv, prop, mtype=2)
        quad.update_KG_given_stress(-1., -1., 0., KGr, KGc, KGv)
    N = DOF*nx*ny
    bk = np.zeros(N, dtype=bool)
    bk[0::DOF] = edge
    bk[1::DOF] = edge
    bk[2::DOF] = edge
    bk[3::DOF] = np.isclose(x, 0.) | np.isclose(x, a)
    bk[4::DOF] = np.isclose(y, 0.) | np.isclose(y, b)
    bu = ~bk
    out = []
    for r, c, v in ((KC0r, KC0c, KC0v), (Mr, Mc, Mv), (KGr, KGc, KGv)):
        A = coo_matrix((v, (r, c)), shape=(N, N)).tocsc()
        out.append(A[bu, :][:, bu])
    return out + [num_reentrant]


def test_reentrant_elements():
    r"""The re-entrant elements of the former distortion of
    ``test_quad4r_natural_freq_distorted.py`` make the stiffness matrix of
    the 40x50 mesh indefinite, measured with 119 re-entrant elements of 2000
    and 3 negative pivots: the frequency analysis, for which the solvers
    fail, the linear buckling analysis and the linear static analysis raise
    errors that say so, with the likely causes, instead of the former
    message of the frequency analysis, "Frequency analysis failed, no
    eigenvalue solver passed the verification. Attempted: eigsh() in
    shift-invert mode: 6 eigenvalues omega**2 lower than 5.56422e+07 exist,
    3 were found; eigh(): dense fallback skipped, size 11822 >
    max_dense_size=2000. Try sparse_solver=False or a larger
    max_dense_size.", and the former results of the two others"""
    from pyfe3d.solver import linear_buckling, linear_static
    K, M, KG, num_reentrant = reentrant_plate()
    assert num_reentrant > 0
    count = eigen._negative_pivots(K)
    assert count > 0
    causes = (r'its factorization has %d negative pivot\(s\).*Likely '
              r'causes: insufficient boundary conditions.*mechanism.*'
              r're-entrant'
              % count)
    with pytest.raises(RuntimeError, match=r'^Frequency analysis: the '
                       r'stiffness matrix K is not positive definite, '
                       + causes):
        natural_frequency(K, M, num_eigvalues=4)
    with pytest.raises(LinAlgError, match=r'^Linear buckling analysis: the '
                       r'stiffness matrix K is not positive definite, '
                       + causes):
        linear_buckling(K, KG, num_eigvalues=4)
    with pytest.raises(LinAlgError, match=r'^Linear static analysis: the '
                       r'stiffness matrix K is not positive definite, '
                       + causes):
        linear_static(K, np.ones(K.shape[0]))
