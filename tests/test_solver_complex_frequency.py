r"""Tests for :func:`pyfe3d.solver.complex_frequency`: damped free vibrations,
`(\lambda^2 \boldsymbol{M} + \lambda \boldsymbol{C} + \boldsymbol{K}) \boldsymbol{u} = \boldsymbol{0}`

A single degree-of-freedom `m \lambda^2 + c \lambda + k = 0` has the roots
`\lambda = -\zeta \omega_n \pm i \omega_n \sqrt{1 - \zeta^2}`, with `\omega_n
= \sqrt{k/m}` and `\zeta = c/(2 \sqrt{k m})`, for `\zeta < 1`, and the real
roots `\lambda = -\zeta \omega_n \pm \omega_n \sqrt{\zeta^2 - 1}` for `\zeta
> 1`. With proportional (Rayleigh) damping, `\boldsymbol{C} = \alpha \boldsymbol{M} + \beta \boldsymbol{K}`,
each undamped mode `\omega_j` is a single degree-of-freedom with `\zeta_j =
\alpha/(2 \omega_j) + \beta \omega_j/2`. The other cases are compared with
:func:`scipy.linalg.eig` on the second companion linearization ([1]_,
Section 3.4, Eq. (3.7) with `\boldsymbol{N} = \boldsymbol{I}`, p. 253),

.. math::

    \begin{bmatrix} -\boldsymbol{K} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{I} \end{bmatrix} -
    \lambda \begin{bmatrix} \boldsymbol{C} & \boldsymbol{M} \\ \boldsymbol{I} & \boldsymbol{0} \end{bmatrix}

independent of the first companion form used by the solver.

.. [1] Tisseur, F. and Meerbergen, K., The quadratic eigenvalue problem,
   SIAM Review, 43(2), 235-286, 2001.
   https://doi.org/10.1137/S0036144500381988
"""
import sys
sys.path.append('..')

import warnings

import numpy as np
import pytest
from numpy.linalg import LinAlgError
from scipy.linalg import eig
from scipy.sparse import csr_matrix, diags, random as sprandom

from pyfe3d.solver import (complex_frequency, natural_frequency,
                           DenseFallbackWarning)
from pyfe3d.solver import eigen

from test_solver import plate_matrices


def sdof_roots(k, c, m):
    r"""The two roots of `m \lambda^2 + c \lambda + k = 0`, the one with
    the larger imaginary part, then the larger real part, first"""
    omegan = np.sqrt(k/m)
    zeta = c/(2*np.sqrt(k*m))
    if zeta < 1:
        r = omegan*np.sqrt(1 - zeta**2)
        return -zeta*omegan + 1j*r, -zeta*omegan - 1j*r
    r = omegan*np.sqrt(zeta**2 - 1)
    return -zeta*omegan + r + 0j, -zeta*omegan - r + 0j


def reference(K, C, M):
    r"""All the finite eigenvalues with `Im(\lambda) \ge 0`, from the second
    companion linearization"""
    K, C, M = (np.asarray(csr_matrix(A).toarray()) for A in (K, C, M))
    n = K.shape[0]
    I, Z = np.eye(n), np.zeros((n, n))
    w = eig(np.block([[-K, Z], [Z, I]]), np.block([[C, M], [I, Z]]),
            right=False)
    w = w[np.isfinite(w)]
    return w[w.imag >= -1e-9*np.abs(w)]


def nearest(ref, sigma, k):
    r"""The ``k`` eigenvalues of ``ref`` nearest ``sigma``, sorted as
    :func:`pyfe3d.solver.complex_frequency` sorts them"""
    ref = ref[np.argsort(np.abs(ref - sigma), kind='stable')[:k]]
    return ref[np.lexsort((np.abs(ref), np.round(ref.imag, 8)))]


def residuals(K, C, M, lam, U):
    return (np.linalg.norm(K @ U + (C @ U)*lam + (M @ U)*lam**2, axis=0)
            / np.linalg.norm(K @ U, axis=0))


def chain(n, k=100., m=2.):
    r"""Chain of ``n`` masses ``m`` and ``n + 1`` springs ``k``, fixed at
    both ends, with `\omega_j = 2 \sqrt{k/m} \sin(j \pi/(2 (n + 1)))` and the
    modes `\sin(i j \pi/(n + 1))`"""
    K = k*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    return csr_matrix(K), csr_matrix(m*np.eye(n))


def test_sdof_damped_oscillator():
    r"""Decoupled single degrees-of-freedom, light to heavy and overdamped:
    the analytical roots, one per complex conjugate pair, the real roots
    of the overdamped one first; a single degree-of-freedom with the dense
    fallback, ARPACK requiring at least 2 degrees-of-freedom"""
    k = np.array([1., 4., 9., 16., 25., 36., 49., 64.])*100
    m = 2.*np.ones_like(k)
    zeta = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 2., 0.02])
    c = 2*zeta*np.sqrt(k*m)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(diags(k), diags(c), diags(m),
                                   num_eigvalues=8)
    assert lam.shape == (8,) and U.shape == (8, 8)
    expected = []
    for ki, ci, mi in zip(k, c, m):
        expected.extend(sdof_roots(ki, ci, mi))
    expected = np.array(expected)
    expected = nearest(expected[expected.imag >= 0], 1., 8)
    np.testing.assert_allclose(lam, expected, rtol=1e-10)
    # the overdamped root nearest the shift is real and first
    assert lam[0].imag == 0 and lam[0].real < 0
    assert np.all(np.diff(lam.imag) >= 0)
    omegad = np.abs(lam.imag)
    zeta_found = -lam.real/np.abs(lam)
    for z, w, ki, mi in zip(zeta, np.sqrt(k/m), k, m):
        if z < 1:
            i = np.argmin(np.abs(omegad - w*np.sqrt(1 - z**2)))
            assert np.isclose(zeta_found[i], z, rtol=1e-10)
    # one eigenvector per degree-of-freedom, unit norm, real positive pivot
    np.testing.assert_allclose(np.abs(U), np.abs(U).round(), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(U, axis=0), 1., rtol=1e-12)
    # all the roots with the dense solver
    lam, _ = complex_frequency(diags(k), diags(c), diags(m),
                               sparse_solver=False)
    assert lam.size == 9
    for v in zip(k, c, m):
        for root in sdof_roots(*v):
            if root.imag >= 0:
                assert np.min(np.abs(lam - root)) <= 1e-10*abs(root)
    with pytest.warns(DenseFallbackWarning, match='eigs'):
        lam, U = complex_frequency(diags([400.]), diags([4.]), diags([1.]),
                                   num_eigvalues=1)
    np.testing.assert_allclose(lam, [sdof_roots(400., 4., 1.)[0]],
                               rtol=1e-12)
    np.testing.assert_allclose(U, [[1.]])


def test_rayleigh_damping_chain():
    r"""Rayleigh damping of a chain of masses: the analytical modal
    eigenvalues and modes"""
    n, k, m = 40, 100., 2.
    K, M = chain(n, k, m)
    alpha, beta = 0.2, 2e-3
    C = alpha*M + beta*K
    j = np.arange(1, 7)
    omega = 2*np.sqrt(k/m)*np.sin(j*np.pi/(2*(n + 1)))
    zeta = alpha/(2*omega) + beta*omega/2
    lam_ref = -zeta*omega + 1j*omega*np.sqrt(1 - zeta**2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(K, C, M, num_eigvalues=6)
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-10)
    np.testing.assert_allclose(-lam.real/np.abs(lam), zeta, rtol=1e-10)
    np.testing.assert_allclose(np.abs(lam), omega, rtol=1e-10)
    i = np.arange(1, n + 1)
    for col, jj in enumerate(j):
        mode = np.sin(i*jj*np.pi/(n + 1))
        mode /= np.linalg.norm(mode)
        np.testing.assert_allclose(np.abs(mode @ U[:, col]), 1., rtol=1e-10)
    assert np.all(residuals(K, C, M, lam, U) < 1e-10)


def nonproportional(n=60, seed=0, skew=0.):
    rng = np.random.RandomState(seed)
    A = sprandom(n, n, density=0.05, random_state=rng)
    K = A @ A.T + diags(np.linspace(1., 100., n))
    M = diags(1. + rng.rand(n))
    # NOTE dashpots at a few degrees-of-freedom: non-proportional
    c = np.zeros(n)
    c[rng.choice(n, 5, replace=False)] = 2. + rng.rand(5)
    C = diags(c) + 0.01*K
    if skew:
        G = sprandom(n, n, density=0.05, random_state=rng)
        C = C + skew*(G - G.T)
    return csr_matrix(K), csr_matrix(C), csr_matrix(M)


@pytest.mark.parametrize('skew', [0., 0.5])
def test_nonproportional_against_dense(skew):
    r"""Non-proportional damping, symmetric or with a skew-symmetric
    (gyroscopic) part: the eigenvalues nearest the shift of the sparse
    solver, without fallback, the dense solver of the solver and the second
    companion linearization agree"""
    K, C, M = nonproportional(skew=skew)
    assert eigen.is_symmetric(C) == (skew == 0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(K, C, M, num_eigvalues=10,
                                   max_dense_size=0)
    ref = reference(K, C, M)
    np.testing.assert_allclose(lam, nearest(ref, 1., 10), rtol=1e-9)
    assert np.all(residuals(K, C, M, lam, U) < 1e-9)
    lam_dense, _ = complex_frequency(K, C, M, sparse_solver=False)
    assert lam_dense.size == ref.size
    np.testing.assert_allclose(lam_dense, nearest(ref, 1., ref.size),
                               rtol=1e-8)


def test_gyroscopic_undamped():
    r"""Symmetric positive definite `\boldsymbol{K}` and `\boldsymbol{M}` with a skew-symmetric
    `\boldsymbol{C}` (gyroscopic, without damping): the eigenvalues are purely
    imaginary"""
    K, _, M = nonproportional(seed=3)
    G = sprandom(60, 60, density=0.05, random_state=np.random.RandomState(4))
    G = csr_matrix(G - G.T)
    lam, U = complex_frequency(K, G, M, num_eigvalues=8)
    assert np.all(np.abs(lam.real) <= 1e-9*np.abs(lam))
    np.testing.assert_allclose(lam, nearest(reference(K, G, M), 1., 8),
                               rtol=1e-9)


def test_null_damping_natural_frequency():
    r"""With `\boldsymbol{C} = \boldsymbol{0}`, `\lambda = i \omega_n` and the modes of
    :func:`pyfe3d.solver.natural_frequency`, for the Quad4 plate of
    ``test_solver.py``"""
    K, _, M = plate_matrices(Nxx=-1., with_mass=True)
    C = csr_matrix(K.shape)
    omegan, eigvecs = natural_frequency(K, M, num_eigvalues=6)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(K, C, M, num_eigvalues=6)
    np.testing.assert_allclose(lam.imag, omegan, rtol=1e-8)
    assert np.all(np.abs(lam.real) <= 1e-8*np.abs(lam))
    for i in range(6):
        u = eigvecs[:, i]/np.linalg.norm(eigvecs[:, i])
        # NOTE double frequencies have the same span, not the same vectors
        if (np.isclose(omegan[i], omegan[i - 1], rtol=1e-6) if i else False) \
                or (np.isclose(omegan[i], omegan[i + 1], rtol=1e-6)
                    if i < 5 else False):
            continue
        assert np.isclose(abs(u @ U[:, i]), 1., rtol=1e-8)
        # NOTE real up to the round-off, the pivot being real
        assert np.linalg.norm(U[:, i].imag) <= 1e-8


@pytest.mark.parametrize('damped', [False, True])
def test_rigid_body_modes(damped):
    r"""A free-free chain: the rigid-body mode gives `\lambda = 0`, a double
    eigenvalue without damping, returned as two eigenvalues null up to the
    round-off, which pass the residual check relative to the matrix-norm
    scale, without the dense fallback; with damping `\lambda = 0` and
    `\lambda = -c/m`"""
    n = 30
    K = 100.*(2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    K[0, 0] = K[-1, -1] = 100.
    K, M = csr_matrix(K), csr_matrix(np.eye(n))
    C = csr_matrix(0.5*np.eye(n) if damped else (n, n))
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(K, C, M, num_eigvalues=5)
    w2 = np.linalg.eigvalsh(K.toarray())
    scale = np.sqrt(w2[1])
    if damped:
        assert abs(lam[0]) <= 1e-12*scale
        assert np.isclose(lam[1], -0.5, rtol=1e-10)
        lam_ref = -0.25 + 1j*np.sqrt(w2[1:4] - 0.25**2)
    else:
        assert np.all(np.abs(lam[:2]) <= 1e-6*scale)
        lam_ref = 1j*np.sqrt(w2[1:4])
    np.testing.assert_allclose(lam[2:], lam_ref, rtol=1e-10)
    res = eigen._quadratic_residuals(K, C, M, lam, U,
                                     rigid_rtol=eigen.RIGID_BODY_RTOL)
    assert np.all(res <= 1e-6)


def test_singular_mass_matrix():
    r"""A mass matrix with null rows, e.g. lumped without rotary inertia:
    the infinite eigenvalues are discarded, by the sparse and the dense
    solvers, and fewer eigenvalues than requested are returned"""
    n = 6
    K = diags(10.*np.arange(1., n + 1))
    M = diags([1., 2., 0., 0., 0., 0.])
    C = diags([0.1, 0.2, 0., 0., 0., 0.])
    ref = [sdof_roots(10., 0.1, 1.)[0], sdof_roots(20., 0.2, 2.)[0]]
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        for num_eigvalues in (2, 4):
            lam, U = complex_frequency(K, C, M, num_eigvalues=num_eigvalues)
            np.testing.assert_allclose(lam, ref, rtol=1e-10)
            assert U.shape == (n, 2)
    lam, U = complex_frequency(K, C, M, sparse_solver=False)
    np.testing.assert_allclose(lam, ref, rtol=1e-10)


def test_complex_shift():
    r"""A complex shift gives the eigenvalues nearest it, one per complex
    conjugate pair, also those found with a negative imaginary part"""
    K, C, M = nonproportional(seed=1)
    ref = reference(K, C, M)
    for sigma in (6j, -6j, 2. + 8j):
        lam, U = complex_frequency(K, C, M, num_eigvalues=4, sigma=sigma)
        dist = np.minimum(np.abs(ref - sigma), np.abs(ref.conj() - sigma))
        expected = ref[np.argsort(dist)[:4]]
        expected = expected[np.lexsort((np.abs(expected), expected.imag))]
        np.testing.assert_allclose(lam, expected, rtol=1e-9)
        assert np.all(lam.imag >= 0)
        assert np.all(residuals(K, C, M, lam, U) < 1e-9)


def test_qep_modes_pairs():
    r"""One eigenvalue per complex conjugate pair, also for a double
    eigenvalue; an eigenvalue without its partner is replaced by its
    conjugate, with the conjugate eigenvector"""
    lam = np.array([1j, -1j, 2 + 3j, 2 - 3j, 2 + 3j, 2 - 3j, -4 - 5j,
                    -1. + 1e-12j, np.inf])
    U = np.eye(9, dtype=complex) + 1j*np.eye(9, k=1)
    lam2, U2 = eigen._qep_modes(lam, U, None, 0.)
    np.testing.assert_allclose(lam2, [-1., 1j, 2 + 3j, 2 + 3j, -4 + 5j])
    assert lam2[0].imag == 0
    i = int(np.flatnonzero(lam2 == -4 + 5j)[0])
    np.testing.assert_allclose(U2[:, i], np.conj(U[:, 6]))
    lam2, _ = eigen._qep_modes(lam, U, 2, 0.)
    np.testing.assert_allclose(lam2, [-1., 1j])
    lam2, _ = eigen._qep_modes(np.array([-1j, -2j]), np.eye(2), None, 0.)
    np.testing.assert_allclose(lam2, [1j, 2j])


def test_normalize_vectors_null_column():
    U = eigen._normalize_vectors(np.array([[0., 3j], [0., -4.]]))
    np.testing.assert_allclose(U, [[0., -0.6j], [0., 0.8]])


def test_options():
    r"""``precondition=False``, ``check=False``, a real shift and the log
    messages give the same eigenvalues"""
    K, C, M = nonproportional(seed=2)
    lam, _ = complex_frequency(K, C, M, num_eigvalues=5)
    for kwargs in [{'precondition': False}, {'check': False},
                   {'sigma': 0.5}, {'tol': 1e-12}]:
        lam2, _ = complex_frequency(K, C, M, num_eigvalues=5, **kwargs)
        np.testing.assert_allclose(lam2, lam, rtol=1e-9)


def test_silent(capsys):
    lam, _ = complex_frequency(chain(10)[0], csr_matrix((10, 10)),
                               chain(10)[1], num_eigvalues=2)
    assert capsys.readouterr().out == ''
    K = diags([1., 4., 9.])
    C = diags([0., 0., 0.])
    M = diags([1., 1., 0.])
    complex_frequency(K, C, M, num_eigvalues=1, silent=False)
    out = capsys.readouterr().out
    assert 'Complex frequency analysis with eigs() in shift-invert mode, 1 ' \
           'eigenvalue(s):' in out
    assert 'omegad = 1 rad/s, zeta = ' in out
    complex_frequency(K, C, M, sparse_solver=False, silent=False)
    assert 'eig() of the linearization' in capsys.readouterr().out


def test_errors():
    K, C, M = nonproportional(n=10)
    K0 = K.tolil()
    K0[3, :] = 0.
    K0[:, 3] = 0.
    with pytest.raises(LinAlgError, match=r'K is singular: null row or '
                       r'column at 1 degree\(s\)-of-freedom, of indices 3\.'):
        complex_frequency(K0.tocsr(), C, M)
    with pytest.raises(ValueError, match='must be real'):
        complex_frequency(K, 1j*C, M)


def test_singular_shift(monkeypatch):
    r"""A shift that is an eigenvalue: the factorization fails and the dense
    solver gives the eigenvalues nearest it"""
    K = diags([1., 4., 9., 16., 25.])
    C = csr_matrix((5, 5))
    M = diags([1., 1., 1., 1., 1.])
    with pytest.warns(DenseFallbackWarning, match='singular at the shift'):
        lam, _ = complex_frequency(K, C, M, num_eigvalues=2, sigma=2j)
    np.testing.assert_allclose(lam, [1j, 2j], atol=1e-12)
    with pytest.raises(RuntimeError, match='singular at the shift'):
        complex_frequency(K, C, M, num_eigvalues=2, sigma=2j,
                          max_dense_size=0)


def test_dense_fallback(monkeypatch):
    r"""A sparse solver returning wrong eigenpairs: the dense fallback
    returns the same eigenvalues with a warning, a RuntimeError is raised
    without it, and ``check=False`` returns the wrong eigenvalues"""
    K, C, M = nonproportional(n=40, seed=5)
    ref, _ = complex_frequency(K, C, M, num_eigvalues=6)

    def wrong(A, k, *args, **kwargs):
        rng = np.random.RandomState(0)
        return (rng.rand(k) + 1j*rng.rand(k),
                rng.randn(A.shape[0], k) + 0j)

    monkeypatch.setattr(eigen, 'eigs', wrong)
    with pytest.warns(DenseFallbackWarning, match='relative residual'):
        lam, _ = complex_frequency(K, C, M, num_eigvalues=6)
    np.testing.assert_allclose(lam, ref, rtol=1e-8)
    for max_dense_size in (0, 79):
        with pytest.raises(RuntimeError, match='Attempted:'):
            complex_frequency(K, C, M, num_eigvalues=6,
                              max_dense_size=max_dense_size)
    lam, _ = complex_frequency(K, C, M, num_eigvalues=6, check=False)
    assert not np.allclose(lam, ref)


def test_check_qep():
    K = diags([1., 4.]).tocsr()
    C = csr_matrix((2, 2))
    M = diags([1., 1.]).tocsr()
    assert (eigen._check_qep(K, C, M, np.zeros(0), np.zeros((2, 0)), 1e-8)
            == 'no eigenpair was found')
    U = np.array([[1.], [0.]])
    assert eigen._check_qep(K, C, M, np.array([1j]), U, 1e-8) is None
    assert 'relative residual' in eigen._check_qep(K, C, M,
                                                   np.array([1.01j]), U, 1e-8)


def test_warnings_shown_once():
    r"""A warning of the caller, shown once per location with the
    ``'default'`` action, is not repeated after each call: the solver does
    not reset the registry of the warnings already shown, as the exit of
    :func:`warnings.catch_warnings` does"""
    K, C, M = nonproportional(n=20)

    def emit():
        warnings.warn('caller warning', UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')
        for _ in range(3):
            emit()
            complex_frequency(K, C, M, num_eigvalues=2)
    assert [str(w.message) for w in caught] == ['caller warning']


@pytest.mark.parametrize('damping', [0., 1e-6])
def test_near_buckling(damping):
    r"""Regression test: the cantilever of
    ``test_beamc_natural_freq_cantilever_pre_stress.py``, saved in
    ``tests/data``, pre-stressed to 0.9999 of its buckling load, with
    `\boldsymbol{C} = c \boldsymbol{K}_{C0}`: the lowest eigenpairs, `|\lambda| = 0.30`, have
    relative residuals larger than the default ``check_rtol=1e-6`` but
    backward errors at the round-off, see
    :func:`pyfe3d.solver.eigen._quadratic_backward_errors`, and pass the
    verification of the sparse solver without the dense fallback; with
    `\boldsymbol{C}` null, `\lambda = i \omega_n` of
    :func:`pyfe3d.solver.natural_frequency`"""
    from test_solver_natural_frequency import beam_cantilever
    from pyfe3d.solver import linear_buckling

    K0, KG, M = beam_cantilever()
    lambda_cr = linear_buckling(K0, KG, num_eigvalues=1)[0][0]
    K = (K0 + 0.9999*lambda_cr*KG).tocsr()
    C = (damping*K0).tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        lam, U = complex_frequency(K, C, M, num_eigvalues=3,
                                   max_dense_size=0)
        omegan = natural_frequency(K, M, num_eigvalues=3,
                                   max_dense_size=0)[0]
    np.testing.assert_allclose(np.abs(lam), omegan, rtol=1e-4)
    assert np.all(lam.imag > 0)
    res = eigen._quadratic_residuals(K, C, M, lam, U,
                                     rigid_rtol=eigen.RIGID_BODY_RTOL)
    be = eigen._quadratic_backward_errors(K, C, M, lam, U)
    assert np.all((res <= 1e-6) | (be <= eigen.BACKWARD_ERROR_RTOL))
