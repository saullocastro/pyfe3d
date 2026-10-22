r"""Tests for :func:`pyfe3d.solver.natural_frequency`: verification, dense
fallback, rigid-body modes and unsymmetric matrices

Ported from the tests of structsolve, whose ``freq`` returns `\lambda^2 =
-\omega_n^2` whereas :func:`pyfe3d.solver.natural_frequency` returns
`\omega_n`, and completed with the Quad4 plate of ``test_solver.py``.

Ziegler's double pendulum: two rigid bars of length ``l``, rotational springs
``k`` at the base and at the middle hinge, masses ``2m`` at the middle hinge
and ``m`` at the tip, and a follower force ``P`` at the tip, tangential to
the upper bar. With the absolute rotations of the bars as
degrees-of-freedom:

.. math::

    [M] = m l^2 \begin{bmatrix} 3 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
    [K(P)] = \begin{bmatrix} 2k - P l & -k + P l \\ -k & k \end{bmatrix}

The stiffness matrix is unsymmetric and never singular, ``det K = k**2``,
and the system loses stability by flutter at ``P_f = (7/2 - sqrt(2)) k/l``,
where two natural frequencies coalesce (Ziegler, 1952, as cited by
structsolve).
"""
import sys
sys.path.append('..')

import warnings

import numpy as np
import pytest
from scipy.linalg import block_diag, eigh, eig
from scipy.sparse import csc_matrix, csr_matrix, diags, random as sprandom

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
    r"""The null rows and columns are removed and the eigenvectors have the
    size of the original matrices"""
    n = 8
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    active = [1, 2, 3, 4, 5]
    for i in active:
        K[i, i] = float(i)*100
        M[i, i] = 1.0
    omegan, eigvecs = natural_frequency(csc_matrix(K), csc_matrix(M),
                                        num_eigvalues=3)
    np.testing.assert_allclose(omegan, np.sqrt([100., 200., 300.]),
                               rtol=1e-10)
    assert eigvecs.shape == (n, 3)
    assert np.all(eigvecs[[0, 6, 7], :] == 0)


def freq_problem(symmetric, n=300, seed=0):
    rng = np.random.RandomState(seed)
    A = sprandom(n, n, density=0.02, random_state=rng)
    K = A @ A.T + diags(np.linspace(1., 100., n))
    if not symmetric:
        K = K + 1e-2*sprandom(n, n, density=0.02, random_state=rng)
    M = diags(1. + rng.rand(n))
    return csc_matrix(K), csc_matrix(M)


@pytest.mark.parametrize('symmetric, num_eigvalues', [
    (True, 5), (True, 12), (False, 5), (False, 25), (False, 40)])
def test_lowest_frequencies(symmetric, num_eigvalues):
    r"""The sparse solver, without dense fallback, returns the lowest
    frequencies of the full dense solution. NOTE more than 15 eigenpairs of
    symmetric matrices exceed ``ncv = 32`` of ARPACK, see the note on Intel
    MKL of :mod:`pyfe3d.solver.eigen`"""
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
    solvers without the dense fallback, the matrix-norm floor of the
    relative residual being meaningful when `\omega_n^2` vanishes"""
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
    res = eigen._relative_residuals(K, M, -omegan**2, eigvecs)
    assert np.all(res <= 1e-6)


def test_quad4_plate_prestress():
    r"""The Quad4 plate of ``test_solver.py`` under a compressive prestress
    `\lambda [K_G]`: the frequencies decrease with `\lambda`, and the
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
    calls = spy(monkeypatch, ['eigsh', 'eigs', '_dense_freq'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', DenseFallbackWarning)
        natural_frequency(K, M, symmetric=symmetric,
                          max_dense_size=max_dense_size)
    assert calls == ['eigsh' if symmetric else 'eigs']


@pytest.mark.parametrize('symmetric', [True, False])
def test_explicit_dense(monkeypatch, symmetric):
    K, M = freq_problem(symmetric, n=120)
    calls = spy(monkeypatch, ['eigsh', 'eigs', '_dense_freq'])
    natural_frequency(K, M, symmetric=symmetric, sparse_solver=False)
    assert calls == ['_dense_freq']


@pytest.mark.parametrize('symmetric', [True, False])
def test_dense_fallback(monkeypatch, symmetric):
    r"""A sparse solver returning wrong eigenpairs: the dense fallback
    returns the lowest frequencies with a warning, a RuntimeError is raised
    without it"""
    K, M = freq_problem(symmetric, n=120)
    ref, _ = natural_frequency(K, M, symmetric=symmetric, num_eigvalues=10)
    name = 'eigsh' if symmetric else 'eigs'

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
    r"""The inertia of ``K - s M`` detects a missing lowest mode"""
    K, M = freq_problem(True, n=120)
    sparse = eigen.eigsh

    def missing_lowest(A, k, *args, **kwargs):
        w2, eigvecs = sparse(A, k + 1, *args, **kwargs)
        order = np.argsort(w2)
        return w2[order[1:]], eigvecs[:, order[1:]]

    monkeypatch.setattr(eigen, 'eigsh', missing_lowest)
    with pytest.raises(RuntimeError, match='eigenvalues omega'):
        natural_frequency(K, M, symmetric=True, num_eigvalues=10,
                          max_dense_size=0)
    with pytest.warns(DenseFallbackWarning, match='eigenvalues omega'):
        omegan, _ = natural_frequency(K, M, symmetric=True, num_eigvalues=10)
    w_ref = eigh(K.toarray(), M.toarray(), eigvals_only=True)[:10]
    np.testing.assert_allclose(omegan**2, w_ref, rtol=1e-10)
