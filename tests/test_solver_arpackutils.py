r"""Tests for :mod:`pyfe3d.solver.arpackutils`: ``ncv <= 32`` against the
``dsteqr`` bug of Intel MKL 2024.2.0 to 2025.0.0

With the Quad4 plate of ``test_solver.py``.
"""
import inspect
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.linalg import block_diag, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import ArpackError, eigs, eigsh

from pyfe3d.solver import arpackutils
from pyfe3d.solver import linear_buckling, natural_frequency
from pyfe3d.solver.eigen import (check_eigenpairs, _eigh_condensed,
                                 estimate_cayley_sigma, _sort_eigenpairs,
                                 diagonal_preconditioner)

from test_solver import plate_matrices


def scaled_plate(with_mass=False):
    r"""Preconditioned matrices of the Quad4 plate of ``test_solver.py``"""
    out = plate_matrices(Nxx=-1., with_mass=with_mass)
    D = diagonal_preconditioner(out[0])
    return [csc_matrix(D @ A @ D) for A in out]


@pytest.fixture
def detection(monkeypatch):
    """Mock the inputs of the automatic cap of ncv"""
    def set_detection(faulty, is_mkl, version):
        monkeypatch.setattr(arpackutils, 'dsteqr_is_faulty', lambda: faulty)
        monkeypatch.setattr(arpackutils, 'lapack_is_mkl', lambda: is_mkl)
        monkeypatch.setattr(arpackutils, 'mkl_version', lambda: version)
        monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', None)
        monkeypatch.delenv('PYFE3D_ARPACK_MAX_NCV', raising=False)
        arpackutils._auto_max_ncv.cache_clear()
    yield set_detection
    arpackutils._auto_max_ncv.cache_clear()


@pytest.mark.parametrize('faulty, is_mkl, version, max_ncv', [
    (True, None, None, 32),
    (True, False, None, 32),
    (False, True, None, 32),
    (False, True, (2024, 2, 0), 32),
    (False, True, (2025, 0, 0), 32),
    (False, True, (2024, 1, 0), None),
    (False, True, (2025, 0, 1), None),
    (False, True, (2025, 3, 0), None),
    (False, False, None, None),
    (False, None, None, None),
    (None, None, None, None),
])
def test_arpack_max_ncv_auto(detection, faulty, is_mkl, version, max_ncv):
    """The cap applies to a faulty dsteqr and to any MKL build that is not
    known to be outside 2024.2.0 to 2025.0.0"""
    detection(faulty, is_mkl, version)
    assert arpackutils.arpack_max_ncv() == max_ncv


@pytest.mark.parametrize('capped', [False, True])
def test_arpack_ncv(detection, capped):
    """SciPy's default ncv, capped at 32 only when the detection says so"""
    detection(capped, capped, None)
    assert arpackutils.arpack_ncv(5, 1000) == 20
    assert arpackutils.arpack_ncv(15, 1000) == 31
    assert arpackutils.arpack_ncv(25, 1000) == (32 if capped else 51)
    assert arpackutils.arpack_ncv(25, 30) == 30
    assert arpackutils.arpack_ncv(3, 10) == 10


def test_arpack_max_ncv_override(detection, monkeypatch):
    """The module attribute takes precedence over the environment variable,
    which takes precedence over the automatic detection"""
    detection(True, True, (2025, 0, 0))
    assert arpackutils.arpack_max_ncv() == 32
    for value, expected in [('0', None), ('none', None), ('48', 48),
                            ('auto', 32), ('', 32)]:
        monkeypatch.setenv('PYFE3D_ARPACK_MAX_NCV', value)
        assert arpackutils.arpack_max_ncv() == expected
    monkeypatch.setenv('PYFE3D_ARPACK_MAX_NCV', '0')
    for value, expected in [(0, None), (40, 40), ('auto', 32), (None, None)]:
        monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', value)
        assert arpackutils.arpack_max_ncv() == expected
    monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', 2)
    with pytest.raises(ValueError):
        arpackutils.arpack_max_ncv()


def test_lapack_name_from_config():
    """The LAPACK of the build configuration, robust to missing keys and to
    the numpy.distutils builds of older SciPy versions"""
    from types import SimpleNamespace as NS

    def show_config(lapack):
        def f(mode='stdout'):
            assert mode == 'dicts'
            return {'Build Dependencies': {'lapack': lapack}}
        return f

    name = arpackutils._lapack_name_from_config
    assert name(NS(show_config=show_config(
        {'name': 'mkl-sdl', 'version': '2025'}))) == 'mkl-sdl'
    assert name(NS(show_config=show_config({'name': 'scipy-openblas'}))) \
        == 'scipy-openblas'
    assert name(NS(show_config=show_config({}))) is None

    def old_show_config():
        print('old style')
    assert name(NS(show_config=old_show_config)) is None
    assert name(NS(show_config=old_show_config, __config__=NS(
        lapack_mkl_info={'libraries': ['mkl_rt']}))) == 'mkl'
    assert name(NS(show_config=old_show_config, __config__=NS(
        lapack_mkl_info={}, blas_mkl_info={}))) is None
    assert isinstance(arpackutils.lapack_is_mkl(), (bool, type(None)))


def test_parse_mkl_version():
    parse = arpackutils._parse_mkl_version
    assert parse('2025.0-Product') == (2025, 0, 0)
    assert parse('Version 2024.2.2-Product Build 20240823') == (2024, 2, 2)
    assert parse('unknown') is None


def faulty_environment():
    """MKL 2024.2.0 to 2025.0.0 identified at runtime"""
    version = arpackutils.mkl_version()
    first, last = arpackutils.MKL_FAULTY_VERSIONS
    return version is not None and first <= version <= last


def test_dsteqr_probe_agrees_with_mkl_version():
    """The functional probe of dsteqr detects the faulty MKL versions"""
    if faulty_environment():
        assert arpackutils.dsteqr_is_faulty() is True
        assert arpackutils.arpack_max_ncv() == 32
    else:
        assert arpackutils.dsteqr_is_faulty() in (False, None)


def test_capped_eigsh_single_call_without_cap(monkeypatch):
    """Without the cap, SciPy's default ncv in a single call, whose result is
    only correct when dsteqr is not faulty"""
    from pyfe3d.solver import arpackutils as au

    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 0)
    calls = []
    scipy_eigsh = au._scipy_eigsh

    def spy(**kwargs):
        calls.append((kwargs['k'], kwargs['ncv']))
        return scipy_eigsh(**kwargs)

    monkeypatch.setattr(au, '_scipy_eigsh', spy)
    n = 200
    K = diags(np.linspace(1., 2., n)).tocsc()
    M = diags(np.ones(n)).tocsc()
    w, v = au.capped_eigsh(A=K, k=25, M=M, sigma=-1., which='LM')
    assert calls == [(25, 51)]
    if not arpackutils.dsteqr_is_faulty():
        np.testing.assert_allclose(np.sort(w), np.linspace(1., 2., n)[:25])


def test_release_memory(monkeypatch):
    """In Pyodide the reference cycles of SciPy's ARPACK wrappers, which
    keep the SuperLU factorizations, exhaust the WebAssembly heap before the
    cyclic garbage collector runs: gc.collect() is called after each ARPACK
    call there, and only there"""
    import gc
    import sys

    calls = []
    monkeypatch.setattr(gc, 'collect', lambda: calls.append(1))
    monkeypatch.setattr(sys, 'platform', 'win32')
    arpackutils.release_memory()
    assert calls == []
    monkeypatch.setattr(sys, 'platform', 'emscripten')
    arpackutils.release_memory()
    assert calls == [1]


def test_eigsh_ncv_boundary():
    r"""Root cause: eigsh in Cayley mode is correct with ncv=32 and, with a
    faulty dsteqr, wrong or raising with ncv=33, i.e. as soon as ARPACK
    computes the eigenvectors of a tridiagonal matrix larger than 32 x 32"""
    K, KG = scaled_plate()
    mu_ref, _ = _eigh_condensed(K, KG, 10)
    sigma = estimate_cayley_sigma(K, KG)

    def solve(ncv):
        mu, v = eigsh(A=KG, M=K, k=10, ncv=ncv, which='SM', sigma=sigma,
                      mode='cayley', tol=0,
                      v0=arpackutils.start_vector(K.shape[0]))
        mu, v = _sort_eigenpairs(mu, v)
        error = check_eigenpairs(K, KG, -1/mu, v, 1e-3)
        return error is None and np.allclose(mu, mu_ref, rtol=1e-6)

    assert solve(32)
    if arpackutils.dsteqr_is_faulty():
        try:
            assert not solve(33)
        except ArpackError as e:
            assert 'dsteqr' in str(e)


@pytest.mark.parametrize('k', [16, 25, 40])
def test_capped_eigsh_passes(monkeypatch, k):
    r"""With the cap, k > 15 eigenpairs are computed in passes of at most 15,
    with ncv <= 32, giving the dense load multipliers and K-orthonormal
    eigenvectors"""
    au = arpackutils
    monkeypatch.setattr(au, 'ARPACK_MAX_NCV', 32)
    calls = []
    scipy_eigsh = au._scipy_eigsh

    def spy(**kwargs):
        calls.append((kwargs['k'], kwargs['ncv']))
        return scipy_eigsh(**kwargs)

    monkeypatch.setattr(au, '_scipy_eigsh', spy)
    K, KG = scaled_plate()
    mu_ref, _ = _eigh_condensed(K, KG, k)
    sigma = estimate_cayley_sigma(K, KG)
    mu, v = au.capped_eigsh(A=KG, k=k, M=K, sigma=sigma, which='SM',
                            mode='cayley')
    mu, v = _sort_eigenpairs(mu, v)
    assert len(calls) == -(-k//15)
    assert all(kp <= 15 and ncv <= 32 for kp, ncv in calls)
    assert sum(kp for kp, ncv in calls) == k
    np.testing.assert_allclose(mu, mu_ref, rtol=1e-6)
    assert check_eigenpairs(K, KG, -1/mu, v, 1e-4) is None
    np.testing.assert_allclose(v.T @ (K @ v), np.eye(k), atol=1e-6)


def test_linear_buckling_and_natural_frequency_many_modes():
    r"""25 load multipliers and 25 frequencies of the Quad4 plate with the
    sparse solvers alone, also when dsteqr is faulty"""
    KC0uu, KGuu, Muu = plate_matrices(Nxx=-1., with_mass=True)
    lam_ref, _ = linear_buckling(KC0uu, KGuu, sparse_solver=False)
    lam_ref = lam_ref[lam_ref > 0][:25]
    eigvals, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=25,
                                 max_dense_size=0)
    np.testing.assert_allclose(eigvals, lam_ref, rtol=1e-6)
    # NOTE M u = (1/omega**2) K u, the mass matrix being singular
    ev = eigh(Muu.toarray(), KC0uu.toarray(), eigvals_only=True)
    w2_ref = np.sort(1/ev[ev > 0])[:25]
    omegan, _ = natural_frequency(KC0uu, Muu, num_eigvalues=25,
                                  max_dense_size=0)
    np.testing.assert_allclose(omegan**2, w2_ref, rtol=1e-6)


def test_start_kwargs():
    r"""v0 is start_vector(n) and, with SciPy 1.17 or newer, rng a
    Generator with the seed 42, a new one at each call"""
    kwargs = arpackutils.start_kwargs(10)
    np.testing.assert_array_equal(kwargs['v0'], arpackutils.start_vector(10))
    if arpackutils._arpack_takes_rng():
        assert set(kwargs) == {'v0', 'rng'}
        other = arpackutils.start_kwargs(10)['rng']
        assert other is not kwargs['rng']
        np.testing.assert_array_equal(kwargs['rng'].uniform(size=5),
                                      np.random.default_rng(42).uniform(size=5))
    else:
        assert set(kwargs) == {'v0'}


@pytest.mark.skipif('rng' not in inspect.signature(eigs).parameters,
                    reason='ARPACK of SciPy < 1.17 draws the restart vectors '
                           'with its own seed')
def test_eigs_restart_reproducible():
    r"""Regression test: with a nilpotent operator ARPACK finds an invariant
    subspace after two Arnoldi vectors and, since SciPy 1.17, draws the new
    starting vectors from rng, by default from the entropy of the operating
    system, such that the same v0 gave different eigenvalues at each call,
    e.g. a spurious real load multiplier of about 1e16 or none, and
    linear_buckling() on Ziegler's double pendulum raised or not at
    random"""
    from pyfe3d.solver.eigen import _eigs_shifted, _splu

    k, l = 3., 0.5
    K = csc_matrix(block_diag(np.array([[2*k, -k], [-k, k]]),
                              np.diag(np.arange(1., 9.))))
    KG = csr_matrix(block_diag(np.array([[-l, l], [0., 0.]]),
                               np.zeros((8, 8))))
    lu = _splu(K)
    v0 = arpackutils.start_vector(10)
    OPv0 = lu.solve(KG @ v0)
    assert np.linalg.norm(OPv0) > 0 and np.all(lu.solve(KG @ OPv0) == 0)
    mu, eigvecs, radius = _eigs_shifted(K, KG, 0., 6, 0, lu)
    for trial in range(10):
        mu_i, eigvecs_i, radius_i = _eigs_shifted(K, KG, 0., 6, 0, lu)
        np.testing.assert_array_equal(mu_i, mu)
        np.testing.assert_array_equal(eigvecs_i, eigvecs)
        assert radius_i == radius
