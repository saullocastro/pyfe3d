r"""The solvers of :mod:`pyfe3d.solver` leave the warnings of the caller alone

A warning shown once per location, the ``'default'`` action of
:mod:`warnings`, must not be shown again after each call of a solver. The
exit of a :func:`warnings.catch_warnings` block resets the registry of the
warnings already shown, of every module, which is why the solvers use none:
each public entry point is called three times between the emissions of a
warning of the caller, which must be shown once, as every warning issued by
the solvers themselves, e.g. a :class:`.DenseFallbackWarning`.
"""
import sys
sys.path.append('..')

import warnings
from collections import Counter

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from pyfe3d.solver import (Analysis, check_eigenpairs, complex_frequency,
                           estimate_cayley_sigma, is_positive_definite,
                           linear_buckling, linear_static, natural_frequency,
                           solve)
from pyfe3d.solver import arpackutils, eigen

N = 40


def symmetric_problem():
    r"""``K``, ``KG`` with null rows, i.e. degrees-of-freedom condensed out
    by the dense fallback, and ``M``"""
    K = csr_matrix(diags([-1., 2.5, -1.], [-1, 0, 1], shape=(N, N)))
    kg = -np.ones(N)
    kg[::4] = 0.
    return K, diags(kg).tocsr(), diags(1. + np.arange(N)/N).tocsr()


def unsymmetric_problem():
    r"""``K`` and an unsymmetric ``KG`` with null rows and columns, of an
    upper triangular `\boldsymbol{K}^{-1} \boldsymbol{K}_G` with the load multipliers ``1, 2,
    ...``"""
    K = diags(np.arange(1., N + 1)).tocsr()
    KG = -np.eye(N) + 0.05*np.triu(np.ones((N, N)), k=1)
    KG[::4, :] = 0.
    KG[:, ::4] = 0.
    return K, csr_matrix(KG)


def fail(*args, **kwargs):
    raise RuntimeError('forced failure of the sparse solver')


def call_linear_buckling_symmetric(monkeypatch):
    K, KG, _ = symmetric_problem()
    linear_buckling(K, KG, num_eigvalues=3)


def call_linear_buckling_symmetric_dense(monkeypatch):
    monkeypatch.setattr(eigen, 'capped_eigsh', fail)
    K, KG, _ = symmetric_problem()
    linear_buckling(K, KG, num_eigvalues=3)


def call_linear_buckling_unsymmetric(monkeypatch):
    K, KG = unsymmetric_problem()
    eigvals, _ = linear_buckling(K, KG, num_eigvalues=3, symmetric=False)
    np.testing.assert_allclose(eigvals[0], 2., rtol=1e-10)


def call_linear_buckling_unsymmetric_dense(monkeypatch):
    monkeypatch.setattr(eigen, '_eigs_shift_search', fail)
    K, KG = unsymmetric_problem()
    linear_buckling(K, KG, num_eigvalues=3, symmetric=False)


def call_natural_frequency(monkeypatch):
    K, _, M = symmetric_problem()
    natural_frequency(K, M, num_eigvalues=3)
    natural_frequency(K, M, num_eigvalues=3, symmetric=False)


def call_complex_frequency(monkeypatch):
    K, _, M = symmetric_problem()
    complex_frequency(K, 0.01*K, M, num_eigvalues=3)


def call_helpers(monkeypatch):
    K, KG, _ = symmetric_problem()
    assert is_positive_definite(K)
    sigma = estimate_cayley_sigma(K, KG)
    eigvals, eigvecs = linear_buckling(K, KG, num_eigvalues=2, check=False)
    assert check_eigenpairs(K, KG, eigvals, eigvecs) is None
    assert sigma > 0
    # NOTE _condense() directly, the dense fallback being covered above
    eigen._eigh_condensed(K, KG, 2)


def call_capped_eigsh_passes(monkeypatch):
    r"""Several passes, with the factorization of the deflated passes"""
    monkeypatch.setattr(arpackutils, 'ARPACK_MAX_NCV', 32)
    K, _, M = symmetric_problem()
    w, _ = arpackutils.capped_eigsh(A=K, k=16, M=M, sigma=-1., which='LM')
    assert w.size == 16


def call_static(monkeypatch):
    K, _, _ = symmetric_problem()
    solve(K, np.ones(N))
    linear_static(K, np.ones(N))
    with pytest.raises(np.linalg.LinAlgError, match='exactly singular'):
        solve(csr_matrix(np.array([[1., 2.], [2., 4.]])), np.ones(2))


def call_nonlinear_analysis(monkeypatch):
    K, _, _ = symmetric_problem()
    f = np.ones(N)
    an = Analysis(calc_fext=lambda inc=1., silent=True: inc*f,
                  calc_kC=lambda c=None, NLgeom=False, silent=True: K,
                  calc_kG=lambda c=None, NLgeom=False, silent=True:
                  csr_matrix((N, N)),
                  calc_fint=lambda c=None, silent=True: K @ c)
    an.initialInc = 0.5
    increments, cs = an.static(NLgeom=True, silent=True)
    np.testing.assert_allclose(cs[-1], solve(K, f), rtol=1e-10)


@pytest.mark.parametrize('call', [
    call_linear_buckling_symmetric, call_linear_buckling_symmetric_dense,
    call_linear_buckling_unsymmetric, call_linear_buckling_unsymmetric_dense,
    call_natural_frequency, call_complex_frequency, call_helpers,
    call_capped_eigsh_passes, call_static, call_nonlinear_analysis])
def test_warnings_shown_once(monkeypatch, call):
    def emit():
        warnings.warn('caller warning', UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')
        for _ in range(3):
            emit()
            call(monkeypatch)
    shown = Counter((str(w.message), w.filename, w.lineno) for w in caught)
    assert sum(1 for key in shown if key[0] == 'caller warning') == 1
    assert all(count == 1 for count in shown.values()), shown
