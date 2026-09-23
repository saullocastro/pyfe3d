r"""Tests for :mod:`pyfe3d.solver`

The plate used for the two end-to-end tests is the one of
``test_quad4_linear_buckling_plate_rotated.py``, so that the load multiplier
can be compared with the classical buckling coefficient

.. math::
    k_c = \left(\frac{m b}{a} + \frac{a}{m b}\right)^2
    \quad , \quad
    \sigma_{cr} = -k_c \frac{\pi^2 E}{12(1 - \nu^2)} \frac{h^2}{b^2}

and, with the mass matrix added, with the natural frequency of a simply
supported plate

.. math::
    \omega_{mn} = \left(\frac{m^2}{a^2} + \frac{n^2}{b^2}\right)
                  \frac{1}{2}\sqrt{\frac{D \pi^4}{2 \rho h}}

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix, diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF
from pyfe3d.solver import (diagonal_preconditioner, estimate_cayley_sigma,
                           is_positive_definite, check_eigenpairs,
                           linear_buckling, natural_frequency)

# NOTE plate of test_quad4_linear_buckling_plate_rotated.py
NX, NY = 21, 7
A_PLATE, B_PLATE, H_PLATE = 2.0, 0.5, 0.002
E_PLATE, NU_PLATE, RHO_PLATE = 203.e9, 0.33, 7830.


def plate_matrices(Nxx=-1., with_mass=False):
    r"""Assemble the simply supported plate and return ``KC0uu``, ``KGuu``,
    optionally ``Muu``, and the free-dof mask"""
    prop = isotropic_plate(E=E_PLATE, nu=NU_PLATE, thickness=H_PLATE,
                           rho=RHO_PLATE)
    data = Quad4Data()
    probe = Quad4Probe()
    xtmp = np.linspace(0, A_PLATE, NX)
    ytmp = np.linspace(0, B_PLATE, NY)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(),
                         np.zeros(NX*NY))).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    ncoords_flatten = ncoords.flatten()
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    nids_mesh = nids.reshape(NX, NY)
    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()
    num_elements = len(n1s)
    N = DOF*ncoords.shape[0]
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_KG = 0
    init_k_M = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        quad = Quad4(probe)
        quad.n1, quad.n2, quad.n3, quad.n4 = n1, n2, n3, n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.init_k_KG = init_k_KG
        quad.init_k_M = init_k_M
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quad.update_KG_given_stress(Nxx, 0, 0, KGr, KGc, KGv)
        if with_mass:
            quad.update_M(Mr, Mc, Mv, prop)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE
        init_k_M += data.M_SPARSE_SIZE
    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    bk = np.zeros(N, dtype=bool)
    edge = (np.isclose(x, 0.) | np.isclose(x, A_PLATE)
            | np.isclose(y, 0.) | np.isclose(y, B_PLATE))
    bk[0::DOF] = edge
    bk[1::DOF] = edge
    bk[2::DOF] = edge
    bu = ~bk
    out = [KC0[bu, :][:, bu], KG[bu, :][:, bu]]
    if with_mass:
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()
        out.append(M[bu, :][:, bu])
    return out


def test_diagonal_preconditioner():
    r"""The preconditioner normalises the diagonal and leaves the
    eigenvalues untouched, which is the property that makes it safe"""
    KC0uu, KGuu = plate_matrices()
    D = diagonal_preconditioner(KC0uu)
    scaled = (D @ KC0uu @ D).diagonal()
    print('scaled diagonal min %.6f max %.6f' % (scaled.min(), scaled.max()))
    assert np.allclose(scaled, 1., rtol=1e-12, atol=1e-12)
    # NOTE the condition number of the plate, which is what motivates it
    d = KC0uu.diagonal()
    print('unscaled diagonal spans %.3e' % (d.max()/d.min()))
    assert d.max()/d.min() > 1.e3

    # NOTE same eigenvalues with and without the scaling
    lam_pre, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4,
                                 precondition=True)
    lam_raw, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4,
                                 precondition=False)
    print('preconditioned', lam_pre[:3], 'raw', lam_raw[:3])
    assert np.allclose(lam_pre[:3], lam_raw[:3], rtol=1e-6)


def test_estimate_cayley_sigma_fallback():
    r"""A singular or indefinite matrix gives no information to the power
    iteration, so the safe default of 1 is returned"""
    n = 20
    K = diags(np.concatenate(([0.], np.ones(n-1)))).tocsc()
    KG = -diags(np.ones(n)).tocsc()
    assert estimate_cayley_sigma(K, KG) == 1.
    # NOTE negative definite is equally uninformative
    assert estimate_cayley_sigma((-diags(np.ones(n))).tocsc(), KG) == 1.


def test_estimate_cayley_sigma_exceeds_the_critical_eigenvalue():
    r"""The shift must be larger than `|\mu|` of the critical eigenvalues,
    which is the whole point of estimating it"""
    KC0uu, KGuu = plate_matrices()
    sigma = estimate_cayley_sigma(KC0uu, KGuu)
    lam, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4)
    mu_crit = 1./lam[0]
    print('sigma %.6e  |mu| of the critical mode %.6e' % (sigma, mu_crit))
    assert sigma > abs(mu_crit)


def test_check_eigenpairs_detects_a_missing_multiplier():
    r"""The completeness check is the one that catches a solver that
    converged to the interior of the spectrum

    With `[K] = diag(1, 2, 3, ...)` and `[K_G] = -[I]` the load multipliers
    are `1, 2, 3, ...`. Handing the second one in as if it were the lowest
    must be rejected.

    """
    n = 6
    K = diags(np.arange(1., n + 1.)).tocsr()
    KG = (-diags(np.ones(n))).tocsr()
    eye = np.eye(n)
    # NOTE the correct lowest pair passes
    assert check_eigenpairs(K, KG, np.array([1.]), eye[:, :1]) is None
    # NOTE the second pair alone must be reported as incomplete
    error = check_eigenpairs(K, KG, np.array([2.]), eye[:, 1:2])
    print('error =', error)
    assert error is not None and 'missing' in error
    # NOTE a wrong eigenvector is caught by the residual instead
    error = check_eigenpairs(K, KG, np.array([1.]), eye[:, 2:3])
    print('error =', error)
    assert error is not None and 'residual' in error


def test_is_positive_definite():
    n = 8
    assert is_positive_definite(diags(np.ones(n)).tocsc())
    assert not is_positive_definite(
        diags(np.concatenate(([-1.], np.ones(n-1)))).tocsc())
    assert not is_positive_definite(
        diags(np.concatenate(([0.], np.ones(n-1)))).tocsc())


def test_linear_buckling_plate():
    r"""End-to-end against the classical buckling coefficient, and against
    the hand-rolled recipe used elsewhere in this suite"""
    Nxx = -1.
    KC0uu, KGuu = plate_matrices(Nxx=Nxx)
    eigvals, eigvecs = linear_buckling(KC0uu, KGuu, num_eigvalues=4)
    P_cr_calc = eigvals[0]*Nxx*B_PLATE

    kcmin = 1e6
    for m in range(1, 21):
        kc = (m*B_PLATE/A_PLATE + A_PLATE/(m*B_PLATE))**2
        kcmin = min(kc, kcmin)
    sigma_cr = -kcmin*np.pi**2*E_PLATE/(12*(1 - NU_PLATE**2))*(
        H_PLATE**2/B_PLATE**2)
    P_cr_theory = sigma_cr*H_PLATE*B_PLATE
    print('P_cr_calc %.4f  P_cr_theory %.4f' % (P_cr_calc, P_cr_theory))
    assert np.isclose(P_cr_theory, P_cr_calc, rtol=0.03)

    # NOTE the same answer as the recipe spelled out in the other tests of
    #      this suite. The two are not bit-identical because the shift
    #      differs, estimated here and hardcoded to 1 there, so they agree
    #      only to the tolerance requested of the eigensolver, which is
    #      1e-9 in both calls below
    lam_est, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4, tol=1e-9)
    D = diagonal_preconditioner(KC0uu)
    mu, _ = eigsh(A=D @ KGuu @ D, k=4, which='SM', M=D @ KC0uu @ D,
                  tol=1e-9, sigma=1., mode='cayley')
    lam = np.sort(-1./mu)
    lam = lam[lam > 0]
    print('linear_buckling %.9f  hand-rolled %.9f  rel diff %.2e'
          % (lam_est[0], lam[0], abs(lam_est[0]/lam[0] - 1)))
    assert np.isclose(lam_est[0], lam[0], rtol=1e-6)

    # NOTE the eigenvectors come back in the original scaling
    residual = np.linalg.norm(KC0uu @ eigvecs[:, 0]
                              + eigvals[0]*(KGuu @ eigvecs[:, 0]))
    residual /= np.linalg.norm(KC0uu @ eigvecs[:, 0])
    print('eigenvector residual %.3e' % residual)
    assert residual < 1e-8


def test_natural_frequency_plate():
    r"""End-to-end against the natural frequency of a simply supported
    plate"""
    KC0uu, KGuu, Muu = plate_matrices(with_mass=True)
    omegan, eigvecs = natural_frequency(KC0uu, Muu, num_eigvalues=4)
    Dp = 2*H_PLATE**3*E_PLATE/(3*(1 - NU_PLATE**2))
    wmn = (1./A_PLATE**2 + 1./B_PLATE**2)*np.sqrt(
        Dp*np.pi**4/(2*RHO_PLATE*H_PLATE))/2
    print('omegan[0] %.4f  theory %.4f' % (omegan[0], wmn))
    assert np.isclose(wmn, omegan[0], rtol=0.05)
    assert np.all(np.diff(omegan) >= -1e-9)


def test_linear_buckling_raises_when_the_spectrum_is_incomplete():
    r"""``check=True`` must turn a silently wrong answer into an error

    The shift is forced far below the critical `|\mu|`, which is exactly the
    mistake the module exists to prevent, and the verification has to catch
    it.

    """
    KC0uu, KGuu = plate_matrices()
    good, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4)
    try:
        bad, _ = linear_buckling(KC0uu, KGuu, num_eigvalues=4,
                                 sigma=1.e-12, check=True)
    except RuntimeError as err:
        print('raised as expected:', err)
        return
    # NOTE if it did not raise, the answer must at least be the right one,
    #      otherwise the check failed to do its job
    print('did not raise, good %.6f bad %.6f' % (good[0], bad[0]))
    assert np.isclose(good[0], bad[0], rtol=1e-6), (good[0], bad[0])


if __name__ == '__main__':
    test_diagonal_preconditioner()
    test_estimate_cayley_sigma_fallback()
    test_estimate_cayley_sigma_exceeds_the_critical_eigenvalue()
    test_check_eigenpairs_detects_a_missing_multiplier()
    test_is_positive_definite()
    test_linear_buckling_plate()
    test_natural_frequency_plate()
    test_linear_buckling_raises_when_the_spectrum_is_incomplete()
