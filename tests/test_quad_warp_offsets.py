r"""Rigid offsets of warped Quad4 and Quad4R, attribute ``warp_offsets``

With ``warp_offsets = 1``, the default, the flat element of a warped
quadrilateral is linked to the nodes by rigid offsets, ``J`` of the
element documentation: ``KC0 = J^T KC0_flat J``, ``M = J^T M_flat J``,
``fint = J^T fint_flat(J ue)``, also with the von Kármán terms of
``nonlinear=1``, ``KG = KG_flat(J ue) + K_E``, with the eccentricity
geometric stiffness ``K_E`` on the rotations, and ``KCNL = J^T
KCNL_flat(J ue) J - K_E``, such that ``KC0 + KCNL + KG`` is the exact
Jacobian of the von Kármán internal forces. Checked here:

- flat elements: every matrix and force bit for bit equal with
  ``warp_offsets`` 0 and 1;
- warped element: ``KC0`` congruent with ``J``, annihilating the six rigid
  modes of the warped geometry, equal to the co-rotational tangent at
  ``u = 0`` of :class:`.CorotationalShell` with ``warping='offsets'``, with
  the non-zero eigenvalues of the flat element kept;
- ``fint`` and ``KG``, with ``K_E`` against the second derivative of the
  work of the nodal forces of the flat element on the second-order rigid
  link and against finite differences of the co-rotational internal force;
- ``M`` congruent with ``J``, for the three mass matrices;
- the von Kármán internal forces and ``KCNL`` against those of the flat
  element at ``J ue``, the tangent ``KC0 + KCNL + KG`` against central
  differences of the internal forces, also with the ``hgfactor_*`` of
  Quad4R, and ``warp_offsets = 0`` against the flat element, bit for bit;
- the strain rows of ``Quad4Probe.update_BL`` times ``J``;
- twisted cantilever of length 12, width 1.1 and thickness 0.32, 90 degrees
  of twist, the twisted beam of MacNeal and Harder (1985), Fig. 6, p. 8,
  with the tip displacements of their Table 3, p. 10, linear, and
  geometrically non-linear with the von Kármán strains against
  ``kinematics='corotational'`` of :class:`.CorotationalModel`.

References
----------
.. [1] MacNeal, R. H. and Harder, R. L., A proposed standard set of
   problems to test finite element accuracy, Finite Elements in Analysis
   and Design, 1, 3-20, 1985. https://doi.org/10.1016/0168-874X(85)90003-4

"""
import sys
sys.path.append('..')
import warnings

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, INT, DOUBLE, DOF)
from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d.corotational import CorotationalShell, expso3

CLS = {'Quad4': (Quad4, Quad4Probe, Quad4Data),
       'Quad4R': (Quad4R, Quad4RProbe, Quad4RData)}
NAMES = list(CLS)

# NOTE warp ratio about 0.025, see CorotationalShell.warp_ratio
X_WARPED = np.array([[0., 0, 0], [1.0, 0.05, 0.03], [1.1, 0.9, -0.02],
                     [-0.05, 1.0, 0.04]])
X_FLAT = np.array([[0., 0, 0], [1.0, 0.05, 0], [1.1, 0.9, 0],
                   [-0.05, 1.0, 0]])


def laminate():
    r"""Laminate with membrane-bending coupling, offset and density"""
    return laminated_plate(stack=[0., 45., -30.], plyts=[0.03, 0.04, 0.03],
                           laminaprop=(140e9, 10e9, 0.3, 5e9, 5e9, 4e9),
                           rho=1600., offset=0.01)


def element(name, X, warp=1, nodes=None, probe=None):
    r"""Element on the nodes ``nodes`` (default ``0, 1, 2, 3``) of ``X``,
    with its rotation matrix and probe coordinates updated"""
    E, P, _ = CLS[name]
    e = E(P() if probe is None else probe)
    nodes = list(range(4)) if nodes is None else nodes
    for i, n in enumerate(nodes):
        setattr(e, 'n%d' % (i + 1), n)
        setattr(e, 'c%d' % (i + 1), DOF*n)
    x = np.asarray(X, dtype=float).ravel()
    e.update_rotation_matrix(x)
    e.update_probe_xe(x)
    e.warp_offsets = warp
    return e


def sparse_size(name, what):
    return getattr(CLS[name][2](), what + '_SPARSE_SIZE')


def dense(name, what, func, n=24):
    size = sparse_size(name, what)
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    func(r, c, v)
    K = np.zeros((n, n))
    np.add.at(K, (r, c), v)
    return K, v


def rotation(e):
    return np.array([[e.r11, e.r12, e.r13], [e.r21, e.r22, e.r23],
                     [e.r31, e.r32, e.r33]])


def heights(e):
    r"""Heights of the nodes above the flat element, element coordinates"""
    z = np.array(e.probe.xe)[2::3]
    return z - 0.25*((z[0] + z[2]) + (z[1] + z[3]))


def J_local(z):
    J = np.eye(24)
    for a in range(4):
        J[DOF*a + 0, DOF*a + 4] = -z[a]
        J[DOF*a + 1, DOF*a + 3] = z[a]
    return J


def J_global(e):
    T = np.kron(np.eye(8), rotation(e))
    return T @ J_local(heights(e)) @ T.T


def rigid_modes(X):
    r"""Six rigid modes of the nodes ``X``, translations and rotations"""
    xc = X - X.mean(axis=0)
    S = np.zeros((DOF*len(X), 6))
    for a, (x, y, z) in enumerate(xc):
        S[DOF*a:DOF*a + 3, :3] = np.eye(3)
        S[DOF*a:DOF*a + 3, 3:] = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
        S[DOF*a + 3:DOF*a + 6, 3:] = np.eye(3)
    return S


def matrices(name, X, prop, warp, u):
    r"""KC0, M (mtype 0, 1, 2), KG, KG of given stress, fint (linear and
    von Kármán) and finte of one element, as (dense, values)"""
    e = element(name, X, warp)
    out = {'KC0': dense(name, 'KC0', lambda r, c, v:
                        e.update_KC0(r, c, v, prop))}
    for mt in (0, 1, 2):
        out['M%d' % mt] = dense(name, 'M', lambda r, c, v:
                                e.update_M(r, c, v, prop, mtype=mt))
    e.update_probe_ue(u)
    out['KG'] = dense(name, 'KG', lambda r, c, v: e.update_KG(r, c, v, prop))
    out['KCNL'] = dense(name, 'KCNL', lambda r, c, v:
                        e.update_KCNL(r, c, v, prop))
    out['KGs'] = dense(name, 'KG', lambda r, c, v:
                       e.update_KG_given_stress(-1e3, 500., 300., r, c, v))
    for nl in (0, 1):
        f = np.zeros(24)
        e.update_fint(f, prop, nonlinear=nl)
        out['fint%d' % nl] = (f, f)
    e.update_probe_finte(prop)
    out['finte'] = (np.array(e.probe.finte), np.array(e.probe.finte))
    return out


def test_sparse_sizes():
    for name in NAMES:
        assert sparse_size(name, 'KC0') == 576
        assert sparse_size(name, 'KG') == 180
        assert sparse_size(name, 'M') == 480
        e = element(name, X_FLAT)
        assert e.warp_offsets == 1
        E, P, _ = CLS[name]
        assert E(P()).warp_offsets == 1


@pytest.mark.parametrize('name', NAMES)
def test_flat_bit_for_bit(name):
    r"""Flat elements: ``warp_offsets`` 0 and 1 give the same bits, also in
    a plane away from the origin; a flat element rotated in space, whose
    local heights are round-off, within round-off"""
    prop = laminate()
    u = np.random.default_rng(3).standard_normal(24)*1e-3
    for X in (X_FLAT, X_FLAT + [0.3, -0.2, 5.]):
        m0 = matrices(name, X, prop, 0, u)
        m1 = matrices(name, X, prop, 1, u)
        for k in m0:
            assert np.array_equal(m0[k][1], m1[k][1]), k
        # NOTE the 36 entries of the eccentricity stiffness stay zero
        assert not np.any(m1['KG'][1][144:])
    R = expso3(np.array([0.3, -0.7, 1.1]))
    X = X_FLAT @ R.T + [1., 2., 3.]
    m0 = matrices(name, X, prop, 0, u)
    m1 = matrices(name, X, prop, 1, u)
    for k in m0:
        scale = np.abs(m0[k][1]).max()
        assert np.abs(m0[k][1] - m1[k][1]).max() <= 1e-13*scale, k


@pytest.mark.parametrize('name', NAMES)
def test_warped_KC0(name):
    r"""``KC0`` of a warped element: ``J^T KC0_flat J``, the six rigid modes
    of the warped nodes annihilated, which ``KC0_flat`` does not, equal to
    the co-rotational tangent at ``u = 0`` with ``warping='offsets'``, and
    the same number of zero eigenvalues, six for both elements (eight for
    Quad4 with ``shear_model = 0``, whose two extra zero-energy modes are
    those of the flat element)"""
    prop = laminate()
    e0 = element(name, X_WARPED, 0)
    e1 = element(name, X_WARPED, 1)
    K0, _ = dense(name, 'KC0', lambda r, c, v: e0.update_KC0(r, c, v, prop))
    K1, _ = dense(name, 'KC0', lambda r, c, v: e1.update_KC0(r, c, v, prop))
    scale = np.abs(K0).max()
    Jg = J_global(e1)
    assert np.abs(K1 - Jg.T @ K0 @ Jg).max() <= 1e-14*scale
    S = rigid_modes(X_WARPED)
    res1 = np.abs(K1 @ S).max()/scale
    res0 = np.abs(K0 @ S).max()/scale
    print('%s: |K S|/|K| %.1e with offsets, %.1e without' % (name, res1, res0))
    assert res1 <= 1e-14
    assert res0 >= 1e-3
    # NOTE co-rotational tangent; the wrapper switches off the element's
    #      own offsets, so it gets an element of its own
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = CorotationalShell(element(name, X_WARPED, 1), prop,
                              X_WARPED.ravel(), warping='offsets')
    assert w.elem.warp_offsets == 0
    f, KM, KG = w.evaluate(np.zeros(24))
    assert np.abs(KM + KG - K1).max() <= 1e-14*scale
    # NOTE eigenvalues: zero modes and the non-zero ones
    ev0 = np.linalg.eigvalsh(K0)
    ev1 = np.linalg.eigvalsh(K1)
    nz0 = int((np.abs(ev0) < 1e-10*ev0.max()).sum())
    nz1 = int((np.abs(ev1) < 1e-10*ev1.max()).sum())
    assert nz0 == nz1 == 6
    rel = np.abs(ev1[nz1:]/ev0[nz0:] - 1)
    print('%s: non-zero eigenvalues, max relative change %.2e, lowest %.2e'
          % (name, rel.max(), rel[0]))
    assert rel[0] < 1e-3


@pytest.mark.parametrize('name', NAMES)
def test_warped_fint(name):
    r"""Linear internal forces of a warped element ``KC0 ue``; the von
    Kármán terms of ``nonlinear=1`` are ``J^T`` times those of the flat
    element at ``J ue``"""
    prop = laminate()
    u = np.random.default_rng(4).standard_normal(24)*1e-3
    m0 = matrices(name, X_WARPED, prop, 0, u)
    m1 = matrices(name, X_WARPED, prop, 1, u)
    K1 = m1['KC0'][0]
    f1 = m1['fint0'][0]
    assert np.abs(f1 - K1 @ u).max() <= 1e-13*np.abs(f1).max()
    e1 = element(name, X_WARPED, 1)
    T = np.kron(np.eye(8), rotation(e1))
    assert np.abs(T @ m1['finte'][0] - f1).max() <= 1e-13*np.abs(f1).max()
    Jg = J_global(e1)
    m0f = matrices(name, X_WARPED, prop, 0, Jg @ u)
    dnl0 = m0f['fint1'][0] - m0f['fint0'][0]
    dnl1 = m1['fint1'][0] - m1['fint0'][0]
    assert np.abs(dnl1 - Jg.T @ dnl0).max() <= 1e-12*np.abs(dnl1).max()
    dnl_raw = m0['fint1'][0] - m0['fint0'][0]
    assert np.abs(dnl1 - dnl_raw).max() >= 1e-3*np.abs(dnl1).max()


@pytest.mark.parametrize('name', NAMES)
def test_warped_mass(name):
    r"""``M = J^T M_flat J`` for the consistent, lumped and reduced mass
    matrices, with the mass, its first moment and the rotary inertia about
    the centroid of the nodes kept for a rigid motion"""
    prop = laminate()
    u = np.zeros(24)
    m0 = matrices(name, X_WARPED, prop, 0, u)
    m1 = matrices(name, X_WARPED, prop, 1, u)
    Jg = J_global(element(name, X_WARPED, 1))
    for mt in (0, 1, 2):
        M0 = m0['M%d' % mt][0]
        M1 = m1['M%d' % mt][0]
        scale = np.abs(M0).max()
        assert np.abs(M1 - Jg.T @ M0 @ Jg).max() <= 1e-14*scale
        assert np.abs(M1 - M1.T).max() <= 1e-14*scale
        S = rigid_modes(X_WARPED)
        # NOTE total mass for the three translations
        mass = np.diag(S[:, :3].T @ M1 @ S[:, :3])
        assert np.allclose(mass, prop.intrho*element(name, X_WARPED).area,
                           rtol=1e-13)


def flat_forces(name, X, prop, u):
    r"""Nodal forces of the flat element, element coordinates, at the
    displacements J ue, computed without offsets"""
    e = element(name, X, 0)
    e1 = element(name, X, 1)
    e.update_probe_ue(u)
    ue = np.array(e.probe.ue)
    uf = J_local(heights(e1)) @ ue
    e.probe.ue[:] = uf
    e.update_probe_finte(prop)
    return np.array(e.probe.finte), uf, e


@pytest.mark.parametrize('name', NAMES)
def test_warped_KG(name):
    r"""``KG`` of a warped element: the geometric stiffness of the flat
    element at ``J ue``, plus ``K_E`` on the rotations of each node, equal
    to the second derivative of the work of the nodal forces ``n_a`` of
    the flat element on the displacement of the flat node, ``-(R_a - I)
    r_a``, at ``psi = 0``, by central differences"""
    prop = laminate()
    u = np.random.default_rng(5).standard_normal(24)*1e-3
    e1 = element(name, X_WARPED, 1)
    e1.update_probe_ue(u)
    KG1, v1 = dense(name, 'KG', lambda r, c, v: e1.update_KG(r, c, v, prop))
    fflat, uf, e0 = flat_forces(name, X_WARPED, prop, u)
    KG0, v0 = dense(name, 'KG', lambda r, c, v: e0.update_KG(r, c, v, prop))
    scale = np.abs(v1).max()
    assert np.abs(v1[:144] - v0[:144]).max() <= 1e-13*scale
    assert not np.any(v0[144:])
    # NOTE the probe finte is kept by update_KG
    e1.update_probe_finte(prop)
    f_before = np.array(e1.probe.finte)
    e1.update_KG(np.zeros(180, dtype=INT), np.zeros(180, dtype=INT),
                 np.zeros(180), prop)
    assert np.array_equal(f_before, e1.probe.finte)
    # NOTE second derivative of the prestress work, global coordinates
    R = rotation(e1)
    z = heights(e1)
    h = 1e-4
    for a in range(4):
        n = R @ fflat[DOF*a:DOF*a + 3]
        r = z[a]*R[:, 2]

        def W(psi):
            return n @ (-(expso3(psi) - np.eye(3)) @ r)
        H = np.zeros((3, 3))
        I3 = np.eye(3)
        for i in range(3):
            for j in range(3):
                H[i, j] = (W(h*(I3[i] + I3[j])) - W(h*(I3[i] - I3[j]))
                           - W(h*(-I3[i] + I3[j]))
                           + W(-h*(I3[i] + I3[j])))/(4*h**2)
        KE = v1[144 + 9*a:144 + 9*a + 9].reshape(3, 3)
        ra = slice(DOF*a + 3, DOF*a + 6)
        assert np.allclose(KG1[ra, ra] - KG0[ra, ra], KE, rtol=0,
                           atol=1e-14*scale)
        assert np.abs(KE - H).max() <= 1e-6*np.abs(KE).max()


@pytest.mark.parametrize('name', NAMES)
def test_warped_KG_given_stress(name):
    r"""``update_KG_given_stress`` and ``update_KG`` agree for a linear
    in-plane displacement field of the flat element without rotations,
    whose stress resultants are uniform, eccentricity term included"""
    prop = isotropic_plate(thickness=0.1, E=70e9, nu=0.3)
    e1 = element(name, X_WARPED, 1)
    R = rotation(e1)
    xe = np.array(e1.probe.xe).reshape(4, 3)
    # NOTE symmetric gradient: no drilling strain
    exx, eyy, gxy = 1e-4, -3e-5, 4e-5
    ue = np.zeros(24)
    for a in range(4):
        x, y = xe[a, :2]
        ue[DOF*a + 0] = exx*x + 0.5*gxy*y
        ue[DOF*a + 1] = 0.5*gxy*x + eyy*y
    u = np.kron(np.eye(8), R) @ ue
    e1.update_probe_ue(u)
    assert np.allclose(e1.probe.ue, ue, rtol=0, atol=1e-18)
    KG, v = dense(name, 'KG', lambda r, c, v: e1.update_KG(r, c, v, prop))
    A11, A12, A22, A66 = prop.A11, prop.A12, prop.A22, prop.A66
    Nxx = A11*exx + A12*eyy
    Nyy = A12*exx + A22*eyy
    Nxy = A66*gxy
    KGs, vs = dense(name, 'KG', lambda r, c, v:
                    e1.update_KG_given_stress(Nxx, Nyy, Nxy, r, c, v))
    scale = np.abs(v).max()
    assert np.abs(v[144:]).max() > 1e-3*scale
    assert np.abs(v - vs).max() <= 1e-10*scale


def fd_jacobian(func, u, h=1e-7):
    n = u.size
    K = np.zeros((func(u).size, n))
    for i in range(n):
        du = np.zeros(n)
        du[i] = h
        K[:, i] = (func(u + du) - func(u - du))/(2*h)
    return K


@pytest.mark.parametrize('name', NAMES)
def test_warped_KG_corotational(name):
    r"""``K_E`` against the co-rotational internal force of
    :class:`.CorotationalShell` with ``warping='offsets'``, whose offsets
    are exact: with the flat element on the projected nodes, its internal
    force is ``f(u) = J(psi)^T f_flat(u_f(u))``, so that the tangent
    ``K(lambda u0)``, from central differences of ``f``, is ``J^T K_flat
    J`` plus the term of the derivative of ``J(psi)^T``. The derivative of
    ``K`` along ``lambda`` at ``lambda = 0``, minus that of ``J(psi)^T
    K_flat(u_f) J(psi)``, the initial-displacement terms, with ``K_flat``
    from central differences of the internal force of a co-rotational flat
    element on the projected nodes, at the exact ``u_f(lambda u0)``,
    leaves the eccentricity term, linear in the forces of the flat
    element: ``K_E`` of ``update_KG`` at ``u0``"""
    prop = isotropic_plate(thickness=0.1, E=70e9, nu=0.3)
    rng = np.random.default_rng(6)
    u0 = rng.standard_normal(24)*1e-3
    e1 = element(name, X_WARPED, 1)
    R = rotation(e1)
    z = heights(e1)
    ea = [z[a]*R[:, 2] for a in range(4)]
    Xf = X_WARPED - np.array(ea)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = CorotationalShell(element(name, X_WARPED, 1), prop,
                              X_WARPED.ravel(), warping='offsets')
        wf = CorotationalShell(element(name, Xf, 1), prop, Xf.ravel(),
                               warping=None)
    assert wf.warp_ratio < 1e-15

    def flat_dofs(u):
        uf = u.copy()
        for a in range(4):
            psi = u[DOF*a + 3:DOF*a + 6]
            uf[DOF*a:DOF*a + 3] -= (expso3(psi) - np.eye(3)) @ ea[a]
        return uf

    def J_of(u):
        return fd_jacobian(flat_dofs, u, h=1e-6)

    def K_cr(lam):
        u = lam*u0
        K = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u, h=1e-4)
        uf = flat_dofs(u)
        Kf = fd_jacobian(lambda v: wf.evaluate(v, tangent=False), uf,
                         h=1e-4)
        J = J_of(u)
        return K - J.T @ Kf @ J
    # NOTE the error is 7e-7 with lam = 0.3, about lam**2 above it and the
    #      round-off of the differences below it
    lam = 0.3
    KE_cr = (K_cr(lam) - K_cr(-lam))/(2*lam)
    e1.update_probe_ue(u0)
    _, v = dense(name, 'KG', lambda r, c, v: e1.update_KG(r, c, v, prop))
    KE = np.zeros((24, 24))
    for a in range(4):
        ra = slice(DOF*a + 3, DOF*a + 6)
        KE[ra, ra] = v[144 + 9*a:153 + 9*a].reshape(3, 3)
    scale = np.abs(KE).max()
    err = np.abs(KE_cr - KE).max()/scale
    print('%s: K_E from the co-rotational force, relative error %.1e'
          % (name, err))
    assert err < 1e-5


# ---------------------------------------------------------------------------
# twisted cantilever, MacNeal and Harder (1985)
# ---------------------------------------------------------------------------
L_TW, B_TW, T_TW, E_TW, NU_TW = 12., 1.1, 0.32, 29.e6, 0.22
#: MacNeal and Harder (1985), Table 3, p. 10: tip displacement along the
#: load, "in-plane" load (along z, the width at the tip) and "out-of-plane"
#: load (along y, the width at the root)
MH_REF = {'z': 0.005424, 'y': 0.001754}


def twisted_mesh(nx, ny):
    s = np.linspace(0, L_TW, nx + 1)
    eta = np.linspace(-B_TW/2, B_TW/2, ny + 1)
    S, E = np.meshgrid(s, eta, indexing='ij')
    th = 0.5*np.pi*S/L_TW
    X = np.column_stack((S.ravel(), (E*np.cos(th)).ravel(),
                         (E*np.sin(th)).ravel()))
    nid = np.arange(X.shape[0]).reshape(nx + 1, ny + 1)
    quads = [[nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]]
             for i in range(nx) for j in range(ny)]
    return X, nid, quads


def assemble(name, X, quads, prop, warp, what='KC0', u=None, mtype=0):
    E, P, D = CLS[name]
    probe = P()
    size = getattr(D(), what + '_SPARSE_SIZE')
    n = len(quads)
    r = np.zeros(size*n, dtype=INT)
    c = np.zeros(size*n, dtype=INT)
    v = np.zeros(size*n, dtype=DOUBLE)
    N = DOF*len(X)
    x = X.ravel()
    for i, nodes in enumerate(quads):
        e = element(name, X, warp, nodes, probe)
        setattr(e, 'init_k_' + what, size*i)
        if what == 'KC0':
            e.update_KC0(r, c, v, prop)
        elif what == 'M':
            e.update_M(r, c, v, prop, mtype=mtype)
        else:
            e.update_probe_ue(u)
            e.update_KG(r, c, v, prop)
    return coo_matrix((v, (r, c)), shape=(N, N)).tocsr()


def twisted_tip(name, nx, ny, warp):
    r"""Tip displacements along the load, unit tip loads along y and z
    distributed with the weights of the end nodes, root clamped"""
    X, nid, quads = twisted_mesh(nx, ny)
    prop = isotropic_plate(thickness=T_TW, E=E_TW, nu=NU_TW)
    K = assemble(name, X, quads, prop, warp)
    N = K.shape[0]
    bu = np.ones(N, dtype=bool)
    for n in nid[0]:
        bu[DOF*n:DOF*n + DOF] = False
    wts = np.ones(ny + 1)
    wts[[0, -1]] = 0.5
    wts /= wts.sum()
    Kuu = K[bu][:, bu].tocsc()
    out = {}
    for comp, label in ((1, 'y'), (2, 'z')):
        f = np.zeros(N)
        f[DOF*nid[-1] + comp] = wts
        uu = np.zeros(N)
        uu[bu] = spsolve(Kuu, f[bu])
        out[label] = wts @ uu[DOF*nid[-1] + comp]
    return out


@pytest.mark.parametrize('name', NAMES)
def test_twisted_cantilever_6x1(name):
    r"""Linear twisted cantilever with 6x1 elements, warp ratio 0.049, with
    the linear path ``KC0`` and a solve: with offsets the tip
    displacements of Quad4 are within 1 % of those of a 96x16 Quad4 mesh
    with offsets (``1.75261e-3, 5.41735e-3``, see
    ``tests/test_corotational_core.py``), without them they are more than
    70 % too small. Quad4R with offsets is 15.3 % too stiff
    in ``uy`` and 1.9 % in ``uz``: its one-point membrane with the hourglass
    stiffness of Brockman (1987) bends in its plane stiffly with one element
    across the width, e.g. 0.40 of the beam solution for the untwisted
    strip under the in-plane tip load with 6x1 elements, 0.72 with 12x2"""
    ref = {'y': 1.75261e-3, 'z': 5.41735e-3}
    tol = {'Quad4': {'y': 0.01, 'z': 0.01}, 'Quad4R': {'y': 0.16, 'z': 0.025}}
    on = twisted_tip(name, 6, 1, 1)
    off = twisted_tip(name, 6, 1, 0)
    for k in ref:
        err_on = on[k]/ref[k] - 1
        err_off = off[k]/ref[k] - 1
        print('%s 6x1 u%s: offsets %+.2f %%, without %+.1f %%'
              % (name, k, 100*err_on, 100*err_off))
        assert abs(err_on) < tol[name][k]
        assert err_off < -0.7


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('nx, ny', [(12, 2), (24, 4)])
def test_macneal_harder_twisted_beam(name, nx, ny):
    r"""Twisted beam of MacNeal and Harder (1985), Fig. 6, p. 8, 12x2 mesh
    of the benchmark and 24x4, against their Table 3, p. 10: with offsets
    Quad4 is within 0.4 % (their QUAD4, Table 10, p. 15: 0.993 and 0.985),
    and Quad4R within 7.1 and 2.3 %, limited by its stiff in-plane bending,
    see ``test_twisted_cantilever_6x1``; without offsets both are more than
    65 % too small with 12x2 elements. With 96x16 elements and offsets,
    0.9992 and 0.9988 for Quad4, 0.9978 and 0.9987 for Quad4R. Since
    0.12.0, with the MITC4 transverse shear of ``shear_model = 1``, Quad4
    gives 0.9963 and 0.9974 with 12x2 elements, instead of 0.9974 and
    0.9980 with the one-point transverse shear of ``shear_model = 0``,
    hence the tolerance of 0.4 % (measured on 2026-10-30); the 96x16 values
    are the same to four digits"""
    tol = {('Quad4', 12): 0.004, ('Quad4', 24): 0.003,
           ('Quad4R', 12): 0.075, ('Quad4R', 24): 0.025}
    on = twisted_tip(name, nx, ny, 1)
    off = twisted_tip(name, nx, ny, 0)
    for k in MH_REF:
        r_on = on[k]/MH_REF[k]
        r_off = off[k]/MH_REF[k]
        print('%s %dx%d u%s: normalised %.4f with offsets, %.4f without'
              % (name, nx, ny, k, r_on, r_off))
        assert abs(r_on - 1) < tol[(name, nx)]
        if nx == 12:
            assert r_off < 0.35


# ---------------------------------------------------------------------------
# von Kármán strains: internal forces, KCNL and tangent
# ---------------------------------------------------------------------------
def von_karman(e, name, prop, u, hg=None):
    r"""``fint`` with ``nonlinear=1``, ``KC0``, ``KCNL`` and ``KG`` of one
    element at ``u``, global coordinates; ``hg``, the ``hgfactor_*`` of
    Quad4R, passed to every method"""
    kw = {} if hg is None else dict(hgfactor_membrane=hg[0],
                                    hgfactor_physical=hg[1])
    e.update_probe_ue(u)
    f = np.zeros(24)
    e.update_fint(f, prop, nonlinear=1, **kw)
    K0, _ = dense(name, 'KC0', lambda r, c, v: e.update_KC0(r, c, v, prop,
                                                            **kw))
    KN, _ = dense(name, 'KCNL', lambda r, c, v: e.update_KCNL(r, c, v, prop,
                                                              **kw))
    KG, _ = dense(name, 'KG', lambda r, c, v: e.update_KG(r, c, v, prop,
                                                          **kw))
    return f, K0, KN, KG


def eccentricity(e, f_flat):
    r"""``K_E`` of the four nodes, 24x24 global, from the forces of the flat
    element in element coordinates"""
    R = rotation(e)
    z = heights(e)
    KE = np.zeros((24, 24))
    for a in range(4):
        n = R @ f_flat[DOF*a:DOF*a + 3]
        r = z[a]*R[:, 2]
        ra = slice(DOF*a + 3, DOF*a + 6)
        KE[ra, ra] = (n @ r)*np.eye(3) - 0.5*(np.outer(n, r)
                                              + np.outer(r, n))
    return KE


@pytest.mark.parametrize('name', NAMES)
def test_von_karman_flat_element_at_J_ue(name):
    r"""Warped element, von Kármán strains: the internal forces are ``J^T``
    times those of the flat element (``warp_offsets = 0``) at ``J ue``, and
    ``KCNL = J^T KCNL_flat(J ue) J - K_E``, with ``K_E`` from the linear
    nodal forces of the flat element, the term that ``update_KG`` adds"""
    prop = laminate()
    u = np.random.default_rng(7).standard_normal(24)*2e-3
    e1 = element(name, X_WARPED, 1)
    e0 = element(name, X_WARPED, 0)
    Jg = J_global(e1)
    f1, _, KN1, KG1 = von_karman(e1, name, prop, u)
    f0, _, KN0, KG0 = von_karman(e0, name, prop, Jg @ u)
    assert np.abs(f1 - Jg.T @ f0).max() <= 1e-13*np.abs(f1).max()
    e0.update_probe_finte(prop)
    KE = eccentricity(e0, np.array(e0.probe.finte))
    scale = np.abs(KN1).max()
    assert np.abs(KE).max() > 1e-4*scale
    assert np.abs(KN1 - (Jg.T @ KN0 @ Jg - KE)).max() <= 1e-13*scale
    assert np.abs(KG1 - (KG0 + KE)).max() <= 1e-13*np.abs(KG1).max()


@pytest.mark.parametrize('name', NAMES)
def test_von_karman_tangent(name):
    r"""``KC0 + KCNL + KG`` of a warped element against central differences
    of the von Kármán internal forces, with the ``hgfactor_*`` of Quad4R
    passed to all the methods, or only to ``update_KC0`` and
    ``update_fint``: ``K_E`` of ``KG`` and that subtracted by ``KCNL``
    cancel out for any factors. The relative error with ``warp_offsets =
    1`` before the von Kármán terms used the offsets was 6e-5"""
    prop = isotropic_plate(thickness=0.01, E=70e9, nu=0.3)
    u = np.random.default_rng(1).standard_normal(24)*2e-3
    cases = [(None, None)]
    if name == 'Quad4R':
        cases += [((0.3, 4.), (0.3, 4.)), ((0.3, 4.), None)]
    for hg_f, hg_k in cases:
        e = element(name, X_WARPED, 1)
        kw = {} if hg_f is None else dict(hgfactor_membrane=hg_f[0],
                                          hgfactor_physical=hg_f[1])

        def fint(v):
            e.update_probe_ue(v)
            f = np.zeros(24)
            e.update_fint(f, prop, nonlinear=1, **kw)
            return f
        Kfd = fd_jacobian(fint, u, h=1e-7)
        _, K0, KN, KG = von_karman(e, name, prop, u, hg_k)
        if hg_f is not None and hg_k is None:
            K0, _ = dense(name, 'KC0', lambda r, c, v: e.update_KC0(
                r, c, v, prop, **kw))
        K = K0 + KN + KG
        err = np.abs(K - Kfd).max()/np.abs(K).max()
        print('%s hgfactors %s / %s: |K - K_fd|/|K| %.1e'
              % (name, hg_f, hg_k, err))
        assert err < 1e-10


@pytest.mark.parametrize('name', NAMES)
def test_von_karman_warp_offsets_zero(name):
    r"""``warp_offsets = 0`` gives, bit for bit, the results of an element
    whose four nodes have the same height in the probe, for which the
    offsets vanish: the code path of the flat element"""
    prop = laminate()
    u = np.random.default_rng(8).standard_normal(24)*2e-3
    e0 = element(name, X_WARPED, 0)
    e1 = element(name, X_WARPED, 1)
    for a in range(4):
        e1.probe.xe[3*a + 2] = 0.37
    out = []
    for e in (e0, e1):
        f, K0, KN, KG = von_karman(e, name, prop, u)
        e.update_probe_ue(u)
        e.update_probe_finte(prop, nonlinear=1)
        out.append((f, K0, KN, KG, np.array(e.probe.finte)))
    for a, b in zip(*out):
        assert np.array_equal(a, b)


def test_KG_hgfactors():
    r"""``update_KG`` of Quad4R with ``hgfactor_*``: ``K_E`` from the
    linear internal forces with the same factors; the flat part of KG does
    not depend on them"""
    name = 'Quad4R'
    prop = laminate()
    u = np.random.default_rng(9).standard_normal(24)*1e-3
    e = element(name, X_WARPED, 1)
    e.update_probe_ue(u)
    hg = dict(hgfactor_membrane=0.2, hgfactor_physical=5.)
    _, v1 = dense(name, 'KG', lambda r, c, v: e.update_KG(r, c, v, prop))
    _, v2 = dense(name, 'KG', lambda r, c, v: e.update_KG(r, c, v, prop,
                                                          **hg))
    assert np.array_equal(v1[:144], v2[:144])
    e.update_probe_finte(prop, **hg)
    KE = eccentricity(e, np.array(e.probe.finte))
    for a in range(4):
        ra = slice(DOF*a + 3, DOF*a + 6)
        assert np.allclose(v2[144 + 9*a:153 + 9*a].reshape(3, 3), KE[ra, ra],
                           rtol=0, atol=1e-13*np.abs(KE).max())
    assert np.abs(v2[144:] - v1[144:]).max() > 1e-3*np.abs(v1[144:]).max()


def test_update_BL_warp_offsets():
    r"""Strain rows of ``Quad4Probe.update_BL``: times ``J`` with
    ``warp_offsets=1``, the default, such that a rigid rotation of the
    warped element about an in-plane axis has no membrane or drilling
    strain; the rows of the flat element with ``0`` and for a flat element;
    the ``r_x``, ``r_y`` columns are overwritten at every call"""
    e = element('Quad4', X_WARPED, 1)
    R = rotation(e)
    xc = X_WARPED - X_WARPED.mean(axis=0)
    psi = 1e-3*R[:, 0]
    u = np.zeros(24)
    for a in range(4):
        u[DOF*a:DOF*a + 3] = np.cross(psi, xc[a])
        u[DOF*a + 3:DOF*a + 6] = psi
    e.update_probe_ue(u)
    ue = np.array(e.probe.ue)
    Jl = J_local(heights(e))
    rows = ('BLexx', 'BLeyy', 'BLgxy', 'BLkxx', 'BLkyy', 'BLkxy',
            'BLgyz_grad', 'BLgyz_rot', 'BLgxz_grad', 'BLgxz_rot',
            'BLdrilling')
    for dm in (0, 1):
        e.probe.update_BL(0.3, -0.2, dm, 0)
        flat = {k: np.array(getattr(e.probe, k)) for k in rows}
        e.probe.update_BL(0.3, -0.2, dm)
        e.probe.update_BL(0.3, -0.2, dm, 1)
        for k in rows:
            B = np.array(getattr(e.probe, k))
            assert np.allclose(B, flat[k] @ Jl, rtol=0, atol=1e-15), k
        strain = [np.array(getattr(e.probe, k)) @ ue
                  for k in ('BLexx', 'BLeyy', 'BLgxy', 'BLdrilling')]
        assert np.abs(strain).max() < 1e-15
        assert np.abs(flat['BLeyy'] @ ue) > 1e-6
        # NOTE back to the flat rows
        e.probe.update_BL(0.3, -0.2, dm, 0)
        for k in rows:
            assert np.array_equal(getattr(e.probe, k), flat[k]), k
    ef = element('Quad4', X_FLAT, 1)
    ef.probe.update_BL(0.3, -0.2, 0, 0)
    flat = {k: np.array(getattr(ef.probe, k)) for k in rows}
    ef.probe.update_BL(0.3, -0.2, 0, 1)
    for k in rows:
        assert np.array_equal(getattr(ef.probe, k), flat[k]), k


def test_corotational_model_von_karman_tangent():
    r"""Warped Quad4R with ``kinematics='von_karman'`` of
    :class:`.CorotationalModel` and ``hgfactor_*``, which the model passes
    to ``update_KC0`` and ``update_fint`` only: ``calc_kC + calc_kG`` is the
    Jacobian of ``calc_fint``; with ``'corotational_von_karman'`` the
    wrapper switches the offsets of the element off"""
    from pyfe3d.corotational import CorotationalModel
    prop = laminate()
    x = X_WARPED.ravel()
    model = CorotationalModel(x)
    model.add(element('Quad4R', X_WARPED, 1), prop, kinematics='von_karman',
              hgfactor_membrane=0.3, hgfactor_physical=4.)
    u = np.random.default_rng(10).standard_normal(24)*2e-3
    K = (model.calc_kC(u, NLgeom=True) + model.calc_kG(u, NLgeom=True)
         ).toarray()
    Kfd = fd_jacobian(model.calc_fint, u, h=1e-7)
    assert np.abs(K - Kfd).max() <= 1e-10*np.abs(K).max()
    model_cr = CorotationalModel(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = model_cr.add(element('Quad4R', X_WARPED, 1), prop,
                         kinematics='corotational_von_karman')
    assert w.elem.warp_offsets == 0


def newton(model, bu, fext, loads, tol=1e-8, maxit=20):
    r"""Load-controlled Newton-Raphson with ``calc_kC + calc_kG``; returns
    the configurations and the iterations of each load level"""
    u = np.zeros(model.N)
    us, its = [], []
    for lam in loads:
        for it in range(maxit):
            res = model.calc_fint(u) - lam*fext
            if np.linalg.norm(res[bu]) <= tol*lam*np.linalg.norm(fext[bu]):
                break
            K = (model.calc_kC(u, NLgeom=True)
                 + model.calc_kG(u, NLgeom=True)).tocsc()
            u[bu] -= spsolve(K[bu][:, bu], res[bu])
        else:
            raise RuntimeError('no convergence at %g' % lam)
        us.append(u.copy())
        its.append(it)
    return us, its


def twisted_tip_nonlinear(name, kinematics, comp, loads, nx=12, ny=2):
    r"""Tip displacement along the load of the twisted beam, tip load along
    ``comp`` (1: y, 2: z) with the weights of the end nodes, and the
    iterations of each load level"""
    from pyfe3d.corotational import CorotationalModel
    X, nid, quads = twisted_mesh(nx, ny)
    prop = isotropic_plate(thickness=T_TW, E=E_TW, nu=NU_TW)
    x = X.ravel()
    model = CorotationalModel(x)
    E, P, _ = CLS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for q in quads:
            e = E(P())
            for i, n in enumerate(q):
                setattr(e, 'n%d' % (i + 1), n)
                setattr(e, 'c%d' % (i + 1), DOF*n)
            e.update_rotation_matrix(x)
            model.add(e, prop, kinematics=kinematics)
    bu = np.ones(model.N, dtype=bool)
    for n in nid[0]:
        bu[DOF*n:DOF*n + DOF] = False
    wts = np.ones(ny + 1)
    wts[[0, -1]] = 0.5
    wts /= wts.sum()
    fext = np.zeros(model.N)
    fext[DOF*nid[-1] + comp] = wts
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        us, its = newton(model, bu, fext, loads)
    return [wts @ u[DOF*nid[-1] + comp] for u in us], its


#: tip displacements of the non-linear twisted beam with 12x2 elements, von
#: Kármán strains with offsets and co-rotational with exact offsets,
#: pre-determined by test_twisted_beam_von_karman; Quad4 with the default
#: shear_model = 1 since 0.12.0, (0.541658, 0.539894) and (0.525850,
#: 0.525783) with the one-point transverse shear of shear_model = 0
TW_VK = {('Quad4', 2): (0.540811, 0.539268),
         ('Quad4', 1): (0.524690, 0.525317),
         ('Quad4R', 2): (0.537782, 0.536316),
         ('Quad4R', 1): (0.489198, 0.490039)}


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('comp, P', [(2, 100.), (1, 300.)])
def test_twisted_beam_von_karman(name, comp, P):
    r"""Twisted beam of MacNeal and Harder (1985), 12x2 elements, warp ratio
    0.024, geometrically non-linear, tip loads of 100 along z and 300 along
    y in two increments, rotations up to 0.07: the von Kármán strains of the
    elements with offsets (``kinematics='von_karman'``) against
    ``kinematics='corotational'``, whose offsets are exact, within 0.4 %,
    with the quadratic convergence of the exact tangent. Before the von
    Kármán terms used the offsets, Newton-Raphson diverged above a load of
    about 60 along z (measured with a numpy prototype of that state)"""
    loads = [0.5*P, P]
    vk, its_vk = twisted_tip_nonlinear(name, 'von_karman', comp, loads)
    cr, its_cr = twisted_tip_nonlinear(name, 'corotational', comp, loads)
    ref_vk, ref_cr = TW_VK[(name, comp)]
    print('%s load %g along %s: tip %.6f von Kármán (%s iterations), %.6f '
          'co-rotational (%s), %+.2f %%' % (name, P, 'yz'[comp - 1], vk[-1],
                                           its_vk, cr[-1], its_cr,
                                           100*(vk[-1]/cr[-1] - 1)))
    assert abs(vk[-1]/cr[-1] - 1) < 0.004
    assert max(its_vk) <= 5
    assert np.isclose(vk[-1], ref_vk, rtol=1e-5)
    assert np.isclose(cr[-1], ref_cr, rtol=1e-5)
