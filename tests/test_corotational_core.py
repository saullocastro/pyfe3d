r"""Core of the co-rotational (EICR) layer, :mod:`pyfe3d.corotational`

Verified:

- the finite-rotation utilities against closed forms: the exponential map
  against :class:`scipy.spatial.transform.Rotation`, the logarithm as its
  inverse, `\boldsymbol{\Lambda}`, `\boldsymbol{H}`, `\boldsymbol{L}` and
  `\partial(\boldsymbol{\Lambda}^T \boldsymbol{m})/\partial
  \boldsymbol{\psi}` against central finite differences of their
  definitions, and the continuity of the small-angle series;
- the local stiffness matrix of the identity trick: symmetric, rigid
  rotations in its null space for flat elements (Felippa and Haugen 2005,
  Eq. (50), `\boldsymbol{K}_M \boldsymbol{S} = \boldsymbol{0}`), equal to
  the element ``KC0`` rotated to the element frame; `\boldsymbol{G}
  \boldsymbol{S} = \boldsymbol{I}` and `\boldsymbol{P}^2 = \boldsymbol{P}`
  (their Eqs. (17), (34));
- zero internal force under large rigid motions, for every wrapped element;
- the tangent stiffness matrix equal to central finite differences of the
  internal force, for every element and for a model mixing the three
  kinematics, including rotation vectors of magnitude `1.9 \pi`;
- ``kinematics='linear'`` gives exactly ``KC0``, and the co-rotational
  tangent at ``u = 0`` equals ``KC0`` for flat elements;
- the applied-moment helpers, with finite differences of their load
  stiffness, and ``calc_fext`` declaring the configuration only with
  configuration-dependent moments;
- warped quadrilaterals, with rigid offsets and with the projector alone:
  rigid motions, finite differences, the eccentricity geometric stiffness,
  the tangent at ``u = 0`` against ``KC0`` and its eigenvalues, a linear
  twisted cantilever, the warnings;
- the wrappers of Tria3DSG, BeamLR, Truss (chord and mean-triad frames)
  and Spring (axis frame, mean-triad frame, zero length): rigid motions,
  finite differences, ``G S = I`` of the spring frame, and a model mixing
  them with the three kinematics;
- ``kinematics='corotational_von_karman'`` (``local='von_karman'`` of the
  wrappers): rigid motions, finite differences for every element alone and
  in a model mixing it with the other kinematics, the tangent at ``u = 0``
  equal to that of ``'corotational'``, the same force and tangent as
  ``'corotational'`` for the shells and BeamLR, the axial force of BeamC
  against Crisfield, Vol. 2, Eqs. (17.56a)-(17.56b), the order of the
  agreement with ``'von_karman'`` for small rotations, Spring and Truss
  treated as ``'corotational'``;
- the cantilever of Sze, Liu and Lo (2004), Section 3.2, under an end
  moment that bends it into a half circle, solved with
  :class:`pyfe3d.solver.Analysis` and Newton-Raphson, against the closed
  form of the circular arc and against the exact solution of the
  discretised problem.

References: Felippa, C. A. and Haugen, B., CMAME 194, 2285-2335, 2005,
https://doi.org/10.1016/j.cma.2004.07.035; Sze, K. Y., Liu, X. H. and Lo,
S. H., Finite Elements in Analysis and Design, 40(11), 1551-1569, 2004,
https://doi.org/10.1016/j.finel.2003.11.001.
"""
import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from pyfe3d import (Tria3R, Tria3RProbe, Quad4R, Quad4RProbe, Quad4,
                    Quad4Probe, BeamC, BeamCProbe, Tria3DSG, Tria3DSGProbe,
                    BeamLR, BeamLRProbe, Truss, TrussProbe, Spring,
                    SpringProbe, DOF)
from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d.beamprop import BeamProp
from pyfe3d.solver import Analysis, callbacks
from pyfe3d import corotational as cr
from pyfe3d.corotational import (CorotationalModel, CorotationalShell,
                                 CorotationalBeam, CorotationalTruss,
                                 CorotationalSpring, spin, axial, expso3,
                                 logso3, Lambda, dLambdaT_m, Hmat, Lmat)

ELEMENTS = ['Tria3R', 'Quad4R', 'Quad4', 'BeamC']
ANGLES = [0., 1e-12, 1e-6, 1e-3, 0.1, 0.2*(1 - 1e-9), 0.2*(1 + 1e-9), 1.,
          2.5, np.pi - 1e-6, np.pi, 4., 1.9*np.pi]


def unit(rng):
    n = rng.standard_normal(3)
    return n/np.linalg.norm(n)


def shell_prop():
    return isotropic_plate(thickness=0.1, E=200e9, nu=0.3)


def beam_prop():
    p = BeamProp()
    p.A = 0.01
    p.E = 200e9
    p.G = 80e9
    p.Iyy = 2e-6
    p.Izz = 1e-6
    p.J = 3e-6
    return p


def make_element(name, X0, nodes=None):
    r"""Element of type ``name`` on the nodes ``nodes`` (default ``0, 1,
    ...``) of the coordinates ``X0``, with its rotation matrix set"""
    nodes = list(range(len(X0))) if nodes is None else nodes
    elem = {'Tria3R': lambda: Tria3R(Tria3RProbe()),
            'Quad4R': lambda: Quad4R(Quad4RProbe()),
            'Quad4': lambda: Quad4(Quad4Probe()),
            'BeamC': lambda: BeamC(BeamCProbe())}[name]()
    for i, n in enumerate(nodes):
        setattr(elem, 'n%d' % (i + 1), n)
        setattr(elem, 'c%d' % (i + 1), DOF*n)
    x = np.asarray(X0, dtype=float).ravel()
    if name == 'BeamC':
        elem.update_rotation_matrix(0.2, 1., 0.3, x)
    else:
        elem.update_rotation_matrix(x)
    return elem


def single(name, flat=False):
    r"""Wrapped element alone, on a skewed or (quadrilaterals) warped
    geometry, or a flat one rotated in space"""
    if name == 'Tria3R':
        X0 = np.array([[0., 0, 0], [1.2, 0.1, 0.05], [0.3, 0.9, -0.02]])
    elif name == 'BeamC':
        X0 = np.array([[0., 0, 0], [1.0, 0.3, -0.2]])
    else:
        w = 0. if flat else 1.
        X0 = np.array([[0., 0, 0], [1.0, 0.05, 0.03*w], [1.1, 0.9, -0.02*w],
                       [-0.05, 1.0, 0.04*w]])
        if flat:
            X0 = X0 @ expso3([0.3, -0.2, 0.5]).T
    elem = make_element(name, X0)
    x = X0.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if name == 'BeamC':
            return CorotationalBeam(elem, beam_prop(), x)
        return CorotationalShell(elem, shell_prop(), x)


def rigid_motion(w, psi, t):
    r"""Degrees-of-freedom of a rigid motion of the element ``w``: rotation
    ``expso3(psi)`` about the centroid and translation ``t``, with the
    nodal rotation vectors set to ``psi`` (any magnitude)"""
    R = expso3(psi)
    xC0 = w.X0.mean(axis=0)
    u = np.zeros((w.nn, DOF))
    u[:, :3] = (w.X0 - xC0) @ R.T + xC0 + t - w.X0
    u[:, 3:] = psi
    return u.ravel()


def fd_jacobian(func, u, h=1e-6):
    J = np.zeros((func(u).size, u.size))
    for j in range(u.size):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        J[:, j] = (func(up) - func(um))/(2*h)
    return J


# ---------------------------------------------------------------------------
# rotation utilities
# ---------------------------------------------------------------------------
def test_spin_axial():
    rng = np.random.default_rng(0)
    v, w = rng.standard_normal((2, 3))
    np.testing.assert_allclose(spin(v) @ w, np.cross(v, w), atol=1e-15)
    np.testing.assert_allclose(axial(spin(v)), v, atol=1e-15)


@pytest.mark.parametrize('t', ANGLES)
def test_exp_log(t):
    r"""Rodrigues' formula against scipy; the logarithm as its inverse,
    with ``|theta| <= pi``"""
    rng = np.random.default_rng(1)
    for _ in range(5):
        psi = t*unit(rng)
        R = expso3(psi)
        np.testing.assert_allclose(R, Rotation.from_rotvec(psi).as_matrix(),
                                   rtol=0, atol=1e-15)
        theta = logso3(R)
        np.testing.assert_allclose(expso3(theta), R, rtol=0, atol=2e-15)
        assert np.linalg.norm(theta) <= np.pi + 1e-12
        if t < np.pi - 1e-3:
            np.testing.assert_allclose(theta, psi, rtol=0,
                                       atol=1e-15*max(1., t))


def test_log_at_pi():
    r"""Rotations of exactly ``pi`` about the axes and about a skew axis"""
    for R in (np.diag([1., -1, -1]), np.diag([-1., 1, -1]),
              np.diag([-1., -1, 1])):
        theta = logso3(R)
        assert np.isclose(np.linalg.norm(theta), np.pi, rtol=1e-15)
        np.testing.assert_allclose(expso3(theta), R, atol=1e-15)
    n = np.array([1., 2., -2.])/3
    R = 2*np.outer(n, n) - np.eye(3)
    theta = logso3(R)
    np.testing.assert_allclose(np.abs(theta), np.pi*np.abs(n), rtol=1e-14)


@pytest.mark.parametrize('t', ANGLES)
def test_Lambda_H(t):
    r"""``dR R^T = Spin(Lambda dpsi)`` by central differences; ``H`` is the
    inverse of ``Lambda``; ``Lambda(psi)^T = Lambda(-psi)`` and ``Lambda^T
    R = Lambda``"""
    rng = np.random.default_rng(2)
    h = 1e-6
    for _ in range(3):
        psi = t*unit(rng)
        R = expso3(psi)
        Lam = Lambda(psi)
        for j in range(3):
            dpsi = np.zeros(3)
            dpsi[j] = h
            dR = (expso3(psi + dpsi) - expso3(psi - dpsi))/(2*h)
            np.testing.assert_allclose(axial(dR @ R.T), Lam[:, j], rtol=0,
                                       atol=1e-9)
        np.testing.assert_allclose(Lam.T, Lambda(-psi), rtol=0, atol=1e-15)
        np.testing.assert_allclose(Lam.T @ R, Lam, rtol=0, atol=1e-15)
        if t < 2*np.pi*0.95:
            np.testing.assert_allclose(Hmat(psi) @ Lam, np.eye(3), rtol=0,
                                       atol=1e-13)


@pytest.mark.parametrize('t', [x for x in ANGLES if x < 3.])
def test_H_log(t):
    r"""``dtheta = H domega`` by central differences of the logarithm,
    ``theta = log(exp(Spin(domega)) R)``"""
    rng = np.random.default_rng(3)
    h = 1e-6
    theta = t*unit(rng)
    R = expso3(theta)
    H = Hmat(theta)
    for j in range(3):
        dw = np.zeros(3)
        dw[j] = h
        dth = (logso3(expso3(dw) @ R) - logso3(expso3(-dw) @ R))/(2*h)
        np.testing.assert_allclose(dth, H[:, j], rtol=0, atol=1e-9)


@pytest.mark.parametrize('t', [x for x in ANGLES if x < 3.])
def test_L(t):
    r"""``L = d(H^T m)/dtheta H`` by central differences along ``H e_j``"""
    rng = np.random.default_rng(4)
    h = 1e-6
    theta = t*unit(rng)
    m = rng.standard_normal(3)
    H = Hmat(theta)
    L = Lmat(theta, m)
    for j in range(3):
        dth = h*H[:, j]
        fd = (Hmat(theta + dth).T @ m - Hmat(theta - dth).T @ m)/(2*h)
        np.testing.assert_allclose(fd, L[:, j], rtol=0, atol=1e-9)


@pytest.mark.parametrize('t', ANGLES)
def test_dLambdaT_m(t):
    r"""``d(Lambda^T m)/dpsi`` by central differences; ``Spin(m)/2`` at
    ``psi = 0``, Crisfield (1997), Eq. (16.96)"""
    rng = np.random.default_rng(5)
    h = 1e-6
    psi = t*unit(rng)
    m = rng.standard_normal(3)
    fd = fd_jacobian(lambda p: Lambda(p).T @ m, psi, h)
    np.testing.assert_allclose(dLambdaT_m(psi, m), fd, rtol=0, atol=1e-9)
    if t == 0.:
        np.testing.assert_allclose(dLambdaT_m(psi, m), 0.5*spin(m),
                                   atol=1e-16)


def test_series_continuity():
    r"""The Taylor series and the closed forms agree at the switch angle"""
    for t0 in (cr.SERIES_ANGLE, cr.SERIES_ANGLE_ETA_MU):
        lo, hi = (np.array(cr._coefficients(t) + cr._eta_mu(t))
                  for t in (t0*(1 - 1e-12), t0*(1 + 1e-12)))
        np.testing.assert_allclose(lo, hi, rtol=1e-11, atol=0)


# ---------------------------------------------------------------------------
# element wrappers
# ---------------------------------------------------------------------------
def S_matrix(xl):
    n = len(xl)
    S = np.zeros((DOF*n, 3))
    for a in range(n):
        S[DOF*a:DOF*a + 3] = -spin(xl[a])
        S[DOF*a + 3:DOF*a + 6] = np.eye(3)
    return S


@pytest.mark.parametrize('name', ELEMENTS)
def test_local_stiffness(name):
    r"""Identity trick: ``Ke`` symmetric, ``Ke S0 = 0`` on a flat geometry,
    and ``T0 Ke T0^T`` equal to the ``KC0`` of the element"""
    w = single(name, flat=True)
    Ke = w.Ke
    scale = np.abs(Ke).max()
    assert np.abs(Ke - Ke.T).max() <= 1e-14*scale
    assert np.abs(Ke @ S_matrix(w.xtilde)).max() <= 1e-13*scale
    model = CorotationalModel(w.X0.ravel())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model.add(w.elem, w.prop, kinematics='linear')
    T = np.kron(np.eye(2*w.nn), w.T0)
    KC0 = model.KC0.toarray()[np.ix_(w.dofs, w.dofs)]
    assert np.abs(T @ Ke @ T.T - KC0).max() <= 1e-13*scale


@pytest.mark.parametrize('name', ELEMENTS)
def test_projector(name):
    r"""``G S = I`` and ``P^2 = P`` in deformed configurations"""
    w = single(name)
    rng = np.random.default_rng(6)
    for _ in range(3):
        u = rigid_motion(w, 2*unit(rng), rng.standard_normal(3))
        u += 0.02*rng.standard_normal(u.size)
        ue, psie = w._split(u)
        TR, xl, d, aux = w._kinematics(ue, psie)
        S = S_matrix(xl)
        G = w._G(xl, aux)
        np.testing.assert_allclose(G @ S, np.eye(3), rtol=0, atol=1e-13)
        P = w._Pu - S @ G
        np.testing.assert_allclose(P @ P, P, rtol=0, atol=1e-13)


@pytest.mark.parametrize('name', ELEMENTS)
@pytest.mark.parametrize('angle', [0.7, np.pi/2, np.pi, np.pi - 1e-7,
                                   1.9*np.pi])
def test_rigid_motion(name, angle):
    r"""No internal force under a large rigid rotation and translation"""
    w = single(name)
    rng = np.random.default_rng(7)
    for _ in range(3):
        u = rigid_motion(w, angle*unit(rng), rng.standard_normal(3))
        f = w.evaluate(u, tangent=False)
        TR, d = w.kinematics(u)
        assert np.abs(d).max() < 1e-13
        assert np.abs(f).max() <= 1e-13*np.abs(w.Ke).max()


@pytest.mark.parametrize('name', ELEMENTS)
@pytest.mark.parametrize('case', ['large', 'near_2pi'])
def test_tangent_fd(name, case):
    r"""Tangent stiffness matrix equal to central differences of the
    internal force, relative to the largest entry of the tangent and to
    the largest entry of its geometric part; at ``|psi| = 1.9 pi`` too"""
    w = single(name)
    rng = np.random.default_rng(8)
    if case == 'large':
        psi = np.array([0.7, -1.9, 2.4])
        amp = 0.03
    else:
        psi = np.array([0., 1.9*np.pi, 0.])
        amp = 1e-3
    u = rigid_motion(w, psi, np.array([0.3, -2., 1.]))
    u += amp*rng.standard_normal(u.size)
    f, KM, KG = w.evaluate(u)
    K = KM + KG
    Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
    err = np.abs(K - Kfd).max()
    print('%s %s: FD error %.1e of max|K|, %.1e of max|KG|'
          % (name, case, err/np.abs(K).max(), err/np.abs(KG).max()))
    assert err <= 1e-8*np.abs(K).max()
    if case == 'large':
        assert err <= 1e-6*np.abs(KG).max()


def test_beam_projection_term():
    r"""The variation of ``G`` of the beam frame matters: without it (F&H
    Eq. (47) alone) the tangent is not consistent"""
    w = single('BeamC')
    rng = np.random.default_rng(9)
    u = rigid_motion(w, np.array([0.7, -1.9, 2.4]), np.zeros(3))
    u += 0.05*rng.standard_normal(u.size)
    errors = {}
    for exact in (True, False):
        w.exact_KGP = exact
        f, KM, KG = w.evaluate(u)
        Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
        errors[exact] = np.abs(KM + KG - Kfd).max()/np.abs(KG).max()
    w.exact_KGP = True
    print('BeamC FD error of max|KG|: %.1e exact, %.1e F&H (47)'
          % (errors[True], errors[False]))
    assert errors[True] < 1e-7
    assert errors[False] > 100*errors[True]


def test_beam_q2_guard():
    r"""Nodal triads rotated by 90 degrees about the local z axis make the
    mean triad parallel to the chord: NaN and a warning"""
    w = single('BeamC')
    u = np.zeros(12)
    for a in range(2):
        u[6*a + 3:6*a + 6] = 0.5*np.pi*w.T0[:, 2]
    with pytest.warns(RuntimeWarning, match='undefined'):
        f = w.evaluate(u, tangent=False)
    assert np.all(np.isnan(f))
    f, KM, KG = w.evaluate(u)
    assert np.all(np.isnan(KM)) and np.all(np.isnan(KG))


def test_frame_mismatch():
    r"""A shell whose ``update_rotation_matrix`` was called with other
    coordinates is rejected"""
    X0 = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]])
    elem = make_element('Tria3R', X0)
    with pytest.raises(ValueError, match='update_rotation_matrix'):
        CorotationalShell(elem, shell_prop(), (X0 @ expso3([0, 0, 1.]).T).ravel())


# ---------------------------------------------------------------------------
# assembler
# ---------------------------------------------------------------------------
MIXED_X = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0],
                    [2., 0, 0.1], [2., 1, 0.1], [0.5, 2, 0], [0., 0, 1]])


def mixed_model():
    r"""Co-rotational Quad4R, Quad4, Tria3R and BeamC, a von Kármán Quad4R
    and Tria3R and a linear BeamC, sharing nodes"""
    x = MIXED_X.ravel()
    model = CorotationalModel(x)
    sp, bp = shell_prop(), beam_prop()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model.add(make_element('Quad4R', MIXED_X, [0, 1, 2, 3]), sp)
        model.add(make_element('Quad4', MIXED_X, [1, 4, 5, 2]), sp)
        model.add(make_element('Tria3R', MIXED_X, [3, 2, 6]), sp)
        model.add(make_element('BeamC', MIXED_X, [0, 7]), bp)
        model.add(make_element('Quad4R', MIXED_X, [1, 4, 5, 2]), sp,
                  kinematics='von_karman')
        model.add(make_element('Tria3R', MIXED_X, [2, 5, 6]), sp,
                  kinematics='von_karman')
        model.add(make_element('BeamC', MIXED_X, [3, 7]), bp,
                  kinematics='linear')
    return model


@pytest.mark.parametrize('case', ['moderate', 'near_2pi'])
def test_mixed_model_tangent_fd(case):
    r"""``calc_kC + calc_kG`` of a model with the three kinematics equal to
    central differences of ``calc_fint``"""
    model = mixed_model()
    rng = np.random.default_rng(10)
    u = np.zeros(model.N)
    if case == 'near_2pi':
        X = MIXED_X
        R = expso3([0., 0., 1.9*np.pi])
        u.reshape(-1, DOF)[:, :3] = X @ R.T - X
        u.reshape(-1, DOF)[:, 3:] = [0., 0., 1.9*np.pi]
    u += 0.02*rng.standard_normal(model.N)
    with pytest.warns(RuntimeWarning, match='shared'):
        K = (model.calc_kC(u, NLgeom=True) + model.calc_kG(u, NLgeom=True))
    K = K.toarray()
    Kfd = fd_jacobian(model.calc_fint, u)
    err = np.abs(K - Kfd).max()/np.abs(K).max()
    print('mixed model %s: FD error %.1e of max|K|' % (case, err))
    assert err < 1e-8


def test_linear_kinematics():
    r"""``kinematics='linear'``: ``calc_fint = KC0 u`` and ``calc_kC =
    KC0``, ``calc_kG = 0`` at any configuration"""
    x = MIXED_X.ravel()
    model = CorotationalModel(x)
    sp, bp = shell_prop(), beam_prop()
    model.add(make_element('Quad4R', MIXED_X, [0, 1, 2, 3]), sp,
              kinematics='linear')
    model.add(make_element('Tria3R', MIXED_X, [3, 2, 6]), sp,
              kinematics='linear')
    model.add(make_element('BeamC', MIXED_X, [0, 7]), bp, kinematics='linear')
    rng = np.random.default_rng(11)
    u = 1e-4*rng.standard_normal(model.N)
    KC0 = model.KC0
    np.testing.assert_allclose(model.calc_fint(u), KC0 @ u, rtol=0,
                               atol=1e-15*np.abs(KC0 @ u).max())
    assert abs(model.calc_kC(u, NLgeom=True) - KC0).max() == 0
    assert model.calc_kG(u, NLgeom=True).nnz == 0
    assert abs(model.calc_kC() - KC0).max() == 0


def flat_model():
    r"""Flat Quad4R, Quad4 and Tria3R plus a BeamC, all co-rotational"""
    X = np.array([[0., 0, 0], [1., 0.1, 0], [1.1, 0.9, 0], [0., 1., 0],
                  [2., 0., 0], [2.1, 1.2, 0], [0.5, 2., 0], [0., 0., 1.]])
    X = X @ expso3([0.4, 0.3, -0.2]).T
    model = CorotationalModel(X.ravel())
    sp, bp = shell_prop(), beam_prop()
    model.add(make_element('Quad4R', X, [0, 1, 2, 3]), sp)
    model.add(make_element('Quad4', X, [1, 4, 5, 2]), sp)
    model.add(make_element('Tria3R', X, [3, 2, 6]), sp)
    model.add(make_element('BeamC', X, [0, 7]), bp)
    return model


def test_tangent_at_base_equals_KC0():
    r"""At ``u = 0`` the co-rotational tangent of flat elements equals
    ``KC0``, and the internal force at small displacements ``eps*u``
    differs from ``KC0 eps u`` by a relative error of the order of
    ``eps``"""
    model = flat_model()
    u0 = np.zeros(model.N)
    K = (model.calc_kC(u0, NLgeom=True) + model.calc_kG(u0, NLgeom=True))
    KC0 = model.KC0
    assert abs(K - KC0).max() <= 1e-13*abs(KC0).max()
    assert model.calc_kG(u0, NLgeom=True).count_nonzero() == 0 or \
        abs(model.calc_kG(u0, NLgeom=True)).max() <= 1e-13*abs(KC0).max()
    rng = np.random.default_rng(12)
    u = rng.standard_normal(model.N)
    for eps in (1e-4, 1e-6):
        f = model.calc_fint(eps*u)
        f0 = KC0 @ (eps*u)
        rel = np.abs(f - f0).max()/np.abs(f0).max()
        print('eps %.0e: |fint - KC0 u|/|KC0 u| = %.1e' % (eps, rel))
        assert rel < 10*eps


def test_postprocessing():
    r"""Nodal rotation matrices and deformational displacements; the probe
    written by ``update_probe`` gives the local forces ``Ke d``"""
    model = flat_model()
    rng = np.random.default_rng(13)
    u = 0.01*rng.standard_normal(model.N)
    R = model.nodal_rotations(u)
    assert R.shape == (model.N//DOF, 3, 3)
    np.testing.assert_allclose(R[3], expso3(u[DOF*3 + 3:DOF*3 + 6]))
    ds = model.deformational_displacements(u)
    assert len(ds) == len(model.corotational)
    for w, d in zip(model.corotational, ds):
        w.update_probe(u)
        w.elem.update_probe_finte(w.prop, **w.elem_kwargs)
        np.testing.assert_allclose(np.asarray(w.elem.probe.finte), w.Ke @ d,
                                   rtol=0, atol=1e-9*np.abs(w.Ke @ d).max())


# ---------------------------------------------------------------------------
# applied moments
# ---------------------------------------------------------------------------
def test_moment_helpers():
    r"""Moments conjugate to the rotation vector go to ``fext``; the
    configuration-dependent moments about fixed axes and following the node
    have the load stiffness ``-inc dF/dc`` in ``calc_kG``, checked by
    central differences of ``calc_fint``; ``calc_fext(c=c)`` is
    ``-dR/dlbd``"""
    model = CorotationalModel(np.zeros(6))
    M1 = np.array([0.3, -1.2, 0.8])
    M2 = np.array([-0.5, 0.4, 1.1])
    model.add_moment(0, M1, kind='fixed_axes')
    model.add_moment(1, M2, kind='follower')
    model.add_moment(1, [1., 2., 3.])
    np.testing.assert_allclose(model.fext[9:12], [1., 2., 3.])
    rng = np.random.default_rng(14)
    lbd = 0.7
    for t in (0.5, 3., 1.9*np.pi):
        u = np.zeros(model.N)
        u[3:6] = t*unit(rng)
        u[9:12] = t*unit(rng)
        F = np.concatenate([np.zeros(3), Lambda(u[3:6]).T @ M1,
                            np.zeros(3),
                            Lambda(u[9:12]).T @ expso3(u[9:12]) @ M2])
        np.testing.assert_allclose(model.calc_fext(c=u) - model.fext, F,
                                   atol=1e-14)
        # NOTE the residual and its derivative with respect to lbd
        R = lambda c, l: model.calc_fint(c, inc=l) - l*model.calc_fext()
        np.testing.assert_allclose(R(u, lbd), -lbd*(F + model.fext),
                                   atol=1e-14)
        dRdl = (R(u, lbd + 1e-6) - R(u, lbd - 1e-6))/2e-6
        np.testing.assert_allclose(-dRdl, model.calc_fext(c=u), atol=1e-8)
        K = (model.calc_kC(u, NLgeom=True)
             + model.calc_kG(u, NLgeom=True, inc=lbd)).toarray()
        Kfd = fd_jacobian(lambda c: model.calc_fint(c, inc=lbd), u)
        err = np.abs(K - Kfd).max()/np.abs(K).max()
        print('moments at |psi| = %.2f: FD error %.1e' % (t, err))
        assert err < 1e-8
        # NOTE the follower load stiffness equals its general form
        psi, m = u[9:12], expso3(u[9:12]) @ M2
        Lam = Lambda(psi)
        np.testing.assert_allclose(-K[9:12, 9:12]/lbd,
                                   dLambdaT_m(psi, m) - Lam.T @ spin(m) @ Lam,
                                   atol=1e-12)
    # NOTE about a fixed axis, Lambda^T n = n
    n = unit(rng)
    np.testing.assert_allclose(Lambda(2.*n).T @ n, n, atol=1e-15)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def test_sze_cantilever_half_circle():
    r"""Cantilever of Sze et al. (2004), Section 3.2, Fig. 3a (``L = 12``,
    ``b = 1``, ``h = 0.1``, ``E = 1.2e6``, ``nu = 0``), 16 co-rotational
    BeamC, under the end moment ``M = M_max/2 = pi E I/L`` that bends it
    into a half circle, the tip rotation reaching ``pi``. Newton-Raphson
    with :class:`pyfe3d.solver.Analysis` in 10 increments.

    Closed form of the circular arc (Sze et al. Section 3.2; Table 3a at
    ``M/M_max = 0.5``: ``-U = 12.000``, ``W = 7.639``): ``U/L =
    sin(phi)/phi - 1``, ``W/L = (1 - cos(phi))/phi``, ``phi = M L/(E I)``.
    The discretised problem has its exact solution too: the axial and shear
    forces vanish, so every chord keeps its length ``l0 = L/16`` and turns
    by ``phi/16``, the nodes lying on the polygon inscribed in a circle;
    its tip differs from the arc by about ``(phi/16)**2/24`` relative.
    """
    L, b, h, E = 12., 1., 0.1, 1.2e6
    EI = E*b*h**3/12
    ne = 16
    prop = BeamProp()
    prop.A = b*h
    prop.E = E
    prop.G = E/2
    prop.Iyy = b*h**3/12
    prop.Izz = h*b**3/12
    prop.J = b*h**3/3
    xs = np.linspace(0, L, ne + 1)
    x = np.vstack((xs, 0*xs, 0*xs)).T.ravel()
    model = CorotationalModel(x)
    probe = BeamCProbe()
    for k in range(ne):
        elem = BeamC(probe)
        elem.n1, elem.n2 = k, k + 1
        elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
        elem.update_rotation_matrix(0., 1., 0., x)
        model.add(elem, prop)
    phi = np.pi
    # NOTE the moment about -y bends the cantilever towards +z
    model.add_moment(ne, [0., -phi*EI/L, 0.])
    bu = np.ones(model.N, dtype=bool)
    bu[:DOF] = False
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC,
                  model.calc_kG, bu=bu)
    an.initialInc = 0.1
    an.maxInc = 0.1
    increments, cs = an.static(NLgeom=True, silent=True)
    assert np.isclose(increments[-1], 1.)
    tip = DOF*ne
    U, W, psi_y = cs[-1][tip], cs[-1][tip + 2], cs[-1][tip + 4]
    U_arc, W_arc = L*(np.sin(phi)/phi - 1), L*(1 - np.cos(phi))/phi
    j = np.arange(ne) + 0.5
    U_pol = L/ne*np.cos(j*phi/ne).sum() - L
    W_pol = L/ne*np.sin(j*phi/ne).sum()
    print('-U %.6f (arc %.6f, polygon %.6f), W %.6f (arc %.6f, polygon %.6f)'
          % (-U, -U_arc, -U_pol, W, W_arc, W_pol))
    assert np.isclose(-U_arc, 12.000, atol=5e-4)
    assert np.isclose(W_arc, 7.639, atol=5e-4)
    assert np.isclose(psi_y, -phi, rtol=1e-9)
    assert abs(U - U_pol) <= 1e-7*L
    assert abs(W - W_pol) <= 1e-7*L
    assert abs(W - W_arc) <= 2e-3*W_arc
    assert abs(U - U_arc) <= 1e-7*L


# ---------------------------------------------------------------------------
# warped quadrilaterals
# ---------------------------------------------------------------------------
def warped_quad(name, warp_ratio, warping, thickness=0.1):
    r"""Quadrilateral of ``single`` with its warping scaled to the given
    warp ratio"""
    X0 = np.array([[0., 0, 0], [1.0, 0.05, 0.03], [1.1, 0.9, -0.02],
                   [-0.05, 1.0, 0.04]])
    prop = isotropic_plate(thickness=thickness, E=200e9, nu=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = CorotationalShell(make_element(name, X0), prop, X0.ravel(),
                              warping=None)
        X0[:, 2] *= warp_ratio/w.warp_ratio
        return CorotationalShell(make_element(name, X0), prop, X0.ravel(),
                                 warping=warping)


@pytest.mark.parametrize('name', ['Quad4', 'Quad4R'])
@pytest.mark.parametrize('warping', [None, 'offsets'])
def test_warped_quad_rigid_motion_and_fd(name, warping):
    r"""Warped quadrilateral (warp ratio 0.05), both treatments: zero force
    under large rigid motions and the tangent equal to central differences,
    at large rotations and at ``|psi| = 1.9 pi``"""
    w = warped_quad(name, 0.05, warping)
    rng = np.random.default_rng(20)
    for angle in (0.7, np.pi, 1.9*np.pi):
        u = rigid_motion(w, angle*unit(rng), rng.standard_normal(3))
        assert np.abs(w.kinematics(u)[1]).max() < 1e-13
        f = w.evaluate(u, tangent=False)
        assert np.abs(f).max() <= 1e-13*np.abs(w.Ke).max()
    for psi, amp in (([0.7, -1.9, 2.4], 0.03), ([0., 1.9*np.pi, 0.], 1e-3)):
        u = rigid_motion(w, np.array(psi), np.array([0.3, -2., 1.]))
        u += amp*rng.standard_normal(u.size)
        f, KM, KG = w.evaluate(u)
        K = KM + KG
        Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
        err = np.abs(K - Kfd).max()
        assert err <= 1e-8*np.abs(K).max()
        if amp > 0.01:
            assert err <= 1e-6*np.abs(KG).max()


def test_eccentricity_stiffness_needed():
    r"""Without the eccentricity geometric stiffness the tangent of the
    warped quadrilateral with offsets is not consistent"""
    w = warped_quad('Quad4', 0.05, 'offsets')
    rng = np.random.default_rng(25)
    u = rigid_motion(w, np.array([0.7, -1.9, 2.4]), np.array([0.3, -2., 1.]))
    u += 0.03*rng.standard_normal(u.size)
    f, KM, KG = w.evaluate(u)
    K = KM + KG
    Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
    _, psie = w._split(u)
    Kecc = np.zeros((24, 24))
    for a in range(4):
        r = expso3(psie[a]) @ w._e[a]
        n = f[DOF*a:DOF*a + 3]
        Lam = Lambda(psie[a])
        Kecc[DOF*a + 3:DOF*a + 6, DOF*a + 3:DOF*a + 6] = (
            dLambdaT_m(psie[a], np.cross(n, r)) - Lam.T @ spin(n) @ spin(r)
            @ Lam)
    scale = np.abs(K).max()
    assert np.abs(K - Kfd).max() <= 1e-8*scale
    assert np.abs(K - Kecc - Kfd).max() >= 1e-4*scale


@pytest.mark.parametrize('name', ['Quad4', 'Quad4R'])
@pytest.mark.parametrize('warp_ratio', [0.02, 0.05])
def test_warped_quad_base_tangent(name, warp_ratio):
    r"""At ``u = 0``: with ``'offsets'`` the tangent is ``J^T KC0 J``, with
    ``J`` the rigid links, and with ``None`` it is ``P^T KC0 P``; both
    annihilate the rigid rotations of the warped element, which ``KC0``
    does not, differ from ``KC0`` by the order of the warp ratio and keep
    its lowest non-zero eigenvalue (Quad4 and Quad4R have six zero
    eigenvalues in this geometry, Quad4 eight with ``shear_model = 0``, the
    one-point transverse shear of its versions up to 0.11)"""
    for warping in (None, 'offsets'):
        w = warped_quad(name, warp_ratio, warping)
        f, KM, KG = w.evaluate(np.zeros(24))
        K = KM + KG
        assert np.abs(f).max() <= 1e-14*np.abs(w.Ke).max()
        assert np.abs(KG).max() <= 1e-14*np.abs(KM).max()
        T = np.kron(np.eye(8), w.T0)
        KC0 = T @ w.Ke @ T.T
        scale = np.abs(KC0).max()
        S0 = S_matrix(w.X0 - w.X0.mean(axis=0))
        assert np.abs(K @ S0).max() <= 1e-14*scale
        assert np.abs(KC0 @ S0).max() >= 0.5*warp_ratio*scale
        if warping == 'offsets':
            J = np.eye(24)
            for a in range(4):
                e = w.xtilde[a, 2]*w.T0[:, 2]
                J[DOF*a:DOF*a + 3, DOF*a + 3:DOF*a + 6] = spin(e)
            assert np.abs(K - J.T @ KC0 @ J).max() <= 1e-14*scale
        else:
            xl = (w.X0 - w.X0.mean(axis=0)) @ w.T0
            P = w._Pu - S_matrix(xl) @ w._G(xl, None)
            Pg = T @ P @ T.T
            assert np.abs(K - Pg.T @ KC0 @ Pg).max() <= 1e-13*scale
        diff = np.abs(K - KC0).max()/scale
        assert 0.1*warp_ratio < diff < 2*warp_ratio
        ev0 = np.linalg.eigvalsh(KC0)
        nz = int((np.abs(ev0) < 1e-10*ev0.max()).sum())
        assert nz == 6
        ev = np.linalg.eigvalsh(0.5*(K + K.T))
        assert int((np.abs(ev) < 1e-10*ev.max()).sum()) == nz
        ratio = ev[nz]/ev0[nz]
        print('%s warp %.2f %s: lowest non-zero eigenvalue ratio %.4f, '
              '|K - KC0|/|KC0| %.1e' % (name, warp_ratio, warping, ratio,
                                        diff))
        assert abs(ratio - 1) < 1e-3


def twisted_cantilever(nx, ny, name, warping):
    r"""Cantilever of length 12, width 1.1 and thickness 0.32, twisted by
    90 degrees, clamped at ``x = 0``, as ``nx`` by ``ny`` quadrilaterals;
    linear tip displacements under unit tip loads along `y` and `z`, with
    the co-rotational tangent at ``u = 0`` and with ``KC0``"""
    L, b = 12., 1.1
    s = np.linspace(0, L, nx + 1)
    eta = np.linspace(-b/2, b/2, ny + 1)
    S, E = np.meshgrid(s, eta, indexing='ij')
    th = 0.5*np.pi*S/L
    X = np.column_stack((S.ravel(), (E*np.cos(th)).ravel(),
                         (E*np.sin(th)).ravel()))
    nid = np.arange(X.shape[0]).reshape(nx + 1, ny + 1)
    model = CorotationalModel(X.ravel())
    prop = isotropic_plate(thickness=0.32, E=29e6, nu=0.22)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(nx):
            for j in range(ny):
                nodes = [nid[i, j], nid[i + 1, j], nid[i + 1, j + 1],
                         nid[i, j + 1]]
                model.add(make_element(name, X, nodes), prop,
                          warping=warping)
    bu = np.ones(model.N, dtype=bool)
    for n in nid[0]:
        bu[DOF*n:DOF*n + DOF] = False
    wts = np.ones(ny + 1)
    wts[[0, -1]] = 0.5
    wts /= wts.sum()
    u0 = np.zeros(model.N)
    out = {}
    for label, K in (('K0', model.calc_kC(u0, NLgeom=True)
                      + model.calc_kG(u0, NLgeom=True)),
                     ('KC0', model.KC0)):
        Kb = K.tocsr()[bu][:, bu].tocsc()
        tip = []
        for comp in (1, 2):
            f = np.zeros(model.N)
            f[DOF*nid[-1] + comp] = wts
            u = np.zeros(model.N)
            u[bu] = spsolve(Kb, f[bu])
            tip.append(wts @ u[DOF*nid[-1] + comp])
        out[label] = np.array(tip)
    return out, max(w.warp_ratio for w in model.corotational)


def test_twisted_cantilever_offsets():
    r"""Linear twisted cantilever, 6x1 Quad4 (warp ratio 0.049): with the
    rigid offsets the tip displacements are within 1 % of those of a
    96x16 Quad4 mesh with offsets (computed once, ``uy = 1.75261e-3``,
    ``uz = 5.41735e-3``; a 48x8x2 Tria3R mesh gives ``1.7430e-3``,
    ``5.4117e-3``), with ``warping=None`` they are 8-11 % too small and
    with the ``KC0`` of the flat element more than 50 % too small; with
    ``'offsets'`` the ``KC0`` of the model, with the element's own rigid
    offsets, gives the same displacements as the tangent at ``u = 0``"""
    ref = np.array([1.75261e-3, 5.41735e-3])
    out, warp = twisted_cantilever(6, 1, 'Quad4', 'offsets')
    out_none, _ = twisted_cantilever(6, 1, 'Quad4', None)
    err = out['K0']/ref - 1
    err_none = out_none['K0']/ref - 1
    err_kc0 = out_none['KC0']/ref - 1
    print('warp %.3f: error offsets %s, None %s, flat KC0 %s'
          % (warp, err, err_none, err_kc0))
    assert np.isclose(warp, 0.049, atol=1e-3)
    assert np.all(np.abs(err) < 0.01)
    assert np.all(err_none < -0.05)
    assert np.all(err_kc0 < -0.5)
    np.testing.assert_allclose(out['KC0'], out['K0'], rtol=1e-10)


def test_warping_option_and_warnings():
    r"""``warping`` accepts ``'offsets'`` (default) and ``None``; the
    warnings follow :data:`WARP_RATIO_WARNING` and
    :data:`WARP_RATIO_WARNING_OFFSETS`; flat elements get no offsets"""
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        w = warped_quad_warn('Quad4', 0.05, 'offsets')
        assert w.warping == 'offsets' and w._e is not None
        X0 = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
        w = CorotationalShell(make_element('Quad4', X0), shell_prop(),
                              X0.ravel())
        assert w.warping == 'offsets' and w.warp_ratio == 0.
        assert w._e is None
    with pytest.warns(RuntimeWarning, match='warping=None'):
        warped_quad_warn('Quad4R', 0.02, None)
    with pytest.warns(RuntimeWarning, match='verified up to'):
        warped_quad_warn('Quad4R', 0.08, 'offsets')
    with pytest.raises(ValueError, match='warping'):
        X0 = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0]])
        CorotationalShell(make_element('Quad4', X0), shell_prop(),
                          X0.ravel(), warping='rigid')


def warped_quad_warn(name, warp_ratio, warping):
    r"""As ``warped_quad``, without silencing the warnings of the last
    wrapper"""
    X0 = np.array([[0., 0, 0], [1.0, 0.05, 0.03], [1.1, 0.9, -0.02],
                   [-0.05, 1.0, 0.04]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        w = CorotationalShell(make_element(name, X0), shell_prop(),
                              X0.ravel(), warping=None)
    X0[:, 2] *= warp_ratio/w.warp_ratio
    return CorotationalShell(make_element(name, X0), shell_prop(),
                             X0.ravel(), warping=warping)


# ---------------------------------------------------------------------------
# Tria3DSG, BeamLR, Truss and Spring
# ---------------------------------------------------------------------------
NEW = ['Tria3DSG', 'BeamLR', 'Truss_chord', 'Truss_mean_triad',
       'Spring_axis', 'Spring_triad', 'Spring_zero_length']


def set_nodes(elem, nodes):
    for i, n in enumerate(nodes):
        setattr(elem, 'n%d' % (i + 1), n)
        setattr(elem, 'c%d' % (i + 1), DOF*n)


def spring(X0, nodes, orientation, frame='auto', axial_only=False):
    r"""Spring with distinct stiffnesses and the given orientation ``(xi,
    xj, xk, vxyi, vxyj, vxyk)``"""
    e = Spring(SpringProbe())
    set_nodes(e, nodes)
    e.kxe, e.kye, e.kze = 1e6, 2e6, 3e6
    if axial_only:
        e.kye = e.kze = 0.
    e.krxe, e.krye, e.krze = 4e5, 5e5, 6e5
    e.update_rotation_matrix(*orientation)
    return e


def new_element(case, X0, nodes):
    r"""Element of the case ``case`` of ``NEW`` on ``nodes``, and its
    property and wrapper keyword arguments"""
    x = X0.ravel()
    if case == 'Tria3DSG':
        e = Tria3DSG(Tria3DSGProbe())
        set_nodes(e, nodes)
        e.alpha_shear_locking = 0.3
        e.update_rotation_matrix(x)
        return e, shell_prop(), {}
    if case == 'BeamLR':
        e = BeamLR(BeamLRProbe())
        set_nodes(e, nodes)
        e.update_rotation_matrix(0.2, 1., 0.3, x)
        return e, beam_prop(), {}
    if case.startswith('Truss'):
        e = Truss(TrussProbe())
        set_nodes(e, nodes)
        e.update_rotation_matrix(x)
        p = beam_prop()
        if case == 'Truss_chord':
            p.J = 0.
        return e, p, {}
    d = X0[nodes[1]] - X0[nodes[0]]
    if case == 'Spring_axis':
        return spring(X0, nodes, (*d, 0.2, 1., 0.3), axial_only=True), \
            None, {}
    return spring(X0, nodes, (0.3, -0.5, 0.8, 0.2, 1., 0.3)), None, {}


def single_new(case):
    r"""Wrapped element of the case ``case`` of ``NEW`` alone"""
    if case == 'Tria3DSG':
        X0 = np.array([[0., 0, 0], [1.2, 0.1, 0.05], [0.3, 0.9, -0.02]])
    elif case == 'Spring_zero_length':
        X0 = np.array([[0.4, -0.3, 0.2], [0.4, -0.3, 0.2]])
    else:
        X0 = np.array([[0., 0, 0], [1.0, 0.3, -0.2]])
    elem, prop, kw = new_element(case, X0, list(range(len(X0))))
    return cr._wrapper_class(elem)(elem, prop, X0.ravel(), **kw)


def test_new_wrappers_frames():
    r"""Wrapper classes and frames chosen by default"""
    expected = {'Tria3DSG': (CorotationalShell, None),
                'BeamLR': (CorotationalBeam, None),
                'Truss_chord': (CorotationalTruss, 'chord'),
                'Truss_mean_triad': (CorotationalTruss, 'mean_triad'),
                'Spring_axis': (CorotationalSpring, 'axis'),
                'Spring_triad': (CorotationalSpring, 'triad'),
                'Spring_zero_length': (CorotationalSpring, 'triad')}
    for case, (cls, frame) in expected.items():
        w = single_new(case)
        assert type(w) is cls
        assert getattr(w, 'frame', None) == frame
    X0 = np.array([[0., 0, 0], [1.0, 0.3, -0.2]])
    e, p, _ = new_element('Truss_mean_triad', X0, [0, 1])
    with pytest.raises(ValueError, match='torsional'):
        CorotationalTruss(e, p, X0.ravel(), frame='chord')
    e, _, _ = new_element('Spring_triad', X0, [0, 1])
    with pytest.raises(ValueError, match='axis'):
        CorotationalSpring(e, None, X0.ravel(), frame='axis')
    d = X0[1] - X0[0]
    e = spring(X0, [0, 1], (*d, 0.2, 1., 0.3))
    with pytest.raises(ValueError, match='kye'):
        CorotationalSpring(e, None, X0.ravel(), frame='axis')
    # NOTE an explicit 'triad' frame is accepted along the axis too
    w = CorotationalSpring(e, None, X0.ravel(), frame='triad')
    assert w.frame == 'triad'


def test_tria3dsg_options_carried():
    r"""``Ke`` of the Tria3DSG wrapper is the element's own ``KC0`` in
    element axes, including ``alpha_shear_locking``"""
    X0 = np.array([[0., 0, 0], [1.2, 0.1, 0.05], [0.3, 0.9, -0.02]])
    Ks = []
    for alpha in (0., 0.3):
        e, p, _ = new_element('Tria3DSG', X0, [0, 1, 2])
        e.alpha_shear_locking = alpha
        w = CorotationalShell(e, p, X0.ravel())
        model = CorotationalModel(X0.ravel())
        model.add(e, p, kinematics='linear')
        T = np.kron(np.eye(6), w.T0)
        KC0 = model.KC0.toarray()
        assert np.abs(T @ w.Ke @ T.T - KC0).max() <= 1e-13*np.abs(KC0).max()
        Ks.append(w.Ke)
    assert np.abs(Ks[0] - Ks[1]).max() > 1e-3*np.abs(Ks[0]).max()


@pytest.mark.parametrize('case', NEW)
@pytest.mark.parametrize('angle', [0.7, np.pi, 1.9*np.pi])
def test_new_wrappers_rigid_motion(case, angle):
    r"""No internal force under a large rigid rotation and translation"""
    w = single_new(case)
    rng = np.random.default_rng(21)
    for _ in range(3):
        u = rigid_motion(w, angle*unit(rng), rng.standard_normal(3))
        f = w.evaluate(u, tangent=False)
        assert np.abs(w.kinematics(u)[1]).max() < 1e-12
        assert np.abs(f).max() <= 1e-13*np.abs(w.Ke).max()


@pytest.mark.parametrize('case', NEW)
@pytest.mark.parametrize('sub', ['large', 'near_2pi'])
def test_new_wrappers_tangent_fd(case, sub):
    r"""Tangent stiffness matrix equal to central differences"""
    w = single_new(case)
    rng = np.random.default_rng(22)
    if sub == 'large':
        psi, amp = np.array([0.7, -1.9, 2.4]), 0.03
    else:
        psi, amp = np.array([0., 1.9*np.pi, 0.]), 1e-3
    u = rigid_motion(w, psi, np.array([0.3, -2., 1.]))
    u += amp*rng.standard_normal(u.size)
    f, KM, KG = w.evaluate(u)
    K = KM + KG
    Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
    err = np.abs(K - Kfd).max()
    print('%s %s: FD error %.1e of max|K|, %.1e of max|KG|'
          % (case, sub, err/np.abs(K).max(), err/np.abs(KG).max()))
    assert err <= 1e-8*np.abs(K).max()
    if sub == 'large':
        assert err <= 1e-6*np.abs(KG).max()


def test_spring_triad_projector():
    r"""Spring with the mean-triad frame: ``G S = I`` and ``P^2 = P`` in
    deformed configurations, for distinct and coincident nodes"""
    for case in ('Spring_triad', 'Spring_zero_length'):
        w = single_new(case)
        rng = np.random.default_rng(23)
        for _ in range(3):
            u = rigid_motion(w, 2*unit(rng), rng.standard_normal(3))
            u += 0.05*rng.standard_normal(u.size)
            TR, xl, d, aux = w._kinematics(*w._split(u))
            S = S_matrix(xl)
            G = w._G(xl, aux)
            np.testing.assert_allclose(G @ S, np.eye(3), rtol=0, atol=1e-13)
            P = w._Pu - S @ G
            np.testing.assert_allclose(P @ P, P, rtol=0, atol=1e-13)


def test_spring_base_tangent():
    r"""At ``u = 0`` the tangent of a zero-length spring and of an axial
    spring along its axis equals ``KC0``; that of a spring with transverse
    stiffnesses between distinct nodes differs, because the linear element
    does not balance the moments of its transverse forces"""
    for case, equal in (('Spring_zero_length', True), ('Spring_axis', True),
                        ('Spring_triad', False)):
        w = single_new(case)
        f, KM, KG = w.evaluate(np.zeros(12))
        T = np.kron(np.eye(4), w.T0)
        KC0 = T @ w.Ke @ T.T
        diff = np.abs(KM + KG - KC0).max()/np.abs(KC0).max()
        assert (diff < 1e-13) == equal


def test_new_kinematics_accepted():
    r"""Spring and Truss have no von Kármán internal force vector, so
    ``kinematics='von_karman'`` is rejected for them, with the accepted
    kinematics in the message"""
    X0 = np.array([[0., 0, 0], [1.0, 0.3, -0.2]])
    model = CorotationalModel(X0.ravel())
    for case in ('Truss_chord', 'Spring_triad'):
        e, p, _ = new_element(case, X0, [0, 1])
        with pytest.raises(ValueError, match="'corotational' or 'linear'"):
            model.add(e, p, kinematics='von_karman')


NEW_X = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0., 1, 0],
                  [2., 0, 0.1], [2., 1, 0.1], [0.5, 2, 0], [0., 0, 1],
                  [1., 0, 1], [1., 1, 1]])


def new_mixed_model():
    r"""Co-rotational Quad4 (warped, with offsets), Tria3DSG, BeamLR,
    Truss and Spring (distinct and coincident nodes), von Kármán Tria3DSG
    and BeamLR, linear Truss and Spring"""
    X = NEW_X.copy()
    X[2, 2] = 0.04
    model = CorotationalModel(X.ravel())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model.add(make_element('Quad4', X, [0, 1, 2, 3]), shell_prop())
        for case, nodes, kin in (
                ('Tria3DSG', [1, 4, 5], 'corotational'),
                ('Tria3DSG', [1, 5, 2], 'von_karman'),
                ('BeamLR', [0, 7], 'corotational'),
                ('BeamLR', [7, 8], 'von_karman'),
                ('Truss_mean_triad', [8, 9], 'corotational'),
                ('Truss_chord', [3, 6], 'corotational'),
                ('Truss_chord', [2, 6], 'linear'),
                ('Spring_triad', [7, 9], 'corotational'),
                ('Spring_axis', [5, 9], 'corotational'),
                ('Spring_triad', [4, 8], 'linear')):
            e, p, kw = new_element(case, X, nodes)
            model.add(e, p, kinematics=kin, **kw)
    return model, X


def test_new_mixed_model_tangent_fd():
    r"""``calc_kC + calc_kG`` of a model with the new elements and the three
    kinematics equal to central differences of ``calc_fint``, plus a
    zero-length spring in a model of its own"""
    model, X = new_mixed_model()
    rng = np.random.default_rng(24)
    u = 0.02*rng.standard_normal(model.N)
    with pytest.warns(RuntimeWarning, match='shared'):
        K = (model.calc_kC(u, NLgeom=True) + model.calc_kG(u, NLgeom=True))
    K = K.toarray()
    Kfd = fd_jacobian(model.calc_fint, u)
    err = np.abs(K - Kfd).max()/np.abs(K).max()
    print('new mixed model: FD error %.1e of max|K|' % err)
    assert err < 1e-8
    # NOTE zero-length spring with a co-rotational BeamLR at each node
    X = np.array([[0., 0, 0], [1., 0.2, 0.], [1., 0.2, 0.], [2., 0.5, 0.3]])
    model = CorotationalModel(X.ravel())
    e, p, _ = new_element('BeamLR', X, [0, 1])
    model.add(e, p)
    e, p, _ = new_element('BeamLR', X, [2, 3])
    model.add(e, p)
    e, _, _ = new_element('Spring_zero_length', X, [1, 2])
    w = model.add(e, None)
    assert w.frame == 'triad'
    u = 0.05*rng.standard_normal(model.N)
    K = (model.calc_kC(u, NLgeom=True) + model.calc_kG(u, NLgeom=True))
    K = K.toarray()
    Kfd = fd_jacobian(model.calc_fint, u)
    assert np.abs(K - Kfd).max() < 1e-8*np.abs(K).max()


# ---------------------------------------------------------------------------
# configuration dependence of the load
# ---------------------------------------------------------------------------
def cantilever_model(ne=4):
    r"""Co-rotational BeamC cantilever along `x`, clamped at node 0"""
    xs = np.linspace(0, 1., ne + 1)
    x = np.vstack((xs, 0*xs, 0*xs)).T.ravel()
    model = CorotationalModel(x)
    probe = BeamCProbe()
    for k in range(ne):
        elem = BeamC(probe)
        elem.n1, elem.n2 = k, k + 1
        elem.c1, elem.c2 = DOF*k, DOF*(k + 1)
        elem.update_rotation_matrix(0., 1., 0., x)
        model.add(elem, beam_prop())
    bu = np.ones(model.N, dtype=bool)
    bu[:DOF] = False
    return model, bu


def test_calc_fext_declares_c_only_with_moments():
    r"""``calc_fext`` declares ``c`` only with configuration-dependent
    moments: without them the linear static analysis of
    :class:`pyfe3d.solver.Analysis` is a single solution, with them it
    iterates on the load; ``c`` is accepted and ignored without them"""
    for dependent in (False, True):
        model, bu = cantilever_model()
        model.fext[DOF*4 + 2] = 1e3
        model.add_moment(4, [0., 10., 0.])
        if dependent:
            model.add_moment(4, [5., 0., 0.], kind='follower')
        assert callbacks.accepts_kwarg(model.calc_fext, 'c') == dependent
        # NOTE the same signature as model.calc_fext, counting the load
        #      evaluations at a configuration
        calls = []
        orig = model.calc_fext
        if dependent:
            def calc_fext(inc=1., silent=True, c=None):
                calls.append(c is not None)
                return orig(inc=inc, silent=silent, c=c)
        else:
            def calc_fext(inc=1., silent=True):
                calls.append(False)
                return orig(inc=inc, silent=silent)
        an = Analysis(calc_fext, model.calc_fint, model.calc_kC,
                      model.calc_kG, bu=bu)
        increments, cs = an.static(NLgeom=False, silent=True)
        print('configuration-dependent %s: %d load corrections'
              % (dependent, sum(calls)))
        assert (sum(calls) > 0) == dependent
        if not dependent:
            u = np.zeros(model.N)
            K = model.KC0.tocsr()[bu][:, bu].tocsc()
            u[bu] = spsolve(K, model.fext[bu])
            np.testing.assert_allclose(cs[-1], u, rtol=1e-12,
                                       atol=1e-15*np.abs(u).max())
            np.testing.assert_array_equal(
                model.calc_fext(inc=0.5, c=np.ones(model.N)),
                0.5*model.fext)
            with pytest.raises(TypeError, match='lbd'):
                model.calc_fext(lbd=1.)
        else:
            u = np.zeros(model.N)
            u[DOF*4 + 3:DOF*4 + 6] = [0.3, 0.2, -0.1]
            F = model.calc_fext(c=u) - model.calc_fext()
            np.testing.assert_allclose(
                F[DOF*4 + 3:DOF*4 + 6],
                (Lambda(u[DOF*4 + 3:DOF*4 + 6]) - np.eye(3)) @ [5., 0, 0],
                atol=1e-14)


# ---------------------------------------------------------------------------
# kinematics='corotational_von_karman'
# ---------------------------------------------------------------------------
VK = ['Tria3R', 'Quad4R', 'Quad4', 'BeamC', 'Tria3DSG', 'BeamLR']


def single_vk(name, local='von_karman'):
    r"""Wrapped element alone, as :func:`single` or :func:`single_new`, with
    the local strains ``local``"""
    w = single(name) if name in ELEMENTS else single_new(name)
    x = np.zeros(3*(w.nodes.max() + 1))
    for a, n in enumerate(w.nodes):
        x[3*n:3*n + 3] = w.X0[a]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return type(w)(w.elem, w.prop, x, local=local)


@pytest.mark.parametrize('name', VK)
@pytest.mark.parametrize('angle', [0.7, np.pi, 1.9*np.pi])
def test_cr_von_karman_rigid_motion(name, angle):
    r"""``local='von_karman'``: no internal force under a large rigid
    rotation and translation"""
    w = single_vk(name)
    rng = np.random.default_rng(31)
    for _ in range(3):
        u = rigid_motion(w, angle*unit(rng), rng.standard_normal(3))
        f = w.evaluate(u, tangent=False)
        assert np.abs(w.kinematics(u)[1]).max() < 1e-12
        assert np.abs(f).max() <= 1e-13*np.abs(w.Ke).max()


@pytest.mark.parametrize('name', VK)
@pytest.mark.parametrize('case', ['large', 'near_2pi'])
def test_cr_von_karman_tangent_fd(name, case):
    r"""``local='von_karman'``: tangent stiffness matrix equal to central
    differences of the internal force at large rotations, as
    :func:`test_tangent_fd`"""
    w = single_vk(name)
    rng = np.random.default_rng(32)
    if case == 'large':
        psi, amp = np.array([0.7, -1.9, 2.4]), 0.03
    else:
        psi, amp = np.array([0., 1.9*np.pi, 0.]), 1e-3
    u = rigid_motion(w, psi, np.array([0.3, -2., 1.]))
    u += amp*rng.standard_normal(u.size)
    f, KM, KG = w.evaluate(u)
    K = KM + KG
    Kfd = fd_jacobian(lambda v: w.evaluate(v, tangent=False), u)
    err = np.abs(K - Kfd).max()
    print('%s %s: FD error %.1e of max|K|, %.1e of max|KG|'
          % (name, case, err/np.abs(K).max(), err/np.abs(KG).max()))
    assert err <= 1e-8*np.abs(K).max()
    if case == 'large':
        assert err <= 1e-6*np.abs(KG).max()


@pytest.mark.parametrize('name', VK)
def test_cr_von_karman_against_linear(name):
    r"""At ``u = 0`` the tangent with ``local='von_karman'`` equals that
    with ``'linear'``; at a large deformed configuration both give the same
    force and tangent to round-off for the shells and BeamLR, whose von
    Kármán terms vanish in the CR frame (see :class:`CorotationalShell`),
    and differ for BeamC; the probe is left unchanged"""
    wv, wl = single_vk(name), single_vk(name, local='linear')
    n6 = wv.dofs.size
    fv, KMv, KGv = wv.evaluate(np.zeros(n6))
    fl, KMl, KGl = wl.evaluate(np.zeros(n6))
    scale = np.abs(KMl).max()
    assert np.abs(fv - fl).max() <= 1e-13*scale
    assert np.abs(KMv - KMl).max() <= 1e-13*scale
    assert np.abs(KGv - KGl).max() <= 1e-13*scale
    rng = np.random.default_rng(33)
    u = rigid_motion(wv, np.array([0.7, -1.9, 2.4]), np.array([0.3, -2., 1.]))
    u += 0.05*rng.standard_normal(n6)
    p = wv.elem.probe
    ue, xe = np.array(p.ue, copy=True), np.array(p.xe, copy=True)
    fv, KMv, KGv = wv.evaluate(u)
    assert np.array_equal(p.ue, ue) and np.array_equal(p.xe, xe)
    fl, KMl, KGl = wl.evaluate(u)
    df = np.abs(fv - fl).max()/np.abs(fl).max()
    dK = np.abs(KMv + KGv - KMl - KGl).max()/np.abs(KMl + KGl).max()
    print('%s: von Kármán minus linear local strains, force %.1e, tangent '
          '%.1e' % (name, df, dK))
    if name == 'BeamC':
        assert df > 1e-3 and dK > 1e-4
    else:
        assert df < 1e-12 and dK < 1e-12


def test_cr_von_karman_beamc_crisfield():
    r"""BeamC with ``local='von_karman'`` and an infinite transverse shear
    stiffness: the local axial force is that of the length-averaged strain
    of Crisfield, Vol. 2, Section 17.1.5, Eqs. (17.56a)-(17.56b), `E A
    (u_l/l_0 + \boldsymbol{\theta}_l^T \boldsymbol{X}
    \boldsymbol{\theta}_l/60)`, with the local rotations
    `\boldsymbol{\theta}_l = (\theta_{x1}, \theta_{y1}, \theta_{z1},
    \theta_{x2}, \theta_{y2}, \theta_{z2})`; with the shear stiffness of
    :func:`beam_prop` it differs by about 1e-4"""
    X = np.zeros((6, 6))
    X[1, 1] = X[2, 2] = X[4, 4] = X[5, 5] = 4.
    X[1, 4] = X[4, 1] = X[2, 5] = X[5, 2] = -1.
    theta = np.array([0.01, 0.02, -0.03, 0.015, -0.01, 0.025])
    d = np.zeros(12)
    d[0], d[6] = -1e-4, 1e-4
    d[3:6], d[9:12] = theta[:3], theta[3:]
    w0 = single('BeamC')
    l0 = np.linalg.norm(w0.X0[1] - w0.X0[0])
    for G, rtol in ((80e9*1e6, 1e-9), (80e9, None)):
        p = beam_prop()
        p.G = G
        w = CorotationalBeam(w0.elem, p, w0.X0.ravel(), local='von_karman')
        N = w._local_von_karman(d, tangent=False)[6]
        N_ref = p.E*p.A*((d[6] - d[0])/l0 + theta @ X @ theta/60)
        print('G %.0e: N/N_Crisfield - 1 = %.1e' % (G, N/N_ref - 1))
        if rtol is not None:
            assert abs(N/N_ref - 1) < rtol
        else:
            assert 1e-5 < abs(N/N_ref - 1) < 1e-3


def beamc_models(slender):
    r"""Straight BeamC of 4 elements along `x`, with the three non-linear
    kinematics; ``slender`` divides the bending and torsion stiffnesses by
    1e4"""
    xs = np.linspace(0, 1., 5)
    X = np.vstack((xs, 0*xs, 0*xs)).T
    models = {}
    for kin in ('corotational', 'corotational_von_karman', 'von_karman'):
        model = CorotationalModel(X.ravel())
        for k in range(4):
            p = beam_prop()
            if slender:
                p.Iyy *= 1e-4
                p.Izz *= 1e-4
                p.J *= 1e-4
            elem = make_element('BeamC', X, [k, k + 1])
            elem.update_rotation_matrix(0., 1., 0., X.ravel())
            model.add(elem, p, kinematics=kin)
        models[kin] = model
    return models, X


def test_cr_von_karman_small_rotations():
    r"""Agreement with ``kinematics='von_karman'`` for small rotations, in
    the scaling of the von Kármán theory: plane bending of a BeamC model
    with `w` and the rotations of order `\varepsilon`, `u` of order
    `\varepsilon^2`, so the internal forces are of order `\varepsilon^2`

    Expected (derived here): the transverse forces and the moments of the
    two kinematics differ by `O(\varepsilon^3)`. The axial forces differ
    by `O(\varepsilon^4)` plus the axial component of the transverse shear
    force rotated by the chord, `V \sin\alpha`, of relative order
    `(r/l)^2`, `r` the radius of gyration, which the von Kármán strains
    neglect; with ``slender=True`` it is negligible. Without the local von
    Kármán strains (``'corotational'``) the axial forces miss the
    length-average of `w_{d,x}^2/2`, Crisfield, Vol. 2, Eq. (17.56a), and
    differ by `O(\varepsilon^2)`, a relative error independent of
    `\varepsilon`. Measured (``slender=True``, `\varepsilon` = 4e-3, 2e-3,
    1e-3): axial, relative to the largest force, 2.1e-5, 5.2e-6, 1.3e-6
    with ``'corotational_von_karman'`` and 4.6e-3 with ``'corotational'``;
    transverse and moments, observed order 3.0"""
    models, X = beamc_models(slender=True)
    s = X[:, 0]
    rel_axial, rel_cr, transverse = [], [], []
    epsilons = (4e-3, 2e-3, 1e-3)
    for eps in epsilons:
        u = np.zeros((X.shape[0], DOF))
        u[:, 2] = eps*np.sin(1.3*s + 0.2)
        u[:, 4] = -eps*1.3*np.cos(1.3*s + 0.2)
        u[:, 0] = 0.8*eps**2*s
        u = u.ravel()
        f = {k: m.calc_fint(u).reshape(-1, DOF) for k, m in models.items()}
        ref = np.abs(f['von_karman']).max()
        d = np.abs(f['corotational_von_karman'] - f['von_karman'])
        rel_axial.append(d[:, 0].max()/ref)
        rel_cr.append(np.abs(f['corotational']
                             - f['von_karman'])[:, 0].max()/ref)
        transverse.append(d[:, [2, 4]].max())
        print('eps %.0e: axial %.1e (corotational %.1e), transverse and '
              'moments %.1e' % (eps, rel_axial[-1], rel_cr[-1],
                                transverse[-1]))
    order_axial = np.log2(np.array(rel_axial[:-1])/rel_axial[1:])
    order_transverse = np.log2(np.array(transverse[:-1])/transverse[1:])
    print('observed orders: axial (relative) %s, transverse %s'
          % (order_axial.round(2), order_transverse.round(2)))
    assert np.all(order_axial > 1.9)
    assert np.all(order_transverse > 2.8)
    assert rel_axial[-1] < 2e-6
    assert min(rel_cr) > 1e-3 and max(rel_cr)/min(rel_cr) < 1.01


def test_cr_von_karman_kinematics_accepted():
    r"""``'corotational_von_karman'`` is accepted for every element: Spring
    and Truss, without von Kármán strains, are treated as
    ``'corotational'`` (wrapper with ``local='linear'``); the wrappers
    reject ``local='von_karman'`` for them and an unknown ``local``"""
    X0 = np.array([[0., 0, 0], [1.0, 0.3, -0.2]])
    model = CorotationalModel(X0.ravel())
    for case in ('Truss_chord', 'Truss_mean_triad', 'Spring_triad',
                 'Spring_axis'):
        e, p, _ = new_element(case, X0, [0, 1])
        w = model.add(e, p, kinematics='corotational_von_karman')
        assert w.local == 'linear' and w is model.corotational[-1]
        with pytest.raises(ValueError, match='von Kármán'):
            cr._wrapper_class(e)(e, p, X0.ravel(), local='von_karman')
    e = make_element('BeamC', X0)
    w = model.add(e, beam_prop(), kinematics='corotational_von_karman')
    assert w.local == 'von_karman'
    with pytest.raises(ValueError, match='local must be'):
        CorotationalBeam(e, beam_prop(), X0.ravel(), local='green')
    e, p, _ = new_element('Truss_chord', X0, [0, 1])
    with pytest.raises(ValueError, match="treated as 'corotational'"):
        model.add(e, p, kinematics='von_karman')


def vk_mixed_model():
    r"""``'corotational_von_karman'`` Quad4 (warped, with offsets),
    Quad4R, Tria3R, Tria3DSG, BeamC, BeamLR, Truss and Spring, with
    co-rotational, von Kármán and linear elements, sharing nodes"""
    X = NEW_X.copy()
    X[2, 2] = 0.04
    model = CorotationalModel(X.ravel())
    sp, bp = shell_prop(), beam_prop()
    kin = 'corotational_von_karman'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model.add(make_element('Quad4', X, [0, 1, 2, 3]), sp, kinematics=kin)
        model.add(make_element('Quad4R', X, [1, 4, 5, 2]), sp, kinematics=kin,
                  hgfactor_membrane=0.5)
        model.add(make_element('Tria3R', X, [3, 2, 6]), sp, kinematics=kin)
        model.add(make_element('BeamC', X, [0, 7]), bp, kinematics=kin)
        model.add(make_element('BeamC', X, [7, 8]), bp)
        model.add(make_element('Tria3R', X, [2, 5, 6]), sp,
                  kinematics='von_karman')
        model.add(make_element('BeamC', X, [3, 9]), bp, kinematics='linear')
        for case, nodes in (('Tria3DSG', [4, 5, 9]), ('BeamLR', [8, 9]),
                            ('Truss_mean_triad', [1, 8]),
                            ('Truss_chord', [3, 6]), ('Spring_triad', [7, 9]),
                            ('Spring_axis', [5, 9])):
            e, p, kw = new_element(case, X, nodes)
            model.add(e, p, kinematics=kin, **kw)
    return model


@pytest.mark.parametrize('case', ['moderate', 'near_2pi'])
def test_cr_von_karman_mixed_model_tangent_fd(case):
    r"""``calc_kC + calc_kG`` of a model mixing ``'corotational_von_karman'``
    elements of every type with the other three kinematics equal to
    central differences of ``calc_fint``; at ``u = 0`` it equals the
    tangent of the same model with the local linear strains"""
    model = vk_mixed_model()
    rng = np.random.default_rng(34)
    u = np.zeros(model.N)
    if case == 'near_2pi':
        X = model.x.reshape(-1, 3)
        R = expso3([0., 0., 1.9*np.pi])
        u.reshape(-1, DOF)[:, :3] = X @ R.T - X
        u.reshape(-1, DOF)[:, 3:] = [0., 0., 1.9*np.pi]
    u += 0.02*rng.standard_normal(model.N)
    with pytest.warns(RuntimeWarning, match='shared'):
        K = (model.calc_kC(u, NLgeom=True) + model.calc_kG(u, NLgeom=True))
    K = K.toarray()
    Kfd = fd_jacobian(model.calc_fint, u)
    err = np.abs(K - Kfd).max()/np.abs(K).max()
    print('mixed model with corotational_von_karman, %s: FD error %.1e of '
          'max|K|' % (case, err))
    assert err < 1e-8
    if case == 'moderate':
        u0 = np.zeros(model.N)
        K0 = (model.calc_kC(u0, NLgeom=True)
              + model.calc_kG(u0, NLgeom=True)).toarray()
        for w in model.corotational:
            w.local = 'linear'
        model._cache = None
        K0l = (model.calc_kC(u0, NLgeom=True)
               + model.calc_kG(u0, NLgeom=True)).toarray()
        assert np.abs(K0 - K0l).max() <= 1e-13*np.abs(K0l).max()
