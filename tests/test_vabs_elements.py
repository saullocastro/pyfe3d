r"""Elements, materials and conventions of :mod:`pyfe3d.vabs`

Checks that do not depend on published results: the degree of the
integration rules, the shape functions, the rotation of the material
matrices, the conversion to the convention of pyfe3d, the creation of a
:class:`pyfe3d.beamprop.BeamProp` and the input errors.
"""
from math import factorial

import numpy as np
import pytest

from pyfe3d.vabs import (CrossSection, isotropic, orthotropic,
                         rotate_material, yu_to_pyfe3d, pyfe3d_to_yu,
                         classical_from_timoshenko, beamprop_from_vabs,
                         _quadrature, _shape_functions)

NODES = {
    3: [(0, 0), (1, 0), (0, 1)],
    6: [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)],
    4: [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    8: [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1),
        (-1, 0)],
    9: [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1),
        (-1, 0), (0, 0)],
}


@pytest.mark.parametrize('nen, degree', [(3, 2), (6, 4)])
def test_triangle_rules(nen, degree):
    r"""`\int_T \xi^p \eta^q = p! q!/(p + q + 2)!` on the unit triangle"""
    xi, eta, w = _quadrature(nen)
    for p in range(degree + 1):
        for q in range(degree + 1 - p):
            exact = factorial(p)*factorial(q)/factorial(p + q + 2)
            assert np.isclose(np.sum(w*xi**p*eta**q), exact, rtol=1e-13)


@pytest.mark.parametrize('nen, degree', [(4, 3), (8, 5), (9, 5)])
def test_quadrilateral_rules(nen, degree):
    xi, eta, w = _quadrature(nen)
    for p in range(degree + 1):
        for q in range(degree + 1):
            ex = (2/(p + 1) if p % 2 == 0 else 0)*(2/(q + 1) if q % 2 == 0
                                                  else 0)
            assert np.isclose(np.sum(w*xi**p*eta**q), ex, atol=1e-14)


@pytest.mark.parametrize('nen', [3, 4, 6, 8, 9])
def test_shape_functions(nen):
    nodes = NODES[nen]
    for a, (xa, ea) in enumerate(nodes):
        N, _, _ = _shape_functions(nen, xa, ea)
        expected = np.zeros(nen)
        expected[a] = 1
        assert np.allclose(N, expected, atol=1e-14)
    rng = np.random.default_rng(1)
    xs = np.array(nodes, dtype=float)
    for _ in range(5):
        xi, eta = rng.uniform(0, 0.5, 2) if nen in (3, 6) else \
            rng.uniform(-1, 1, 2)
        N, Nxi, Neta = _shape_functions(nen, xi, eta)
        assert np.isclose(N.sum(), 1)
        # NOTE linear fields are reproduced, with their derivatives
        assert np.allclose(N @ xs, [xi, eta])
        assert np.allclose(Nxi @ xs, [1, 0]) and np.allclose(Neta @ xs,
                                                             [0, 1])
        # NOTE derivatives against finite differences
        d = 1e-6
        Np, _, _ = _shape_functions(nen, xi + d, eta)
        Nm, _, _ = _shape_functions(nen, xi - d, eta)
        assert np.allclose((Np - Nm)/(2*d), Nxi, atol=1e-8)
        Np, _, _ = _shape_functions(nen, xi, eta + d)
        Nm, _, _ = _shape_functions(nen, xi, eta - d)
        assert np.allclose((Np - Nm)/(2*d), Neta, atol=1e-8)


def test_materials():
    E, nu = 70e9, 0.3
    D = isotropic(E, nu)
    G = E/(2*(1 + nu))
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    # NOTE ordering {11, 12, 13, 22, 23, 33}
    assert np.isclose(D[0, 0], lam + 2*G)
    assert np.isclose(D[0, 3], lam) and np.isclose(D[3, 5], lam)
    assert np.isclose(D[1, 1], G) and np.isclose(D[2, 2], G)
    assert np.isclose(D[4, 4], G)
    assert np.allclose(rotate_material(D, 33., -71.), D, atol=1e-12*E)
    Do = orthotropic(20.59e6, 1.42e6, 1.3e6, 0.87e6, 0.8e6, 0.696e6, 0.30,
                     0.28, 0.34)
    S = np.linalg.inv(Do)
    assert np.isclose(1/S[0, 0], 20.59e6)
    assert np.isclose(1/S[3, 3], 1.42e6)
    assert np.isclose(1/S[5, 5], 1.3e6)
    assert np.isclose(1/S[1, 1], 0.87e6)
    assert np.isclose(1/S[2, 2], 0.8e6)
    assert np.isclose(1/S[4, 4], 0.696e6)
    assert np.isclose(-S[0, 3]*20.59e6, 0.30)
    assert np.isclose(-S[0, 5]*20.59e6, 0.28)
    assert np.isclose(-S[3, 5]*1.42e6, 0.34)


@pytest.mark.parametrize('theta1, theta3', [(0., 15.), (90., 15.),
                                            (180., -15.), (37., 22.)])
def test_rotate_material(theta1, theta3):
    r"""The strain energy of any strain given in the material axes is
    the same as that of the same strain written in the beam axes, with the
    material axes of :func:`pyfe3d.vabs.rotate_material`"""
    D = orthotropic(20.59e6, 1.42e6, 1.42e6, 8.7e5, 8.7e5, 6.96e5, 0.42,
                    0.42, 0.42)
    Dr = rotate_material(D, theta1, theta3)
    t1, t3 = np.deg2rad(theta1), np.deg2rad(theta3)
    y1 = np.array([1., 0, 0])
    y2 = np.array([0, np.cos(t1), np.sin(t1)])
    y3 = np.cross(y1, y2)
    m1 = np.cos(t3)*y1 + np.sin(t3)*y2
    m2 = -np.sin(t3)*y1 + np.cos(t3)*y2
    R = np.array([m1, m2, y3])
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def vec(eps):
        return np.array([eps[i, j]*(1 if i == j else 2) for i, j in pairs])
    for a in range(3):
        for b in range(3):
            em = np.zeros((3, 3))
            em[a, b] += 0.5
            em[b, a] += 0.5
            eb = R.T @ em @ R
            assert np.isclose(vec(em) @ D @ vec(em), vec(eb) @ Dr @ vec(eb),
                              rtol=1e-12)


def test_conventions():
    r"""`\boldsymbol{T}_s` of the book, `\kappa_1 = \kappa_x`, `\kappa_2 =
    \kappa_z`, `\kappa_3 = -\kappa_y`, and the scaling `\boldsymbol{H}`"""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 6))
    C = M @ M.T
    Cs = yu_to_pyfe3d(C)
    assert np.isclose(Cs[0, 4], -C[0, 5]) and np.isclose(Cs[0, 5], C[0, 4])
    assert np.isclose(Cs[4, 4], C[5, 5]) and np.isclose(Cs[5, 5], C[4, 4])
    assert np.isclose(Cs[4, 5], -C[4, 5]) and np.isclose(Cs[0, 3], C[0, 3])
    eps_s = rng.normal(size=6)
    eps = np.array([eps_s[0], eps_s[1], eps_s[2], eps_s[3], eps_s[5],
                    -eps_s[4]])
    assert np.isclose(eps @ C @ eps, eps_s @ Cs @ eps_s)
    h = 0.7
    assert np.allclose(pyfe3d_to_yu(yu_to_pyfe3d(C, h), h), C)
    Ch = yu_to_pyfe3d(C, h)
    assert np.isclose(Ch[4, 4], h**2*C[5, 5]) and np.isclose(Ch[0, 4],
                                                            -h*C[0, 5])
    C4 = C[np.ix_([0, 3, 4, 5], [0, 3, 4, 5])]
    C4s = yu_to_pyfe3d(C4)
    assert np.allclose(C4s, Cs[np.ix_([0, 3, 4, 5], [0, 3, 4, 5])])
    assert np.allclose(pyfe3d_to_yu(C4s), C4)
    # NOTE condensation: Schur complement of the shear block
    cl = [0, 3, 4, 5]
    sh = [1, 2]
    Kc = C[np.ix_(cl, cl)] - C[np.ix_(cl, sh)] @ np.linalg.solve(
        C[np.ix_(sh, sh)], C[np.ix_(sh, cl)])
    assert np.allclose(classical_from_timoshenko(C), Kc)
    with pytest.raises(ValueError):
        yu_to_pyfe3d(np.eye(5))
    for convert in (yu_to_pyfe3d, pyfe3d_to_yu):
        with pytest.raises(ValueError, match='shape'):
            convert(np.ones((4, 6)))
    with pytest.raises(ValueError):
        classical_from_timoshenko(np.eye(4))


def test_beamprop_from_vabs_errors():
    Cs = np.diag([7e9, 2e9, 2e9, 1e6, 6e6, 6e6])
    prop = beamprop_from_vabs(Cs, E=70e9)
    assert np.isclose(prop.A, 0.1) and np.isclose(prop.G12, 2e10)
    assert np.isclose(prop.GJ, 1e6) and np.isclose(prop.Izz, 6e6/70e9)
    Cs[0, 3] = Cs[3, 0] = 1e6
    with pytest.raises(ValueError):
        beamprop_from_vabs(Cs, E=70e9)
    with pytest.raises(ValueError):
        beamprop_from_vabs(np.eye(4), E=1.)
    with pytest.raises(ValueError):
        beamprop_from_vabs(np.eye(6), E=0.)


def test_input_errors():
    D = isotropic(1., 0.3)
    x2 = np.array([0., 1., 1., 0.])
    x3 = np.array([0., 0., 1., 1.])
    with pytest.raises(ValueError):
        isotropic(1., 0.5)
    with pytest.raises(ValueError):
        isotropic(-1., 0.3)
    with pytest.raises(ValueError):
        orthotropic(1., 1., 1., 1., 1., 1., 0.9, 0.9, 0.9)
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 3, 0]], [D], [0])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2]], [D], [0])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 4]], [D], [0])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [1])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [0], theta1=[0., 0.])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [0], densities=[1., 2.])
    with pytest.raises(ValueError):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [np.eye(5)], [0])
    with pytest.raises(ValueError):
        CrossSection(x2[:3], x3, [[0, 1, 2]], [D], [0])
    with pytest.raises(NotImplementedError):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [0], k1=0.01)
    with pytest.raises(ValueError, match='at least one element'):
        CrossSection(x2, x3, [], [D], [])
    with pytest.raises(ValueError, match='one entry per element'):
        CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [0, 0])
    with pytest.raises(ValueError, match='shape'):
        rotate_material(np.eye(5), 0., 0.)
    # NOTE clockwise element
    sec = CrossSection(x2, x3, [[0, 3, 2, 1]], [D], [0])
    with pytest.raises(ValueError):
        sec.calc_stiffness()
    with pytest.raises(ValueError):
        sec.calc_mass()
    sec = CrossSection(x2, x3, [[0, 1, 2, 3]], [D], [0])
    with pytest.raises(ValueError):
        sec.calc_stiffness('other')
    # NOTE two disconnected elements
    x2d = np.concatenate((x2, x2 + 2))
    x3d = np.concatenate((x3, x3))
    sec = CrossSection(x2d, x3d, [[0, 1, 2, 3], [4, 5, 6, 7]], [D], [0, 0])
    with pytest.raises(RuntimeError, match='not connected'):
        sec.calc_stiffness()
    # NOTE a material without stiffness
    sec = CrossSection(x2, x3, [[0, 1, 2, 3]], [np.zeros((6, 6))], [0])
    with pytest.raises(RuntimeError, match='singular'):
        sec.calc_stiffness()


def test_mixed_elements():
    r"""Triangles and quadrilaterals of the same order in one mesh give the
    same section properties as a mesh of one type"""
    D = isotropic(1., 0.3)
    # NOTE unit square: a quadrilateral and two triangles side by side
    x2 = np.array([0., 1., 2., 0., 1., 2.])
    x3 = np.array([0., 0., 0., 1., 1., 1.])
    mixed = CrossSection(x2, x3, [[0, 1, 4, 3], [1, 2, 5], [1, 5, 4]], [D],
                         [0, 0, 0])
    mixed.calc_stiffness()
    mixed.calc_mass()
    assert np.isclose(mixed.area, 2.)
    assert np.allclose(mixed.centroid, [1., 0.5])
    assert np.isclose(mixed.Kc[0, 0], 2.)
    assert np.isclose(mixed.Kc[0, 3], -2.)
    assert np.isclose(mixed.Kc[0, 2], 1.)
