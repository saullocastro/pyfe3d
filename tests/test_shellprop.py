import sys
sys.path.append('..')

import copy
import pickle

import numpy as np
import pytest

from pyfe3d.shellprop import (Lamina, shellprop_from_LaminationParameters,
        shellprop_from_lamination_parameters, make_balanced_LP,
        make_orthotropic_LP, make_symmetric_LP, LaminationParameters,
        GradABD)
from pyfe3d.shellprop_utils import (read_laminaprop, laminated_plate,
        isotropic_plate)

# Melo, J. D. D., Bi, J. and Tsai, S. W., "A novel invariant-based design
# approach to carbon fiber reinforced laminates", Composite Structures, 159,
# 44-52, 2017, https://doi.org/10.1016/j.compstruct.2016.09.055, Table 1,
# p. 45: Ex, Ey, nux, Es and the trace Tr of Eq. (1), p. 45, of the
# plane-stress stiffness matrix, in GPa, printed as integers
data = {
'IM6/epoxy': dict(Ex=203e9, Ey=11.20e9, vx=0.32, Es=8.40e9, tr=232e9),
'IM7/977-3': dict(Ex=191e9, Ey=9.94e9, vx=0.35, Es=7.79e9, tr=218e9),
'T4708/MR60H': dict(Ex=142e9, Ey=7.72e9, vx=0.34, Es=3.80e9, tr=158e9),
}


# Classical lamination theory
# ---------------------------
#
# Reference stiffnesses computed in the tests, independently of pyfe3d, with
#
#     (A_ij, B_ij, D_ij) = sum_k int_{z_{k-1}}^{z_k} Qbar_ij^(k) (1, z, z^2) dz
#
# Reddy, J. N., Chandrashekhara, K. and Chao, W. C., "Geometrically nonlinear
# analysis of laminated elastic structures", NASA CR-191055, 1993, Part 1,
# Eq. (12), p. 8, where Qbar^(k) is the plane-stress reduced stiffness of ply
# k in the laminate axes, see ``qbar``. The homogeneous plate is the special
# case of one ply: A = h Q, B = 0, D = h^3/12 Q.

def qbar(E1, E2, nu12, G12, thetadeg):
    r"""Plane-stress reduced stiffness of a ply in the laminate axes

    ``Q = [[E1, nu12 E2, 0], [nu12 E2, E2, 0], [0, 0, G12 (1 - nu12
    nu21)]]/(1 - nu12 nu21)`` in the material axes, with ``nu21 = nu12
    E2/E1``, and ``Qbar = Ts^-1 Q Te``, where ``Ts`` and ``Te`` transform the
    stresses and the engineering strains ``{xx, yy, xy}`` from the laminate
    axes to the material axes, whose direction 1 is at ``thetadeg`` from the
    laminate x axis, counterclockwise.
    """
    nu21 = nu12*E2/E1
    den = 1 - nu12*nu21
    Q = np.array([[E1/den, nu12*E2/den, 0.],
                  [nu12*E2/den, E2/den, 0.],
                  [0., 0., G12]])
    c = np.cos(np.deg2rad(thetadeg))
    s = np.sin(np.deg2rad(thetadeg))
    Ts = np.array([[c*c, s*s, 2*c*s], [s*s, c*c, -2*c*s],
                   [-c*s, c*s, c*c - s*s]])
    Te = np.array([[c*c, s*s, c*s], [s*s, c*c, -c*s],
                   [-2*c*s, 2*c*s, c*c - s*s]])
    return np.linalg.solve(Ts, Q @ Te)


def clt_abd(stack, plyts, laminaprops, offset=0.):
    r"""A, B, D of Eq. (12) of Reddy et al. (1993), plies from the bottom,
    ``laminaprops`` with the entries ``(E1, E2, nu12, G12)`` first"""
    h = sum(plyts)
    z = offset - h/2 + np.concatenate(([0.], np.cumsum(plyts)))
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for k, (thetadeg, lp) in enumerate(zip(stack, laminaprops)):
        Qb = qbar(lp[0], lp[1], lp[2], lp[3], thetadeg)
        A += Qb*(z[k+1] - z[k])
        B += Qb*(z[k+1]**2 - z[k]**2)/2
        D += Qb*(z[k+1]**3 - z[k]**3)/3
    return A, B, D


def assert_abd(A, B, D, A_ref, B_ref, D_ref, h):
    r"""Round-off tolerance, where the terms that vanish are compared with
    an absolute tolerance relative to the scale of the respective matrix"""
    scale = np.abs(A_ref).max()
    assert np.allclose(A, A_ref, rtol=1e-12, atol=1e-14*scale)
    assert np.allclose(B, B_ref, rtol=1e-12, atol=1e-14*scale*h)
    assert np.allclose(D, D_ref, rtol=1e-12, atol=1e-14*scale*h**2)

def test_lampar_tri_axial():
    E = 71e9
    nu = 0.33
    G = E/(2*(1+nu))
    lamprop = (E, E, nu, G, G, G, E, nu, nu)
    rho = 0
    thickness = 1
    matlamina = read_laminaprop(lamprop, rho)
    matlamina.get_constitutive_matrix()
    matlamina.get_invariant_matrix()
    ply = Lamina()
    ply.thetadeg = 45.
    ply.h = 3.
    ply.matlamina = matlamina
    ply.get_transf_matrix_displ_to_lamina()
    ply.get_constitutive_matrix()
    ply.get_transf_matrix_stress_to_lamina()
    ply.get_transf_matrix_stress_to_laminate()

    prop = shellprop_from_lamination_parameters(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    # NOTE with e3 given, the invariants are by default those of the
    #      plane-stress reduced stiffnesses, see MatLamina.rebuild, as in
    #      test_lampar_plane_stress: Q = E/(1 - nu^2) [[1, nu, 0], [nu, 1,
    #      0], [0, 0, (1 - nu)/2]], Jones (1999), Eq. (2.71), p. 73. For an
    #      isotropic material u2 = u3 = u7 = 0, such that the lamination
    #      parameters have no effect: A = h Q, B = 0, D = h^3/12 Q
    assert matlamina.plane_stress
    Q = E/(1 - nu**2)*np.array([[1., nu, 0.],
                                [nu, 1., 0.],
                                [0., 0., (1 - nu)/2]])
    Abar_ts = G*thickness*np.eye(2)
    assert_abd(prop.A, prop.B, prop.D, thickness*Q, np.zeros((3, 3)),
               thickness**3/12*Q, thickness)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12,
                       atol=1e-14*G*thickness)

    # NOTE three-dimensional option: the invariants are those of the
    #      three-dimensional stiffness, which for an isotropic material are
    #      the Lame constants: C11 = lambda + 2 mu, C12 = lambda, C66 = mu
    matlamina.rebuild(plane_stress=False)
    assert not matlamina.plane_stress
    prop = shellprop_from_lamination_parameters(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    C = np.array([[lam + 2*G, lam, 0.],
                  [lam, lam + 2*G, 0.],
                  [0., 0., G]])
    A = thickness*C
    B = np.zeros((3, 3))
    D = thickness**3/12*C
    assert_abd(prop.A, prop.B, prop.D, A, B, D, thickness)
    prop.make_symmetric()
    assert np.array_equal(prop.B, B)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12,
                       atol=1e-14*G*thickness)
    ABD = prop.ABD
    assert_abd(ABD[:3, :3], ABD[:3, 3:], ABD[3:, 3:], A, B, D, thickness)


def test_lampar_plane_stress():
    E = 71e9
    nu = 0.33
    lamprop = (E, nu)
    rho = 0
    thickness = 1
    matlamina = read_laminaprop(lamprop, rho)
    matlamina.get_constitutive_matrix()
    matlamina.get_invariant_matrix()
    ply = Lamina()
    ply.thetadeg = 45.
    ply.h = 3.
    ply.matlamina = matlamina
    ply.get_transf_matrix_displ_to_lamina()
    ply.get_constitutive_matrix()
    ply.get_transf_matrix_stress_to_lamina()
    ply.get_transf_matrix_stress_to_laminate()

    prop = shellprop_from_lamination_parameters(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    # NOTE isotropic plane stress, u2 = u3 = u7 = 0 such that the lamination
    #      parameters have no effect: A = E h/(1 - nu^2) [[1, nu, 0], [nu,
    #      1, 0], [0, 0, (1 - nu)/2]], D = h^2/12 A, Abar_ts = G h I
    G = E/(2*(1 + nu))
    Q = E/(1 - nu**2)*np.array([[1., nu, 0.],
                                [nu, 1., 0.],
                                [0., 0., (1 - nu)/2]])
    A = thickness*Q
    B = np.zeros((3, 3))
    D = thickness**3/12*Q
    Abar_ts = G*thickness*np.eye(2)
    assert_abd(prop.A, prop.B, prop.D, A, B, D, thickness)
    prop.make_symmetric()
    assert np.array_equal(prop.B, B)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12,
                       atol=1e-14*G*thickness)
    ABD = prop.ABD
    assert_abd(ABD[:3, :3], ABD[:3, 3:], ABD[3:, 3:], A, B, D, thickness)


def test_laminated_plate_tri_axial():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9, 7e9, 0.28, 0.28)
    stack = [0, 45, 90]
    plyt = 0.000125
    prop = laminated_plate(stack, plyt, lamprop)
    # NOTE the plies use the plane-stress reduced stiffnesses of E1, E2,
    #      nu12 and G12 whether or not e3 is given, see Lamina.rebuild, such
    #      that the result is the same as in the plane-stress test below
    A, B, D = clt_abd(stack, [plyt]*3, [lamprop]*3)
    # NOTE G13 = G23 = G, constant-strain transverse shear stiffness G h
    Abar_ts = 7e9*3*plyt*np.eye(2)
    assert_abd(prop.A, prop.B, prop.D, A, B, D, prop.h)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12,
                       atol=1e-14*Abar_ts.max())


def test_laminated_plate_plane_stress():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9)
    stack = [0, 45, 90]
    plyt = 0.000125
    prop = laminated_plate(stack, plyt, lamprop)
    A, B, D = clt_abd(stack, [plyt]*3, [lamprop]*3)
    Abar_ts = 7e9*3*plyt*np.eye(2)
    assert_abd(prop.A, prop.B, prop.D, A, B, D, prop.h)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12,
                       atol=1e-14*Abar_ts.max())
    prop.calc_equivalent_properties()
    lp = prop.calc_lamination_parameters()
    matlamina = prop.plies[0].matlamina
    thickness = prop.h
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)

    prop.make_balanced()
    make_balanced_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)

    prop.make_orthotropic()
    make_orthotropic_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)

    prop = laminated_plate(stack, plyt, lamprop)
    lp = prop.calc_lamination_parameters()
    prop.make_symmetric()
    make_symmetric_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)


# Materials given with the nine constants of read_laminaprop, e3 != 0: an
# isotropic one and an orthotropic one with E2 != E3, nu13 != nu23 and G13 !=
# G23, the material of tests/test_matlamina_stiffness.py
G_ISO = 71e9/(2*(1 + 0.33))
LAMINAPROPS_9 = {
    'isotropic': (71e9, 71e9, 0.33, G_ISO, G_ISO, G_ISO, 71e9, 0.33, 0.33),
    'orthotropic': (150e9, 8.7e9, 0.28, 5.0e9, 5.5e9, 3.1e9, 10.1e9, 0.3,
                    0.45),
}
# stack from the bottom ply, ply thicknesses
LAYUPS = {
    'cross-ply symmetric': ([0, 90, 90, 0], [0.125e-3]*4),
    'unsymmetric': ([0, 45, 90], [0.125e-3]*3),
    'antisymmetric': ([45, -45], [0.125e-3]*2),
    'unbalanced unsymmetric': ([15, -70, 35, 90, 0],
                               [0.1e-3, 0.2e-3, 0.125e-3, 0.15e-3, 0.3e-3]),
}


def abar_ts(stack, plyts, G13, G23):
    r"""Constant-strain transverse shear stiffness ``sum_k Qbar_s^(k) h_k``,
    ``[[Qbar44, Qbar45], [Qbar45, Qbar55]]`` of Reddy (2003), Eq. (3.4.17b),
    p. 137, with ``Q44 = G23`` and ``Q55 = G13``, Eq. (2.4.4b), p. 101"""
    Abar = np.zeros((2, 2))
    for thetadeg, hk in zip(stack, plyts):
        c = np.cos(np.deg2rad(thetadeg))
        s = np.sin(np.deg2rad(thetadeg))
        Abar += hk*np.array([[G23*c*c + G13*s*s, (G13 - G23)*c*s],
                             [(G13 - G23)*c*s, G23*s*s + G13*c*c]])
    return Abar


@pytest.mark.parametrize('layup', LAYUPS.keys())
@pytest.mark.parametrize('material', LAMINAPROPS_9.keys())
def test_lamination_parameters_three_dimensional_constants(material, layup):
    r"""Both routes give the plane-stress stiffnesses when e3 is given

    The shell elements are plane stress, `\sigma_{33} = 0`: the reduced
    stiffnesses depend on E1, E2, nu12, G12, G13 and G23 only (Reddy, 2003,
    Eq. (2.4.4), p. 101; Jones, 1999, Eqs. (2.64) and (2.66), p. 72), and the
    invariants of Tsai and Pagano are those of the reduced stiffnesses
    (Jones, 1999, Eq. (2.99), p. 86). :func:`.laminated_plate` and
    :func:`.shellprop_from_LaminationParameters`, with the lamination
    parameters of the same layup, must give A, B and D of the classical
    lamination theory, ``clt_abd``, and the constant-strain transverse shear
    stiffness ``abar_ts``, to round-off, also for e3 != 0, and the same
    values as for the six-entry ``laminaprop`` without e3, nu13 and nu23.
    """
    lamprop = LAMINAPROPS_9[material]
    stack, plyts = LAYUPS[layup]
    h = sum(plyts)
    # NOTE no shear correction, such that Ats = Abar_ts in both routes
    prop = laminated_plate(stack, plyts=plyts, laminaprop=lamprop,
                           shear_correction=None)
    mat = read_laminaprop(lamprop)
    assert mat.e3 != 0 and mat.plane_stress
    lp = prop.calc_lamination_parameters()
    prop_lp = shellprop_from_LaminationParameters(prop.h, mat, lp)

    A, B, D = clt_abd(stack, plyts, [lamprop]*len(stack))
    Abar = abar_ts(stack, plyts, lamprop[4], lamprop[5])
    for p in (prop, prop_lp):
        assert_abd(p.A, p.B, p.D, A, B, D, h)
        assert np.allclose(p.Abar_ts, Abar, rtol=1e-12,
                           atol=1e-14*np.abs(Abar).max())
    assert np.allclose(prop_lp.Ats, prop.Ats, rtol=1e-12,
                       atol=1e-14*np.abs(Abar).max())

    # NOTE e3, nu13 and nu23 do not change the plane-stress invariants
    mat6 = read_laminaprop(lamprop[:6])
    for name in ('q11', 'q12', 'q21', 'q22', 'q44', 'q55', 'q66', 'u1', 'u2',
                 'u3', 'u4', 'u5', 'u6', 'u7'):
        assert getattr(mat, name) == getattr(mat6, name), name

    # NOTE the gradients use the same invariants: at constant lamination
    #      parameters A is linear in h and D cubic, dA/dh = A/h and dD/dh =
    #      3 D/h
    grad = GradABD()
    grad.calc_LP_grad(prop.h, mat, lp)
    ij = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    gradA = np.array([A[i, j] for i, j in ij])/h
    gradD = np.array([D[i, j] for i, j in ij])*3/h
    assert np.allclose(np.asarray(grad.gradAij)[:, 0], gradA, rtol=1e-12,
                       atol=1e-14*np.abs(gradA).max())
    assert np.allclose(np.asarray(grad.gradDij)[:, 0], gradD, rtol=1e-12,
                       atol=1e-14*np.abs(gradD).max())

    # NOTE three-dimensional option: stiffer than plane stress, the
    #      lamination-parameter route then no longer gives the stiffnesses of
    #      the shell elements
    mat.rebuild(plane_stress=False)
    prop_3d = shellprop_from_LaminationParameters(prop.h, mat, lp)
    assert prop_3d.A11 > (1 + 1e-3)*prop.A11
    assert prop_3d.A22 > (1 + 1e-3)*prop.A22


@pytest.mark.parametrize('material', LAMINAPROPS_9.keys())
def test_trace_normalized_three_dimensional_constants(material):
    r"""Trace normalization with e3 given

    The trace of Melo et al. (2017), Eq. (1), p. 45, is that of the
    plane-stress stiffness matrix, ``tr = Q11 + Q22 + 2 Q66`` with ``Q11 =
    E1/(1 - nu12 nu21)``, ``Q22 = E2/(1 - nu12 nu21)``, ``Q66 = G12`` (Jones,
    1999, Eq. (2.66), p. 72), independent of e3. The stiffnesses of the
    lamination-parameter route are linear in the invariants, such that with
    the trace-normalized invariants they are those of laminated_plate
    divided by tr. After ``rebuild(plane_stress=False)`` the trace
    normalization is refused.
    """
    lamprop = LAMINAPROPS_9[material]
    E1, E2, nu12, G12 = lamprop[:4]
    nu21 = nu12*E2/E1
    tr = (E1 + E2)/(1 - nu12*nu21) + 2*G12
    stack, plyts = LAYUPS['unbalanced unsymmetric']
    prop = laminated_plate(stack, plyts=plyts, laminaprop=lamprop,
                           shear_correction=None)
    m = read_laminaprop(lamprop)
    assert np.isclose(m.q11 + m.q22 + 2*m.q66, tr, rtol=1e-14, atol=0)
    m.trace_normalize_plane_stress()
    assert np.isclose(2*(m.u1 + m.u5), 1., rtol=1e-14, atol=0)
    prop_lp = shellprop_from_LaminationParameters(prop.h, m,
            prop.calc_lamination_parameters())
    assert_abd(prop_lp.A*tr, prop_lp.B*tr, prop_lp.D*tr, prop.A, prop.B,
               prop.D, prop.h)
    assert np.allclose(prop_lp.Abar_ts*tr, prop.Abar_ts, rtol=1e-12,
                       atol=1e-14*np.abs(prop.Abar_ts).max())

    m3d = read_laminaprop(lamprop)
    m3d.rebuild(plane_stress=False)
    with pytest.raises(ValueError, match='plane-stress stiffnesses'):
        m3d.trace_normalize_plane_stress()


def helper_isotropic_plate():
    E = 71e9
    nu = 0.28
    thick = 0.000125
    prop = isotropic_plate(thickness=thick, E=E, nu=nu)
    return prop


def test_isotropic_plate():
    prop = helper_isotropic_plate()
    E = 71e9
    nu = 0.28
    h = 0.000125
    G = E/(2*(1 + nu))
    # NOTE A = E h/(1 - nu^2) [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]],
    #      B = 0 and D = E h^3/(12 (1 - nu^2)) [...]
    Q = E/(1 - nu**2)*np.array([[1., nu, 0.],
                                [nu, 1., 0.],
                                [0., 0., (1 - nu)/2]])
    A = h*Q
    B = np.zeros((3, 3))
    D = h**3/12*Q
    Abar_ts = G*h*np.eye(2)
    assert_abd(prop.A, prop.B, prop.D, A, B, D, h)
    assert np.allclose(prop.Abar_ts, Abar_ts, rtol=1e-12, atol=1e-14*G*h)
    # NOTE Rohwer (1988), DFVLR-FB 88-32, Sec. 4.1, p. 17: for a homogeneous
    #      isotropic plate the improved stiffness is 5/6 of the simple ones,
    #      "independent of the Poisson's ratio or the position of the
    #      reference surface"
    assert prop.shear_correction == 'rohwer'
    assert np.allclose(prop.Ats, 5/6*G*h*np.eye(2), rtol=1e-12,
                       atol=1e-14*G*h)


def test_offset_isotropic_plate():
    r"""With the offset `e`, the plate occupies `-h/2 + e \le z \le h/2 + e`
    and the thickness integrals give `B = e A`, `D = D_0 + e^2 A`,
    ``intrhoz`` `= \rho h e` and ``intrhoz2`` `= \rho (h^3/12 + h e^2)`; the
    transverse shear stiffness does not change"""
    E = 71e9
    nu = 0.28
    rho = 2700.
    h = 0.002
    e = 0.6*h
    prop0 = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)
    prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho, offset=e)
    assert np.allclose(prop.A, prop0.A)
    assert np.allclose(prop.B, e*prop0.A, rtol=1e-12)
    assert np.allclose(prop.D, prop0.D + e**2*prop0.A, rtol=1e-12)
    assert np.allclose(prop.Ats, prop0.Ats, rtol=1e-12)
    assert np.isclose(prop.intrho, rho*h)
    assert np.isclose(prop.intrhoz, rho*h*e)
    assert np.isclose(prop.intrhoz2, rho*(h**3/12 + h*e**2))


def test_errors():
    prop = helper_isotropic_plate()
    prop.offset = 1.
    with pytest.raises(RuntimeError, match='cannot be made balanced'):
        prop.make_balanced()
    with pytest.raises(RuntimeError, match='cannot be made orthotropic'):
        prop.make_orthotropic()
    with pytest.raises(RuntimeError, match='cannot be made symmetric'):
        prop.make_symmetric()
    with pytest.raises(NotImplementedError, match='cannot be made smeared'):
        prop.make_smeared()
    prop.plies = []
    with pytest.raises(ValueError, match='0 plies'):
        prop.calc_lamination_parameters()


def test_trace_normalized():
    r"""
    Reference:

        Melo, J. D. D., Bi, J., and Tsai, S. W., 2017, “A Novel Invariant-Based
        Design Approach to Carbon Fiber Reinforced Laminates,” Compos. Struct.,
        159, pp. 44–52. https://doi.org/10.1016/j.compstruct.2016.09.055

    The trace is ``tr = Q11 + Q22 + 2 Q66``, Eq. (1), p. 45, of the
    plane-stress stiffnesses, with the materials of Table 1, p. 45, see
    ``data``; the in-plane invariants are
    written with Q66 = G12, and the transverse shear invariants with Q44 =
    G23 and Q55 = G13, checked below against the rotation of the transverse
    shear moduli. ``data`` has G12 only; G13 and G23 are arbitrary and
    distinct, such that Q44, Q55 and Q66 cannot be interchanged unnoticed.

    """
    for material, d in data.items():
        G12 = d['Es']
        G13 = 0.9*G12
        G23 = 0.5*G12
        laminaprop = (d['Ex'], d['Ey'], d['vx'], G12, G13, G23)
        m = read_laminaprop(laminaprop)
        nu21 = d['vx']*d['Ey']/d['Ex']
        den = 1 - d['vx']*nu21
        Q11 = d['Ex']/den
        Q12 = d['vx']*d['Ey']/den
        Q22 = d['Ey']/den
        Q44 = G23
        Q55 = G13
        Q66 = G12
        tr = Q11 + Q22 + 2*Q66
        assert np.isclose(m.q11 + m.q22 + 2*m.q66, tr, rtol=1e-14)
        # NOTE Tr of Melo et al. (2017), Table 1, p. 45, is printed as an
        #      integer in GPa: the traces of Eq. (1) computed from Ex, Ey,
        #      nux and Es differ from it by +0.09 %, -0.09 % and +0.17 %,
        #      within half a unit of the last printed digit
        assert np.isclose(tr, d['tr'], atol=0.5e9, rtol=0)
        u1 = (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66) / 8.
        u2 = (Q11 - Q22) / 2.
        u3 = (Q11 + Q22 - 2*Q12 - 4*Q66) / 8.
        u4 = (Q11 + Q22 + 6*Q12 - 4*Q66) / 8.
        u5 = (u1 - u4) / 2.
        u6 = (Q44 + Q55) / 2.
        u7 = (Q44 - Q55) / 2.
        # NOTE tau_yz and tau_xz of a ply at theta: G_yz = G23 c^2 + G13 s^2
        #      = u6 + u7 cos 2 theta, G_xz = G13 c^2 + G23 s^2 = u6 - u7 cos 2
        #      theta, G_yz,xz = (G13 - G23) c s = -u7 sin 2 theta
        ply = Lamina()
        ply.thetadeg = 30.
        ply.h = 1.
        ply.matlamina = m
        ply.rebuild()
        c2 = np.cos(np.deg2rad(2*30.))
        s2 = np.sin(np.deg2rad(2*30.))
        assert np.isclose(ply.q44L, u6 + u7*c2, rtol=1e-14)
        assert np.isclose(ply.q55L, u6 - u7*c2, rtol=1e-14)
        assert np.isclose(ply.q45L, -u7*s2, rtol=1e-14)
        tr_norm_inv = np.array((u1, u2, u3, u4, u5, u6, u7))/tr
        m.trace_normalize_plane_stress()
        tr_norm_inv2 = (m.u1, m.u2, m.u3, m.u4, m.u5, m.u6, m.u7)
        assert np.allclose(tr_norm_inv2, tr_norm_inv, rtol=1e-13,
                           atol=1e-15)
        tr_norm_q = np.array((Q11, Q12, Q22, Q44, Q55, Q66))/tr
        assert np.allclose((m.q11, m.q12, m.q22, m.q44, m.q55, m.q66),
                           tr_norm_q, rtol=1e-14, atol=0)


def test_laminate_LP_gradients():
    # NOTE orthotropic plies with G13 != G23, such that no column of the
    #      gradients vanishes identically, compared with central finite
    #      differences of shellprop_from_LaminationParameters
    lamprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)
    rho = 0
    thickness = 1.
    matlamina = read_laminaprop(lamprop, rho)
    lp = LaminationParameters()
    lp.xiA1 = 0.5
    lp.xiA2 = 0.4
    lp.xiA3 = -0.3
    lp.xiA4 = -0.6
    lp.xiB1 = 0.5
    lp.xiB2 = 0.4
    lp.xiB3 = -0.3
    lp.xiB4 = -0.6
    lp.xiD1 = 0.5
    lp.xiD2 = 0.4
    lp.xiD3 = -0.3
    lp.xiD4 = -0.6
    lp.xiAts1 = 0.2
    lp.xiAts2 = -0.1
    gradABD = GradABD()
    gradABD.calc_LP_grad(thickness, matlamina, lp)

    lp_names = ['xi%s%d' % (m, i) for m in 'ABD' for i in (1, 2, 3, 4)]
    lp_names += ['xiAts1', 'xiAts2']
    ij = ('11', '12', '16', '22', '26', '66')
    def stiffnesses(h, lp):
        prop = shellprop_from_LaminationParameters(h, matlamina, lp)
        return dict(A=np.array([getattr(prop, 'A' + s) for s in ij]),
                    B=np.array([getattr(prop, 'B' + s) for s in ij]),
                    D=np.array([getattr(prop, 'D' + s) for s in ij]),
                    Ats=np.array([prop.Abar44, prop.Abar45, prop.Abar55]))

    def central_difference(name, var, step):
        def perturbed(delta):
            lp2 = LaminationParameters()
            for attr in lp_names:
                setattr(lp2, attr, getattr(lp, attr))
            h = thickness
            if var == 'h':
                h += delta
            else:
                setattr(lp2, var, getattr(lp, var) + delta)
            return stiffnesses(h, lp2)[name]
        return (perturbed(step) - perturbed(-step))/(2*step)

    variables = dict(A=['h', 'xiA1', 'xiA2', 'xiA3', 'xiA4'],
                     B=['h', 'xiB1', 'xiB2', 'xiB3', 'xiB4'],
                     D=['h', 'xiD1', 'xiD2', 'xiD3', 'xiD4'],
                     Ats=['h', 'xiAts1', 'xiAts2'])
    grads = dict(A=gradABD.gradAij, B=gradABD.gradBij, D=gradABD.gradDij,
                 Ats=gradABD.gradAtsij)
    for name, names in variables.items():
        grad = np.asarray(grads[name])
        fd = np.array([central_difference(name, var,
                           1e-6*thickness if var == 'h' else 1e-3)
                       for var in names]).T
        # NOTE A, B, D and Abar_ts are linear in the lamination parameters
        #      and at most cubic in h, such that the central differences are
        #      exact to round-off and to (step/h)^2 for D
        scale = np.abs(fd).max()
        assert scale > 0
        assert np.allclose(grad, fd, rtol=1e-7, atol=1e-7*scale), name
        # NOTE no column is identically zero
        assert np.all(np.abs(fd).max(axis=0) > 1e-3*scale), name


def test_laminated_plate_length_error():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9)
    try:
        laminated_plate(stack=[0, 45], plyts=[0.000125, 0.000125],
                        laminaprops=[lamprop], rhos=[0., 0.])
    except ValueError:
        pass
    else:
        assert False, 'ValueError not raised'


def test_transf_matrix_displ_to_lamina():
    ply = Lamina()
    ply.thetadeg = 30.
    ply.h = 1.
    ply.matlamina = read_laminaprop((142e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.2e9))
    ply.rebuild()
    T = np.asarray(ply.get_transf_matrix_displ_to_lamina())
    c, s = np.cos(np.deg2rad(30.)), np.sin(np.deg2rad(30.))
    # a displacement along the material direction 1, given in laminate axes
    assert np.allclose(T @ np.array([c, s, 0.]), [1., 0., 0.])
    assert np.allclose(T @ T.T, np.eye(3))


def test_transf_matrix_stress_order():
    # {s11, s22, s33, t23, t13, t12} = T {sxx, syy, szz, tyz, txz, txy}
    ply = Lamina()
    ply.thetadeg = 30.
    ply.h = 1.
    ply.matlamina = read_laminaprop((142e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.2e9))
    ply.rebuild()
    c, s = np.cos(np.deg2rad(30.)), np.sin(np.deg2rad(30.))
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    sig = np.array([[1.0, 0.4, -0.3], [0.4, -2.0, 0.7], [-0.3, 0.7, 0.5]])
    sig_mat = R @ sig @ R.T
    order = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    v = np.array([sig[i, j] for i, j in order])
    v_mat = np.array([sig_mat[i, j] for i, j in order])
    T_lamina = np.asarray(ply.get_transf_matrix_stress_to_lamina())
    T_laminate = np.asarray(ply.get_transf_matrix_stress_to_laminate())
    assert np.allclose(T_lamina @ v, v_mat)
    assert np.allclose(T_laminate @ v_mat, v)


def test_make_symmetric_keeps_even_powers():
    # [30/-45/60]s: A and D are integrals of even powers of z
    stack = [30, -45, 60, 60, -45, 30]
    plyt = 0.125e-3
    prop = laminated_plate(stack, plyt=plyt,
            laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9))
    A, D = prop.A, prop.D
    prop.make_symmetric()
    assert np.allclose(prop.B, 0, rtol=0, atol=0)
    assert np.array_equal(prop.A, A)
    assert np.array_equal(prop.D, D)


def test_make_orthotropic_zeroes_all_couplings():
    # terms that are integrals of q16L, q26L or q45L, all zero for 0/90 plies
    names16 = ['%s%s' % (m, ij) for m in 'ABD' for ij in ('16', '26')]
    names45 = ['A45', 'Abar45', 'Abarbar45']
    others = (['%s%s' % (m, ij) for m in 'ABD'
               for ij in ('11', '12', '22', '66')]
              + ['A44', 'A55', 'Abar44', 'Abar55', 'Abarbar44', 'Abarbar55',
                 'h', 'intrho', 'intrhoz2'])
    laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)
    plyt = 0.125e-3
    # [0/90/0/90]: the integrated values are round-off, for every correction
    for mode in ('rohwer', 'vlachoutsis', 'whitney', 'birman_bert',
                 'thickness_shear', 'constant', None):
        prop = laminated_plate([0, 90, 0, 90], plyt=plyt,
                               laminaprop=laminaprop, rho=1600.,
                               shear_correction=mode)
        for name in names16:
            assert (abs(getattr(prop, name))
                    < 1e-14*abs(getattr(prop, name[0] + '11')))
        for name in names45:
            assert (abs(getattr(prop, name))
                    < 1e-14*abs(getattr(prop, name[:-2] + '44')))
    # [30/-45/60]s: none of them is zero before make_orthotropic
    prop = laminated_plate([30, -45, 60, 60, -45, 30], plyt=plyt,
                           laminaprop=laminaprop, rho=1600.)
    before = {name: getattr(prop, name) for name in others}
    for name in names16 + names45:
        if name[0] != 'B':
            assert abs(getattr(prop, name)) > 0
    prop.make_orthotropic()
    for name in names16 + names45:
        assert getattr(prop, name) == 0
    for name in others:
        assert getattr(prop, name) == before[name]


def test_make_orthotropic_LP_transverse_shear():
    # G13 != G23, such that xiAts2 gives Abar45
    laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)
    prop = laminated_plate([30, -45, 60, 60, -45, 30], plyt=0.125e-3,
                           laminaprop=laminaprop)
    matlamina = prop.plies[0].matlamina
    lp = prop.calc_lamination_parameters()
    assert abs(lp.xiAts2) > 0
    prop.make_orthotropic()
    make_orthotropic_LP(lp)
    assert lp.xiAts2 == 0
    prop_2 = shellprop_from_LaminationParameters(prop.h, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)


def test_make_balanced_zeroes_only_vanishing_terms():
    # in a balanced laminate only the integrals of z^0 of the odd q16L, q26L
    # and q45L vanish, for any position of the +theta and -theta plies
    laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)
    plyt = 0.125e-3
    names = (['%s%s' % (m, ij) for m in 'ABD'
              for ij in ('11', '12', '16', '22', '26', '66')]
             + ['A44', 'A45', 'A55', 'Abar44', 'Abar45', 'Abar55',
                'Abarbar44', 'Abarbar45', 'Abarbar55', 'h', 'intrho',
                'intrhoz2'])
    modes = ('rohwer', 'vlachoutsis', 'whitney', 'birman_bert',
             'thickness_shear', 'constant', None)
    stacks = ([30, -30, 0], [45, 0, -45, 90], [45, -45],
              [30, -30, 0, 0, -30, 30])
    for stack in stacks:
        symmetric = stack == stack[::-1]
        for mode in modes + (('chow',) if symmetric else ()):
            zeroed = ['A16', 'A26', 'Abar45', 'Abarbar45']
            if mode != 'rohwer':
                zeroed.append('A45')
            # the integrated values are round-off, also with an offset
            for offset in (0.37e-3, 0.):
                prop = laminated_plate(stack, plyt=plyt,
                        laminaprop=laminaprop, rho=1600., offset=offset,
                        shear_correction=mode)
                for name in zeroed:
                    ref = prop.A11 if name[1] in '12' else prop.Abar44
                    assert abs(getattr(prop, name)) < 1e-14*abs(ref)
            before = {name: getattr(prop, name) for name in names}
            prop.make_balanced()
            for name in names:
                if name in zeroed:
                    assert getattr(prop, name) == 0
                else:
                    assert getattr(prop, name) == before[name]
    # antisymmetric [+45/-45]: B16 and B26 do not vanish and are kept
    prop = laminated_plate([45, -45], plyt=plyt, laminaprop=laminaprop)
    z = -prop.h/2 + plyt*np.arange(3)
    prop.make_balanced()
    for attr, name in (('B16', 'q16L'), ('B26', 'q26L')):
        ref = sum(getattr(ply, name)*(z[k+1]**2 - z[k]**2)/2
                  for k, ply in enumerate(prop.plies))
        assert abs(ref) > 1e-3*prop.A11*prop.h
        assert np.isclose(getattr(prop, attr), ref, rtol=1e-12, atol=0)
    # 'rohwer': A45 does not vanish for [+30/-30/0]s, even with g13 = g23,
    # and the stiffness in the element coordinate system is not modified
    prop = laminated_plate([30, -30, 0, 0, -30, 30], plyt=plyt,
                           laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9,
                                       5.1e9))
    A45 = prop.A45
    Ats30 = np.array(prop.calc_Ats_element(30.))
    assert prop.Abar45 == 0
    assert abs(A45) > 0.1*prop.Abar44
    prop.make_balanced()
    assert prop.A45 == A45
    assert np.array_equal(np.array(prop.calc_Ats_element(30.)), Ats30)


def test_make_balanced_LP():
    # antisymmetric [+45/-45]: xiA2, xiA4 and xiAts2 vanish, xiB2 does not
    laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)
    prop = laminated_plate([45, -45], plyt=0.125e-3, laminaprop=laminaprop)
    matlamina = prop.plies[0].matlamina
    lp = prop.calc_lamination_parameters()
    for name in ('xiA2', 'xiA4', 'xiAts2'):
        assert abs(getattr(lp, name)) < 1e-14
    xiB2 = lp.xiB2
    assert abs(xiB2) > 0.1
    make_balanced_LP(lp)
    for name in ('xiA2', 'xiA4', 'xiAts2'):
        assert getattr(lp, name) == 0
    assert lp.xiB2 == xiB2
    prop.make_balanced()
    prop_2 = shellprop_from_LaminationParameters(prop.h, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.Abar_ts, prop.Abar_ts)


def test_make_smeared():
    prop = laminated_plate([0, 45, 90], plyt=0.000125,
                           laminaprop=(71e9, 7e9, 0.28, 7e9, 7e9, 7e9))
    A = prop.A
    prop.make_smeared()
    assert np.allclose(prop.B, 0, rtol=0, atol=0)
    assert np.array_equal(prop.A, A)
    assert np.allclose(prop.D, prop.h**2/12*A, rtol=1e-15)


def test_gradabd_pickle():
    prop = laminated_plate([0, 45, 90], plyt=0.000125,
                           laminaprop=(71e9, 7e9, 0.28, 7e9, 7e9, 7e9))
    grad = GradABD()
    grad.calc_LP_grad(prop.h, prop.plies[0].matlamina,
                      prop.calc_lamination_parameters())
    assert np.any(np.asarray(grad.gradDij) != 0)
    for grad2 in (pickle.loads(pickle.dumps(grad)), copy.deepcopy(grad)):
        for name in ('gradAij', 'gradBij', 'gradDij', 'gradAtsij'):
            np.testing.assert_array_equal(getattr(grad, name),
                                          getattr(grad2, name))
    # created without __init__, as by copyreg.__newobj__ when unpickling
    grad3 = GradABD.__new__(GradABD)
    np.testing.assert_array_equal(grad3.gradDij, np.zeros((6, 5)))


def test_make_orthotropic_make_balanced_keep_nan_Abarbar45():
    # a ply with G13 = 0 has a singular C_s: the constant-stress Abarbar44,
    # Abarbar45, Abarbar55 are nan, and both methods keep Abarbar45 nan
    laminaprops = [(142.5e9, 8.7e9, 0.28, 5.1e9, 0., 3.0e9)]*4
    for stack, method in (([0, 90, 0, 90], 'make_orthotropic'),
                          ([30, -30, -30, 30], 'make_balanced')):
        prop = laminated_plate(stack, plyt=0.125e-3, laminaprops=laminaprops,
                               shear_correction=None)
        assert np.isnan(prop.Abarbar44)
        assert np.isnan(prop.Abarbar45)
        assert np.isnan(prop.Abarbar55)
        getattr(prop, method)()
        assert np.isnan(prop.Abarbar44)
        assert np.isnan(prop.Abarbar45)
        assert np.isnan(prop.Abarbar55)
        assert prop.Abar45 == 0
        assert prop.A45 == 0
