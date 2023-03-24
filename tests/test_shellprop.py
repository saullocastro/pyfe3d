import sys
sys.path.append('..')

import numpy as np

from pyfe3d.shellprop import (Lamina, shellprop_from_LaminationParameters,
        shellprop_from_lamination_parameters, force_balanced_LP,
        force_orthotropic_LP, force_symmetric_LP, LaminationParameters,
        GradABDE)
from pyfe3d.shellprop_utils import (read_laminaprop, laminated_plate,
        isotropic_plate)

data = {
'IM6/epoxy': dict(Ex=203e9, Ey=11.20e9, vx=0.32, Es=8.40e9, tr=232e9),
'IM7/977-3': dict(Ex=191e9, Ey=9.94e9, vx=0.35, Es=7.79e9, tr=218e9),
'T4708/MR60H': dict(Ex=142e9, Ey=7.72e9, vx=0.34, Es=3.80e9, tr=158e9),
}

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
    ply.get_transf_matrix_displ_to_laminate()
    ply.get_constitutive_matrix()
    ply.get_transf_matrix_stress_to_lamina()
    ply.get_transf_matrix_stress_to_laminate()

    prop = shellprop_from_lamination_parameters(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    A = np.array([[1.05196816e+11, 5.18133569e+10, 0.00000000e+00],
                  [5.18133569e+10, 1.05196816e+11, 0.00000000e+00],
                  [0.00000000e+00, 0.00000000e+00, 2.66917293e+10]])
    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    D = np.array([[8.76640130e+09, 4.31777974e+09, 0.00000000e+00],
                  [4.31777974e+09, 8.76640130e+09, 0.00000000e+00],
                  [0.00000000e+00, 0.00000000e+00, 2.22431078e+09]])
    E = np.array([[2.66917293e+10,  0.00000000e+00],
                  [0.00000000e+00,  2.66917293e+10]])
    assert np.allclose(prop.A, A)
    prop.force_symmetric()
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.E, E)
    ABD = prop.ABD
    assert np.allclose(ABD[:3, :3], A)
    assert np.allclose(ABD[3:, 3:], D)
    ABDE = prop.ABDE
    assert np.allclose(ABDE[:3, :3], A)
    assert np.allclose(ABDE[3:6, 3:6], D)


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
    ply.get_transf_matrix_displ_to_laminate()
    ply.get_constitutive_matrix()
    ply.get_transf_matrix_stress_to_lamina()
    ply.get_transf_matrix_stress_to_laminate()

    prop = shellprop_from_lamination_parameters(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    A = np.array([[7.96768040e+10, 2.62933453e+10, 1.14440918e-06],
                  [2.62933453e+10, 7.96768040e+10, -1.14440918e-06],
                  [1.14440918e-06, -1.14440918e-06, 2.66917293e+10]])
    B = np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])
    D = np.array([[6.63973366e+09, 2.19111211e+09, 9.53674316e-08],
                  [2.19111211e+09, 6.63973366e+09, -9.53674316e-08],
                  [9.53674316e-08, -9.53674316e-08, 2.22431078e+09]])
    E = np.array([[2.66917293e+10, 0.00000000e+00],
                  [0.00000000e+00, 2.66917293e+10]])
    assert np.allclose(prop.A, A)
    prop.force_symmetric()
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.E, E)
    ABD = prop.ABD
    assert np.allclose(ABD[:3, :3], A)
    assert np.allclose(ABD[3:, 3:], D)
    ABDE = prop.ABDE
    assert np.allclose(ABDE[:3, :3], A)
    assert np.allclose(ABDE[3:6, 3:6], D)


def test_laminated_plate_tri_axial():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9, 7e9, 0.28, 0.28)
    stack = [0, 45, 90]
    plyt = 0.000125
    prop = laminated_plate(stack, plyt, lamprop)
    A = np.array([[ 13280892.30559593, 2198758.85719477, 2015579.57848837],
                  [  2198758.85719477,13280892.30559593, 2015579.57848837],
                  [  2015579.57848837, 2015579.57848837, 4083033.36210029]])
    B = np.array([[ -1.00778979e+03, 0.00000000e+00, 8.53496487e-15],
                  [  0.00000000e+00, 1.00778979e+03, 5.31743621e-14],
                  [  8.53496487e-15, 5.31743621e-14, 0.00000000e+00]])
    D = np.array([[ 0.1708233 , 0.01057886, 0.00262445],
                  [ 0.01057886, 0.1708233 , 0.00262445],
                  [ 0.00262445, 0.00262445, 0.0326602 ]])
    E = np.array([[ 2625000.,       0.],
                  [       0., 2625000.]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.E, E)


def test_laminated_plate_plane_stress():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9)
    stack = [0, 45, 90]
    plyt = 0.000125
    prop = laminated_plate(stack, plyt, lamprop)
    A = np.array([[ 13280892.30559593, 2198758.85719477, 2015579.57848837],
                  [  2198758.85719477,13280892.30559593, 2015579.57848837],
                  [  2015579.57848837, 2015579.57848837, 4083033.36210029]])
    B = np.array([[ -1.00778979e+03, 0.00000000e+00, 8.53496487e-15],
                  [  0.00000000e+00, 1.00778979e+03, 5.31743621e-14],
                  [  8.53496487e-15, 5.31743621e-14, 0.00000000e+00]])
    D = np.array([[ 0.1708233 , 0.01057886, 0.00262445],
                  [ 0.01057886, 0.1708233 , 0.00262445],
                  [ 0.00262445, 0.00262445, 0.0326602 ]])
    E = np.array([[ 2625000.,       0.],
                  [       0., 2625000.]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.E, E)
    prop.calc_scf()
    prop.calc_equivalent_properties()
    lp = prop.calc_lamination_parameters()
    matlamina = prop.plies[0].matlamina
    thickness = prop.h
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.E, prop.E)

    prop.force_balanced()
    force_balanced_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.E, prop.E)

    prop.force_orthotropic()
    force_orthotropic_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.E, prop.E)

    prop = laminated_plate(stack, plyt, lamprop)
    lp = prop.calc_lamination_parameters()
    prop.force_symmetric()
    force_symmetric_LP(lp)
    prop_2 = shellprop_from_LaminationParameters(thickness, matlamina, lp)
    assert np.allclose(prop_2.A, prop.A)
    assert np.allclose(prop_2.B, prop.B)
    assert np.allclose(prop_2.D, prop.D)
    assert np.allclose(prop_2.E, prop.E)


def test_isotropic_plate():
    E = 71e9
    nu = 0.28
    thick = 0.000125
    prop = isotropic_plate(thickness=thick, E=E, nu=nu)
    A = np.array([[9629991.31944444, 2696397.56944444,       0.   ],
                  [2696397.56944444, 9629991.31944444,       0.   ],
                  [      0.        ,       0.        , 3466796.875]])
    D = np.array([[0.01253905, 0.00351093, 0.        ],
                  [0.00351093, 0.01253905, 0.        ],
                  [0.        , 0.        , 0.00451406]])
    E = np.array([[3466796.875,       0.   ],
                  [      0.   , 3466796.875]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, 0)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.E, E)
    return prop

def test_errors():
    prop = test_isotropic_plate()
    prop.offset = 1.
    try:
        prop.force_balanced()
    except RuntimeError:
        pass
    try:
        prop.force_orthotropic()
    except RuntimeError:
        pass
    try:
        prop.force_symmetric()
    except RuntimeError:
        pass
    try:
        prop.plies = []
        prop.calc_lamination_parameters()
    except ValueError:
        pass

def test_trace_normalized():
    r"""
    Reference:

        Melo, J. D. D., Bi, J., and Tsai, S. W., 2017, “A Novel Invariant-Based
        Design Approach to Carbon Fiber Reinforced Laminates,” Compos. Struct.,
        159, pp. 44–52.

    """
    for material, d in data.items():
        m = read_laminaprop((d['Ex'], d['Ey'], d['vx'], d['Es'], d['Es'], d['Es']))
        tr = m.q11 + m.q22 + 2*m.q66
        assert np.isclose(tr, d['tr'], rtol=0.01)
        q11 = m.q11
        q12 = m.q12
        q22 = m.q22
        q44 = m.q44
        q55 = m.q55
        q66 = m.q66
        q11 /= tr
        q12 /= tr
        q22 /= tr
        q44 /= tr
        q55 /= tr
        q66 /= tr
        u1 = (3*q11 + 3*q22 + 2*q12 + 4*q44) / 8.
        u2 = (q11 - q22) / 2.
        u3 = (q11 + q22 - 2*q12 - 4*q44) / 8.
        u4 = (q11 + q22 + 6*q12 - 4*q44) / 8.
        u5 = (u1 - u4) / 2.
        u6 = (q55 + q66) / 2.
        u7 = (q55 - q66) / 2.
        tr_norm_inv = (u1, u2, u3, u4, u5, u6, u7)
        m.trace_normalize_plane_stress()
        tr_norm_inv2 = (m.u1, m.u2, m.u3, m.u4, m.u5, m.u6, m.u7)
        assert np.allclose(tr_norm_inv, tr_norm_inv2)


def test_laminate_LP_gradients():
    E = 71e9
    nu = 0.33
    lamprop = (E, nu)
    rho = 0
    thickness = 1
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
    gradABDE = GradABDE()
    gradABDE.calc_LP_grad(thickness, matlamina, lp)
    print(gradABDE.gradAij)
