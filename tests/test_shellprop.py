import sys
sys.path.append('..')

import numpy as np

from pyfe3d.shellprop import (laminate_from_lamination_parameters,
        laminate_from_lamination_parameters2,
        force_balanced_LP, force_symmetric_LP, Lamina, read_laminaprop,
        laminated_plate, isotropic_plate)


def test_lampar():
    lamprop = (71e9, 71e9, 0.33)
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

    lam = laminate_from_lamination_parameters2(thickness, matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6)
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
    assert np.allclose(lam.A, A)
    lam.force_symmetric()
    assert np.allclose(lam.B, B)
    assert np.allclose(lam.D, D)
    assert np.allclose(lam.E, E)
    ABD = lam.ABD
    assert np.allclose(ABD[:3, :3], A)
    assert np.allclose(ABD[3:, 3:], D)
    ABDE = lam.ABDE
    assert np.allclose(ABDE[:3, :3], A)
    assert np.allclose(ABDE[3:6, 3:6], D)



def test_laminated_plate():
    lamprop = (71e9, 7e9, 0.28, 7e9, 7e9, 7e9)
    stack = [0, 45, 90]
    plyt = 0.000125
    lam = laminated_plate(stack, plyt, lamprop)
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
    assert np.allclose(lam.A, A)
    assert np.allclose(lam.B, B)
    assert np.allclose(lam.D, D)
    assert np.allclose(lam.E, E)
    lam.calc_scf()
    lam.calc_equivalent_properties()
    lp = lam.calc_lamination_parameters()
    matlamina = lam.plies[0].matlamina
    thickness = lam.h
    lam = laminate_from_lamination_parameters(thickness, matlamina, lp)
    #TODO A, B and D are changing from the original, check!
    A = np.array([[ 13589503.90225179,  2502486.88587513,  2026742.01957523],
                  [  2502486.88587513, 13589503.90225179,  2026742.01957523],
                  [  2026742.01957523,  2026742.01957523,  4084254.25409417]])
    B = np.array([[ -1.01337101e+03,  0.00000000e+00,  8.68715094e-15],
                  [  0.00000000e+00,  1.01337101e+03,  5.33639272e-14],
                  [  8.68715094e-15,  5.33639272e-14,  0.00000000e+00]])
    D = np.array([[ 0.17445256, 0.01412545, 0.00263899],
                  [ 0.01412545, 0.17445256, 0.00263899],
                  [ 0.00263899, 0.00263899, 0.03266179]])
    E = np.array([[ 2625000.,       0.],
                  [       0., 2625000.]])
    assert np.allclose(lam.A, A)
    assert np.allclose(lam.B, B), print(np.asarray(lam.B), B)
    assert np.allclose(lam.D, D)
    assert np.allclose(lam.E, E)

    lam.force_orthotropic()
    A = np.array([[ 13589503.90225179,  2502486.88587513,  0],
                  [  2502486.88587513, 13589503.90225179,  0],
                  [  0,  0,  4084254.25409417]])
    B = np.array([[ -1.01337101e+03,  0.00000000e+00,  0],
                  [  0.00000000e+00,  1.01337101e+03,  0],
                  [  0,  0,  0.00000000e+00]])
    D = np.array([[ 0.17445256, 0.01412545, 0],
                  [ 0.01412545, 0.17445256, 0],
                  [ 0, 0, 0.03266179]])
    assert np.allclose(lam.A, A)
    assert np.allclose(lam.B, B)
    assert np.allclose(lam.D, D)

    lam.force_symmetric()
    assert np.allclose(lam.B, 0*B)

    force_balanced_LP(lp)
    force_symmetric_LP(lp)

def test_isotropic_plate():
    E = 71e9
    nu = 0.28
    thick = 0.000125
    lam = isotropic_plate(thickness=thick, E=E, nu=nu)
    A = np.array([[9629991.31944444, 2696397.56944444,       0.   ],
                  [2696397.56944444, 9629991.31944444,       0.   ],
                  [      0.        ,       0.        , 3466796.875]])
    D = np.array([[0.01253905, 0.00351093, 0.        ],
                  [0.00351093, 0.01253905, 0.        ],
                  [0.        , 0.        , 0.00451406]])
    E = np.array([[3466796.875,       0.   ],
                  [      0.   , 3466796.875]])
    assert np.allclose(lam.A, A)
    assert np.allclose(lam.B, 0)
    assert np.allclose(lam.D, D)
    assert np.allclose(lam.E, E)
    return lam

def test_errors():
    lam = test_isotropic_plate()
    lam.offset = 1.
    try:
        lam.force_orthotropic()
    except RuntimeError:
        pass
    try:
        lam.force_symmetric()
    except RuntimeError:
        pass
    try:
        lam.plies = []
        lam.calc_lamination_parameters()
    except ValueError:
        pass

if __name__ == '__main__':
    test_lampar()
    test_laminated_plate()
    test_isotropic_plate()
    test_errors()

