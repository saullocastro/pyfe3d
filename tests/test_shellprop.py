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
    ply.get_transf_matrix_displ_to_lamina()
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
    Abar_ts = np.array([[2.66917293e+10,  0.00000000e+00],
                  [0.00000000e+00,  2.66917293e+10]])
    assert np.allclose(prop.A, A)
    prop.make_symmetric()
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.Abar_ts, Abar_ts)
    ABD = prop.ABD
    assert np.allclose(ABD[:3, :3], A)
    assert np.allclose(ABD[3:, 3:], D)


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
    A = np.array([[7.96768040e+10, 2.62933453e+10, 1.14440918e-06],
                  [2.62933453e+10, 7.96768040e+10, -1.14440918e-06],
                  [1.14440918e-06, -1.14440918e-06, 2.66917293e+10]])
    B = np.array([[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]])
    D = np.array([[6.63973366e+09, 2.19111211e+09, 9.53674316e-08],
                  [2.19111211e+09, 6.63973366e+09, -9.53674316e-08],
                  [9.53674316e-08, -9.53674316e-08, 2.22431078e+09]])
    Abar_ts = np.array([[2.66917293e+10, 0.00000000e+00],
                  [0.00000000e+00, 2.66917293e+10]])
    assert np.allclose(prop.A, A)
    prop.make_symmetric()
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.Abar_ts, Abar_ts)
    ABD = prop.ABD
    assert np.allclose(ABD[:3, :3], A)
    assert np.allclose(ABD[3:, 3:], D)


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
    Abar_ts = np.array([[ 2625000.,       0.],
                  [       0., 2625000.]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.Abar_ts, Abar_ts)


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
    Abar_ts = np.array([[ 2625000.,       0.],
                  [       0., 2625000.]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, B)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.Abar_ts, Abar_ts)
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


def helper_isotropic_plate():
    E = 71e9
    nu = 0.28
    thick = 0.000125
    prop = isotropic_plate(thickness=thick, E=E, nu=nu)
    return prop


def test_isotropic_plate():
    prop = helper_isotropic_plate()
    A = np.array([[9629991.31944444, 2696397.56944444,       0.   ],
                  [2696397.56944444, 9629991.31944444,       0.   ],
                  [      0.        ,       0.        , 3466796.875]])
    D = np.array([[0.01253905, 0.00351093, 0.        ],
                  [0.00351093, 0.01253905, 0.        ],
                  [0.        , 0.        , 0.00451406]])
    Abar_ts = np.array([[3466796.875,       0.   ],
                  [      0.   , 3466796.875]])
    assert np.allclose(prop.A, A)
    assert np.allclose(prop.B, 0)
    assert np.allclose(prop.D, D)
    assert np.allclose(prop.Abar_ts, Abar_ts)
    assert np.allclose(prop.Ats, 5/6*Abar_ts)


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
    try:
        prop.make_balanced()
    except RuntimeError:
        pass
    try:
        prop.make_orthotropic()
    except RuntimeError:
        pass
    try:
        prop.make_symmetric()
    except RuntimeError:
        pass
    with pytest.raises(NotImplementedError):
        prop.make_smeared()
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
    gradABD = GradABD()
    gradABD.calc_LP_grad(thickness, matlamina, lp)
    print(gradABD.gradAij)
    print(np.asarray(gradABD.gradAtsij))


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
