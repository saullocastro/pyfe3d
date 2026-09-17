import sys
sys.path.append('..')

import numpy as np
import pytest

from pyfe3d.shellprop import shellprop_from_lamination_parameters
from pyfe3d.shellprop_utils import (read_laminaprop, laminated_plate,
        isotropic_plate)
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
        Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF)

# NOTE CFRP of Rohwer (1988), kN/mm^2 and mm
CFRP = (138., 9.3, 0.3, 4.6, 4.6, 2.3)
PLYT = 0.25
CROSS_PLY = [0, 90, 90, 0]
QUASI_ISO = [0, 45, -45, 90, 90, -45, 45, 0]
UNSYM = [30, -60, 15, 75, 0]


def tensor_rotation(Ats, thetadeg):
    c = np.cos(np.deg2rad(thetadeg))
    s = np.sin(np.deg2rad(thetadeg))
    Ts = np.array([[c, s], [-s, c]])
    return Ts @ Ats @ Ts.T


def test_isotropic_five_sixths():
    E = 71e9
    nu = 0.33
    h = 0.002
    G = E/(2*(1 + nu))
    for offset in [0., 0.37*h]:
        for mode in ['rohwer', 'vlachoutsis', 'constant']:
            prop = isotropic_plate(thickness=h, E=E, nu=nu, offset=offset,
                                   shear_correction=mode)
            assert np.allclose(prop.Ats, 5/6*G*h*np.eye(2), rtol=1e-12)
            assert np.isclose(prop.scf_k13, 5/6)
            assert np.isclose(prop.scf_k23, 5/6)
            assert np.allclose(prop.Abar_ts, G*h*np.eye(2))
            assert np.allclose(prop.Abarbar_ts, G*h*np.eye(2))
        prop = isotropic_plate(thickness=h, E=E, nu=nu, offset=offset,
                               shear_correction=None)
        assert np.allclose(prop.Ats, G*h*np.eye(2))
        assert np.isclose(prop.scf_k13, 1.)


def test_rohwer_cross_ply_reference_values():
    # NOTE Rohwer (1988), and Table plate-shear-rotation of the theory
    prop = laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=CFRP)
    assert np.isclose(prop.Abar44, 3.450, rtol=1e-4)
    assert np.isclose(prop.Abar55, 3.450, rtol=1e-4)
    assert np.isclose(prop.Abarbar44, 3.067, rtol=1e-3)
    assert np.isclose(prop.Abarbar55, 3.067, rtol=1e-3)
    assert np.isclose(prop.A44, 2.5252, rtol=1e-4)
    assert np.isclose(prop.A55, 2.3131, rtol=1e-4)
    assert np.isclose(prop.A45, 0., atol=1e-12)
    assert np.isclose(prop.scf_k13, prop.A55/prop.Abar55)
    assert np.isclose(prop.scf_k23, prop.A44/prop.Abar44)


def test_rohwer_rotated_plies_reference_values():
    # NOTE "direct" columns of Table plate-shear-rotation, with the plies at
    #      theta_k - phi
    ref = {
        (tuple(CROSS_PLY), 45): (2.5885, 0.1204, 2.5885),
        (tuple(CROSS_PLY), 30): (2.6109, 0.0849, 2.4500),
        (tuple(QUASI_ISO), 0): (5.0341, -0.0724, 4.9906),
        (tuple(QUASI_ISO), 60): (4.8049, -0.3695, 3.8566),
    }
    for (stack, phi), (A44, A45, A55) in ref.items():
        prop = laminated_plate(stack=[t - phi for t in stack], plyt=PLYT,
                               laminaprop=CFRP)
        assert np.allclose(prop.Ats, [[A44, A45], [A45, A55]], atol=1.5e-4)


@pytest.mark.parametrize('stack, offset', [(CROSS_PLY, 0.), (QUASI_ISO, 0.),
                                           (UNSYM, 0.3)])
def test_element_frame_equals_shifted_plies(stack, offset):
    # NOTE a material direction at theta from the element x axis is the same
    #      laminate with all ply angles shifted by +theta
    prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                           offset=offset)
    assert np.allclose(prop.calc_Ats_element(0.), prop.Ats, rtol=1e-14)
    for theta in [-75., -30., 15., 45., 60., 90., 135., 180.]:
        ref = laminated_plate(stack=[t + theta for t in stack], plyt=PLYT,
                              laminaprop=CFRP, offset=offset)
        Ats_e = prop.calc_Ats_element(theta)
        assert np.allclose(Ats_e, ref.Ats, rtol=1e-10, atol=1e-12)
        # NOTE the same ABD is obtained with the tensor rotation of the ABD
        #      or from the shifted plies, the frame dependence is exclusive
        #      of the corrected transverse shear stiffness
        assert np.allclose(tensor_rotation(prop.Abar_ts, theta), ref.Abar_ts)
        # NOTE bound Ats <= Abar_ts in any frame
        eig = np.linalg.eigvalsh(ref.Abar_ts - Ats_e)
        assert eig.min() > -1e-10*np.abs(eig).max()
        assert ref.scf_k13 <= 1. and ref.scf_k23 <= 1.
        # NOTE periodic with 180 degrees
        assert np.allclose(prop.calc_Ats_element(theta + 180.), Ats_e)


def test_constitutive_element_equals_shifted_plies():
    # NOTE A, B, D and Ats obtained with the single function used by all
    #      elements, compared against the laminate with shifted plies, at
    #      arbitrary angles to verify the Fourier evaluation of Ats
    rng = np.random.default_rng(1)
    for stack, offset in [(QUASI_ISO, 0.), (UNSYM, 0.3), ([0, 90], 0.)]:
        prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                               offset=offset)
        for theta in np.concatenate(([0., 90., 180.], rng.uniform(-180, 180, 20))):
            ref = laminated_plate(stack=[t + theta for t in stack], plyt=PLYT,
                                  laminaprop=CFRP, offset=offset)
            A, B, D, Ats = prop.calc_constitutive_element(theta)
            scale = np.abs(ref.ABD).max()
            assert np.allclose(A, ref.A, rtol=1e-12, atol=1e-12*scale)
            assert np.allclose(B, ref.B, rtol=1e-12, atol=1e-12*scale)
            assert np.allclose(D, ref.D, rtol=1e-12, atol=1e-12*scale)
            assert np.allclose(Ats, ref.Ats, rtol=1e-11, atol=1e-12)


def test_quasi_isotropic_frame_dependence():
    # NOTE the tensor rotation of the stored result cannot reproduce the
    #      element-frame evaluation for this laminate
    prop = laminated_plate(stack=QUASI_ISO, plyt=PLYT, laminaprop=CFRP)
    Ats_e = prop.calc_Ats_element(60.)
    Ats_t = tensor_rotation(prop.Ats, 60.)
    dev = np.abs(Ats_e - Ats_t).max()/np.abs(prop.Ats).max()
    assert dev > 0.1
    # NOTE the trace is not preserved
    assert not np.isclose(np.trace(Ats_e), np.trace(prop.Ats), rtol=1e-2)


@pytest.mark.parametrize('mode', ['constant', None, 'vlachoutsis'])
def test_element_frame_tensor_rotation(mode):
    prop = laminated_plate(stack=UNSYM, plyt=PLYT, laminaprop=CFRP,
                           shear_correction=mode)
    for theta in [0., 20., 45., 90., 110.]:
        assert np.allclose(prop.calc_Ats_element(theta),
                           tensor_rotation(prop.Ats, theta))
        if mode in ('constant', None):
            ref = laminated_plate(stack=[t + theta for t in UNSYM], plyt=PLYT,
                                  laminaprop=CFRP, shear_correction=mode)
            assert np.allclose(prop.calc_Ats_element(theta), ref.Ats)


def test_lamination_parameters_no_correction():
    matlamina = read_laminaprop(CFRP)
    prop = shellprop_from_lamination_parameters(1., matlamina,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4, -0.3, -0.6,
        0.5, 0.4)
    assert prop.shear_correction is None
    assert np.allclose(prop.Ats, prop.Abar_ts)
    assert np.all(np.isnan(prop.Abarbar_ts))
    for theta in [0., 30., 90.]:
        assert np.allclose(prop.calc_Ats_element(theta),
                           tensor_rotation(prop.Ats, theta))


def test_vlachoutsis_cross_ply():
    # NOTE for a symmetric cross-ply both methods should be close
    prop_r = laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=CFRP)
    prop_v = laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=CFRP,
                             shear_correction='vlachoutsis')
    assert np.allclose(prop_v.Ats, prop_r.Ats, rtol=0.02)
    assert prop_v.scf_k13 < 1 and prop_v.scf_k23 < 1


def test_transverse_shear_stress():
    E = 71e9
    nu = 0.33
    h = 0.002
    offset = 0.1*h
    prop = isotropic_plate(thickness=h, E=E, nu=nu, offset=offset)
    Qx = 3.
    Qy = -2.
    for zbar in np.linspace(-h/2, h/2, 11):
        tau_yz, tau_xz = prop.calc_transverse_shear_stress(zbar + offset, Qy, Qx)
        assert np.isclose(tau_xz, 3*Qx/(2*h)*(1 - 4*zbar**2/h**2), atol=1e-8)
        assert np.isclose(tau_yz, 3*Qy/(2*h)*(1 - 4*zbar**2/h**2), atol=1e-8)

    for stack in [CROSS_PLY, QUASI_ISO, UNSYM]:
        for mode in ['rohwer', None]:
            prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                                   offset=0.1, shear_correction=mode)
            z0 = -prop.h/2 + 0.1
            z1 = +prop.h/2 + 0.1
            # NOTE traction free at both faces
            assert np.allclose(prop.calc_transverse_shear_stress(z0, 1., 1.), 0, atol=1e-10)
            assert np.allclose(prop.calc_transverse_shear_stress(z1, 1., 1.), 0, atol=1e-10)
            # NOTE the distribution carries the shear forces
            zs = np.linspace(z0, z1, 2001)
            tau = np.array([prop.calc_transverse_shear_stress(z, 0.7, -1.3) for z in zs])
            assert np.allclose(np.trapezoid(tau, zs, axis=0), [0.7, -1.3], rtol=1e-3)
    with pytest.raises(ValueError):
        prop.calc_transverse_shear_stress(z1 + 1., 1., 1.)


def test_errors():
    with pytest.raises(ValueError):
        laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=CFRP,
                        shear_correction='invalid')
    no_g13 = (138., 9.3, 0.3, 4.6, 0., 2.3)
    for mode in ['rohwer', 'vlachoutsis']:
        with pytest.raises(ValueError):
            laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=no_g13,
                            shear_correction=mode)
    prop = laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=no_g13,
                           shear_correction=None)
    assert np.all(np.isnan(prop.Abarbar_ts))


def test_pickle_and_deepcopy():
    import copy
    import pickle
    prop = laminated_plate(stack=UNSYM, plyt=PLYT, laminaprop=CFRP, offset=0.1)
    for prop2 in [pickle.loads(pickle.dumps(prop)), copy.deepcopy(prop)]:
        assert np.allclose(prop2.ABD, prop.ABD)
        assert np.allclose(prop2.Ats, prop.Ats)
        assert np.allclose(prop2.Abar_ts, prop.Abar_ts)
        assert prop2.shear_correction == prop.shear_correction
        for theta in [0., 35., 90.]:
            assert np.allclose(prop2.calc_Ats_element(theta),
                               prop.calc_Ats_element(theta))
        assert np.allclose(prop2.calc_transverse_shear_stress(0.1, 1., 2.),
                           prop.calc_transverse_shear_stress(0.1, 1., 2.))


def _element_KC0(element, stack, thetadeg, offset):
    r"""KC0 of a single element with a material direction at thetadeg"""
    if element == 'Quad4':
        data, probe, cls, nnodes = Quad4Data(), Quad4Probe(), Quad4, 4
    elif element == 'Quad4R':
        data, probe, cls, nnodes = Quad4RData(), Quad4RProbe(), Quad4R, 4
    else:
        data, probe, cls, nnodes = Tria3RData(), Tria3RProbe(), Tria3R, 3
    h = PLYT*len(stack)
    ncoords = np.array([[0, 0, 0], [3*h, 0, 0], [3*h, 2*h, 0], [0, 2*h, 0]],
                       dtype=DOUBLE)[:nnodes]
    ncoords_flatten = ncoords.flatten()
    N = DOF*nnodes
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                           offset=offset)
    elem = cls(probe)
    elem.n1 = 1
    elem.n2 = 2
    elem.n3 = 3
    elem.c1 = 0
    elem.c2 = DOF
    elem.c3 = 2*DOF
    if nnodes == 4:
        elem.n4 = 4
        elem.c4 = 3*DOF
    elem.init_k_KC0 = 0
    c = np.cos(np.deg2rad(thetadeg))
    s = np.sin(np.deg2rad(thetadeg))
    if thetadeg == 0:
        elem.update_rotation_matrix(ncoords_flatten)
    else:
        elem.update_rotation_matrix(ncoords_flatten, c, s, 0.)
        assert np.isclose(elem.m11, c) and np.isclose(elem.m21, s)
    elem.update_probe_xe(ncoords_flatten)
    elem.update_KC0(KC0r, KC0c, KC0v, prop)
    K = np.zeros((N, N))
    np.add.at(K, (KC0r, KC0c), KC0v)
    u = np.linspace(-1, 1, N)**3*1e-3
    elem.update_probe_ue(u)
    fint = np.zeros(N)
    elem.update_fint(fint, prop)
    return K, fint


@pytest.mark.parametrize('element', ['Quad4', 'Quad4R', 'Tria3R'])
def test_element_material_direction_consistency(element):
    # NOTE an element with material direction theta and plies theta_k must
    #      be identical to an element without material direction and plies
    #      theta_k + theta, including the transverse shear terms
    for stack, offset in [(QUASI_ISO, 0.), (UNSYM, 0.2)]:
        for theta in [30., 60.]:
            K, fint = _element_KC0(element, stack, theta, offset)
            K_ref, fint_ref = _element_KC0(element, [t + theta for t in stack],
                                           0., offset)
            scale = np.abs(K_ref).max()
            assert np.allclose(K, K_ref, rtol=1e-9, atol=1e-11*scale)
            assert np.allclose(fint, fint_ref, rtol=1e-9,
                               atol=1e-11*np.abs(fint_ref).max())


if __name__ == '__main__':
    test_isotropic_five_sixths()
    test_rohwer_cross_ply_reference_values()
    test_rohwer_rotated_plies_reference_values()
