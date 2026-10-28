import sys
sys.path.append('..')
from decimal import Decimal

import numpy as np
import pytest

from pyfe3d.shellprop import ShellProp, shellprop_from_lamination_parameters
from pyfe3d.shellprop_utils import (read_laminaprop, laminated_plate,
        isotropic_plate)
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
        Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF)

from test_shellprop import qbar

# Rohwer, K. "Improved transverse shear stiffness for layered finite elements",
# DFVLR-FB 88-32, 1988. Table 1, p. 19: "CFRP material constants", kN/mm^2
# E_L, E_T, G_LT, G_TT, nu_LT, as laminaprop = (E11, E22, nu12, G12, G13, G23)
# of a transversely isotropic ply
CFRP = (138., 9.3, 0.3, 4.6, 4.6, 2.3)
PLYT = 0.25
CROSS_PLY = [0, 90, 90, 0]
QUASI_ISO = [0, 45, -45, 90, 90, -45, 45, 0]
UNSYM = [30, -60, 15, 75, 0]

# Rohwer (1988), p. 18: "A laminate with a stacking sequence of [0,90]s is
# built from this material where each of the four layers is 0.25mm thick. The
# simple tansverse shear stiffnesses as calculated from Eqs. (22) and (25) and
# the improved values according to Eq. (43) are given in Table 2. Coupling
# terms are zero." Table 2, p. 19, "Shear stiffnesses in kN/mm", printed as
# Hbar11 = Hbar22, Hbarbar11 = Hbarbar22, H11, H22. Index 1 of Rohwer is x
# and 2 is y, Eq. (41), i.e. H11 is A55 and H22 is A44.
ROHWER_1988_TABLE_2 = dict(Abar='3.450', Abarbar='3.067', A55='2.313',
                           A44='2.521')


def assert_printed(value, printed, rtol=None):
    r"""Assert that ``value`` agrees with the number ``printed`` in a paper

    The tolerance is half a unit of the last printed digit, i.e. ``value`` must
    round to ``printed``, unless the relative tolerance ``rtol`` is given.
    """
    ref = float(printed)
    if rtol is None:
        tol = 0.5*10.**Decimal(printed).as_tuple().exponent
    else:
        tol = rtol*abs(ref)
    assert abs(value - ref) <= tol*(1 + 1e-12), (value, printed)


def tensor_rotation(Ats, thetadeg):
    c = np.cos(np.deg2rad(thetadeg))
    s = np.sin(np.deg2rad(thetadeg))
    Ts = np.array([[c, s], [-s, c]])
    return Ts @ Ats @ Ts.T


def rohwer_stiffness(stack, plyts, laminaprops, offset=0.):
    r"""Improved transverse shear stiffness of Rohwer (1988), Eq. (43)

    Written out here independently of pyfe3d, with the notation of Rohwer
    (1988), Sec. 3.4, pp. 15-17: the shear stresses of Eqs. (35) and (36),
    ``tau_xz = -int Q1 [A* + z B*, B* + z D*] RFXX R dz`` and ``tau_yz = -int
    Q2 [...] RFYY R dz``, with ``Q1`` and ``Q2`` the first two rows of the
    plane-stress stiffness of the ply in the laminate axes, the compliance
    ``[[A*, B*], [B*^T, D*]] = ABD^-1`` and ``R = {Rx, Ry}`` the shear forces,
    vanishing at the bottom face; the cylindrical bending states of Eq. (34),
    ``M_x,x = Rx``, ``M_xy,x = Ry`` for ``tau_xz`` and ``M_y,y = Ry``,
    ``M_xy,y = Rx`` for ``tau_yz``, with no membrane force gradients; and Eq.
    (43), ``H = [sum_k int f^T G^-1 f dz]^-1``, where ``tau = f R``, Eq. (41),
    with the 3-point Gauss-Legendre rule, exact for the polynomial of degree 4
    in each ply.

    Returns ``Ats = [[A44, A45], [A45, A55]]``, 4 <-> yz, 5 <-> xz, and the
    shear stresses at the top face, which must vanish.
    """
    h = sum(plyts)
    z = offset - h/2 + np.concatenate(([0.], np.cumsum(plyts)))
    Qbs = [qbar(p[0], p[1], p[2], p[3], t)
           for t, p in zip(stack, laminaprops)]
    ABD = np.zeros((6, 6))
    for k, Qb in enumerate(Qbs):
        ABD[:3, :3] += Qb*(z[k+1] - z[k])
        ABD[:3, 3:] += Qb*(z[k+1]**2 - z[k]**2)/2
        ABD[3:, 3:] += Qb*(z[k+1]**3 - z[k]**3)/3
    ABD[3:, :3] = ABD[:3, 3:].T
    compliance = np.linalg.inv(ABD)
    # NOTE strain gradients from the moment gradients only: z-dependent
    #      part (B* + z D*) {M_x, M_y, M_xy}_,x
    Bs, Ds = compliance[:3, 3:], compliance[3:, 3:]
    RFXX = np.array([[1., 0.], [0., 0.], [0., 1.]])
    RFYY = np.array([[0., 0.], [0., 1.], [1., 0.]])
    xg, wg = np.polynomial.legendre.leggauss(3)
    F = np.zeros((2, 2))
    f0 = np.zeros((2, 2))
    for k, (thetadeg, p) in enumerate(zip(stack, laminaprops)):
        Qb = Qbs[k]
        z0, z1 = z[k], z[k+1]
        def f(zz):
            # rows tau_xz, tau_yz; columns Rx, Ry
            integral = Bs*(zz - z0) + Ds*(zz**2 - z0**2)/2
            return np.vstack((f0[0] - Qb[0] @ integral @ RFXX,
                              f0[1] - Qb[1] @ integral @ RFYY))
        c = np.cos(np.deg2rad(thetadeg))
        s = np.sin(np.deg2rad(thetadeg))
        G13, G23 = p[4], p[5]
        # transverse shear moduli in the laminate axes, rows xz, yz
        G = np.array([[G13*c**2 + G23*s**2, (G13 - G23)*c*s],
                      [(G13 - G23)*c*s, G13*s**2 + G23*c**2]])
        Ginv = np.linalg.inv(G)
        for xi, wi in zip(xg, wg):
            fz = f((z0 + z1)/2 + xi*(z1 - z0)/2)
            F += wi*(z1 - z0)/2*fz.T @ Ginv @ fz
        f0 = f(z1)
    H = np.linalg.inv(F)
    Ats = np.array([[H[1, 1], H[0, 1]], [H[1, 0], H[0, 0]]])
    return Ats, f0


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
    # NOTE Rohwer (1988), [0,90]s of Table 2, p. 19
    ref = ROHWER_1988_TABLE_2
    prop = laminated_plate(stack=CROSS_PLY, plyt=PLYT, laminaprop=CFRP)
    G12, G13, G23 = CFRP[3:]
    h = PLYT*len(CROSS_PLY)
    # NOTE simple stiffnesses, constant strain sum_k G_k h_k and constant
    #      stress h^2/sum_k (h_k/G_k): in the yz plane G23 for the 0 plies
    #      and G13 for the 90 plies, the reverse in the xz plane
    Abar = 2*PLYT*G23 + 2*PLYT*G13
    Abarbar = h**2/(2*PLYT/G23 + 2*PLYT/G13)
    for name in ['Abar44', 'Abar55']:
        assert np.isclose(getattr(prop, name), Abar, rtol=1e-14)
        assert_printed(getattr(prop, name), ref['Abar'])
    for name in ['Abarbar44', 'Abarbar55']:
        assert np.isclose(getattr(prop, name), Abarbar, rtol=1e-14)
        assert_printed(getattr(prop, name), ref['Abarbar'])
    # NOTE improved stiffness, Eq. (43), written out in rohwer_stiffness
    Ats, top = rohwer_stiffness(CROSS_PLY, [PLYT]*4, [CFRP]*4)
    assert np.allclose(top, 0., atol=1e-14)
    assert np.allclose(prop.Ats, Ats, rtol=1e-12, atol=1e-14*Ats.max())
    assert_printed(prop.A55, ref['A55'])
    # NOTE A44 = 2.52525, and the independent rohwer_stiffness gives the same,
    #      which Rohwer prints as H22 = 2.521; the other three values round
    #      correctly. Both 2.52525/3.067 and 2.521/3.067 round to the
    #      reduction 0.82 given for the y-direction on p. 18-19
    assert_printed(prop.A44, ref['A44'], rtol=1.7e-3)
    # NOTE "Coupling terms are zero", p. 18
    assert abs(prop.A45) < 1e-14*prop.A44
    assert np.isclose(prop.scf_k13, prop.A55/prop.Abar55, rtol=1e-14)
    assert np.isclose(prop.scf_k23, prop.A44/prop.Abar44, rtol=1e-14)


def test_rohwer_rotated_plies_reference_values():
    # NOTE plies at theta_k - phi, the cases of Table plate-shear-rotation of
    #      the theory, whose values are computed with pyfe3d and therefore
    #      not used here; the reference is Eq. (43) of Rohwer (1988) written
    #      out in rohwer_stiffness, also with an offset and an unsymmetric
    #      laminate, for which A45 does not vanish
    cases = [(CROSS_PLY, 45, 0.), (CROSS_PLY, 30, 0.), (QUASI_ISO, 0, 0.),
             (QUASI_ISO, 60, 0.), (QUASI_ISO, 60, 0.3), (UNSYM, 0, 0.),
             (UNSYM, 20, 0.3)]
    for stack, phi, offset in cases:
        stack = [t - phi for t in stack]
        prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                               offset=offset)
        Ats, top = rohwer_stiffness(stack, [PLYT]*len(stack),
                                    [CFRP]*len(stack), offset=offset)
        assert np.allclose(top, 0., atol=1e-13)
        assert abs(Ats[0, 1]) > 1e-3*Ats[0, 0]
        assert np.allclose(prop.Ats, Ats, rtol=1e-12,
                           atol=1e-12*np.abs(Ats).max())


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


@pytest.mark.parametrize('stack', [[-60, 90], [-30, -60]])
def test_rohwer_independent_of_offset(stack):
    # NOTE with these offsets the elimination of the ABD matrix requires row
    #      pivoting, the transverse shear stiffness and distribution do not
    #      depend on the reference surface
    ref = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP)
    prop = laminated_plate(stack=stack, plyt=PLYT, laminaprop=CFRP,
                           offset=0.5)
    assert np.allclose(prop.Ats, ref.Ats, rtol=1e-10)
    for theta in [0., 30., 75.]:
        assert np.allclose(prop.calc_Ats_element(theta),
                           ref.calc_Ats_element(theta), rtol=1e-10)
    for zbar in np.linspace(-ref.h/2, ref.h/2, 7):
        assert np.allclose(prop.calc_transverse_shear_stress(zbar + 0.5, 1., -2.),
                           ref.calc_transverse_shear_stress(zbar, 1., -2.),
                           rtol=1e-10, atol=1e-12)


def test_errors_no_plies():
    prop = ShellProp()
    prop.calc_transverse_shear_stiffness()
    assert prop.A44 == 0 and prop.A45 == 0 and prop.A55 == 0
    with pytest.raises(ValueError, match='0 plies'):
        prop.calc_transverse_shear_stress(0., 1., 1.)


@pytest.mark.parametrize('thetadeg', [0., 45.])
def test_errors_singular_ABD(thetadeg):
    # NOTE with g12 = 0 the ABD matrix has a zero diagonal term for a 0-degree
    #      ply, and a positive diagonal but it is still singular for 45 degrees
    no_g12 = (138., 9.3, 0.3, 0., 4.6, 2.3)
    with pytest.raises(ValueError, match='ABD matrix'):
        laminated_plate(stack=[thetadeg], plyt=PLYT, laminaprop=no_g12)


@pytest.mark.parametrize('thetadeg', [0., 45.])
def test_errors_nearly_singular_Cs(thetadeg):
    # NOTE the check of C_s does not depend on the ply angle, such that it is
    #      not singular in the laminate frame and singular in an element frame
    near_singular = (138., 9.3, 0.3, 4.6, 2.3e-13, 2.3)
    with pytest.raises(ValueError, match='singular transverse shear'):
        laminated_plate(stack=[thetadeg], plyt=PLYT, laminaprop=near_singular)


def test_laminated_plate_missing_input():
    with pytest.raises(ValueError, match='plyt or plyts'):
        laminated_plate(stack=CROSS_PLY, laminaprop=CFRP)
    with pytest.raises(ValueError, match='laminaprop or laminaprops'):
        laminated_plate(stack=CROSS_PLY, plyt=PLYT)


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
