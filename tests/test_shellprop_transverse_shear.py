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


# NOTE Rohwer (1988), Sec. 4.3, pp. 22-23, gives a second example, the
#      sandwich solar panel of Fig. 9, p. 22: faces of four CFRP layers of
#      0.125 mm, [0,90,+45,-45] on top and [-45,+45,90,0] at the bottom, the
#      -45 layers adhered to the core, a foam core of 9 mm with E = 0.1
#      kN/mm^2 and nu = 0.3 (Table 3, p. 22), and the CFRP of Table 1. Its
#      stiffnesses, Table 4, p. 23, are not used as a test, because they
#      cannot be reproduced from these data and are not consistent with the
#      stated lay-up:
#      - the transverse shear stiffnesses, H11 = 0.416, H12 = 0.008 and
#        H22 = 0.403 kN/mm, depend on the foam only through its shear
#        modulus; with nu = 0.3, G = 0.0385 kN/mm^2, Eq. (43) gives 0.400,
#        0.008 and 0.387; all three round to the printed values for G
#        between 0.03999 and 0.04005 kN/mm^2, i.e. nu = 0.25 with E = 0.1,
#        which is the suspected value of the computation;
#      - D13 = D23 = 9.36 kNmm, the bending-twisting couplings, written D16
#        and D26 in pyfe3d, are reproduced, 9.3573, but D11 = 1366.08, D12
#        = 395.09 and D22 = 1321.97 kNmm are not, +0.12, +0.62 and +0.47 per
#        cent with nu = 0.3 and nearly the same with nu = 0.25;
#      - the printed bending entries contradict each other for this lay-up:
#        for any ply material, the +-45 layers and the isotropic core add the
#        same amount to D11 and D22 and Qbar16 = +-(Q11 - Q22)/4 at +-45
#        degrees, such that (D11 - D22)/D13 = 4 (I0 - I90)/(I+45 - I-45),
#        with I the integral of z^2 over the layers of each angle, a number
#        given by the layer positions alone: 4.2162 for Fig. 9, against
#        4.709 to 4.716 from the printed values within half a unit of their
#        last digits. No change of the material constants, of the foam or of
#        the CFRP, can reproduce Table 4; its bending entries were computed
#        for layer positions, or a lay-up, other than those of Fig. 9, or
#        are misprinted. A search over all orderings of the face layers and
#        over the material constants and thicknesses found no single
#        correction of the data that reproduces the whole table.
#      The only other publication of this panel found is by the same author,
#      Rohwer, K., "Improved transverse shear stiffnesses for anisotropic
#      sandwich plates and shells", Sandwich Constructions 2, Vol. 1, EMAS,
#      1992, pp. 465-484, not available here. The sandwich case is tested
#      instead against Vlachoutsis (1992) and Allen (1969), see
#      test_sandwich_vlachoutsis_1992_reference_values and
#      test_sandwich_antiplane_core_converges_to_allen_1969.


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


# Vlachoutsis, S. "Shear correction factors for plates and shells", Int. J.
# Numer. Meth. Engng 33(7), 1537-1552, 1992, https://doi.org/10.1002/nme.1620330712.
# Symmetrical sandwich of Fig. 7, p. 1544, faces of thickness t and core of
# thickness c, h = c + 2t, p = c/h, Eq. (31), "homogeneous and isotropic
# materials", p. 1547, with nu1 = nu2 = 0.3, Fig. 11 caption, p. 1548; index 1
# the faces and 2 the core. K is the single factor of the section, A = K
# sum_i G_i h_i, Eqs. (24)-(25), p. 1541, i.e. scf_k13 = A55/Abar55, and r_f =
# U_f/(U_f + U_c) the share of the faces in the transverse shear energy, Eq.
# (48), p. 1545. Rows (E1/E2, c/h, K, r_f): Table I (c/h = 0.8), p. 1548;
# Table II (E1/E2 = 15), p. 1548, without its row c/h = 0.8, which repeats
# Table I; and the example of p. 1546, Eqs. (49)-(50), "Dc/Df = Gc/Gf =
# 10^-3 and p = c/h = 0.95", i.e. E1/E2 = 1000 with equal Poisson ratios
VLACHOUTSIS_1992_SANDWICH = [
    (1, 0.8, '0.8333', '0.01712'),
    (5, 0.8, '0.5324', '0.01130'),
    (10, 0.8, '0.3525', '0.006974'),
    (15, 0.8, '0.2625', '0.005012'),
    (50, 0.8, '0.09385', '0.001678'),
    (15, 0.9, '0.4087', '0.001779'),
    (15, 0.6, '0.1636', '0.01570'),
    (15, 0.4, '0.1400', '0.03883'),
    (1000, 0.95, '0.01964', '1.755e-5'),
    ]


def vlachoutsis_sandwich_K(p, Dc_Df, Gc_Gf):
    r"""Shear correction factors of a symmetrical sandwich, Vlachoutsis (1992)

    Closed forms of pp. 1543-1545: ``A(p)``, Eq. (32); ``T1``-``T5``, Eqs.
    (37)-(41); the single factor ``K``, Eq. (44); the factor ``K_ant`` of the
    antiplane core, ``Dc/Df = 0``, Eq. (45); and the energy ratio of the
    faces ``r_f``, Eq. (48). ``D = E/(1 - nu12 nu21)`` of each material, Eq.
    (33), and ``G`` its transverse shear modulus, Eq. (34).
    """
    A = (1 - p)**3*(3*p**2 + 9*p + 8)/15
    T1 = (1 - p**3) + p**3*Dc_Df
    T2 = (1 - p)/Gc_Gf + p
    T3 = ((1 - p**2)**2 + 8/15*Dc_Df**2*p**4
          + 4/3*Dc_Df*p**2*(1 - p**2))
    T4 = A*Gc_Gf + p*T3
    T5 = A*Gc_Gf + p*(1 - p**2)**2
    K = 4/9*T1**2/(T2*T4)
    K_ant = 4/9*(1 - p**3)**2/(T2*T5)
    r_f = A*Gc_Gf/T4
    return K, K_ant, r_f


def _shear_energies(prop, plyts, Gs):
    r"""Complementary shear energy ``2U`` of each ply for ``Qx = 1``, ``Qy = 0``

    Uses the equilibrium distribution of :meth:`.calc_transverse_shear_stress`
    and the 3-point Gauss-Legendre rule, exact for ``tau_xz**2``, of degree 4
    within each ply.
    """
    xg, wg = np.polynomial.legendre.leggauss(3)
    z = -sum(plyts)/2 + np.concatenate(([0.], np.cumsum(plyts)))
    U = []
    for k, G in enumerate(Gs):
        zg = (z[k] + z[k+1])/2 + xg*plyts[k]/2
        tau = np.array([prop.calc_transverse_shear_stress(zi, 0., 1.)[1]
                        for zi in zg])
        U.append(np.sum(wg*tau**2/G)*plyts[k]/2)
    return np.array(U)


@pytest.mark.parametrize('E1_E2, c_h, K_printed, rf_printed',
                         VLACHOUTSIS_1992_SANDWICH)
def test_sandwich_vlachoutsis_1992_reference_values(E1_E2, c_h, K_printed,
                                                    rf_printed):
    r"""Isotropic sandwich of Vlachoutsis (1992), Tables I and II, p. 1548

    A strongly heterogeneous section, E1/E2 up to 1000, for which the factor
    K = A55/Abar55 falls far below 5/6, down to 0.01964. For isotropic layers
    with one Poisson ratio, the cylindrical bending of Rohwer (1988), Eq.
    (34), with M_y = 0, gives the bending stress E_k z M_x/D of a beam, as
    assumed by Vlachoutsis, Sec. 2, such that ``'rohwer'`` and
    ``'vlachoutsis'`` both reproduce K of Eq. (44) and r_f of Eq. (48) to
    round-off (rtol 1e-12, 1e-10 for r_f, a difference of energies), and K
    and r_f must round to the printed digits (half a unit of the last printed
    digit). Measured: all 9 rows agree with Eq. (44) within 2e-15, r_f
    within 5e-15, and round to the printed K and r_f; e.g. E1/E2 = 50, K =
    0.093854 (printed 0.09385),
    and p = 0.95, K = 0.0196419 (printed 0.01964) and r_f = 1.7552e-5
    (printed 1.755e-5), where the constant 5/6 overestimates the shear
    stiffness 42 times.
    """
    nu = 0.3
    h = 0.04  # Fig. 11 caption, p. 1548; K does not depend on h or on E2
    E2 = 1e9
    E1 = E1_E2*E2
    G1 = E1/(2*(1 + nu))
    G2 = E2/(2*(1 + nu))
    t = (1 - c_h)*h/2
    c = c_h*h
    plyts = [t, c, t]
    laminaprops = [(E1, E1, nu, G1, G1, G1), (E2, E2, nu, G2, G2, G2),
                   (E1, E1, nu, G1, G1, G1)]
    K, K_ant, r_f = vlachoutsis_sandwich_K(c_h, 1/E1_E2, 1/E1_E2)
    Abar = 2*t*G1 + c*G2
    for mode in ['rohwer', 'vlachoutsis']:
        prop = laminated_plate(stack=[0, 0, 0], plyts=plyts,
                               laminaprops=laminaprops, shear_correction=mode)
        assert np.isclose(prop.Abar55, Abar, rtol=1e-14)
        for name in ['scf_k13', 'scf_k23']:
            assert np.isclose(getattr(prop, name), K, rtol=1e-12)
            assert_printed(getattr(prop, name), K_printed)
        assert np.isclose(prop.A44, prop.A55, rtol=1e-12)
        assert abs(prop.A45) <= 1e-14*prop.A55
    # NOTE independent implementation of Eq. (43) of Rohwer (1988)
    Ats, top = rohwer_stiffness([0, 0, 0], plyts, laminaprops)
    assert np.allclose(top, 0., atol=1e-12*np.abs(Ats).max())
    assert np.allclose(Ats, K*Abar*np.eye(2), rtol=1e-12,
                       atol=1e-14*K*Abar)
    # NOTE the equilibrium stress distribution: energy share of the faces,
    #      Eq. (48), and the complementary energy 1/A55 for Qx = 1
    prop = laminated_plate(stack=[0, 0, 0], plyts=plyts,
                           laminaprops=laminaprops)
    U = _shear_energies(prop, plyts, [G1, G2, G1])
    assert np.isclose(U.sum(), 1/prop.A55, rtol=1e-12)
    rf = (U[0] + U[2])/U.sum()
    assert np.isclose(rf, r_f, rtol=1e-10)
    assert_printed(rf, rf_printed)
    # NOTE 'constant' is wrong by the factor (5/6)/K, from 1 to 42
    prop = laminated_plate(stack=[0, 0, 0], plyts=plyts,
                           laminaprops=laminaprops, shear_correction='constant')
    assert np.isclose(prop.A55, 5/6*Abar, rtol=1e-14)


def test_sandwich_vlachoutsis_1992_example_close_to_allen_1969():
    r"""Example of Vlachoutsis (1992), p. 1546, and Allen (1969), Eq. (2.15b)

    For ``Dc/Df = Gc/Gf = 1e-3`` and ``p = c/h = 0.95`` Vlachoutsis prints K
    = 0.01964, Eq. (50), and K = 0.01865, Eqs. (52)-(53), for a constant core
    shear stress Q/c ignoring the faces, "5 per cent below". The shear
    stiffness of Allen (1969), Eq. (2.15b), p. 17, ``A G = G_c d^2/c`` per
    unit width, with ``d = (h + c)/2``, Eq. (2.3), p. 9, the distance between
    the centroids of the faces, includes the lever arm d of the faces and
    rounds to the same printed K = 0.01964; pyfe3d gives A55/(G_c d^2/c) - 1
    = 9.6e-5. The deviation from Allen is bounded with the terms that Allen
    neglects in :func:`test_sandwich_antiplane_core_converges_to_allen_1969`;
    here the core is not antiplane, so only the printed digits are compared.
    """
    nu = 0.3
    h = 1.
    p = 0.95
    t = (1 - p)*h/2
    c = p*h
    d = (h + c)/2
    Ec, Ef = 1., 1000.
    Gc, Gf = Ec/(2*(1 + nu)), Ef/(2*(1 + nu))
    prop = laminated_plate(stack=[0, 0, 0], plyts=[t, c, t],
                           laminaprops=[(Ef, nu), (Ec, nu), (Ef, nu)])
    Abar = 2*t*Gf + c*Gc
    assert np.isclose(prop.Abar55, Abar, rtol=1e-14)
    assert_printed(prop.scf_k13, '0.01964')
    assert_printed(Gc*d**2/c/Abar, '0.01964')
    assert_printed(Gc*c/Abar, '0.01865')
    assert 9e-5 < prop.A55/(Gc*d**2/c) - 1 < 1e-4


def test_sandwich_antiplane_core_converges_to_allen_1969():
    r"""Antiplane core: closed form and convergence to Allen (1969)

    Allen, H. G., "Analysis and Design of Structural Sandwich Panels",
    Pergamon Press, 1969, Chap. 2, pp. 8-47,
    https://doi.org/10.1016/B978-0-08-012870-2.50006-7. An antiplane core is
    "an idealized core in which the modulus of elasticity in planes parallel
    with the faces is zero but the shear modulus in planes perpendicular to
    the faces is finite", ``E_c = 0``, p. 14, given here as ``laminaprop =
    (0, 0, nu, 0, G_c, G_c)``, see :func:`.read_laminaprop`. Then
    the shear stress is constant in the core, Eq. (2.12), p. 13, ``tau_c = Q
    E_f t d/(2 D)``, with ``D = E_f (t^3/6 + t d^2/2)``, Eq. (2.2), p. 9,
    i.e. ``tau_c = Q/(d phi)``, ``phi = 1 + t^2/(3 d^2)``, and parabolic in
    the faces, Eq. (2.9), p. 11, ``tau_f = Q E_f (h^2/4 - z^2)/(2 D)``. The
    complementary energy, Rohwer (1988), Eq. (43), gives::

        A = (G_c d^2/c) phi^2/(1 + eps),
        eps = (G_c/G_f) 2 t (h^2/3 - h t/2 + t^2/5)/(c d^2)

    with eps the ratio of the face to the core shear energy. This is Eq. (45)
    of Vlachoutsis (1992), p. 1545, ``A = K_ant sum_i G_i h_i``, verified
    here to round-off. Allen's shear stiffness ``G_c d^2/c``, Eq. (2.15b),
    p. 17, neglects the local bending stiffness of the faces, the first term
    of Eq. (2.2), "less than 1%" of the second when ``3 (d/t)^2 > 100``, Eq.
    (2.4), p. 10, i.e. when ``phi - 1 < 0.01``, and the shear strains of the
    faces, footnote of p. 17, i.e. eps. Hence the derived bounds, which tend
    to zero as ``d/t`` and ``G_f/G_c`` grow::

        -eps <= A/(G_c d^2/c) - 1 <= phi^2 - 1

    The family below has ``t/c = 10^-k/2`` and ``G_c/G_f = 10^-k/10``, i.e. a
    constant share of the core in ``Abar = sum_i G_i h_i``. Measured, with
    ``E_f = 70 GPa``, ``nu = 0.3``, ``c = 10 mm``:

    ===  =======  =========  =============  ===========  ======
     k    eps      phi^2-1    A/A_Allen - 1  pyfe3d vs    K
                                             Eq. (45)
    ===  =======  =========  =============  ===========  ======
     1    3.4e-4   1.51e-3    1.17e-3        7e-16        0.1003
     2    3.3e-6   1.65e-5    1.32e-5        1e-14        0.0918
     3    3.3e-8   1.67e-7    1.33e-7        7e-14        0.0910
    ===  =======  =========  =============  ===========  ======

    The deviation from Allen decreases by a factor 100 per step, as both
    bounds, while K tends to 1/11, such that the constant 5/6 overestimates
    the shear stiffness 9.2 times. The column "pyfe3d vs Eq. (45)" is the
    largest relative difference of ``A44`` and ``A55`` of ``'rohwer'`` and
    ``'vlachoutsis'`` and of the independent ``rohwer_stiffness`` from
    ``K_ant sum_i G_i h_i``. With ``E_c = 0``, Eq. (44) of Vlachoutsis
    reduces to Eq. (45), ``K = K_ant``, checked exactly, and the only
    allowance added to the derived bounds is 1e-12 for round-off, also the
    tolerance against Eq. (45).
    """
    Ef = 70e3
    nu = 0.3
    Gf = Ef/(2*(1 + nu))
    c = 10.
    lf = (Ef, Ef, nu, Gf, Gf, Gf)
    devs = []
    for k in [1, 2, 3]:
        t = c*10.**(-k)/2
        Gc = Gf*10.**(-k)/10
        # NOTE antiplane core, E_c = 0 in the plane of the faces
        lc = (0., 0., nu, 0., Gc, Gc)
        plyts = [t, c, t]
        h = c + 2*t
        p = c/h
        d = (h + c)/2
        phi = 1 + t**2/(3*d**2)
        eps = Gc/Gf*2*t*(h**2/3 - h*t/2 + t**2/5)/(c*d**2)
        A_allen = Gc*d**2/c
        Abar = 2*t*Gf + c*Gc
        K, K_ant, _ = vlachoutsis_sandwich_K(p, 0., Gc/Gf)
        # NOTE the closed form above is Eq. (45) of Vlachoutsis (1992), and
        #      Eq. (44) reduces to it for Dc/Df = 0
        assert np.isclose(A_allen*phi**2/(1 + eps), K_ant*Abar, rtol=1e-13)
        assert K == K_ant
        slack = 1e-12
        for mode in ['rohwer', 'vlachoutsis']:
            prop = laminated_plate(stack=[0, 0, 0], plyts=plyts,
                                   laminaprops=[lf, lc, lf],
                                   shear_correction=mode)
            assert np.isclose(prop.A55, K*Abar, rtol=1e-12)
            assert np.isclose(prop.A44, K*Abar, rtol=1e-12)
        Ats, top = rohwer_stiffness([0, 0, 0], plyts, [lf, lc, lf])
        assert np.allclose(Ats, K*Abar*np.eye(2), rtol=1e-12,
                           atol=1e-12*K*Abar)
        dev = prop.A55/A_allen - 1
        assert -eps - slack <= dev <= phi**2 - 1 + slack
        devs.append(dev)
        # NOTE Allen's condition (2.4) holds
        assert 3*(d/t)**2 > 100
        assert (5/6)/K > 8
    # NOTE convergence: the deviation from Allen decreases with k, as the
    #      bounds, by a factor 100 per step
    assert devs[0] > 50*devs[1] > 0 and devs[1] > 50*devs[2] > 0
    assert devs[-1] < 2e-7


if __name__ == '__main__':
    test_isotropic_five_sixths()
    test_rohwer_cross_ply_reference_values()
    test_rohwer_rotated_plies_reference_values()
