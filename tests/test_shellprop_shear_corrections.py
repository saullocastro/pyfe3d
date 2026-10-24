r"""
Shear correction factors of Whitney, Chow, Birman and Bert and of the
thickness-shear frequency, a posteriori recovery of the transverse shear
stresses, and evaluation about the mid-surface for large offsets

Vlachoutsis' and Chow's ``K13`` and ``K23``, and Whitney's ``k1^2`` and
``k2^2``, are ``scf_k13`` and ``scf_k23``. Conventions: ``Ats = [[A44, A45],
[A45, A55]]``, 4 <-> yz, 5 <-> xz.
"""
import sys
sys.path.append('..')
from decimal import Decimal

import numpy as np
import pytest

from pyfe3d.shellprop import ShellProp, Lamina, _thickness_shear_frequency
from pyfe3d.shellprop_utils import (laminated_plate, isotropic_plate,
        read_laminaprop)


# Published values
# ----------------
#
# Results computed in the papers are kept as strings, exactly as printed, so
# that each comparison uses half a unit of the last printed digit as tolerance,
# see ``assert_printed``. Input data are kept as floats.

# Rohwer, K. "Improved transverse shear stiffness for layered finite elements",
# DFVLR-FB 88-32, 1988. Table 1, p. 19: "CFRP material constants", kN/mm^2
# E_L, E_T, G_LT, G_TT, nu_LT, as laminaprop = (E11, E22, nu12, G12, G13, G23)
# of a transversely isotropic ply
CFRP = (138.0, 9.3, 0.3, 4.6, 4.6, 2.3)

# Chow, T. S. "On the propagation of flexural waves in an orthotropic laminated
# plate and its response to an impulsive load", Journal of Composite
# Materials, Vol. 5, 306-319, 1971. Pages as printed in the journal.
CHOW_1971 = {
    # Section VII, p. 316: "a 4-layer symmetric cross-ply fiber-reinforced
    # composite in the directions (0, 90, 90, 0), and 0 degree being parallel
    # to x-axis of the body; the thickness of each layer is the same", with
    # Eq. (32): "G12/E22 = 0.6, G23/E22 = 0.5, nu12 = 0.25 together with
    # E11/E22 = 3 for glass-epoxy and E11/E22 = 40 for graphite-epoxy".
    # Table 1, p. 316: "Values of k1 and k2 Calculated from the Equations
    # (12)", printed as k1^2 and k2^2, which are scf_k13 and scf_k23
    'cross_ply': dict(stack=[0, 90, 90, 0], G12_E2=0.6, G23_E2=0.5,
        nu12=0.25, rows=[
            # E11/E22, k1^2, k2^2
            (3., '0.815', '0.807'),
            (40., '0.828', '0.521'),
            ]),
    # Table 1, p. 316, and p. 312: "homogeneous plate isotropic or
    # anisotropic", k1^2 = 5/6
    'homogeneous': dict(K=5/6),
}

# Whitney, J. M. "Shear correction factors for orthotropic laminates under
# static load", Journal of Applied Mechanics, Vol. 40, 302-304, 1973. p. 303:
# "each ply has the following unidirectional properties: EL = 25 x 10^6 psi,
# ET = 10^6 psi, nu_LT = 0.25, G_LT = 0.5 x 10^6 psi, G_TT = 0.2 x 10^6 psi",
# "Both a 4-layer symmetric and a 2-layer unsymmetric square plate are
# considered", [0/90/90/0] and [0/90] in his Fig. 1, "For the two-ply
# laminate k1^2 = k2^2 = 0.8212, and for the four-ply laminate k1^2 = 0.5952
# and k2^2 = 0.7205".
WHITNEY_1973 = {
    'ply': (25., 1., 0.25, 0.5, 0.5, 0.2),
    'laminates': [([0, 90], '0.8212', '0.8212'),
                  ([0, 90, 90, 0], '0.5952', '0.7205')],
}

# Birman, V. and Bert, C. W. "On the choice of shear correction factor in
# sandwich structures", Journal of Sandwich Structures and Materials, Vol. 4,
# 83-95, 2002. Pages as printed in the journal.
BIRMAN_BERT_2002 = {
    # p. 89, below Eq. (13): "It is easy to show that in the case of a single
    # layer, K = 1."
    'homogeneous': dict(K=1.),
    # p. 90: "Type 2 sandwich is a typical shipbuilding configuration with
    # E-glass/vinyl ester facings and a balsa core. The properties of the
    # facings and core for Type 2 are: Ef = 24.4 GPa, Gf = 2.89 GPa and Ec =
    # 2.41 GPa, Gc = 0.103 GPa. Two geometries considered for this beam were
    # hf = 5 mm, hc = 20 mm and hf = hc = 10 mm." Table 1, p. 90, column
    # "Average shear strain", Eq. (14). The beam stiffness Q11 is read as E,
    # i.e. nu = 0.
    'type_2': dict(Ef=24.4, Gf=2.89, Ec=2.41, Gc=0.103, rows=[
        # hf, hc, K
        (5., 20., '0.124'),
        (10., 10., '0.109'),
        ]),
}

# Noor, A. K., "Stability of multilayered composite plates", Fibre Sci.
# Technol., 8, 81-89, 1975, Table 1, p. 83: "composite shear correction
# factors" of cross-ply laminates in cylindrical bending, following Chow and
# Whitney, with EL/ET = 30, GLT/ET = 0.6, GTT/ET = 0.5, nu_LT = 0.25, the total
# thickness of the 0 and 90 deg layers being the same (p. 84)
NOOR_1975 = {
    'ply': (30., 1., 0.25, 0.6, 0.6, 0.5),
    'rows': [
        # NL, kappa(1), kappa(2); even NL skew-symmetric, odd NL symmetric
        (2, '0.6421', '0.6421'), (4, '0.6523', '0.6523'),
        (6, '0.7422', '0.7422'), (10, '0.7947', '0.7947'),
        (3, '0.8274', '0.5412'), (5, '0.8732', '0.5914'),
        (7, '0.8769', '0.6749'), (9, '0.8736', '0.7170'),
    ],
}

SEEDS = list(range(6))


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


def random_laminate(seed, offset=None):
    rng = np.random.default_rng(seed)
    nplies = int(rng.integers(2, 13))
    stack = rng.choice([0., 90., 45., -45., 30., -60., 15.], nplies)
    stack[rng.random(nplies) < 0.3] = rng.uniform(-90, 90)
    plyts = rng.uniform(0.1, 0.4, nplies)
    laminaprops = []
    for i in range(nplies):
        E1 = rng.uniform(40., 180.)
        E2 = rng.uniform(5., 12.)
        G12 = rng.uniform(3., 6.)
        G13 = G12
        G23 = rng.uniform(1.5, 4.)
        laminaprops.append((E1, E2, rng.uniform(0.2, 0.35), G12, G13, G23))
    if offset is None:
        offset = rng.uniform(-1., 1.)
    return laminated_plate(stack, plyts=plyts, laminaprops=laminaprops,
            offset=offset)


def _interfaces(prop):
    return prop.offset - prop.h/2 + np.concatenate(([0.],
        np.cumsum([ply.h for ply in prop.plies])))


def _manual_shellprop(stack, plyt, laminaprops, shear_correction):
    prop = ShellProp()
    prop.shear_correction = shear_correction
    for thetadeg, laminaprop in zip(stack, laminaprops):
        ply = Lamina()
        ply.thetadeg = thetadeg
        ply.h = plyt
        ply.matlamina = read_laminaprop(laminaprop)
        ply.rebuild()
        prop.plies.append(ply)
    prop.calc_constitutive_matrix()
    return prop


# Chow (1971), Birman and Bert (2002) and thickness-shear frequency
# -----------------------------------------------------------------

@pytest.mark.parametrize('E1_E2, k13, k23', CHOW_1971['cross_ply']['rows'])
def test_chow_published_cross_ply(E1_E2, k13, k23):
    ref = CHOW_1971['cross_ply']
    E2 = 1.
    laminaprop = (E1_E2*E2, E2, ref['nu12'], ref['G12_E2']*E2,
                  ref['G12_E2']*E2, ref['G23_E2']*E2)
    prop = laminated_plate(ref['stack'], plyt=0.25, laminaprop=laminaprop,
            shear_correction='chow')
    assert_printed(prop.scf_k13, k13)
    # NOTE k2^2 = 0.52165 for graphite-epoxy, which Chow prints as 0.521,
    #      the other three values round correctly
    assert_printed(prop.scf_k23, k23, rtol=None if E1_E2 == 3. else 1.3e-3)


@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.parametrize('offset', [0., 0.8])
def test_chow_equals_vlachoutsis_symmetric(seed, offset):
    # for a symmetric laminate the neutral surfaces of Vlachoutsis coincide
    # with the mid-surface used by Chow
    rng = np.random.default_rng(2000 + seed)
    half = list(rng.uniform(-90, 90, int(rng.integers(1, 5))))
    plyts = list(rng.uniform(0.1, 0.4, len(half)))
    stack = half + half[::-1]
    plyts = plyts + plyts[::-1]
    props = [laminated_plate(stack, plyts=plyts, laminaprop=CFRP,
                 offset=offset, shear_correction=mode)
             for mode in ['chow', 'vlachoutsis']]
    assert np.allclose(props[0].Ats, props[1].Ats, rtol=1e-12, atol=0)


@pytest.mark.parametrize('stack', [[0, 90], [0, 90, 45], [30, -30]])
def test_chow_unsymmetric(stack):
    with pytest.raises(ValueError, match='symmetric'):
        laminated_plate(stack, plyt=0.25, laminaprop=CFRP,
                shear_correction='chow')


@pytest.mark.parametrize('mode, K', [
    ('chow', CHOW_1971['homogeneous']['K']),
    ('whitney', 5/6),
    ('birman_bert', BIRMAN_BERT_2002['homogeneous']['K']),
    ('thickness_shear', np.pi**2/12)])
@pytest.mark.parametrize('nu', [0.0, 0.3, 0.49])
@pytest.mark.parametrize('offset', [0., +0.37, -1.5])
def test_homogeneous_limits(mode, K, nu, offset):
    # pi^2/12 is the value of Mindlin (1951), independent of nu
    prop = isotropic_plate(thickness=2., E=70e3, nu=nu, rho=2.7e-6,
            offset=offset, shear_correction=mode)
    assert np.isclose(prop.scf_k13, K, rtol=1e-12, atol=0)
    assert np.isclose(prop.scf_k23, K, rtol=1e-12, atol=0)
    assert np.allclose(prop.Ats, K*prop.Abar_ts, rtol=1e-12)


def _birman_bert_eq14(Qf, Gf, Qc, Gc, hf, hc):
    r"""Eq. (14) of Birman and Bert (2002), symmetric sandwich beam"""
    x = hc/(2*hf + hc)
    num = 2/3*(Qf*(1 - x**3) + Qc*x**3)
    den = ((Gf*(1 - x) + Gc*x)*(Qf*(2/3 - x + x**3/3)/Gf
           + Qf*x*(1 - x**2)/Gc + 2/3*Qc*x**3/Gc))
    return num/den


def _beam_sandwich(Ef, Gf, Ec, Gc, hf, hc, **kwargs):
    faces = (Ef, Ef, 0., Gf, Gf, Gf)
    core = (Ec, Ec, 0., Gc, Gc, Gc)
    return laminated_plate([0, 0, 0], plyts=[hf, hc, hf],
            laminaprops=[faces, core, faces], **kwargs)


@pytest.mark.parametrize('hf, hc, K', BIRMAN_BERT_2002['type_2']['rows'])
def test_birman_bert_published_type_2(hf, hc, K):
    ref = BIRMAN_BERT_2002['type_2']
    prop = _beam_sandwich(ref['Ef'], ref['Gf'], ref['Ec'], ref['Gc'], hf, hc,
            shear_correction='birman_bert')
    assert_printed(prop.scf_k13, K)
    assert_printed(prop.scf_k23, K)


@pytest.mark.parametrize('Ef_Ec, Gf_Gc, hf_hc', [(1e1, 1e1, 0.25),
    (1e3, 30., 0.05), (5e2, 1e3, 1.), (1., 1., 0.5)])
def test_birman_bert_closed_form(Ef_Ec, Gf_Gc, hf_hc):
    Ef, Gf, hc = 70., 26., 1.
    prop = _beam_sandwich(Ef, Gf, Ef/Ef_Ec, Gf/Gf_Gc, hf_hc*hc, hc,
            shear_correction='birman_bert')
    K = _birman_bert_eq14(Ef, Gf, Ef/Ef_Ec, Gf/Gf_Gc, hf_hc*hc, hc)
    assert np.isclose(prop.scf_k13, K, rtol=1e-12)


@pytest.mark.parametrize('mode', ['birman_bert', 'thickness_shear'])
@pytest.mark.parametrize('seed', SEEDS[:4])
def test_new_modes_offset_invariance(mode, seed):
    rng = np.random.default_rng(3000 + seed)
    nplies = int(rng.integers(2, 9))
    stack = rng.uniform(-90, 90, nplies)
    rhos = rng.uniform(1e-6, 3e-6, nplies)
    Ats = [laminated_plate(stack, plyt=0.25, laminaprop=CFRP, rhos=rhos,
               offset=offset, shear_correction=mode).Ats
           for offset in [0., +0.4, -2.0]]
    for other in Ats[1:]:
        assert np.allclose(other, Ats[0], rtol=1e-9,
                atol=1e-9*np.abs(Ats[0]).max())


def test_thickness_shear_density():
    with pytest.raises(ValueError, match='densit'):
        laminated_plate([0, 90], plyt=0.25, laminaprop=CFRP,
                shear_correction='thickness_shear')
    with pytest.raises(ValueError, match='densit'):
        laminated_plate([0, 90], plyt=0.25, laminaprop=CFRP,
                rhos=[1.6e-6, 0.], shear_correction='thickness_shear')


def test_thickness_shear_frequency():
    # homogeneous layer with traction-free faces: omega = pi/h sqrt(G/rho)
    omega = _thickness_shear_frequency([4.], [1.], [2.], 1.)
    assert np.isclose(omega, np.pi/2.*np.sqrt(4./1.), rtol=1e-14)
    # the search stops at 100 times the reference frequency
    with pytest.raises(RuntimeError, match='No thickness-shear frequency'):
        _thickness_shear_frequency([1.], [1.], [1.], 1e-3)


# Whitney (1973) and Noor (1975)
# ------------------------------

@pytest.mark.parametrize('stack, k13, k23', WHITNEY_1973['laminates'])
@pytest.mark.parametrize('mode', ['whitney', 'vlachoutsis'])
def test_whitney_published_laminates(stack, k13, k23, mode):
    prop = laminated_plate(stack, plyt=0.25, laminaprop=WHITNEY_1973['ply'],
            shear_correction=mode)
    assert_printed(prop.scf_k13, k13)
    assert_printed(prop.scf_k23, k23)


@pytest.mark.parametrize('seed', SEEDS)
def test_whitney_equals_vlachoutsis(seed):
    props = [random_laminate(seed) for _ in range(2)]
    for prop, mode in zip(props, ['whitney', 'vlachoutsis']):
        prop.shear_correction = mode
        prop.calc_transverse_shear_stiffness()
    assert np.allclose(props[0].Ats, props[1].Ats, rtol=1e-14, atol=0)


@pytest.mark.parametrize('NL, k13, k23', NOOR_1975['rows'])
def test_noor_1975_published_factors(NL, k13, k23):
    stack = [0 if i % 2 == 0 else 90 for i in range(NL)]
    n0 = sum(1 for t in stack if t == 0)
    n90 = NL - n0
    plyts = [0.5/n0 if t == 0 else 0.5/n90 for t in stack]
    prop = laminated_plate(stack, plyts=plyts, laminaprop=NOOR_1975['ply'],
            shear_correction='whitney')
    assert_printed(prop.scf_k13, k13)
    assert_printed(prop.scf_k23, k23)


# A posteriori recovery (Noor and Peters, 1989)
# ---------------------------------------------

@pytest.mark.parametrize('seed', SEEDS)
def test_equilibrium_recovery_cylindrical_bending(seed):
    r"""With the gradients of the two cylindrical bending states of Rohwer
    (1988), the recovered stresses are the distribution f(z) Q of
    calc_transverse_shear_stress, and the top face is traction free"""
    prop = random_laminate(seed)
    Q = np.array([0.7, -1.3]) # Qy, Qx
    Hs = np.linalg.inv(prop.ABD)
    Lx = np.zeros((6, 2)); Lx[3, 1] = 1.; Lx[5, 0] = 1.
    Ly = np.zeros((6, 2)); Ly[4, 0] = 1.; Ly[5, 1] = 1.
    # tau_xz from the x state only, tau_yz from the y state only
    zi = _interfaces(prop)
    zs = np.linspace(zi[0], zi[-1], 17)
    _, _, txz = prop.calc_equilibrium_transverse_shear(Hs @ Lx @ Q,
            np.zeros(6), z=zs)
    _, tyz, _ = prop.calc_equilibrium_transverse_shear(np.zeros(6),
            Hs @ Ly @ Q, z=zs)
    for j, z in enumerate(zs):
        ref = prop.calc_transverse_shear_stress(z, Q[0], Q[1])
        assert np.isclose(tyz[0, j], ref[0], atol=1e-9*np.abs(Q).max()/prop.h)
        assert np.isclose(txz[0, j], ref[1], atol=1e-9*np.abs(Q).max()/prop.h)
    assert abs(txz[0, -1]) < 1e-9*np.abs(Q).max()/prop.h


def test_aposteriori_energy_homogeneous():
    r"""Homogeneous plate in cylindrical bending: parabola, energy
    3/5 Qx^2/(G h), i.e. k = 5/6 with Eq. (8) of Noor and Peters (1989)"""
    h, E, nu = 2., 70e3, 0.3
    G = E/(2*(1 + nu))
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    Qx = 1.7
    D = E*h**3/(12*(1 - nu**2))
    grad_x = np.zeros(6)
    # M_xx,x = Qx with the plane strain curvature of cylindrical bending
    grad_x[3] = Qx/D
    U13, U23, Qy_r, Qx_r = prop.calc_aposteriori_energy(grad_x, np.zeros(6))
    assert np.isclose(Qx_r[0], Qx, rtol=1e-12)
    assert np.isclose(U13[0], 0.6*Qx**2/(G*h), rtol=1e-12)
    assert np.isclose(0.5*Qx**2/(G*h)/U13[0], 5/6, rtol=1e-12)
    assert abs(U23[0]) < 1e-20


def test_equilibrium_z_outside():
    prop = laminated_plate([0, 90], plyt=0.25, laminaprop=CFRP, offset=0.1)
    zero = np.zeros(6)
    # the faces themselves are accepted
    z, _, _ = prop.calc_equilibrium_transverse_shear(zero, zero,
                                                     z=[-0.15, 0.35])
    assert np.allclose(z, [-0.15, 0.35])
    for z in ([0.36], [-0.16]):
        with pytest.raises(ValueError, match='outside'):
            prop.calc_equilibrium_transverse_shear(zero, zero, z=z)


def test_aposteriori_energy_singular_Cs():
    laminaprops = [CFRP, (138., 9.3, 0.3, 4.6, 0., 2.3), CFRP]
    prop = _manual_shellprop([0, 0, 0], 0.25, laminaprops, None)
    grad_x = np.zeros(6)
    grad_x[3] = 1.
    with pytest.raises(ValueError, match='Ply 1'):
        prop.calc_aposteriori_energy(grad_x, np.zeros(6))


# Error branches
# --------------

def test_unknown_shear_correction():
    with pytest.raises(ValueError, match='shear_correction must be one of'):
        laminated_plate([0, 90], plyt=0.25, laminaprop=CFRP,
                        shear_correction='unknown')


@pytest.mark.parametrize('mode', ['vlachoutsis', 'whitney', 'chow',
                                  'birman_bert'])
def test_non_positive_in_plane_stiffness(mode):
    prop = laminated_plate([0, 90, 0], plyt=0.25, laminaprop=CFRP,
                           shear_correction=None)
    for ply in prop.plies:
        ply.q11L = 0.
        ply.q22L = 0.
    prop.shear_correction = mode
    with pytest.raises(ValueError, match='positive in-plane stiffnesses'):
        prop.calc_transverse_shear_stiffness()


@pytest.mark.parametrize('mode', ['whitney', 'chow', 'birman_bert',
                                  'thickness_shear'])
def test_singular_Cs_scalar_modes(mode):
    laminaprops = [CFRP, (138., 9.3, 0.3, 4.6, 0., 2.3), CFRP]
    # positive densities, such that 'thickness_shear' reaches the Cs check
    prop = laminated_plate([0, 0, 0], plyt=0.25, laminaprops=laminaprops,
                           rho=1.6e-6, shear_correction=None)
    prop.shear_correction = mode
    with pytest.raises(ValueError, match='Ply 1'):
        prop.calc_transverse_shear_stiffness()


# Large offsets
# -------------

@pytest.mark.parametrize('mode', ['rohwer', 'vlachoutsis', 'whitney', 'chow',
                                  'birman_bert', 'thickness_shear',
                                  'constant', None])
def test_large_offset(mode):
    # computed about the mid-surface, hence identical for any offset, while
    # the round-off of the heights from the reference surface grows with
    # (offset/h)^2
    stack = [0, 90, 45, 90, 0] if mode == 'chow' else [0, 90, 45, -45, 30]
    props = [laminated_plate(stack, plyt=0.2, laminaprop=CFRP, rho=1.6e-6,
                             offset=offset, shear_correction=mode)
             for offset in [0., 1e2, -1e4, 1e6]]
    for other in props[1:]:
        assert np.array_equal(other.Ats, props[0].Ats)
        # including the stiffness in an element coordinate system
        assert np.array_equal(other.calc_Ats_element(30.),
                              props[0].calc_Ats_element(30.))


def test_large_offset_stresses():
    stack = [0, 90, 45, -45, 30]
    prop0 = laminated_plate(stack, plyt=0.2, laminaprop=CFRP)
    zs = np.linspace(-0.5, 0.5, 11)
    tau0 = np.array([prop0.calc_transverse_shear_stress(z, 0.7, -1.3)
                     for z in zs])
    rng = np.random.default_rng(0)
    grad_x = rng.normal(size=6)
    grad_y = rng.normal(size=6)
    _, tyz0, txz0 = prop0.calc_equilibrium_transverse_shear(grad_x, grad_y,
                                                            z=zs)
    U0 = np.array(prop0.calc_aposteriori_energy(grad_x, grad_y))
    for offset in [1e2, -1e4, 1e6]:
        prop = laminated_plate(stack, plyt=0.2, laminaprop=CFRP,
                               offset=offset)
        # only the representation of z + offset is affected by round-off
        tol = 1e-14*abs(offset)/prop.h
        tau = np.array([prop.calc_transverse_shear_stress(z + offset, 0.7,
                                                          -1.3)
                        for z in zs])
        assert np.allclose(tau, tau0, rtol=0, atol=tol*np.abs(tau0).max())
        # same strain field, eps0 + z eps1, about the new reference surface
        gx = grad_x.copy()
        gy = grad_y.copy()
        gx[:3] -= offset*gx[3:]
        gy[:3] -= offset*gy[3:]
        z, tyz, txz = prop.calc_equilibrium_transverse_shear(gx, gy,
                                                             z=zs + offset)
        assert np.array_equal(z, zs + offset)
        assert np.allclose(txz, txz0, rtol=0, atol=tol*np.abs(txz0).max())
        assert np.allclose(tyz, tyz0, rtol=0, atol=tol*np.abs(tyz0).max())
        U = np.array(prop.calc_aposteriori_energy(gx, gy))
        assert np.allclose(U, U0, rtol=0, atol=tol*np.abs(U0).max())
