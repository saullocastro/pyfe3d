"""The tangent stiffness matrix must be the derivative of the internal forces

KT = KC0 + KCNL(u) + KG(u) is the Jacobian of the geometrically nonlinear
internal force vector given by ``update_fint(..., nonlinear=1)``. If the two
drift apart, a Newton-Raphson iteration built on them still converges, but
linearly instead of quadratically.

The main check is a directional Taylor test,

    |fint(u + h d) - fint(u) - h KT d| / |h KT d|

which falls proportionally to h for a consistent tangent and plateaus for an
inconsistent one. Every element is distorted and arbitrarily oriented in
space, the shells use an unsymmetric laminate with a material direction
different from the element direction, and the beams use an offset reference
axis, so that all the coupling terms carry weight.
"""
import sys
sys.path.append('..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pyfe3d.beamprop import BeamProp
from pyfe3d.shellprop_utils import isotropic_plate, laminated_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe,
                    Tria3DSG, Tria3DSGData, Tria3DSGProbe, BeamC,
                    BeamCData, BeamCProbe, BeamLR, BeamLRData, BeamLRProbe,
                    DOF, INT, DOUBLE)

SHELLS = {
    'Quad4': (Quad4, Quad4Probe, Quad4Data, 4),
    'Quad4R': (Quad4R, Quad4RProbe, Quad4RData, 4),
    'Tria3R': (Tria3R, Tria3RProbe, Tria3RData, 3),
    'Tria3DSG': (Tria3DSG, Tria3DSGProbe, Tria3DSGData, 3),
}
BEAMS = {
    'BeamC': (BeamC, BeamCProbe, BeamCData),
    'BeamLR': (BeamLR, BeamLRProbe, BeamLRData),
}
ELEMENTS = sorted(SHELLS) + sorted(BEAMS)
# NOTE the drilling stiffness is computed in update_KC0 and again in
#      update_probe_finte, the pair that drifted apart in Tria3R, so the
#      shells are also checked with the two non-default settings: an explicit
#      gamma_rz under the physics-based model and the K6ROT penalty
DRILLING = ['default', 'gamma_rz', 'K6ROT']
CASES = ([(name, drilling) for name in sorted(SHELLS) for drilling in DRILLING]
         + [(name, 'default') for name in sorted(BEAMS)])


def rotation_matrix(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def assemble(size, n, update):
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    update(r, c, v)
    return coo_matrix((v, (r, c)), shape=(n, n)).toarray()


def set_drilling(elem, prop, drilling):
    """Select one of the drilling settings of ``DRILLING``"""
    if drilling == 'gamma_rz':
        elem.drilling_model = 0
        elem.gamma_rz = 0.37*prop.A66
    elif drilling == 'K6ROT':
        elem.drilling_model = 1
        elem.K6ROT = 1.e4
    else:
        assert drilling == 'default'


def make_element(name, drilling='default'):
    """One distorted element, arbitrarily oriented in space"""
    Q = rotation_matrix(3)
    origin = np.array([0.3, -0.2, 0.5])
    if name in SHELLS:
        cls, probecls, datacls, num_nodes = SHELLS[name]
        # unsymmetric laminate, with B != 0
        prop = laminated_plate(stack=[0, 30, -60, 90], plyt=0.001,
                               laminaprop=(127.6e9, 11.3e9, 0.3, 6e9, 6e9,
                                           6e9))
        if num_nodes == 4:
            xy = [[0., 0.], [0.1, 0.01], [0.11, 0.09], [-0.01, 0.1]]
        else:
            xy = [[0., 0.], [0.1, 0.015], [0.03, 0.09]]
        X = np.column_stack((xy, np.zeros(num_nodes))) @ Q.T + origin
        x = X.flatten()
        elem = cls(probecls())
        for i in range(num_nodes):
            setattr(elem, 'n%d' % (i + 1), i + 1)
            setattr(elem, 'c%d' % (i + 1), DOF*i)
        # material direction different from the element direction
        xmat = Q @ np.array([1., 0.6, 0.])
        elem.update_rotation_matrix(x, xmat[0], xmat[1], xmat[2])
        set_drilling(elem, prop, drilling)
    else:
        assert drilling == 'default'
        cls, probecls, datacls = BEAMS[name]
        num_nodes = 2
        b, h = 0.05, 0.03
        prop = BeamProp()
        prop.A = b*h
        prop.E = 70e9
        prop.G = 5/6*70e9/2.6
        prop.Izz = b*h**3/12
        prop.Iyy = b**3*h/12
        prop.J = prop.Izz + prop.Iyy
        # offset reference axis, coupling the axial force and the curvatures
        prop.Ay = 0.004*prop.A
        prop.Az = -0.003*prop.A
        prop.Iyz = 0.1*prop.Izz
        X = np.array([origin, origin + 0.4*Q[:, 0]])
        x = X.flatten()
        elem = cls(probecls())
        elem.n1, elem.n2 = 1, 2
        elem.c1, elem.c2 = 0, DOF
        vxy = Q[:, 1] + 0.3*Q[:, 0]
        elem.update_rotation_matrix(vxy[0], vxy[1], vxy[2], x)
    elem.init_k_KC0 = 0
    elem.init_k_KCNL = 0
    elem.init_k_KG = 0
    elem.update_probe_xe(x)
    return elem, datacls(), prop, x, DOF*num_nodes


def make_callables(name, drilling='default'):
    elem, data, prop, x, n = make_element(name, drilling)

    def fint(u, nonlinear=1):
        f = np.zeros(n, dtype=DOUBLE)
        elem.update_probe_xe(x)
        elem.update_probe_ue(u)
        elem.update_fint(f, prop, nonlinear=nonlinear)
        return f

    KC0 = assemble(data.KC0_SPARSE_SIZE, n,
                   lambda r, c, v: elem.update_KC0(r, c, v, prop))

    def KCNL(u):
        elem.update_probe_xe(x)
        elem.update_probe_ue(u)
        return assemble(data.KCNL_SPARSE_SIZE, n,
                        lambda r, c, v: elem.update_KCNL(r, c, v, prop))

    def KG(u):
        elem.update_probe_xe(x)
        elem.update_probe_ue(u)
        return assemble(data.KG_SPARSE_SIZE, n,
                        lambda r, c, v: elem.update_KG(r, c, v, prop))

    return fint, KC0, KCNL, KG, n


# displacement scale, of about 10% of the element size, so that the
# nonlinear terms are not negligible in front of the linear ones
SCALE = 0.01


def taylor_errors(name, seed, steps):
    fint, KC0, KCNL, KG, n = make_callables(name)
    rng = np.random.default_rng(seed)
    u = SCALE*rng.standard_normal(n)
    d = SCALE*rng.standard_normal(n)
    f0 = fint(u)
    KTd = (KC0 + KCNL(u) + KG(u)) @ d
    return [np.linalg.norm(fint(u + h*d) - f0 - h*KTd)/np.linalg.norm(h*KTd)
            for h in steps]


STEPS = [1.e-1, 1.e-2, 1.e-3, 1.e-4]


@pytest.mark.parametrize('name', ELEMENTS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_tangent_is_derivative_of_fint(name, seed):
    """Taylor test: the error must fall by about ten for each decade of h"""
    errors = taylor_errors(name, seed, STEPS)

    # a consistent tangent leaves a second order remainder, so the error
    # falls by ten for every decade of h; an inconsistent one leaves a first
    # order remainder, so the error plateaus instead
    for h, prev, err in zip(STEPS[1:], errors[:-1], errors[1:]):
        assert err < 0.2*prev, (
            '%s: error did not fall by ten going to h=%.0e: %.3e -> %.3e; '
            'the tangent is not the derivative of fint' % (name, h, prev, err))

    # error/h multiplies the second derivative, so it must not drift with h
    coefficients = [err/h for h, err in zip(STEPS, errors)]
    spread = max(coefficients)/min(coefficients)
    assert spread < 2., (
        '%s: error/h drifted by a factor %.1f, the remainder is not second '
        'order' % (name, spread))


@pytest.mark.parametrize('name, drilling', CASES)
def test_tangent_matches_finite_difference_jacobian(name, drilling):
    """Every entry of KT, against a central difference of fint"""
    fint, KC0, KCNL, KG, n = make_callables(name, drilling)
    rng = np.random.default_rng(7)
    u = SCALE*rng.standard_normal(n)

    step = 1.e-6*SCALE
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        J[:, j] = (fint(u + e) - fint(u - e))/(2*step)

    KT = KC0 + KCNL(u) + KG(u)
    err = np.linalg.norm(KT - J)/np.linalg.norm(J)
    assert err < 1.e-6, '%s %s: KT differs from d(fint)/du by %.3e' % (
        name, drilling, err)


@pytest.mark.parametrize('name', ELEMENTS)
def test_tangent_is_symmetric(name):
    """A tangent that comes from a strain energy is symmetric"""
    fint, KC0, KCNL, KG, n = make_callables(name)
    rng = np.random.default_rng(11)
    u = SCALE*rng.standard_normal(n)
    KT = KC0 + KCNL(u) + KG(u)
    assert np.linalg.norm(KT - KT.T)/np.linalg.norm(KT) < 1.e-12


@pytest.mark.parametrize('name, drilling', CASES)
def test_linear_fint_is_KC0_times_u(name, drilling):
    """Without the nonlinear terms, fint is exactly KC0 @ u

    This is the default, and it failed for Tria3R when update_KC0 did not use
    all the drilling terms that update_probe_finte used.
    """
    fint, KC0, KCNL, KG, n = make_callables(name, drilling)
    rng = np.random.default_rng(5)
    u = SCALE*rng.standard_normal(n)
    KC0u = KC0 @ u
    assert np.linalg.norm(fint(u, nonlinear=0) - KC0u)/np.linalg.norm(KC0u) < 1.e-12


@pytest.mark.parametrize('name', sorted(SHELLS))
def test_explicit_gamma_rz(name):
    """``gamma_rz = A66`` is the default spelled out, any other value is read

    A negative ``gamma_rz``, the default, means A66, so setting it to A66
    must reproduce KC0 and fint exactly, while a different value must change
    both, which is what shows that the attribute is not a no-op.

    The plate is isotropic because the default is A66 in the element frame,
    and only then does it coincide with the ``A66`` of the property.
    """
    prop = isotropic_plate(thickness=0.004, E=70e9, nu=0.33)
    results = {}
    for label, factor in (('default', None), ('A66', 1.), ('other', 0.37)):
        elem, data, _, x, n = make_element(name)
        if factor is not None:
            elem.gamma_rz = factor*prop.A66
        KC0 = assemble(data.KC0_SPARSE_SIZE, n,
                       lambda r, c, v: elem.update_KC0(r, c, v, prop))
        # NOTE the same displacements for the three settings
        u = SCALE*np.random.default_rng(17).standard_normal(n)
        f = np.zeros(n, dtype=DOUBLE)
        elem.update_probe_ue(u)
        elem.update_fint(f, prop, nonlinear=1)
        results[label] = KC0, f
    for i, what in enumerate(('KC0', 'fint')):
        ref = results['default'][i]
        same = np.abs(results['A66'][i] - ref).max()/np.abs(ref).max()
        diff = np.abs(results['other'][i] - ref).max()/np.abs(ref).max()
        print('%-8s %-4s gamma_rz=A66 %.3e  gamma_rz=0.37*A66 %.3e'
              % (name, what, same, diff))
        assert same < 1.e-14, (name, what, same)
        assert diff > 1.e-8, (name, what, diff)


@pytest.mark.parametrize('name', ELEMENTS)
def test_nonlinear_terms_vanish_in_the_undeformed_state(name):
    """At u = 0 there is no internal force, and KT reduces to KC0"""
    fint, KC0, KCNL, KG, n = make_callables(name)
    u = np.zeros(n)
    assert np.all(fint(u) == 0)
    assert np.abs(KCNL(u)).max() == 0
    assert np.abs(KG(u)).max() == 0
    assert np.abs(KC0).max() > 0


@pytest.mark.parametrize('name', ELEMENTS)
def test_nonlinear_fint_is_linear_for_small_displacements(name):
    """For displacements much smaller than the element, fint approaches KC0 @ u"""
    fint, KC0, KCNL, KG, n = make_callables(name)
    rng = np.random.default_rng(3)
    u = 1.e-8*SCALE*rng.standard_normal(n)
    KC0u = KC0 @ u
    assert np.linalg.norm(fint(u) - KC0u)/np.linalg.norm(KC0u) < 1.e-6


if __name__ == '__main__':
    for name in ELEMENTS:
        steps = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
        errors = taylor_errors(name, 0, steps)
        print(name)
        prev = None
        for h, err in zip(steps, errors):
            ratio = '' if prev is None else '   (x %.3f)' % (err/prev)
            print('    h %.0e   rel err %.6e%s' % (h, err, ratio))
            prev = err
