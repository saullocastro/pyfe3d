r"""Regression tests for the fictitious (K6ROT penalty) drilling stiffness.

The penalty model is the pre-0.10.0 default and remains available through
``drilling_model = 1``. It is the approach adopted in MSC Nastran and Autodesk
Nastran through the K6ROT parameter, and is very close to Eq. 2.20 of:

    Adam, F. M., Mohamed, A. E., and Hassaballa, A. E., 2013, "Degenerated Four
    Nodes Shell Element with Drilling Degree of Freedom", IOSR J. Eng., 3(8),
    pp. 10-20. doi:10.9790/3021-03831020

The penalty energy per element is

.. math::
    U_{drill} = \frac{1}{2} K6ROT \cdot 10^{-6}
                \int_A A_{66} (r_z - \theta_z)^2 dA

so for the uniform drilling state `u_i = v_i = 0`, `{r_z}_i = \omega_0`, which
makes `\theta_z = 0` and `r_z = \omega_0` everywhere because the shape
functions form a partition of unity, the energy has the closed form

.. math::
    U = \frac{1}{2} K6ROT \cdot 10^{-6} A_{66} A_e \omega_0^2

with `A_e` the element area. That closed form is the reference used below, and
it is independent of the element shape. The second reference used is
cross-element: Quad4 and Quad4R integrate the same drilling operator with the
same 2x2 rule, so their drilling blocks must agree to machine precision on the
same geometry. Since 0.10.0 none of the three shell elements reads ``K6ROT``
by default, which :func:`test_k6rot_penalty_is_not_the_default` checks.

These tests exist because the coefficient `K6ROT \cdot 10^{-6} A_{66}` was
lost from Quad4 between 0.8.0 and 0.9.0, leaving an effective coefficient of 1
and making the ``K6ROT`` attribute a no-op. Only non-coplanar models were
affected, because on a flat mesh the drilling row is decoupled from every other
degree-of-freedom and its magnitude cannot change any displacement.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE,
                    DOF)

# NOTE element geometries in the xy plane, counter-clockwise
RECTANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]])
DISTORTED = np.array([[0., 0., 0.], [2., 0., 0.], [2.3, 1.4, 0.], [0.2, 1.1, 0.]])
TRIANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [0.3, 1.2, 0.]])
TRIANGLE_DISTORTED = np.array([[0.1, -0.2, 0.], [2.4, 0.3, 0.], [0.9, 1.7, 0.]])


def shoelace_area(ncoords):
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    return 0.5*abs(x.dot(np.roll(y, -1)) - y.dot(np.roll(x, -1)))


def build_KC0(cls, Data, Probe, ncoords, K6ROT, prop):
    num_nodes = ncoords.shape[0]
    data = Data()
    probe = Probe()
    el = cls(probe)
    ncoords_flatten = ncoords.flatten()
    for i in range(num_nodes):
        setattr(el, 'n%d' % (i + 1), i + 1)
        setattr(el, 'c%d' % (i + 1), DOF*i)
    el.init_k_KC0 = 0
    el.drilling_model = 1
    el.K6ROT = K6ROT
    el.update_rotation_matrix(ncoords_flatten)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    N = DOF*num_nodes
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    return coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()


def uniform_rz(num_nodes):
    ue = np.zeros(DOF*num_nodes)
    ue[5::DOF] = 1.
    return ue


def test_k6rot_penalty_energy_closed_form():
    r"""Energy of the uniform drilling state against the closed form

    ``U = 0.5*K6ROT*1e-6*A66*area``, for every element and both a regular and
    a distorted shape.

    """
    prop = isotropic_plate(thickness=0.01, E=200e9, nu=0.3)
    cases = [
        ('Quad4', Quad4, Quad4Data, Quad4Probe, RECTANGLE),
        ('Quad4', Quad4, Quad4Data, Quad4Probe, DISTORTED),
        ('Quad4R', Quad4R, Quad4RData, Quad4RProbe, RECTANGLE),
        ('Quad4R', Quad4R, Quad4RData, Quad4RProbe, DISTORTED),
        ('Tria3R', Tria3R, Tria3RData, Tria3RProbe, TRIANGLE),
        ('Tria3R', Tria3R, Tria3RData, Tria3RProbe, TRIANGLE_DISTORTED),
    ]
    for name, cls, Data, Probe, ncoords in cases:
        area = shoelace_area(ncoords)
        ue = uniform_rz(ncoords.shape[0])
        for K6ROT in (1., 100., 1.e4):
            KC0 = build_KC0(cls, Data, Probe, ncoords, K6ROT, prop)
            U = 0.5*ue.dot(KC0.dot(ue))
            U_ref = 0.5*K6ROT*1e-6*prop.A66*area
            print(name, K6ROT, U, U_ref)
            assert np.isclose(U, U_ref, rtol=1e-12), (name, K6ROT, U, U_ref)


def test_k6rot_penalty_scales_linearly():
    r"""The penalty is linear in K6ROT and it is the only term that reads it"""
    prop = isotropic_plate(thickness=0.01, E=200e9, nu=0.3)
    cases = [
        ('Quad4', Quad4, Quad4Data, Quad4Probe, DISTORTED),
        ('Quad4R', Quad4R, Quad4RData, Quad4RProbe, DISTORTED),
        ('Tria3R', Tria3R, Tria3RData, Tria3RProbe, TRIANGLE_DISTORTED),
    ]
    for name, cls, Data, Probe, ncoords in cases:
        K1 = build_KC0(cls, Data, Probe, ncoords, 1., prop)
        K2 = build_KC0(cls, Data, Probe, ncoords, 2., prop)
        K3 = build_KC0(cls, Data, Probe, ncoords, 3., prop)
        # NOTE K(a) = Kother + a*Kdrill, so the second difference vanishes.
        #      It is measured against the scale of the matrix and not against
        #      the scale of the first difference: the drilling operator
        #      contributes to the membrane entries through its u and v terms,
        #      and those entries are of the order of A66/element size, about
        #      8e8 here, whereas the change per unit of K6ROT is about 1e2.
        #      Differencing therefore loses some seven digits, so a relative
        #      tolerance on the difference itself cannot be met in double
        #      precision, while the second difference does vanish to the
        #      machine precision of the entries being differenced
        scale = np.abs(K1).max()
        second = np.abs((K3 - K2) - (K2 - K1)).max()
        print(name, 'second difference/scale =', second/scale)
        assert second/scale < 1.e-14, (name, second/scale)
        assert np.abs(K2 - K1).max() > 0., name


def test_k6rot_penalty_quad4_matches_quad4r():
    r"""Quad4 and Quad4R integrate the same drilling operator at 2x2

    Their drilling blocks must therefore be identical on the same geometry,
    which is an independent check of the coefficient used by each element.

    """
    prop = isotropic_plate(thickness=0.01, E=200e9, nu=0.3)
    for ncoords in (RECTANGLE, DISTORTED):
        # NOTE isolating the drilling block by differencing two values of
        #      K6ROT, since every other term is independent of it
        dq = (build_KC0(Quad4, Quad4Data, Quad4Probe, ncoords, 1.e4, prop)
              - build_KC0(Quad4, Quad4Data, Quad4Probe, ncoords, 100., prop))
        dqr = (build_KC0(Quad4R, Quad4RData, Quad4RProbe, ncoords, 1.e4, prop)
               - build_KC0(Quad4R, Quad4RData, Quad4RProbe, ncoords, 100., prop))
        scale = np.abs(dqr).max()
        assert scale > 0.
        assert np.abs(dq - dqr).max()/scale < 1e-11


def test_k6rot_penalty_is_not_the_default():
    r"""The penalty must not be active unless it is selected

    Documents the 0.10.0 behaviour change: with the default physics-based
    drilling stiffness the ``K6ROT`` attribute is not read at all. This
    applies to all three shell elements. Note that Quad4R reaches it by a
    different route from Quad4 and Tria3R: it applies the coefficient
    `\gamma_{r_z} = A_{66}` of Hughes and Brezzi (1989) to the same operator
    that carries the penalty, without the in-plane enrichment of Allman
    (1984), see the module documentation of :mod:`pyfe3d.quad4r`.

    """
    prop = isotropic_plate(thickness=0.01, E=200e9, nu=0.3)
    for cls, Data, Probe, ncoords in [
            (Quad4, Quad4Data, Quad4Probe, DISTORTED),
            (Quad4R, Quad4RData, Quad4RProbe, DISTORTED),
            (Tria3R, Tria3RData, Tria3RProbe, TRIANGLE_DISTORTED)]:
        num_nodes = ncoords.shape[0]
        Ks = []
        for K6ROT in (1., 1.e6):
            data = Data()
            probe = Probe()
            el = cls(probe)
            ncoords_flatten = ncoords.flatten()
            for i in range(num_nodes):
                setattr(el, 'n%d' % (i + 1), i + 1)
                setattr(el, 'c%d' % (i + 1), DOF*i)
            el.init_k_KC0 = 0
            assert el.drilling_model == 0
            el.K6ROT = K6ROT
            el.update_rotation_matrix(ncoords_flatten)
            el.update_probe_xe(ncoords_flatten)
            el.update_area()
            N = DOF*num_nodes
            KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
            KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
            KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
            el.update_KC0(KC0r, KC0c, KC0v, prop)
            Ks.append(coo_matrix((KC0v, (KC0r, KC0c)),
                                 shape=(N, N)).toarray())
        assert np.array_equal(Ks[0], Ks[1])


if __name__ == '__main__':
    test_k6rot_penalty_energy_closed_form()
    test_k6rot_penalty_scales_linearly()
    test_k6rot_penalty_quad4_matches_quad4r()
    test_k6rot_penalty_is_not_the_default()
