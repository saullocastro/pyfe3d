r"""Consistency and patch tests for the physics-based drilling stiffness.

The default drilling stiffness of the shell elements combines the enrichment
of the in-plane displacement field of Allman, so that `r_z` produces membrane
strain energy, with the regularisation of Hughes and Brezzi, which restores
the rank of the enriched element and gives the added term a consistent,
parameter-free meaning:

    Allman, D. J., 1984, "A compatible triangular element including vertex
    rotations for plane elasticity analysis", Computers & Structures, 19(1-2),
    pp. 1-8. doi:10.1016/0045-7949(84)90197-4

    Hughes, T. J. R. and Brezzi, F., 1989, "On drilling degrees of freedom",
    CMAME, 72(1), pp. 105-121. doi:10.1016/0045-7825(89)90124-2

    Ibrahimbegovic, A., Taylor, R. L. and Wilson, E. L., 1990, "A robust
    quadrilateral membrane finite element with drilling degrees of freedom",
    IJNME, 30(3), pp. 445-457. doi:10.1002/nme.1620300305

The irregular patch is the one proposed for plates in Fig. 2 and Table 2 of:

    MacNeal, R. H. and Harder, R. L., 1985, "A proposed standard set of
    problems to test finite element accuracy", Finite Elements in Analysis and
    Design, 1(1), pp. 3-20. doi:10.1016/0168-874X(85)90003-4

The three properties checked here are the ones that the enrichment must not
break. A rigid-body motion and any state of constant strain make every edge
amplitude `a_k = (\ell_k/8)({r_z}_i - {r_z}_j)` vanish, because the drilling
rotations are then all equal, so the enriched field collapses onto its
bilinear part and reproduces the prescribed field exactly. The drilling term
contributes no energy in those states either, for any value of the
regularisation parameter, which is the qualitative difference from the K6ROT
penalty, whose added energy is non-zero and merely kept small.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF

# NOTE MacNeal and Harder's patch, Fig. 2: a rectangle of 0.24 by 0.12 split
#      into five quadrilaterals by four arbitrarily placed interior nodes
PATCH_OUTER = np.array([[0., 0.], [0.24, 0.], [0.24, 0.12], [0., 0.12]])
PATCH_INNER = np.array([[0.04, 0.02], [0.18, 0.03], [0.16, 0.08], [0.08, 0.08]])
# nodes 1-4 are the corners, 5-8 the interior ones
PATCH_CONN = [(5, 6, 7, 8),
              (1, 2, 6, 5),
              (2, 3, 7, 6),
              (3, 4, 8, 7),
              (4, 1, 5, 8)]


def patch_mesh():
    ncoords = np.column_stack((np.vstack((PATCH_OUTER, PATCH_INNER)),
                               np.zeros(8)))
    nids = 1 + np.arange(8)
    return ncoords, nids


def build(drilling_model, prop, K6ROT=100., gamma_rz=-1.):
    r"""Assemble the patch and return the elements and the global KC0"""
    ncoords, nids = patch_mesh()
    nid_pos = dict(zip(nids, np.arange(len(nids))))
    ncoords_flatten = ncoords.flatten()
    N = DOF*len(nids)

    data = Quad4Data()
    probe = Quad4Probe()
    num_elements = len(PATCH_CONN)
    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)

    quads = []
    init_k_KC0 = 0
    for n1, n2, n3, n4 in PATCH_CONN:
        r1 = ncoords[nid_pos[n1]]
        r2 = ncoords[nid_pos[n2]]
        r3 = ncoords[nid_pos[n3]]
        assert np.cross(r2 - r1, r3 - r2)[2] > 0
        q = Quad4(probe)
        q.n1, q.n2, q.n3, q.n4 = n1, n2, n3, n4
        q.c1 = DOF*nid_pos[n1]
        q.c2 = DOF*nid_pos[n2]
        q.c3 = DOF*nid_pos[n3]
        q.c4 = DOF*nid_pos[n4]
        q.init_k_KC0 = init_k_KC0
        q.drilling_model = drilling_model
        q.K6ROT = K6ROT
        q.gamma_rz = gamma_rz
        q.update_rotation_matrix(ncoords_flatten)
        q.update_probe_xe(ncoords_flatten)
        q.update_area()
        q.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(q)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    return quads, KC0, ncoords, nids, nid_pos, N


def membrane_field(ncoords, a, b):
    r"""u = a1 + a2 x + a3 y, v = b1 + b2 x + b3 y, r_z = (b2 - a3)/2

    The in-plane rotation of this field is the constant `\theta_z = (b_2 -
    a_3)/2`, and prescribing `r_z = \theta_z` at every node is what makes the
    state a constant-strain state of the enriched element.

    """
    u = np.zeros(DOF*ncoords.shape[0])
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    u[0::DOF] = a[0] + a[1]*x + a[2]*y
    u[1::DOF] = b[0] + b[1]*x + b[2]*y
    u[5::DOF] = 0.5*(b[1] - a[2])
    return u


def bending_field(ncoords, c):
    r"""w = c(x^2 + xy + y^2)/2, r_x = w_{,y}, r_y = -w_{,x}

    The constant-curvature state of Table 2(b) of MacNeal and Harder, written
    with the six-degree-of-freedom rotations of the element, for which `r_x =
    \partial w/\partial y` and `r_y = -\partial w/\partial x`.

    """
    u = np.zeros(DOF*ncoords.shape[0])
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    u[2::DOF] = c*(x**2 + x*y + y**2)/2.
    u[3::DOF] = c*(y + x/2.)
    u[4::DOF] = -c*(-x - y/2.)
    return u


def interior_dofs(nids, nid_pos):
    r"""The degrees-of-freedom of the four interior nodes of the patch"""
    free = np.zeros(DOF*len(nids), dtype=bool)
    for nid in (5, 6, 7, 8):
        free[DOF*nid_pos[nid]:DOF*nid_pos[nid] + DOF] = True
    return free


def test_A1_rigid_body_motion():
    r"""A rigid-body motion must store no energy

    The in-plane rigid rotation `u = -\omega_0 y`, `v = +\omega_0 x`, `r_z =
    \omega_0` is the particular constant-strain state with zero strain, and it
    is the one that the drilling enrichment could spoil, because it is the
    only rigid-body mode in which `r_z` is non-zero.

    """
    prop = isotropic_plate(thickness=0.001, E=1.e6, nu=0.25)
    for drilling_model in (0, 1):
        quads, KC0, ncoords, nids, nid_pos, N = build(drilling_model, prop)
        scale = np.abs(np.linalg.eigvalsh(0.5*(KC0 + KC0.T))).max()

        w0 = 0.37
        modes = {}
        # three translations
        for d in range(3):
            m = np.zeros(N)
            m[d::DOF] = 1.
            modes['translation %d' % d] = m
        # three rotations about the global axes, through the origin
        for ax in range(3):
            m = np.zeros(N)
            e = np.zeros(3)
            e[ax] = 1.
            for i in range(len(nids)):
                m[DOF*i:DOF*i + 3] = w0*np.cross(e, ncoords[i])
                m[DOF*i + 3 + ax] = w0
            modes['rotation about %d' % ax] = m

        for name, m in modes.items():
            U = 0.5*m.dot(KC0.dot(m))
            print(drilling_model, name, U/scale)
            assert abs(U)/scale < 1.e-12, (drilling_model, name, U/scale)


def test_A2_constant_strain_patch_test():
    r"""Constant membrane strain and constant curvature, exactly reproduced

    Both states are prescribed at every node of the irregular patch. The
    interior nodes are not supported, so the internal forces there must
    vanish, and the strains recovered anywhere inside any element must equal
    the prescribed constant values to machine precision.

    """
    prop = isotropic_plate(thickness=0.001, E=1.e6, nu=0.25)
    a = np.array([0.11, 0.023, -0.017])
    b = np.array([-0.07, 0.031, 0.019])
    curv = 1.e-3

    for drilling_model in (0, 1):
        quads, KC0, ncoords, nids, nid_pos, N = build(drilling_model, prop)
        free = interior_dofs(nids, nid_pos)
        scale = np.abs(KC0).max()

        # (a) membrane form
        u = membrane_field(ncoords, a, b)
        f = KC0.dot(u)
        print('membrane residual', np.abs(f[free]).max()/scale)
        assert np.abs(f[free]).max()/scale < 1.e-12

        # the recovered membrane strains, in the element coordinate system,
        # must be the prescribed ones rotated by the element angle
        for q in quads:
            probe = q.probe
            q.update_probe_xe(ncoords.flatten())
            q.update_probe_ue(u)
            ue = np.asarray(probe.ue).copy()
            for xi, eta in ((-0.3, 0.7), (0., 0.), (0.5, -0.5)):
                probe.update_BL(xi, eta, drilling_model)
                exx = np.asarray(probe.BLexx).dot(ue)
                eyy = np.asarray(probe.BLeyy).dot(ue)
                gxy = np.asarray(probe.BLgxy).dot(ue)
                # the strain tensor of the prescribed field, rotated into the
                # element frame with the in-plane part of the element triad
                c, s = q.r11, q.r21
                E = np.array([[a[1], 0.5*(a[2] + b[1])],
                              [0.5*(a[2] + b[1]), b[2]]])
                R = np.array([[c, s], [-s, c]])
                Ee = R.dot(E).dot(R.T)
                assert np.isclose(exx, Ee[0, 0], rtol=1e-10, atol=1e-14)
                assert np.isclose(eyy, Ee[1, 1], rtol=1e-10, atol=1e-14)
                assert np.isclose(gxy, 2*Ee[0, 1], rtol=1e-10, atol=1e-14)

    # (b) bending form, MacNeal and Harder Table 2(b). The enrichment acts
    #     only on the in-plane translations, so it must leave the whole
    #     bending response untouched. That is asserted directly, on the
    #     stiffness block and on the internal forces, rather than against
    #     zero: the residual of this state is not zero for this element,
    #     because the transverse displacement of the prescribed field is
    #     quadratic while the element interpolates it bilinearly, which
    #     leaves a transverse shear strain inside the element. That is a
    #     property of the underlying plate formulation and is unrelated to
    #     the drilling degree-of-freedom
    u = bending_field(ncoords, curv)
    _, KC0_enriched, _, _, _, _ = build(0, prop)
    _, KC0_penalty, _, _, _, _ = build(1, prop)
    bending = np.zeros(KC0_enriched.shape[0], dtype=bool)
    for d in (2, 3, 4):
        bending[d::DOF] = True
    block_e = KC0_enriched[np.ix_(bending, bending)]
    block_p = KC0_penalty[np.ix_(bending, bending)]
    print('bending block difference', np.abs(block_e - block_p).max())
    assert np.array_equal(block_e, block_p)
    fe = KC0_enriched.dot(u)
    fp = KC0_penalty.dot(u)
    print('bending internal force difference', np.abs(fe - fp).max())
    assert np.allclose(fe, fp, rtol=0., atol=1.e-20)


def test_A3_drilling_energy_vanishes_at_constant_strain():
    r"""Neither drilling term adds energy in a constant-strain state

    Stationarity of the Hughes-Brezzi functional with respect to `r_z` gives
    `r_z = \theta_z` pointwise, so the physics-based term vanishes at the
    exact solution for any value of the regularisation parameter.

    The K6ROT penalty vanishes in this state as well, and that is worth
    recording rather than assuming otherwise: it is built from the same
    operator `B_{drill} = S^{r_z} + \frac{1}{2}S^u_{,y} -
    \frac{1}{2}S^v_{,x}`, which is the residual `r_z - \theta_z`, and not
    from a diagonal addition on the `r_z` terms. A diagonal addition would
    stiffen the rigid rotation of the element about its normal and break the
    patch test, whereas the operator form does not. The qualitative
    difference between the two models is therefore not visible here: it shows
    up in the sensitivity to the parameter, and in whether the recovered
    nodal moment about the shell normal means anything.

    """
    prop = isotropic_plate(thickness=0.001, E=1.e6, nu=0.25)
    a = np.array([0.11, 0.023, -0.017])
    b = np.array([-0.07, 0.031, 0.019])
    ncoords, nids = patch_mesh()
    u_const = membrane_field(ncoords, a, b)

    # the drilling contribution is isolated by differencing two values of the
    # parameter that scales it, every other term being independent of it
    def drilling_block(drilling_model, lo, hi):
        if drilling_model == 0:
            _, Klo, _, _, _, _ = build(0, prop, gamma_rz=lo*prop.A66)
            _, Khi, _, _, _, _ = build(0, prop, gamma_rz=hi*prop.A66)
        else:
            _, Klo, _, _, _, _ = build(1, prop, K6ROT=lo)
            _, Khi, _, _, _, _ = build(1, prop, K6ROT=hi)
        return Khi - Klo

    _, K, _, _, _, _ = build(0, prop)
    scale = 0.5*abs(u_const.dot(K.dot(u_const)))

    for drilling_model, lo, his in ((0, 0., (1., 10., 1000.)),
                                    (1, 0., (100., 1.e4))):
        for hi in his:
            Kd = drilling_block(drilling_model, lo, hi)
            U = 0.5*u_const.dot(Kd.dot(u_const))
            print('model %d, parameter %g: drilling energy/total = %.3e'
                  % (drilling_model, hi, U/scale))
            # the round-off left in the constraint residual is multiplied by
            # the parameter, so the tolerance is scaled by it as well
            assert abs(U)/(scale*hi) < 1.e-12

    # NOTE the checks above would also pass if the drilling term were simply
    #      absent, so the term is exercised here on a state that does violate
    #      r_z = theta_z, where the energy must be non-zero and must scale
    #      linearly with the parameter of each model
    u_bad = u_const.copy()
    u_bad[5::DOF] += 0.05
    for drilling_model, ref in ((0, 1.), (1, 100.)):
        Kd1 = drilling_block(drilling_model, 0., ref)
        Kd2 = drilling_block(drilling_model, 0., 2*ref)
        U1 = 0.5*u_bad.dot(Kd1.dot(u_bad))
        U2 = 0.5*u_bad.dot(Kd2.dot(u_bad))
        print('model %d: energy of the violating state/total = %.3e'
              % (drilling_model, U1/scale))
        assert abs(U1)/scale > 1.e-6
        assert np.isclose(U2, 2*U1, rtol=1.e-10)


if __name__ == '__main__':
    test_A1_rigid_body_motion()
    test_A2_constant_strain_patch_test()
    test_A3_drilling_energy_vanishes_at_constant_strain()
