r"""The two zero-energy modes of the selectively underintegrated element

These tests record a known and documented property of the element that
:class:`pyfe3d.Quad4` is built on, they do not report an implementation
error. The two modes are the ones of Figure 10 of the paper the integration
scheme comes from:

    Hughes, T. J. R., Taylor, R. L. and Kanoknukulchai, W., 1977, "A simple
    and efficient finite element for plate bending", IJNME, 11(10), pp.
    1529-1543. doi:10.1002/nme.1620111005

whose Section "Application to thick plates" states plainly that "a spectral
analysis of the element stiffness, when one-point Gaussian quadrature is
employed on the shear term, reveals that there are five zero eigenvalues, two
more than the usual three rigid body modes. Thus the element by itself is
rank deficient". The tests exist so that the extent of it is measured rather
than guessed, and so that the behaviour is pinned while the drilling work of
0.10.0 changes the in-plane part of the element: every assertion below holds
identically for both values of ``drilling_model``.

Their Figure 10 uses the rotations `\theta_1` and `\theta_2` of the normal,
which pair with `\partial w/\partial x_1` and `\partial w/\partial x_2` in the
shear energy of their Eq. 7. Against the convention of pyfe3d, where
`\gamma_{xz} = w_{,x} + r_y` and `\gamma_{yz} = w_{,y} - r_x`, that makes
`\theta_1 = -r_y` and `\theta_2 = r_x`, so the two modes read:

- **Mode 1, the hourglass.** `w = x_1 x_2` with `\theta_1 = \theta_2 = 0`,
  that is `w = (x - x_c)(y - y_c)` with the rotations zero. Its transverse
  shear is `\gamma_{xz} = w_{,x}` and `\gamma_{yz} = w_{,y}`, which vanish at
  the centroid.

- **Mode 2, the in-plane twist.** `w = 0` with `\theta_1 = -x_2`, `\theta_2 =
  x_1`, that is `r_x = x - x_c` and `r_y = y - y_c`. Its curvatures vanish,
  since `\kappa_{xy} = r_{y,y} - r_{x,x} = 1 - 1 = 0`, and its shear strains
  are proportional to `x - x_c` and `y - y_c`, so they too vanish at the
  centroid.

Both therefore cost no energy when the rotation part of the shear is sampled
at the centroid alone.

The paper also gives a remedy for Mode 1, the "modified one-point shear"
element, which moves the `(\partial w/\partial x_\alpha)^2` contributions to a
two-by-two rule and leaves the rest at one point, and recommends it "when the
t/h ratio exceeds unity". :class:`pyfe3d.Quad4` applied it, gated at
`h/\ell \ge 1`, up to version 0.11, but the modified rule stores spurious shear
energy in a state of pure twist, see ``tests/test_quad4_thick_twist.py``, and
since 0.12.0 the elements with `h/\ell \ge 1` use the assumed transverse shear
strains of Dvorkin and Bathe given at the end of this docstring, which remove
both modes. The gate is measured below: Modes 1 and 2 cost nothing at `h/\ell
< 1`, and at `h/\ell \ge 1` exactly the energies of their transverse shear
strains interpolated as in Eqs. (6) and (7) of Bathe and Dvorkin (1985).

What the paper says about each mode is also confirmed here. Mode 2 "cannot
persist" in an assembled mesh once the rigid-body modes are removed, and it
does not: the mode that survives assembly is Mode 1. Mode 1 is harmless for
the supported thin plates of their numerical examples, and it is harmless
here too, being suppressed by any support that constrains `w` along a
boundary line, which is why it is invisible in the rest of this test suite.

Where it does bite is a strip one element wide loaded in torsion, because the
twist of such a strip *is* the hourglass pattern `w \propto xy`, so the
element has no stiffness against precisely the deformation being asked for.
On the twist case of the straight cantilever of MacNeal and Harder (1985),
see ``test_torsion_of_a_one_element_wide_strip``, at their 6 by 1 mesh, the
ratio to the reference value is 377 for Quad4, 4.8 for Quad4R and 1.7 for
Tria3R, the last two being helped by the hourglass control of
Brockman (1987) and by the transverse shear stabilisation of Bischoff and
Bletzinger (2004) respectively. Two elements across the width bring Quad4 to
1.02 and Quad4R to 0.92.

Neither of the two obvious remedies works for a thin plate, which is measured
in ``test_no_cheap_quadrature_fix_exists``. Integrating the whole shear term
with a two-by-two rule does remove both modes and does give the right strip
torsion, 0.93 of the reference, but it locks: on a simply supported square
plate with a central load it returns 76% of the one-point deflection at a
slenderness of 10 and 0.04% at 1000, which is the very failure the paper was
written to avoid, and which its Tables I and II report for the beam.
Ungating the paper's own modified scheme is worse still, locking the plate to
the same degree while also making the strip 33 times too stiff. The remedy
that removes the modes without locking is an assumed transverse shear field,
which :class:`pyfe3d.Quad4` uses for `h/\ell \ge 1` since 0.12.0, whereas the
thin elements keep the one-point rule and the two modes:

    Dvorkin, E. N. and Bathe, K. J., 1984, "A continuum mechanics based
    four-node shell element for general nonlinear analysis", Engineering
    Computations, 1(1), pp. 77-88. doi:10.1108/eb023562

    Bathe, K. J. and Dvorkin, E. N., 1985, "A four-node plate bending element
    based on Mindlin/Reissner plate theory and a mixed interpolation", IJNME,
    21(2), pp. 367-383. doi:10.1002/nme.1620210213

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import (Quad4, Quad4Data, Quad4Probe, Quad4R, Quad4RData,
                    Quad4RProbe, Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE,
                    DOF)

SQUARE = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
DISTORTED = np.array([[0., 0., 0.], [2., 0., 0.], [2.3, 1.4, 0.],
                      [0.2, 1.1, 0.]])
TRIANGLE = np.array([[0., 0., 0.], [2., 0., 0.], [0.3, 1.2, 0.]])
GAUSS2 = (-0.5773502691896257, 0.5773502691896257)


def element_K(cls, Data, Probe, ncoords, h, drilling_model=0):
    prop = isotropic_plate(thickness=h, E=1., nu=0.3)
    num_nodes = ncoords.shape[0]
    data = Data()
    probe = Probe()
    el = cls(probe)
    ncoords_flatten = ncoords.flatten()
    for i in range(num_nodes):
        setattr(el, 'n%d' % (i + 1), i + 1)
        setattr(el, 'c%d' % (i + 1), DOF*i)
    el.init_k_KC0 = 0
    # NOTE Quad4R and Tria3R do not carry the attribute yet
    try:
        el.drilling_model = drilling_model
    except AttributeError:
        pass
    el.update_rotation_matrix(ncoords_flatten)
    el.update_probe_xe(ncoords_flatten)
    el.update_area()
    N = DOF*num_nodes
    KC0r = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE, dtype=DOUBLE)
    el.update_KC0(KC0r, KC0c, KC0v, prop)
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).toarray()
    return 0.5*(K + K.T), np.asarray(probe.xe).reshape(-1, 3)


def htk_modes(xe, num_nodes):
    r"""Modes 1 and 2 of Figure 10, in the convention of pyfe3d"""
    xc = xe[:, 0].mean()
    yc = xe[:, 1].mean()
    mode1 = np.zeros(DOF*num_nodes)
    mode2 = np.zeros(DOF*num_nodes)
    for i in range(num_nodes):
        X = xe[i, 0] - xc
        Y = xe[i, 1] - yc
        mode1[DOF*i + 2] = X*Y          # w = x1 x2, rotations zero
        mode2[DOF*i + 3] = X            # r_x = x1
        mode2[DOF*i + 4] = Y            # r_y = x2, w zero
    return mode1/np.linalg.norm(mode1), mode2/np.linalg.norm(mode2)


def test_figure_10_modes_of_hughes_taylor_kanoknukulchai():
    r"""Both modes reproduced, and the thickness gate of the remedy

    Measured, energy of the mode divided by the largest eigenvalue of the
    element matrix:

        h/l      Mode 2 in-plane twist   Mode 1 hourglass
        0.01            5.0e-19              0.0
        0.1             4.6e-18              0.0
        0.5             1.3e-18              0.0
        1.0             0.0187               0.1496
        2.0             0.0187               0.1496

    The step is the switch to the assumed transverse shear strains of the
    MITC4 element, which :class:`.Quad4` applies at `h/\ell \ge 1` since
    0.12.0. The modified one-point shear rule of the paper, applied up to
    0.11, gave the same 0.1496 to Mode 1 but left Mode 2 free, consistently
    with the paper, which offers no remedy for it and argues instead that it
    cannot persist in a mesh.

    Above the gate the energies follow in closed form from the assumed
    shear strains of Bathe and Dvorkin (1985), p. 370, Eq. (6): on a
    rectangle `\gamma_{xz}` is interpolated linearly in `y` between its
    values at the mid-points of the edges `y = \pm \ell/2`, Eq. (7a),
    `\gamma_{yz}` linearly in `x`, Eq. (7b), and these edge values are
    exact for the bilinear fields of both modes. On the unit square, with
    the modes normalised to unit length:

    - Mode 1, nodal `w = \pm 1/2`, is `w = 2 X Y`: `\gamma_{xz} = 2 Y`,
      `\gamma_{yz} = 2 X`, no curvature, `u^T K u = 2 U = A_{55}
      \int (4 Y^2 + 4 X^2) dA = 2 A_{55}/3`;
    - Mode 2, `r_x = X/\sqrt{2}`, `r_y = Y/\sqrt{2}`: `\gamma_{xz} =
      Y/\sqrt{2}`, `\gamma_{yz} = -X/\sqrt{2}`, no curvature and no
      drilling, `u^T K u = A_{55}/12`;

    `A_{44} = A_{55} = k G h`, `k = 5/6`. The ratios of the table follow
    with the largest eigenvalue, the equal biaxial stretching of the
    membrane, `E h/(1 - \nu)` for these `h`: `(5/9)(1 - \nu) G/E = 0.1496`
    and `(5/72)(1 - \nu) G/E = 0.0187`. The energies, not the ratios, are
    asserted, the largest eigenvalue depending on the membrane and drilling
    parts of the element.

    """
    for drilling_model in (0, 1):
        for h in (0.01, 0.1, 0.5):
            K, xe = element_K(Quad4, Quad4Data, Quad4Probe, SQUARE, h,
                              drilling_model)
            mode1, mode2 = htk_modes(xe, 4)
            emax = np.linalg.eigvalsh(K).max()
            U1 = mode1.dot(K.dot(mode1))/emax
            U2 = mode2.dot(K.dot(mode2))/emax
            print('h/l=%g: mode1 %.3e mode2 %.3e' % (h, U1, U2))
            assert abs(U1) < 1.e-14, (h, U1)
            assert abs(U2) < 1.e-14, (h, U2)

        # above the gate the assumed strains remove both modes
        for h in (1., 2.):
            K, xe = element_K(Quad4, Quad4Data, Quad4Probe, SQUARE, h,
                              drilling_model)
            mode1, mode2 = htk_modes(xe, 4)
            emax = np.linalg.eigvalsh(K).max()
            U1 = mode1.dot(K.dot(mode1))
            U2 = mode2.dot(K.dot(mode2))
            A55 = isotropic_plate(thickness=h, E=1., nu=0.3).A55
            print('h/l=%g: mode1 %.4f mode2 %.4f of the largest eigenvalue'
                  % (h, U1/emax, U2/emax))
            assert np.isclose(U1, 2*A55/3, rtol=1.e-10), (h, U1)
            assert np.isclose(U2, A55/12, rtol=1.e-10), (h, U2)


def test_mode_2_is_shared_by_the_three_shell_elements():
    r"""All three sample the rotation part of the shear at the centroid

    Exactly free on a regular shape, and of order `(h/\ell)^2` on a distorted
    one, so still a mechanism for any thin element.

    Quad4R shared it up to 0.11. Since 0.12.0 its transverse shear is that
    of Belytschko and Leviathan (1994), whose stabilisation of the
    nonconstant part of the assumed shear strains gives Mode 2 the energy
    0.018697 times the largest eigenvalue on the square, at both
    thicknesses (measured on 2026-10-29), see
    ``test_twisting_mode_is_stabilised`` in
    ``test_quad4r_hourglass_control.py``.

    """
    for h in (0.01, 0.1):
        K, xe = element_K(Quad4R, Quad4RData, Quad4RProbe, SQUARE, h)
        _, mode2 = htk_modes(xe, 4)
        U = mode2.dot(K.dot(mode2))/np.linalg.eigvalsh(K).max()
        print('Quad4R h/l=%g: %.6f' % (h, U))
        assert np.isclose(U, 0.01869658119658119, rtol=1e-6), (h, U)
    for name, cls, Data, Probe, ncoords, num_nodes in [
            ('Quad4', Quad4, Quad4Data, Quad4Probe, SQUARE, 4),
            ('Tria3R', Tria3R, Tria3RData, Tria3RProbe, TRIANGLE, 3)]:
        for h in (0.01, 0.1):
            K, xe = element_K(cls, Data, Probe, ncoords, h)
            _, mode2 = htk_modes(xe, num_nodes)
            U = mode2.dot(K.dot(mode2))/np.linalg.eigvalsh(K).max()
            print('%s h/l=%g: %.2e' % (name, h, U))
            assert abs(U) < 1.e-14, (name, h, U)

    # on a distorted element the energy is not zero but falls with h squared
    previous = None
    for h in (1., 0.1, 0.01):
        K, xe = element_K(Quad4, Quad4Data, Quad4Probe, DISTORTED, h)
        _, mode2 = htk_modes(xe, 4)
        U = mode2.dot(K.dot(mode2))/np.linalg.eigvalsh(K).max()
        assert U > 0.
        if previous is not None:
            assert 50. < previous/U < 200., (h, previous/U)
        previous = U


def plate_K(cls, Data, Probe, nx, ny, Lx, Ly, h, E, nu, triangles=False):
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    xs = np.linspace(0., Lx, nx + 1)
    ys = np.linspace(0., Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ncoords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    ids = np.arange(X.size).reshape(nx + 1, ny + 1)
    if triangles:
        conn = []
        for i in range(nx):
            for j in range(ny):
                conn.append((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1]))
                conn.append((ids[i, j], ids[i + 1, j + 1], ids[i, j + 1]))
    else:
        conn = [(ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1])
                for i in range(nx) for j in range(ny)]
    N = DOF*ncoords.shape[0]
    ncoords_flatten = ncoords.flatten()
    data = Data()
    probe = Probe()
    size = data.KC0_SPARSE_SIZE*len(conn)
    KC0r = np.zeros(size, dtype=INT)
    KC0c = np.zeros(size, dtype=INT)
    KC0v = np.zeros(size, dtype=DOUBLE)
    k = 0
    for el_nodes in conn:
        el = cls(probe)
        for t, nid in enumerate(el_nodes):
            setattr(el, 'n%d' % (t + 1), nid + 1)
            setattr(el, 'c%d' % (t + 1), DOF*nid)
        el.init_k_KC0 = k
        el.update_rotation_matrix(ncoords_flatten)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        el.update_KC0(KC0r, KC0c, KC0v, prop)
        k += data.KC0_SPARSE_SIZE
    K = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N))
    return 0.5*(K.toarray() + K.toarray().T), ncoords, ids, N


def num_zeros(K, free=None, tol=1.e-9):
    if free is not None:
        K = K[np.ix_(free, free)]
    ev = np.linalg.eigvalsh(K)
    return int((ev < tol*ev.max()).sum())


def test_mode_2_does_not_persist_in_a_mesh_but_mode_1_does():
    r"""Which is what the paper asserts for Mode 2

    Free-free square mesh, zeros below 1e-9 of the largest eigenvalue, at a
    slenderness of 100:

        element     2x2   3x3   4x4
        Quad4         7     7     7
        Quad4R        6     6     6
        Tria3R        6     6     6

    The surviving Quad4 mode is Mode 1, pure `w` with every rotation zero,
    not Mode 2. Quad4R removes it with its hourglass control and Tria3R with
    its shear stabilisation.

    """
    for n in (2, 3, 4):
        K, ncoords, ids, N = plate_K(Quad4, Quad4Data, Quad4Probe, n, n,
                                     1., 1., 0.01, 1., 0.3)
        assert num_zeros(K) == 7, n

        # the extra mode is Mode 1: isolate it by removing the rigid-body span
        basis = []
        for d in range(3):
            m = np.zeros(N)
            m[d::DOF] = 1.
            basis.append(m)
        for ax in range(3):
            m = np.zeros(N)
            e = np.zeros(3)
            e[ax] = 1.
            for i in range(ncoords.shape[0]):
                m[DOF*i:DOF*i + 3] = np.cross(e, ncoords[i])
                m[DOF*i + 3 + ax] = 1.
            basis.append(m)
        Q, _ = np.linalg.qr(np.array(basis).T)
        w, V = np.linalg.eigh(K)
        found = False
        for kk in range(7):
            res = V[:, kk] - Q.dot(Q.T.dot(V[:, kk]))
            if np.linalg.norm(res) > 0.5:
                res /= np.linalg.norm(res)
                # pure w, no rotation and no in-plane component
                assert np.abs(res[2::DOF]).max() > 0.1
                for d in (0, 1, 3, 4, 5):
                    assert np.abs(res[d::DOF]).max() < 1.e-9, d
                found = True
                break
        assert found, n

        assert num_zeros(plate_K(Quad4R, Quad4RData, Quad4RProbe, n, n,
                                 1., 1., 0.01, 1., 0.3)[0]) == 6, n
        assert num_zeros(plate_K(Tria3R, Tria3RData, Tria3RProbe, n, n,
                                 1., 1., 0.01, 1., 0.3,
                                 triangles=True)[0]) == 6, n

    # above the thickness gate the assumed strains remove it here too
    assert num_zeros(plate_K(Quad4, Quad4Data, Quad4Probe, 4, 4,
                             1., 1., 1., 1., 0.3)[0]) == 6


def test_a_supported_boundary_suppresses_mode_1():
    r"""Why it is invisible in the rest of the test suite

    Measured with the drilling rotations left free, at a slenderness of 100:

        boundary conditions              n=4   n=8
        free-free                          7     7
        w on the four edges                3     3
        w on the edges and u, v fixed      0     0
        w at one corner only               6     6

    Constraining `w` along the boundary leaves only the three in-plane
    rigid-body modes and nothing else. A single corner is not enough.

    """
    for n in (4, 8):
        K, ncoords, ids, N = plate_K(Quad4, Quad4Data, Quad4Probe, n, n,
                                     1., 1., 0.01, 1., 0.3)
        x = ncoords[:, 0]
        y = ncoords[:, 1]
        edge = (np.isclose(x, 0.) | np.isclose(x, 1.)
                | np.isclose(y, 0.) | np.isclose(y, 1.))
        assert num_zeros(K) == 7

        bk = np.zeros(N, dtype=bool)
        bk[2::DOF] = edge
        assert num_zeros(K, ~bk) == 3, n

        bk[0::DOF] = True
        bk[1::DOF] = True
        assert num_zeros(K, ~bk) == 0, n

        bk = np.zeros(N, dtype=bool)
        bk[2::DOF] = np.isclose(x, 0.) & np.isclose(y, 0.)
        assert num_zeros(K, ~bk) == 6, n


def strip_twist(cls, Data, Probe, ny, triangles=False):
    r"""Tip twist under a unit torque of the 6 by ny strip of the straight
    cantilever of MacNeal and Harder (1985), reference 0.03208"""
    K, ncoords, ids, N = plate_K(cls, Data, Probe, 6, ny, 6., 0.2, 0.1,
                                 1.e7, 0.3, triangles)
    bk = np.zeros(N, dtype=bool)
    root = np.isclose(ncoords[:, 0], 0.)
    for d in range(DOF):
        bk[d::DOF] |= root
    bu = ~bk
    fext = np.zeros(N)
    tip = ids[-1, :]
    fext[DOF*tip[0] + 2] = -1./0.2
    fext[DOF*tip[-1] + 2] = +1./0.2
    u = np.zeros(N)
    u[bu] = np.linalg.solve(K[np.ix_(bu, bu)], fext[bu])
    return (u[DOF*tip[-1] + 2] - u[DOF*tip[0] + 2])/0.2


def test_torsion_of_a_one_element_wide_strip():
    r"""The twist of such a strip is the hourglass pattern itself

    Straight cantilever of

        MacNeal, R. H. and Harder, R. L., 1985, "A proposed standard set of
        problems to test finite element accuracy", Finite Elements in
        Analysis and Design, 1(1), pp. 3-20. doi:10.1016/0168-874X(85)90003-4

    Fig. 4, p. 7: length 6, width 0.2, thickness 0.1, `E = 10^7`, `\nu =
    0.3`, 6 by 1 mesh, clamped root; twist under a unit tip torque, here a
    couple of normal forces at the two tip corners, theoretical value
    0.03208, Table 3, p. 10. Ratio to it, measured:

        element      ny=1    ny=2    ny=4
        Quad4       377.4    1.019   1.059
        Quad4R        0.928  0.934   1.010
        Tria3R        1.67   1.649   1.645

    (Quad4R up to 0.11: 4.78, 0.923 and 1.001.)

    The grades of MacNeal and Harder (1985), p. 14, are used as the
    thresholds: A for an error of at most 2 %, B at most 10 %, C at most
    20 %, D at most 50 %, F above. Quad4 has no stiffness against the
    deformation being asked for, more than 100 times the reference. Tria3R
    is helped by its stabilisation but still fails at ny=1, grade F, as
    Quad4R did up to 0.11, so the practical rule for these elements is at
    least two elements across the width for any strip that carries torsion;
    with two and four Quad4 gets grade B or better. Since 0.12.0 Quad4R,
    whose transverse shear is that of Belytschko and Leviathan (1994), gets
    grade B with one, two and four. Tria3R stays at 1.65 with 6 elements along
    the length, still grade F, and is not asserted at ny > 1.

    The theoretical value of the table is 0.94 times the Saint-Venant twist
    `T L/(G J)` of the 0.2 x 0.1 rectangle, 0.03411, `J = 0.2287 b h^3`
    from the series of Timoshenko, S. P. and Goodier, J. N., Theory of
    Elasticity, 2nd ed., McGraw-Hill, 1951, p. 278, Eq. (160), whose table
    on p. 277 gives `k_1 = 0.229` for `b/a = 2`, see
    ``tests/test_quad4_thick_twist.py``; Quad4R meshes of 24x4, 48x8 and
    96x16 elements, the torque applied as normal forces proportional to the
    distance from the axis, give 1.049, 1.063 and 1.067 times 0.03208. The
    ratios above are kept against the value of MacNeal and Harder, whose
    grades refer to it.

    The 6 % between Quad4 with 6x4 elements and 0.03208 is not a mesh error
    that refinement removes: Quad4 and Quad4R converge to the same twist,
    about 1.067 times 0.03208, measured on 2026-10-29 with the distributed
    torque above (ratio to 0.03208):

    ========  ======  ======  ======  ======  ======  ======  =======
    element   6x2     6x4     12x4    24x8    48x8    96x16   192x32
    ========  ======  ======  ======  ======  ======  ======  =======
    Quad4     1.0188  1.0583  1.0651  1.0499  1.0518  1.0622  1.0655
    Quad4R    0.9339  1.0101  1.0142  1.0481  1.0505  1.0616  1.0653
    ========  ======  ======  ======  ======  ======  ======  =======

    and with the corner forces of this test, whose deflection is singular,
    1.0700 and 1.0697 with 192x32 elements. The converged value, 0.0342, is
    0.3 % above the Saint-Venant twist; the difference to 0.03208 is one of
    the reference, whose derivation is not given in MacNeal and Harder, and
    no mesh is graded A against it. Following the policy of the tests whose
    reference is not reached on the coarse mesh used, the 6x2 and 6x4 meshes,
    and the 6x1 mesh of Quad4R, are compared with the values pre-determined by this same test, with
    ``rtol = 1e-6`` for the round-off of the dense linear solution, whose
    condition number is about 1e9 (``test_drilling_benchmarks.py``), and with
    grade B against 0.03208 as a bound.

    """
    ref = 0.03208
    # NOTE values pre-determined by strip_twist on 2026-10-29
    predetermined = {(Quad4, 2): 0.03268400685870222,
                     (Quad4, 4): 0.03397594865499767,
                     (Quad4R, 1): 0.02977197292453833,
                     (Quad4R, 2): 0.029960702298012846,
                     (Quad4R, 4): 0.03240695925149363}
    assert strip_twist(Quad4, Quad4Data, Quad4Probe, 1)/ref > 100.
    twist = strip_twist(Quad4R, Quad4RData, Quad4RProbe, 1)
    assert np.isclose(twist, predetermined[Quad4R, 1], rtol=1e-6)
    # NOTE grade B of MacNeal and Harder (1985), error at most 10 %
    assert abs(twist/ref - 1.) <= 0.10
    # NOTE grade F of MacNeal and Harder (1985), error above 50 %
    assert strip_twist(Tria3R, Tria3RData, Tria3RProbe, 1,
                       triangles=True)/ref - 1 > 0.5
    for ny in (2, 4):
        for cls, Data, Probe in ((Quad4, Quad4Data, Quad4Probe),
                                 (Quad4R, Quad4RData, Quad4RProbe)):
            twist = strip_twist(cls, Data, Probe, ny)
            assert np.isclose(twist, predetermined[cls, ny], rtol=1e-6), ny
            # NOTE grade B or better, error at most 10 %
            assert abs(twist/ref - 1.) <= 0.10, ny


def reference_K(nx, ny, Lx, Ly, h, E, nu, scheme):
    r"""The element rebuilt in python with a chosen transverse shear rule

    The rows of ``Quad4Probe`` are reused, so only the quadrature of the
    transverse shear differs from :class:`.Quad4`:

    - ``pure1``: the whole term at the centroid, which is what Quad4 uses
      below the thickness gate
    - ``modified``: two-by-two on the gradient-gradient part and one point on
      the rest, the modified rule of the paper, which Quad4 used above the
      gate up to 0.11; since 0.12.0 Quad4 uses there the assumed strains of
      the MITC4 element, not rebuilt here
    - ``full2x2``: the whole term at two-by-two

    """
    prop = isotropic_plate(thickness=h, E=E, nu=nu)
    A = np.array([[prop.A11, prop.A12, prop.A16],
                  [prop.A12, prop.A22, prop.A26],
                  [prop.A16, prop.A26, prop.A66]])
    Db = np.array([[prop.D11, prop.D12, prop.D16],
                   [prop.D12, prop.D22, prop.D26],
                   [prop.D16, prop.D26, prop.D66]])
    Ats = np.array([[prop.A44, prop.A45], [prop.A45, prop.A55]])
    xs = np.linspace(0., Lx, nx + 1)
    ys = np.linspace(0., Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    ncoords = np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size)))
    ids = np.arange(X.size).reshape(nx + 1, ny + 1)
    conn = [(ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1])
            for i in range(nx) for j in range(ny)]
    N = DOF*ncoords.shape[0]
    ncoords_flatten = ncoords.flatten()
    probe = Quad4Probe()
    K = np.zeros((N, N))
    for el_nodes in conn:
        el = Quad4(probe)
        for t, nid in enumerate(el_nodes):
            setattr(el, 'n%d' % (t + 1), nid + 1)
            setattr(el, 'c%d' % (t + 1), DOF*nid)
        el.update_rotation_matrix(ncoords_flatten)
        el.update_probe_xe(ncoords_flatten)
        el.update_area()
        xe = np.asarray(probe.xe)
        x1, y1, x2, y2 = xe[0], xe[1], xe[3], xe[4]
        x3, y3, x4, y4 = xe[6], xe[7], xe[9], xe[10]

        def detJ(xi, eta):
            J11 = -0.5*x1 + 0.5*x2 + 0.5*(eta + 1)*(0.5*x1 - 0.5*x2 + 0.5*x3 - 0.5*x4)
            J12 = -0.5*y1 + 0.5*y2 + 0.5*(eta + 1)*(0.5*y1 - 0.5*y2 + 0.5*y3 - 0.5*y4)
            J21 = -0.5*x1 + 0.5*x4 - 0.25*(-x1 + x2)*(xi + 1) + 0.25*(x3 - x4)*(xi + 1)
            J22 = -0.5*y1 + 0.5*y4 - 0.25*(xi + 1)*(-y1 + y2) + 0.25*(xi + 1)*(y3 - y4)
            return J11*J22 - J12*J21

        def shear_rows(xi, eta):
            probe.update_BL(xi, eta, 0)
            Bg = np.array([np.asarray(probe.BLgyz_grad),
                           np.asarray(probe.BLgxz_grad)])
            Br = np.array([np.asarray(probe.BLgyz_rot),
                           np.asarray(probe.BLgxz_rot)])
            return Bg.copy(), Br.copy()

        Ke = np.zeros((24, 24))
        for xi in GAUSS2:
            for eta in GAUSS2:
                probe.update_BL(xi, eta, 0)
                Bm = np.array([np.asarray(probe.BLexx),
                               np.asarray(probe.BLeyy),
                               np.asarray(probe.BLgxy)])
                Bb = np.array([np.asarray(probe.BLkxx),
                               np.asarray(probe.BLkyy),
                               np.asarray(probe.BLkxy)])
                Ke += detJ(xi, eta)*(Bm.T.dot(A).dot(Bm)
                                     + Bb.T.dot(Db).dot(Bb))
        if scheme == 'pure1':
            Bg, Br = shear_rows(0., 0.)
            Bs = Bg + Br
            Ke += 4.*detJ(0., 0.)*Bs.T.dot(Ats).dot(Bs)
        elif scheme == 'full2x2':
            for xi in GAUSS2:
                for eta in GAUSS2:
                    Bg, Br = shear_rows(xi, eta)
                    Bs = Bg + Br
                    Ke += detJ(xi, eta)*Bs.T.dot(Ats).dot(Bs)
        elif scheme == 'modified':
            for xi in GAUSS2:
                for eta in GAUSS2:
                    Bg, Br = shear_rows(xi, eta)
                    Ke += detJ(xi, eta)*Bg.T.dot(Ats).dot(Bg)
            Bg, Br = shear_rows(0., 0.)
            Ke += 4.*detJ(0., 0.)*(Bg.T.dot(Ats).dot(Br)
                                   + Br.T.dot(Ats).dot(Bg)
                                   + Br.T.dot(Ats).dot(Br))
        else:
            raise ValueError(scheme)
        gdof = np.concatenate([np.arange(DOF*m, DOF*m + DOF)
                               for m in el_nodes])
        K[np.ix_(gdof, gdof)] += Ke
    return 0.5*(K + K.T), ncoords, ids, N


def test_no_cheap_quadrature_fix_exists():
    r"""Neither alternative rule is usable for a thin plate

    Twist of the 6 by 1 strip, reference 0.03208, and the centre deflection
    of a simply supported square plate under a central load, relative to the
    one-point rule:

        scheme       strip/ref     a/h=10   a/h=50   a/h=200   a/h=1000
        pure1          377.37      1.0000   1.0000   1.0000    1.0000
        modified         0.03      0.7668   0.1678   0.0137    0.0006
        full2x2          0.93      0.7621   0.1461   0.0106    0.0004

    The two-by-two rule gets the strip right and locks the plate. The
    modified rule of the paper, applied without its thickness gate, locks the
    plate just as badly and makes the strip thirty times too stiff as well,
    so it could only be used behind the thickness gate, where :class:`.Quad4`
    applied it up to 0.11 and has applied the MITC4 assumed strains since
    0.12.0.

    """
    # (a) the strip
    strip = {}
    for scheme in ('pure1', 'modified', 'full2x2'):
        K, ncoords, ids, N = reference_K(6, 1, 6., 0.2, 0.1, 1.e7, 0.3,
                                         scheme)
        bk = np.zeros(N, dtype=bool)
        root = np.isclose(ncoords[:, 0], 0.)
        for d in range(DOF):
            bk[d::DOF] |= root
        bk[5::DOF] = True
        bu = ~bk
        fext = np.zeros(N)
        tip = ids[-1, :]
        fext[DOF*tip[0] + 2] = -1./0.2
        fext[DOF*tip[-1] + 2] = +1./0.2
        u = np.zeros(N)
        u[bu] = np.linalg.solve(K[np.ix_(bu, bu)], fext[bu])
        strip[scheme] = ((u[DOF*tip[-1] + 2] - u[DOF*tip[0] + 2])/0.2)/0.03208
        print('strip %s: %.4g x reference' % (scheme, strip[scheme]))
    assert strip['pure1'] > 100.
    assert 0.85 < strip['full2x2'] < 1.05
    assert strip['modified'] < 0.1

    # (b) the plate
    deflection = {}
    for scheme in ('pure1', 'modified', 'full2x2'):
        row = []
        for h in (0.1, 0.02, 0.005, 0.001):
            K, ncoords, ids, N = reference_K(8, 8, 1., 1., h, 200e9, 0.3,
                                             scheme)
            x = ncoords[:, 0]
            y = ncoords[:, 1]
            edge = (np.isclose(x, 0.) | np.isclose(x, 1.)
                    | np.isclose(y, 0.) | np.isclose(y, 1.))
            bk = np.zeros(N, dtype=bool)
            bk[2::DOF] = edge
            bk[0::DOF] = True
            bk[1::DOF] = True
            bk[5::DOF] = True
            bu = ~bk
            fext = np.zeros(N)
            centre = np.isclose(x, 0.5) & np.isclose(y, 0.5)
            fext[2::DOF][centre] = 1.
            u = np.zeros(N)
            u[bu] = np.linalg.solve(K[np.ix_(bu, bu)], fext[bu])
            row.append(u[2::DOF].max())
        deflection[scheme] = row
    for scheme in ('modified', 'full2x2'):
        ratios = [deflection[scheme][i]/deflection['pure1'][i]
                  for i in range(4)]
        print('plate %s: %s' % (scheme, ['%.4f' % r for r in ratios]))
        # locking: the ratio collapses as the plate thins
        assert ratios[0] < 0.8
        assert ratios[-1] < 0.001
        assert ratios[0] > ratios[1] > ratios[2] > ratios[3]


if __name__ == '__main__':
    test_figure_10_modes_of_hughes_taylor_kanoknukulchai()
    test_mode_2_is_shared_by_the_three_shell_elements()
    test_mode_2_does_not_persist_in_a_mesh_but_mode_1_does()
    test_a_supported_boundary_suppresses_mode_1()
    test_torsion_of_a_one_element_wide_strip()
    test_no_cheap_quadrature_fix_exists()
