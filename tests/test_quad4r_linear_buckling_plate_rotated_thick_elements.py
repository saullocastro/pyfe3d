import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import linear_buckling


def test_linear_buckling_plate(plot=False, mode=0):
    r"""Plate with ``a/b = 4`` and ``b/h = 10``, uniform compression
    ``N_xx``, hard simply supported, 30x14 elements rotated by 25
    combinations of pitch and yaw

    The edges are hard simply supported, ``w = 0`` and the rotation about the
    axis of each edge equal to zero, for which the Navier solution of the
    plate with transverse shear deformation is exact. The reference is that
    solution as given by Seide, P., and Stowell, E. Z., Elastic and Plastic
    Buckling of Simply Supported Solid-Core Sandwich Plates in Compression,
    NACA Report 967, 1950, Eq. (1), p. 2, derived in Appendix A, Eq. (A7),
    p. 7: ``N_xx,cr = -k*pi**2*D/b**2``,
    ``k = (m/beta + beta/m)**2/(1 + r*(1 + m**2/beta**2))``,
    ``beta = a/b``, ``r = pi**2*D/(b**2*D_Q)``, with the transverse shear
    stiffness ``D_Q`` of the sandwich core taken equal to ``A55``, minimised
    over ``m``; their governing equation (A5) is that of a plate with
    transverse shear stiffness ``D_Q``. The same load follows from Wang,
    C. M., Wang, C. Y., and Reddy, J. N., Exact Solutions for Buckling of
    Structural Members, CRC Press, 2005, Sec. 4.7.2, Eq. (4.7.17), with the
    foundation moduli equal to zero and ``gamma = 0`` (page not verified,
    the PDF consulted has no page numbers). It is 5.6 % below the Kirchhoff
    solution of Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Sec.
    10.2.1, p. 277, Eq. (10.46).

    The buckling load must not depend on the orientation of the plate. With
    30x14 elements it is 1.41 % above the reference, measured on 2026-10-29
    with the default stabilisation of Quad4R since 0.12.0, the physical
    stabilisation of the bending and of the transverse shear of Belytschko
    and Leviathan (1994), see :mod:`pyfe3d.quad4r`. Up to 0.11, with twice
    the hourglass stiffnesses of Brockman (1987) for `u`, `v`, `r_x` and
    `r_y` and four times that of `w`, it was 10.7 % above the reference
    (-195870012.17574 N), and the flat plate converged to it, 2.64 % above
    it with 60x28 elements, 1.17 % with 90x42 elements, 0.66 % with 120x56
    elements; with the soft simple support used before those, rotations
    free, the 30x14 mesh gave a buckling load 1.8 % below the Kirchhoff value
    and 4.0 % above the reference.

    The 30x14 meshes are compared with the load pre-determined by this same
    test with that mesh, -179433757.7496 N, with ``rtol = 1e-4`` for the
    round-off of the eigenvalue solver, and with the reference within grade
    A of MacNeal, R. H. and Harder, R. L., A proposed standard set of
    problems to test finite element accuracy, Finite Elements in Analysis
    and Design, 1(1), 3-20, 1985,
    https://doi.org/10.1016/0168-874X(85)90003-4, p. 14 (at most 2 %). The
    90x42 flat plate, run until 0.12.0 against the reference within 2 %, is
    no longer run.
    """
    thetas = np.linspace(-np.pi, np.pi, 5)
    # NOTE (pitch, yaw, nx, ny)
    cases = [(pitch, yaw, 31, 15) for pitch in thetas for yaw in thetas]
    P_cr_calcs = []

    a = 2.0
    b = 0.5
    h = 0.05 # m

    E = 203.e9 # Pa
    nu = 0.33
    prop = isotropic_plate(E=E, nu=nu, thickness=h)

    Nxx = -1.

    data = Quad4RData()
    probe = Quad4RProbe()

    for pitch, yaw, nx, ny in cases:
        print('pitch, yaw, nx, ny', pitch, yaw, nx, ny)

        xtmp = np.linspace(0, a, nx)
        ytmp = np.linspace(0, b, ny)
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        ncoords_local = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

        x_local = ncoords_local[:, 0]
        y_local = ncoords_local[:, 1]
        z_local = ncoords_local[:, 2]

        x = np.cos(yaw)*np.cos(pitch)*x_local - np.sin(yaw)*y_local + np.cos(yaw)*np.sin(pitch)*z_local
        y = np.sin(yaw)*np.cos(pitch)*x_local + np.cos(yaw)*y_local + np.sin(yaw)*np.sin(pitch)*z_local
        z = -np.sin(pitch)*x_local + np.cos(pitch)*z_local

        ncoords = np.vstack((x, y, z)).T
        ncoords_flatten = ncoords.flatten()

        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))
        nids_mesh = nids.reshape(nx, ny)
        n1s = nids_mesh[:-1, :-1].flatten()
        n2s = nids_mesh[1:, :-1].flatten()
        n3s = nids_mesh[1:, 1:].flatten()
        n4s = nids_mesh[:-1, 1:].flatten()

        num_elements = len(n1s)
        print('num_elements', num_elements)

        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        init_k_KC0 = 0

        quads = []
        init_k_KG = 0
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            pos3 = nid_pos[n3]
            pos4 = nid_pos[n4]

            quad = Quad4R(probe)
            quad.n1 = n1
            quad.n2 = n2
            quad.n3 = n3
            quad.n4 = n4
            quad.c1 = DOF*nid_pos[n1]
            quad.c2 = DOF*nid_pos[n2]
            quad.c3 = DOF*nid_pos[n3]
            quad.c4 = DOF*nid_pos[n4]
            # NOTE The thick plate (h=0.05 m) has a worse drilling-to-bending
            #      stiffness ratio with K6ROT=1e4, causing an ill-conditioned
            #      KC0uu (condition number ~1e14). A diagonal preconditioner
            #      is used in the eigensolver to normalise KC0 diagonal to 1.
            quad.K6ROT = 1e4
            quad.init_k_KC0 = init_k_KC0
            quad.init_k_KG = init_k_KG
            quad.update_rotation_matrix(ncoords_flatten)
            quad.update_probe_xe(ncoords_flatten)
            # NOTE the default stabilisation: until 0.12.0 this test used
            #      twice the hourglass stiffnesses of u, v, r_x and r_y and
            #      four times that of w of Brockman (1987)
            quad.update_KC0(KC0r, KC0c, KC0v, prop)
            quads.append(quad)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE

        print('elements created')

        Nxx = -1

        for quad in quads:
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KG_given_stress(Nxx, 0, 0, KGr, KGc, KGv)

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

        print('sparse KC0 and KG created')

        # constraining all edge nodes to no translation
        bk = np.zeros(N, dtype=bool)
        check = isclose(x_local, 0.) | isclose(x_local, a) | isclose(y_local, 0) | isclose(y_local, b)
        bk[0::DOF] = check
        bk[1::DOF] = check
        bk[2::DOF] = check
        # NOTE hard simple support: rotation about the local x axis on
        #      the edges x = 0, a and about the local y axis on the edges
        #      y = 0, b; the rotations of the plate are multiples of 90
        #      degrees, such that each local axis is a global one
        ex = np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch),
                       -np.sin(pitch)])
        ey = np.array([-np.sin(yaw), np.cos(yaw), 0.])
        assert isclose(np.abs(ex).max(), 1.) and isclose(np.abs(ey).max(), 1.)
        bk[3 + np.argmax(np.abs(ex))::DOF] |= isclose(x_local, 0.) | isclose(x_local, a)
        bk[3 + np.argmax(np.abs(ey))::DOF] |= isclose(y_local, 0.) | isclose(y_local, b)

        bu = ~bk

        KC0uu = KC0[bu, :][:, bu]
        KGuu = KG[bu, :][:, bu]

        num_eig_lb = max(mode+1, 3)

        # NOTE pyfe3d.solver.linear_buckling equilibrates the diagonal,
        #      estimates the Cayley shift and verifies that no lower load
        #      multiplier was missed, which a hardcoded shift does not
        eigvals, eigvecsu = linear_buckling(KC0uu, KGuu, num_eigvalues=num_eig_lb, tol=1e-9)

        load_mult = eigvals[0]
        P_cr_calc = load_mult*Nxx*b
        print('linear buckling load_mult =', load_mult)
        print('linear buckling P_cr_calc =', P_cr_calc)
        P_cr_calcs.append(P_cr_calc)

    # NOTE Seide and Stowell (1950), Eq. (1), with D_Q = A55
    D = E*h**3/(12*(1 - nu**2))
    r = np.pi**2*D/(b**2*prop.A55)
    beta = a/b
    k = min((m/beta + beta/m)**2/(1 + r*(1 + m**2/beta**2))
            for m in range(1, 21))
    P_cr_theory = -k*np.pi**2*D/b**2*b
    print('Theoretical P_cr_theory', P_cr_theory)
    assert np.allclose(P_cr_calcs, P_cr_calcs[0], rtol=1e-6)
    # NOTE value pre-determined with the 30x14 mesh, see the docstring
    assert isclose(P_cr_calcs[0], -179433757.7496, rtol=1e-4)
    # NOTE grade A against Seide and Stowell (1950)
    assert abs(P_cr_calcs[0]/P_cr_theory - 1) <= 0.02

    if plot:
        import matplotlib.pyplot as plt

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]
        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_linear_buckling_plate(plot=True, mode=0)

