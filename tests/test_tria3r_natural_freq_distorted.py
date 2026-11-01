import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import natural_frequency


def test_tria3r_nat_freq_distorted(plot=False, mode=0, refinement=1):
    r"""Fundamental frequency of a simply supported plate, ``a = 0.3``,
    ``b = 0.5``, ``h = 0.01``, with a distorted mesh and the reduced mass
    matrix

    The inner nodes are moved along ``x`` and ``y`` by independent random
    fractions, up to 0.25, of the element size, which keeps every element
    convex, as asserted at its four corners. Before, the coordinates passed
    to the elements were copied before the distortion, so that the regular
    mesh was analysed, and the distortion moved the nodes by the same
    fraction, up to 0.4, along ``x`` and ``y``, which made up to 7 % of the
    elements re-entrant.

    The reference is the exact solution of Leissa, A. W., Vibration of
    Plates, NASA SP-160, 1969, Sec. 4.1, p. 44, Eq. (4.20),
    ``omega = sqrt(D/rho)*((m*pi/a)**2 + (n*pi/b)**2)``, with ``m = n = 1``
    and Leissa's ``rho`` the mass per unit area, here ``rho*h``.

    The edges are soft simply supported, rotations free. With the default
    transverse shear since 0.12.0, the discrete shear gap field of
    ``shear_model = 1``, the error against the reference, measured with
    this test (quadrilaterals split in two triangles, the inner nodes
    distorted with the same seed, time on one thread), is:

        refinement   mesh      omega_1         error    time
        1            8x10      2412.861       +3.90 %   0.0 s
        2            16x20     2334.011       +0.51 %   0.1 s
        3            24x30     2317.061       -0.22 %   0.2 s
        4            32x40     2309.989       -0.53 %   0.3 s
        6            48x60     2303.533       -0.81 %   0.8 s

    approaching from above the limit of the regular mesh of
    :func:`test_tria3r_natural_freq.test_tria3r_nat_freq`, about -1.15 %,
    the soft simple support of the Mindlin plate. ``refinement=1`` is
    compared with the value pre-determined by this test, with ``rtol=1e-4``
    for the round-off of the eigensolver, and with grade B of MacNeal and
    Harder (1985), p. 14, as the bound against the reference; from
    ``refinement=2`` grade A is asserted against the reference alone.
    Before 0.12.0, with the centroid-sampled shear and
    ``alpha_shear_locking = 0.7``, the frequency was 1.65 % above the
    reference with 8x10 elements and rose to 3.58 % with 16x20 before
    falling to 1.82 % with 48x60.
    """
    data = Tria3RData()
    probe = Tria3RProbe()
    mtype = 2
    if True:
        nx = 8*refinement + 1
        ny = 10*refinement + 1

        a = 0.3
        b = 0.5

        E = 203.e9 # Pa
        nu = 0.33

        rho = 7.83e3 # kg/m3
        h = 0.01 # m

        xtmp = np.linspace(0, a, nx)
        ytmp = np.linspace(0, b, ny)

        dx = xtmp[1] - xtmp[0]
        dy = ytmp[1] - ytmp[0]

        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

        x = ncoords[:, 0]
        y = ncoords[:, 1]
        z = ncoords[:, 2]
        ncoords_flatten = ncoords.flatten()

        inner = np.logical_not(isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b))
        # NOTE independent random fractions along x and y, up to 0.25 of the
        #      element size, which keep every element convex
        np.random.seed(20)
        rdmx = -1 + 2*np.random.rand(x[inner].shape[0])
        rdmy = -1 + 2*np.random.rand(y[inner].shape[0])
        x[inner] += dx*rdmx*0.25
        y[inner] += dy*rdmy*0.25
        # NOTE the coordinates passed to the elements, after the distortion
        ncoords_flatten = ncoords.flatten()

        nids = 1 + np.arange(ncoords.shape[0])
        nid_pos = dict(zip(nids, np.arange(len(nids))))
        nids_mesh = nids.reshape(nx, ny)
        n1s = nids_mesh[:-1, :-1].flatten()
        n2s = nids_mesh[1:, :-1].flatten()
        n3s = nids_mesh[1:, 1:].flatten()
        n4s = nids_mesh[:-1, 1:].flatten()

        num_elements = len(n1s)*2
        print('num_elements', num_elements)

        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)

        trias = []
        init_k_KC0 = 0
        init_k_M = 0
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            pos3 = nid_pos[n3]
            pos4 = nid_pos[n4]
            r1 = ncoords[pos1]
            r2 = ncoords[pos2]
            r3 = ncoords[pos3]
            r4 = ncoords[pos4]
            # NOTE convex element: positive cross products at the four corners
            for ra, rb, rc in ((r4, r1, r2), (r1, r2, r3), (r2, r3, r4), (r3, r4, r1)):
                assert np.cross(rb - ra, rc - rb)[2] > 0

            # first tria
            normal = np.cross(r2 - r1, r3 - r1)[2]
            assert normal > 0
            tria = Tria3R(probe)
            tria.n1 = n1
            tria.n2 = n2
            tria.n3 = n3
            tria.c1 = DOF*nid_pos[n1]
            tria.c2 = DOF*nid_pos[n2]
            tria.c3 = DOF*nid_pos[n3]
            tria.init_k_KC0 = init_k_KC0
            tria.init_k_M = init_k_M
            tria.update_rotation_matrix(ncoords_flatten)
            tria.update_probe_xe(ncoords_flatten)
            tria.update_KC0(KC0r, KC0c, KC0v, prop)
            tria.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            trias.append(tria)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

            # second tria
            normal = np.cross(r3 - r1, r4 - r1)[2]
            assert normal > 0
            tria = Tria3R(probe)
            tria.n1 = n1
            tria.n2 = n3
            tria.n3 = n4
            tria.c1 = DOF*nid_pos[n1]
            tria.c2 = DOF*nid_pos[n3]
            tria.c3 = DOF*nid_pos[n4]
            tria.init_k_KC0 = init_k_KC0
            tria.init_k_M = init_k_M
            tria.update_rotation_matrix(ncoords_flatten)
            tria.update_probe_xe(ncoords_flatten)
            tria.update_KC0(KC0r, KC0c, KC0v, prop)
            tria.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            trias.append(tria)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        # applying boundary conditions
        # simply supported
        bk = np.zeros(N, dtype=bool)
        check = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
        bk[0::DOF] = check
        bk[1::DOF] = check
        bk[2::DOF] = check

        bu = ~bk

        Kuu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]

        num_eigenvalues = max(3, mode+1)
        print('eig solver begin')
        # solves Ax = lambda M x
        # we have Ax - lambda M x = 0, with lambda = omegan**2
        # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
        #      before calling the eigensolver and returns the circular
        #      frequencies already sorted
        omegan, eigvecsu = natural_frequency(Kuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-9)
        print('eig solver end')
        eigvecs = np.zeros((N, eigvecsu.shape[1]), dtype=float)
        eigvecs[bu, :] = eigvecsu

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]

        # NOTE Leissa (1969), Eq. (4.20), with rho*h the mass per unit area
        m = 1
        n = 1
        D = E*h**3/(12*(1 - nu**2))
        wmn = np.sqrt(D/(rho*h))*((m*np.pi/a)**2 + (n*np.pi/b)**2)

        print('Theoretical omega123', wmn)
        print('Numerical omega123', omegan[0:10])
        error = abs(omegan[0]/wmn - 1)
        print('error', omegan[0]/wmn - 1)
        if refinement == 1:
            # NOTE pre-determined by this test with refinement=1 on 2026-10-29
            assert isclose(omegan[0], 2412.86091037, rtol=1e-4)
            assert error <= 0.10
        else:
            assert error <= 0.02

    if plot:
        import matplotlib.pyplot as plt

        plt.clf()
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            pos3 = nid_pos[n3]
            pos4 = nid_pos[n4]
            r1 = ncoords[pos1]
            r2 = ncoords[pos2]
            r3 = ncoords[pos3]
            r4 = ncoords[pos4]
            plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k-')
            plt.plot([r2[0], r3[0]], [r2[1], r3[1]], 'k-')
            plt.plot([r3[0], r4[0]], [r3[1], r4[1]], 'k-')
            plt.plot([r4[0], r1[0]], [r4[1], r1[1]], 'k-')
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_tria3r_nat_freq_distorted(plot=True, mode=0)

