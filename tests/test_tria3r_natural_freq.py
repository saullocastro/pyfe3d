import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import natural_frequency


def test_tria3r_nat_freq(plot=False, mode=0, mtypes=range(3), refinement=1):
    r"""Fundamental frequency of a simply supported plate, ``a = 0.3``,
    ``b = 0.5``, ``h = 0.01``, with the consistent, lumped and reduced mass
    matrices

    The reference is the exact solution of Leissa, A. W., Vibration of
    Plates, NASA SP-160, 1969, Sec. 4.1, p. 44, Eq. (4.20),
    ``omega = sqrt(D/rho)*((m*pi/a)**2 + (n*pi/b)**2)``, with ``m = n = 1``
    and Leissa's ``rho`` the mass per unit area, here ``rho*h``.

    The edges are soft simply supported, rotations free. With the default
    transverse shear since 0.12.0, the discrete shear gap field of
    ``shear_model = 1``, the errors against the reference, measured with
    this test for the consistent (``mtype=0``), lumped (``1``) and reduced
    (``2``) mass matrices (quadrilaterals split in two triangles, time of
    one mass matrix on one thread), are:

        refinement   mesh      mtype=0   mtype=1   mtype=2   time
        1            8x10      +2.75 %   +0.62 %   +3.49 %   0.0 s
        2            17x21     +0.10 %   -0.37 %   +0.26 %   0.1 s
        3            26x32     -0.44 %   -0.64 %   -0.37 %   0.3 s
        4            35x43     -0.67 %   -0.78 %   -0.63 %   0.6 s
        6            53x65     -0.89 %   -0.94 %   -0.87 %   1.6 s
        8            71x87     -0.99 %   -1.01 %   -0.98 %   3.3 s
        12           107x131   -1.07 %                        5.1 s
        16           143x175   -1.10 %                       14.9 s

    The soft simple support of the Mindlin plate, `h/a = 1/30`, lies below
    the Kirchhoff reference: Richardson extrapolation of the last three
    meshes, at second order, gives about -1.15 %, against -0.28 % for the
    hard simple support. That limit is a measurement, not a published
    value. ``refinement=1`` is compared with the values pre-determined by
    this test, with ``rtol=1e-4`` for the round-off of the eigensolver, and
    with grade B of MacNeal and Harder (1985), p. 14, as the bound against
    the reference; from ``refinement=2`` grade A is asserted against the
    reference alone. Before 0.12.0, with the centroid-sampled shear and
    ``alpha_shear_locking = 0.7``, the errors were -1.47 to +1.31 % with
    8x10 elements and rose to +2.65 to +3.29 % with 17x21 before falling,
    see :mod:`pyfe3d.tria3r`.
    """
    # NOTE pre-determined by this test with refinement=1 on 2026-10-29
    predetermined = {0: 2386.04605865, 1: 2336.69402554, 2: 2403.19824795}
    data = Tria3RData()
    probe = Tria3RProbe()
    for mtype in mtypes:
        nx = refinement*9
        ny = refinement*11

        a = 0.3
        b = 0.5

        E = 203.e9 # Pa
        nu = 0.33

        rho = 7.83e3 # kg/m3
        h = 0.01 # m

        xtmp = np.linspace(0, a, nx)
        ytmp = np.linspace(0, b, ny)
        xmesh, ymesh = np.meshgrid(xtmp, ytmp)
        ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

        x = ncoords[:, 0]
        y = ncoords[:, 1]
        z = ncoords[:, 2]
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
            assert isclose(omegan[0], predetermined[mtype], rtol=1e-4)
            assert error <= 0.10
        else:
            assert error <= 0.02

    if plot:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_tria3r_nat_freq(plot=True, mode=0, mtypes=[1], refinement=5)

