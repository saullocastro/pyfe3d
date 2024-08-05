import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve, cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def test_linear_buckling_plate(plot=False, mode=0):
    #
    thetas = np.deg2rad(np.linspace(-np.pi, np.pi, 5))

    nx = 13
    ny = 13

    a = 0.5
    b = 0.5
    h = 0.05 # m

    E = 203.e9 # Pa
    nu = 0.33
    prop = isotropic_plate(E=E, nu=nu, thickness=h, calc_scf=True)
    print(prop.scf_k13)
    print(prop.scf_k23)

    Nxx = -1.

    data = Quad4Data()
    probe = Quad4Probe()

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)
    ncoords_local = np.vstack((xmesh.T.flatten(), ymesh.T.flatten(), np.zeros_like(ymesh.T.flatten()))).T

    x_local = ncoords_local[:, 0]
    y_local = ncoords_local[:, 1]
    z_local = ncoords_local[:, 2]

    for pitch in thetas:
        for yaw in thetas:
            print('pitch, yaw', pitch, yaw)

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

                quad = Quad4(probe)
                quad.n1 = n1
                quad.n2 = n2
                quad.n3 = n3
                quad.n4 = n4
                quad.c1 = DOF*nid_pos[n1]
                quad.c2 = DOF*nid_pos[n2]
                quad.c3 = DOF*nid_pos[n3]
                quad.c4 = DOF*nid_pos[n4]
                quad.K6ROT = 1e4
                quad.init_k_KC0 = init_k_KC0
                quad.init_k_KG = init_k_KG
                quad.update_rotation_matrix(ncoords_flatten)
                quad.update_probe_xe(ncoords_flatten)
                quad.update_KC0(KC0r, KC0c, KC0v, prop)
                quad.update_KG_given_stress(Nxx, 0, 0, KGr, KGc, KGv)
                quads.append(quad)
                init_k_KC0 += data.KC0_SPARSE_SIZE
                init_k_KG += data.KG_SPARSE_SIZE

            print('elements created')

            KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
            KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()

            print('sparse KC0 and KG created')

            # applying simply supported boundary conditions
            bk = np.zeros(N, dtype=bool)
            check = isclose(x_local, 0.) | isclose(x_local, a) | isclose(y_local, 0) | isclose(y_local, b)
            bk[0::DOF] = check
            bk[1::DOF] = check
            bk[2::DOF] = check

            bu = ~bk

            KC0uu = KC0[bu, :][:, bu]
            KGuu = KG[bu, :][:, bu]

            num_eig_lb = max(mode+1, 3)
            PREC = np.max(1/KC0uu.diagonal())
            eigvals, eigvecsu = eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM',
                    M=PREC*KC0uu, tol=1e-9, sigma=1., mode='cayley')
            eigvals = -1./eigvals
            load_mult = eigvals[0]
            P_cr_calc = load_mult*Nxx*b
            print('linear buckling load_mult =', load_mult)
            print('linear buckling P_cr_calc =', P_cr_calc)

            kcmin = 1e6
            mmin = 0
            for m in range(1, 21):
                kc = (m*b/a + a/(m*b))**2
                if kc <= kcmin:
                    kcmin = kc
                    mmin = m
            sigma_cr = -kcmin*np.pi**2*E/(12*(1-nu**2))*h**2/b**2
            P_cr_theory = sigma_cr*h*b
            print('Theoretical P_cr_theory', P_cr_theory)
            assert isclose(P_cr_theory, P_cr_calc, rtol=0.03)



    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]

        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_linear_buckling_plate(plot=True, mode=0)
