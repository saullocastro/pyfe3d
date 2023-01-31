import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF


def test_tria3r_nat_freq_distorted(plot=False, mode=0):
    data = Tria3RData()
    probe = Tria3RProbe()
    mtype = 1
    if True:
        nx = 9
        ny = 11

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
        np.random.seed(20)
        rdm = (-1 + 2*np.random.rand(x[inner].shape[0]))
        np.random.seed(20)
        rdm = (-1 + 2*np.random.rand(y[inner].shape[0]))
        x[inner] += dx*rdm*0.4
        y[inner] += dy*rdm*0.4

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

        prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True, rho=rho)

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
        eigvals, eigvecsu = eigsh(A=Kuu, M=Muu, sigma=-1., which='LM',
                k=num_eigenvalues, tol=1e-9)
        print('eig solver end')
        eigvecs = np.zeros((N, eigvecsu.shape[1]), dtype=float)
        eigvecs[bu, :] = eigvecsu
        omegan = eigvals**0.5

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]

        # theoretical reference
        m = 1
        n = 1
        D = 2*h**3*E/(3*(1 - nu**2))
        wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2

        print('Theoretical omega123', wmn)
        print('Numerical omega123', omegan[0:10])
        assert isclose(wmn, omegan[0], rtol=0.05)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
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
