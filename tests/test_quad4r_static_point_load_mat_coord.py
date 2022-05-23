import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import laminated_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF


def test_static_plate_quad_point_load(plot=False):
    # NOTE keep thetadeg = 0 as first, to work as reference wmax_ref
    thetadegs = [0, -90, -60, -30, 30, 60, 90]
    for thetadeg in thetadegs:
        matx = (np.cos(np.deg2rad(thetadeg)), np.sin(np.deg2rad(thetadeg)), 0)
        print('matx', matx)

        data = Quad4RData()
        probe = Quad4RProbe()
        nx = 7
        ny = 11

        a = 3
        b = 7
        h = 0.005 # m

        E1 = 200e9
        E2 = 50e9
        nu12 = 0.3
        G12 = 8e9

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

        num_elements = len(n1s)

        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        prop = laminated_plate(stack=[-thetadeg], laminaprop=(E1, E2, nu12, G12, G12, G12), plyt=h)

        quads = []
        init_k_KC0 = 0
        for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
            pos1 = nid_pos[n1]
            pos2 = nid_pos[n2]
            pos3 = nid_pos[n3]
            pos4 = nid_pos[n4]
            r1 = ncoords[pos1]
            r2 = ncoords[pos2]
            r3 = ncoords[pos3]
            normal = np.cross(r2 - r1, r3 - r2)[2]
            assert normal > 0
            quad = Quad4R(probe)
            quad.n1 = n1
            quad.n2 = n2
            quad.n3 = n3
            quad.n4 = n4
            quad.c1 = DOF*nid_pos[n1]
            quad.c2 = DOF*nid_pos[n2]
            quad.c3 = DOF*nid_pos[n3]
            quad.c4 = DOF*nid_pos[n4]
            quad.init_k_KC0 = init_k_KC0
            quad.update_rotation_matrix(ncoords_flatten, matx[0], matx[1], matx[2])
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KC0(KC0r, KC0c, KC0v, prop)
            quads.append(quad)
            init_k_KC0 += data.KC0_SPARSE_SIZE

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

        print('elements created')

        bk = np.zeros(N, dtype=bool)
        check = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
        bk[2::DOF] = check

        bk[0::DOF] = True
        bk[1::DOF] = True

        bu = ~bk

        # point load at center node
        f = np.zeros(N)
        fmid = 1.
        check = np.isclose(x, a/2) & np.isclose(y, b/2)
        f[2::DOF][check] = fmid

        KC0uu = KC0[bu, :][:, bu]
        fu = f[bu]
        assert fu.sum() == fmid

        uu, info = cg(KC0uu, fu, atol=1e-9)
        assert info == 0

        u = np.zeros(N)
        u[bu] = uu

        w = u[2::DOF].reshape(nx, ny).T

        if thetadeg == 0:
            wmax_ref = w.max()
        else:
            print('wmax_ref, w.max()', wmax_ref, w.max())
            assert np.isclose(wmax_ref, w.max(), rtol=1e-5)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_static_plate_quad_point_load(plot=True)
