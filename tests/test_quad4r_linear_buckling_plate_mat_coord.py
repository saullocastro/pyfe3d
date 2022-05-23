import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, spsolve, cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import laminated_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF


def test_linear_buckling_plate(plot=False, mode=0):
    # NOTE keep thetadeg = 0 as first, to work as reference wmax_ref
    thetadegs = [0, -90, -60, -30, 30, 60, 90]
    for thetadeg in thetadegs:
        matx = (np.cos(np.deg2rad(thetadeg)), np.sin(np.deg2rad(thetadeg)), 0)
        print('matx', matx)
        data = Quad4RData()
        probe = Quad4RProbe()
        nx = 19
        ny = 7
        if (nx % 2) == 0:
            nx += 1
        if (ny % 2) == 0:
            ny += 1

        a = 2.0
        b = 0.5
        h = 0.002 # m

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
        print('num_elements', num_elements)

        KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
        KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
        KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        prop = laminated_plate(stack=[-thetadeg], laminaprop=(E1, E2, nu12, G12, G12, G12), plyt=h)

        quads = []
        init_k_KC0 = 0
        init_k_KG = 0
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
            quad.init_k_KG = init_k_KG
            quad.update_rotation_matrix(ncoords_flatten, matx[0], matx[1], matx[2])
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KC0(KC0r, KC0c, KC0v, prop)
            quads.append(quad)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

        print('sparse KC0 and M created')

        # applying boundary conditions (leading to a constant Nxx)
        # simply supported in w
        bk = np.zeros(N, dtype=bool)
        check = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
        bk[2::DOF] = check
        # constraining u at x = a/2, y = 0,b
        check = isclose(x, a/2.) & (isclose(y, 0.) | isclose(y, b))
        bk[0::DOF] = check
        # constraining v at x = 0,a y = b/2
        check = isclose(y, b/2.) & (isclose(x, 0.) | isclose(x, a))
        bk[1::DOF] = check
        # removing drilling
        bk[5::DOF] = True

        bu = ~bk

        # applying load along u at x=a
        # nodes at vertices get 1/2 of the force
        fext = np.zeros(N)
        ftotal = -1000.
        print('ftotal', ftotal)
        # at x=0
        check = (isclose(x, 0) & ~isclose(y, 0) & ~isclose(y, b))
        fext[0::DOF][check] = -ftotal/(ny - 1)
        check = ((isclose(x, 0) & isclose(y, 0))
                |(isclose(x, 0) & isclose(y, b)))
        fext[0::DOF][check] = -ftotal/(ny - 1)/2
        assert np.isclose(fext.sum(), -ftotal)
        # at x=a
        check = (isclose(x, a) & ~isclose(y, 0) & ~isclose(y, b))
        fext[0::DOF][check] = ftotal/(ny - 1)
        check = ((isclose(x, a) & isclose(y, 0))
                |(isclose(x, a) & isclose(y, b)))
        fext[0::DOF][check] = ftotal/(ny - 1)/2
        assert np.isclose(fext.sum(), 0)

        Kuu = KC0[bu, :][:, bu]
        fextu = fext[bu]

        PREC = np.max(1/Kuu.diagonal())
        uu, out = cg(PREC*Kuu, PREC*fextu, atol=1e-8)
        assert out == 0, 'cg failed'
        u = np.zeros(N)
        u[bu] = uu

        print('u extremes', u[0::DOF].min(), u[0::DOF].max())
        print('v extremes', u[1::DOF].min(), u[1::DOF].max())
        print('w extremes', u[2::DOF].min(), u[2::DOF].max())
        if False:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.gca().set_aspect('equal')
            uplot = u[2::DOF].reshape(nx, ny).T
            levels = np.linspace(uplot.min(), uplot.max(), 300)
            plt.contourf(xmesh, ymesh, uplot, levels=levels)
            plt.colorbar()
            plt.show()
            raise

        for quad in quads:
            quad.update_probe_ue(u) # NOTE update affects the Quad4RProbe class attribute ue
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KG(KGr, KGc, KGv, prop)
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        KGuu = KG[bu, :][:, bu]
        print('sparse KG created')

        num_eig_lb = max(mode+1, 1)
        eigvals, eigvecsu = eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM',
                M=PREC*Kuu, tol=1e-15, sigma=1., mode='cayley')
        eigvals = -1./eigvals
        load_mult = eigvals[0]
        P_cr_calc = load_mult*ftotal
        print('linear buckling load_mult =', load_mult)
        print('linear buckling P_cr_calc =', P_cr_calc)

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]

        if thetadeg == 0:
            P_cr_ref = P_cr_calc
        else:
            print('P_cr_ref, P_cr_calc', P_cr_ref, P_cr_calc)
            assert np.isclose(P_cr_ref, P_cr_calc, rtol=1e-5)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_linear_buckling_plate(plot=True, mode=0)
