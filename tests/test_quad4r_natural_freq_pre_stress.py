import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import cg
from scipy.sparse import coo_matrix, diags as sp_diags

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF
from pyfe3d.solver import natural_frequency


def test_nat_freq_pre_stress(plot=False, mode=0, mtypes=range(3), refinement=2):
    r"""Fundamental frequency of a simply supported plate under a uniform
    compression ``N_xx`` equal to half of its buckling load

    The reference is the exact solution of Leissa, A. W., Vibration of Plates,
    NASA SP-160, 1969, Sec. 10.2.1, p. 277: the frequency equation (10.44),
    ``rho*h*omega_mn**2 = D*((m*pi/a)**2 + (n*pi/b)**2)**2 + N_xx*(m*pi/a)**2``,
    minimised over ``m`` and ``n``, and the buckling load (10.46).

    With 42x26 elements the frequencies of the three mass matrices are 0.03 to
    0.29 % above the reference, with 20x12 elements 0.21 to 1.39 %, measured on
    2026-10-29 with the physical stabilisation of the bending and of the
    transverse shear of Belytschko and Leviathan (1994) that Quad4R uses since
    0.12.0, see :mod:`pyfe3d.quad4r`; up to 0.11, with the hourglass
    stiffnesses of Brockman (1987), 0.14 to 0.39 % and 0.77 to 1.95 %.
    """
    data = Quad4RData()
    probe = Quad4RProbe()
    for mtype in mtypes:
        nx = refinement*21
        ny = refinement*13
        if (nx % 2) == 0:
            nx += 1
        if (ny % 2) == 0:
            ny += 1

        a = 1.5
        b = 0.5

        E = 203.e9 # Pa
        nu = 0.33

        rho = 7.83e3 # kg/m3
        h = 0.002 # m

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
        Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
        Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        N = DOF*nx*ny

        prop = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)

        quads = []
        init_k_KC0 = 0
        init_k_KG = 0
        init_k_M = 0
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
            quad.init_k_M = init_k_M
            quad.K6ROT = 100.
            quad.update_rotation_matrix(ncoords_flatten)
            quad.update_probe_xe(ncoords_flatten)
            quad.update_KC0(KC0r, KC0c, KC0v, prop)
            quad.update_M(Mr, Mc, Mv, prop, mtype=mtype)
            quads.append(quad)
            init_k_KC0 += data.KC0_SPARSE_SIZE
            init_k_KG += data.KG_SPARSE_SIZE
            init_k_M += data.M_SPARSE_SIZE

        print('elements created')

        KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
        M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

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

        # NOTE buckling load of Leissa (1969), Eq. (10.46), with N_yy = 0
        D = E*h**3/(12*(1 - nu**2))
        Nxx_cr = max(-D*(a/(m*np.pi))**2*((m*np.pi/a)**2 + (np.pi/b)**2)**2
                     for m in range(1, 21))
        Nxx = 0.5*Nxx_cr

        # applying load along u at x=a
        # nodes at vertices get 1/2 of the force
        fext = np.zeros(N)
        ftotal = Nxx*b
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

        KC0uu = KC0[bu, :][:, bu]
        Muu = M[bu, :][:, bu]
        fextu = fext[bu]

        # NOTE pre-conditioning the linear system to improve convergence of the iterative solver
        kuu_diag = KC0uu.diagonal()
        kuu_diag_inv_sqrt = 1.0/np.sqrt(np.maximum(kuu_diag, 1e-30))
        D_inv_sqrt = sp_diags(kuu_diag_inv_sqrt)
        KC0uu_scaled = D_inv_sqrt @ KC0uu @ D_inv_sqrt
        fextu_scaled = D_inv_sqrt @ fextu
        uu_scaled, out = cg(KC0uu_scaled, fextu_scaled, atol=1e-8)
        assert out == 0, 'cg failed'
        uu = D_inv_sqrt @ uu_scaled
        u = np.zeros(N)
        u[bu] = uu

        print('u extremes', u[0::DOF].min(), u[0::DOF].max())
        print('v extremes', u[1::DOF].min(), u[1::DOF].max())
        print('w extremes', u[2::DOF].min(), u[2::DOF].max())

        if False:
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

        num_eigenvalues = max(2, mode+1)

        print('eig solver begin')
        # solves Ax = lambda M x
        # we have Ax - lambda M x = 0, with lambda = omegan**2

        # NOTE pre-conditioning the eigenvalue problem to improve convergence of the eigensolver
        KTuu = KC0uu + KGuu
        kt_diag = KTuu.diagonal()
        kt_diag_inv_sqrt = 1.0/np.sqrt(np.maximum(kt_diag, 1e-30))
        # NOTE pyfe3d.solver.natural_frequency equilibrates the diagonal
        #      before calling the eigensolver and returns the circular
        #      frequencies already sorted
        omegan, eigvecsu = natural_frequency(KTuu, Muu, num_eigvalues=num_eigenvalues, tol=1e-6)

        print('eig solver end')

        eigvecs = np.zeros((N, eigvecsu.shape[1]))
        eigvecs[bu, :] = eigvecsu

        u = np.zeros(N)
        u[bu] = eigvecsu[:, mode]

        # NOTE frequency equation of Leissa (1969), Eq. (10.44), with N_yy = 0
        omega_theory = min(np.sqrt((D*((m*np.pi/a)**2 + (n*np.pi/b)**2)**2
                                    + Nxx*(m*np.pi/a)**2)/(rho*h))
                           for m in range(1, 21) for n in range(1, 6))

        print('Theoretical omega', omega_theory)
        print('Numerical omega', omegan[0:10])
        assert isclose(omega_theory, omegan[0], rtol=0.01)

    if plot:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.contourf(xmesh, ymesh, u[2::DOF].reshape(nx, ny).T)
        plt.show()


if __name__ == '__main__':
    test_nat_freq_pre_stress(plot=True, mode=0, mtypes=[0], refinement=2)

