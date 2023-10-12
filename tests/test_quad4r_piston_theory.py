import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF


def test_quad4r_piston_theory(plot=False, refinement=1):
    data = Quad4RData()
    probe = Quad4RProbe()
    nx = refinement*21
    ny = refinement*21

    a = 0.8
    b = 0.5

    E = 70.e9 # Pa
    nu = 0.3

    rho = 7.8e3 # kg/m3
    h = 0.0035 # m

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

    num_elements = len(n1s)
    print('num_elements', num_elements)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    Mr = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(data.M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KA_betar = np.zeros(data.KA_BETA_SPARSE_SIZE*num_elements, dtype=INT)
    KA_betac = np.zeros(data.KA_BETA_SPARSE_SIZE*num_elements, dtype=INT)
    KA_betav = np.zeros(data.KA_BETA_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KA_gammar = np.zeros(data.KA_GAMMA_SPARSE_SIZE*num_elements, dtype=INT)
    KA_gammac = np.zeros(data.KA_GAMMA_SPARSE_SIZE*num_elements, dtype=INT)
    KA_gammav = np.zeros(data.KA_GAMMA_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    CAr = np.zeros(data.CA_SPARSE_SIZE*num_elements, dtype=INT)
    CAc = np.zeros(data.CA_SPARSE_SIZE*num_elements, dtype=INT)
    CAv = np.zeros(data.CA_SPARSE_SIZE*num_elements, dtype=DOUBLE)

    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True, rho=rho)

    quads = []
    init_k_KC0 = 0
    init_k_M = 0
    init_k_KA_beta = 0
    init_k_KA_gamma = 0
    init_k_CA = 0
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
        quad.init_k_M = init_k_M
        quad.init_k_KA_beta = init_k_KA_beta
        quad.init_k_KA_gamma = init_k_KA_gamma
        quad.init_k_CA = init_k_CA
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quad.update_M(Mr, Mc, Mv, prop)
        quad.update_KA_beta(KA_betar, KA_betac, KA_betav)
        quad.update_KA_gamma(KA_gammar, KA_gammac, KA_gammav)
        quad.update_CA(CAr, CAc, CAv)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_M += data.M_SPARSE_SIZE
        init_k_KA_beta += data.KA_BETA_SPARSE_SIZE
        init_k_KA_gamma += data.KA_GAMMA_SPARSE_SIZE
        init_k_CA += data.CA_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()
    KA_beta = coo_matrix((KA_betav, (KA_betar, KA_betac)), shape=(N, N)).tocsc()
    KA_gamma = coo_matrix((KA_gammav, (KA_gammar, KA_gammac)), shape=(N, N)).tocsc()
    CA = coo_matrix((CAv, (CAr, CAc)), shape=(N, N)).tocsc()

    print('sparse KC0 and M created')

    bk = np.zeros(N, dtype=bool)
    edges = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
    bk[0::DOF][edges] = True
    bk[1::DOF][edges] = True
    bk[2::DOF][edges] = True

    bu = ~bk

    Kuu = KC0[bu, :][:, bu]
    Muu = M[bu, :][:, bu]
    KA_betauu = KA_beta[bu, :][:, bu]
    KA_gammauu = KA_gamma[bu, :][:, bu]
    CAuu = CA[bu, :][:, bu]

    num_eigenvalues = 7
    print('eig solver begins')
    # solves Ax = lambda M x
    # we have Ax - lambda M x = 0, with lambda = omegan**2
    eigvals, eigvecsu = eigsh(A=Kuu, M=Muu, sigma=-1., which='LM',
            k=num_eigenvalues, tol=1e-3)
    print('eig solver end')
    eigvecs = np.zeros((N, eigvecsu.shape[1]), dtype=float)
    eigvecs[bu, :] = eigvecsu
    omegan = eigvals**0.5

    # panel flutter analysis
    def MAC(mode1, mode2):
        return (mode1@mode2)**2/((mode1@mode1)*(mode2@mode2))

    MACmatrix = np.zeros((num_eigenvalues, num_eigenvalues))
    rho_air = 1.225 # kg/m^3
    v_sound = 343 # m/s
    v_air = np.linspace(1.1*v_sound, 5*v_sound, 20)
    Mach = v_air/v_sound
    betas = rho_air*v_air**2/np.sqrt(Mach**2 - 1)
    radius = 1.
    gammas = betas/(2*radius*np.sqrt(Mach**2 - 1))
    mus = betas/(Mach*v_sound)*((Mach**2 - 2)/(Mach**2 - 1))
    mus*=0 # TODO quadractic eigenvalue problem to solve including damping

    omegan_vec = []
    for i, beta in enumerate(betas):
        print('analysis i', i)
        # solving generalized eigenvalue problem
        Keffective = Kuu + beta*KA_betauu + gammas[i]*KA_gammauu
        eigvals, eigvecsu = eigs(A=Keffective, M=Muu,
                k=num_eigenvalues, which='LM', sigma=-1., tol=1e-6, v0=eigvecsu[:, 0])
        # NOTE with v0=eigvecsu[:, 0]) avoids fluctuations in adjacent
        # solutions, while making it a bit slower
        eigvecs = np.zeros((N, num_eigenvalues), dtype=float)
        eigvecs[bu, :] = eigvecsu

        if i == 0:
            eigvecs_ref = eigvecs

        corresp = []
        for j in range(num_eigenvalues):
            for k in range(num_eigenvalues):
                MACmatrix[j, k] = MAC(eigvecs_ref[:, j], eigvecs[:, k])
            if np.isclose(np.max(MACmatrix[j, :]), 1.):
                corresp.append(np.argmax(MACmatrix[j, :]))
            else:
                corresp.append(j)
        omegan_vec.append(eigvals[corresp]**0.5)
        print(np.round(MACmatrix, 2))

        eigvecs_ref = eigvecs[:, corresp].copy()


    omegan_vec = np.array(omegan_vec)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        for i in range(num_eigenvalues):
            plt.plot(Mach, omegan_vec[:, i])
        plt.ylabel('$\omega_n\ [rad/s]$')
        plt.xlabel('Mach')
        plt.show()


if __name__ == '__main__':
    test_quad4r_piston_theory(plot=True, refinement=1)
