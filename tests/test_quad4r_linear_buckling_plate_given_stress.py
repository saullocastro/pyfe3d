import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh, cg
from scipy.sparse import coo_matrix, diags as sp_diags

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4R, Quad4RData, Quad4RProbe, INT, DOUBLE, DOF


def test_linear_buckling(mode=0, refinement=1):
    data = Quad4RData()
    probe = Quad4RProbe()
    nx = refinement*31
    ny = refinement*15
    if (nx % 2) == 0:
        nx += 1
    if (ny % 2) == 0:
        ny += 1

    a = 2.0
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
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True, rho=rho)

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
        quad.K6ROT = 100.
        quad.init_k_KC0 = init_k_KC0
        quad.init_k_KG = init_k_KG
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')
 
    # constraining all edge loads to no translation
    bk = np.zeros(N, dtype=bool)
    check = isclose(x, 0.) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[0::DOF] = check
    bk[1::DOF] = check
    bk[2::DOF] = check

    bu = ~bk

    KC0uu = KC0[bu, :][:, bu]

    Nxx = -1

    for quad in quads:
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KG_given_stress(Nxx, 0, 0, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 3)

    # NOTE pre-conditioning the eigenvalue problem to improve convergence of the eigensolver
    kc0_diag = KC0uu.diagonal()
    kc0_diag_inv_sqrt = 1.0/np.sqrt(np.maximum(kc0_diag, 1e-30))
    D_inv_sqrt = sp_diags(kc0_diag_inv_sqrt)
    KC0uu_scaled = D_inv_sqrt @ KC0uu @ D_inv_sqrt
    KGuu_scaled = D_inv_sqrt @ KGuu @ D_inv_sqrt
    eigvals_inv, eigvecsu_scaled = eigsh(A=KGuu_scaled, k=num_eig_lb, which='SM',
            M=KC0uu_scaled, tol=1e-9, sigma=1., mode='cayley')
    eigvals = -1./eigvals_inv

    # NOTE the eigenvectors are scaled by the preconditioner to recover the original eigenvectors
    eigvecsu = D_inv_sqrt @ eigvecsu_scaled

    load_mult = eigvals[0]
    P_cr_calc = load_mult*Nxx*b
    print('linear buckling load_mult =', load_mult)
    print('linear buckling P_cr_calc =', P_cr_calc)

    u = np.zeros(N)
    u[bu] = eigvecsu[:, mode]

    kcmin = 1e6
    mmin = 0
    for m in range(1, 21):
        kc = (m*b/a + a/(m*b))**2
        if kc <= kcmin:
            kcmin = kc
            mmin = m
    print('kcmin =', kcmin)
    print('m =', mmin)
    sigma_cr = -kcmin*np.pi**2*E/(12*(1-nu**2))*h**2/b**2
    P_cr_theory = sigma_cr*h*b
    print('Theoretical P_cr_theory', P_cr_theory)
    assert isclose(P_cr_theory, P_cr_calc, rtol=0.05)


if __name__ == '__main__':
    test_linear_buckling(mode=0, refinement=1)
