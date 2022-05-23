import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF


def test_tria3r_linear_buckling_plate_given_stress(mode=0, refinement=1):
    data = Tria3RData()
    probe = Tria3RProbe()
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

    num_elements = len(n1s)*2
    print('num_elements', num_elements)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(data.KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True, rho=rho)

    trias = []
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
        tria.init_k_KG = init_k_KG
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

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
        tria.init_k_KG = init_k_KG
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE
        init_k_KG += data.KG_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    # applying boundary conditions (leading to a constant Nxx)
    # simply supported in w
    bk = np.zeros(N, dtype=bool) #array to store known DOFs
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

    Kuu = KC0[bu, :][:, bu]

    Nxx = -1

    for tria in trias:
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KG_given_stress(Nxx, 0, 0, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    print('sparse KG created')

    num_eig_lb = max(mode+1, 1)
    PREC = np.max(1/Kuu.diagonal())
    eigvals, eigvecsu = eigsh(A=PREC*KGuu, k=num_eig_lb, which='SM',
            M=PREC*Kuu, tol=1e-15, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    load_mult = eigvals[0]
    P_cr_calc = load_mult*Nxx*b
    print('linear buckling load_mult =', load_mult)
    print('linear buckling P_cr_calc =', P_cr_calc)

    u = np.zeros(N)
    u[bu] = eigvecsu[:, mode]

    # theoretical reference
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
    test_tria3r_linear_buckling_plate_given_stress(mode=0, refinement=1)
