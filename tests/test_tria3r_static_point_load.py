import sys
sys.path.append('..')

import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Tria3R, Tria3RData, Tria3RProbe, INT, DOUBLE, DOF


def test_static_plate_quad_point_load(plot=False, refinement=1):
    data = Tria3RData()
    probe = Tria3RProbe()
    nx = 7*refinement
    ny = 11*refinement
    if (nx % 2) == 0:
        nx += 1
    if (ny % 2) == 0:
        ny += 1

    # geometry
    a = 3
    b = 7
    h = 0.005 # m

    # material
    E = 200e9
    nu = 0.3

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
    N = DOF*nx*ny

    # creating elements and populating global stiffness

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

    trias = []
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        pos1 = nid_pos[n1]
        pos2 = nid_pos[n2]
        pos3 = nid_pos[n3]
        pos4 = nid_pos[n4]
        r1 = ncoords[pos1]
        r2 = ncoords[pos2]
        r3 = ncoords[pos3]
        r4 = ncoords[pos4]

        #first tria
        normal = np.cross(r2 - r1, r3 - r1)[2]
        assert normal > 0 # guaranteeing that all elements have CCW positive normal
        tria = Tria3R(probe)
        tria.n1 = n1
        tria.n2 = n2
        tria.n3 = n3
        tria.c1 = DOF*nid_pos[n1]
        tria.c2 = DOF*nid_pos[n2]
        tria.c3 = DOF*nid_pos[n3]
        tria.init_k_KC0 = init_k_KC0
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE

        #second tria
        normal = np.cross(r3 - r1, r4 - r1)[2]
        assert normal > 0 # guaranteeing that all elements have CCW positive normal
        tria = Tria3R(probe)
        tria.n1 = n1
        tria.n2 = n3
        tria.n3 = n4
        tria.c1 = DOF*nid_pos[n1]
        tria.c2 = DOF*nid_pos[n3]
        tria.c3 = DOF*nid_pos[n4]
        tria.init_k_KC0 = init_k_KC0
        tria.update_rotation_matrix(ncoords_flatten)
        tria.update_probe_xe(ncoords_flatten)
        tria.update_KC0(KC0r, KC0c, KC0v, prop)
        trias.append(tria)
        init_k_KC0 += data.KC0_SPARSE_SIZE

    print('elements created')

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    print('sparse KC0 created')

    # applying boundary conditions
    # simply supported
    bk = np.zeros(N, dtype=bool) #array to store known DOFs
    check = np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0) | np.isclose(y, b)
    bk[2::DOF] = check

    # eliminating all u,v displacements
    bk[0::DOF] = True
    bk[1::DOF] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # external force vector for point load at center
    f = np.zeros(N)
    fmid = 1.
    # force at center node
    check = np.isclose(x, a/2) & np.isclose(y, b/2)
    f[2::DOF][check] = fmid

    # sub-matrices corresponding to unknown DOFs
    KC0uu = KC0[bu, :][:, bu]
    fu = f[bu]
    assert fu.sum() == fmid

    # solving static problem
    uu, info = cg(KC0uu, fu, atol=1e-9)
    assert info == 0

    # vector u containing displacements for all DOFs
    u = np.zeros(N)
    u[bu] = uu

    w = u[2::DOF].reshape(nx, ny).T

    # obtained with bfsplate2d element, nx=ny=29
    wmax_ref = 6.594931610258557e-05
    # obtained with Tria3R nx=7, ny=11
    wmax_ref = 5.219888791560321e-05
    print('w.max()', w.max())
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    assert np.isclose(wmax_ref, w.max(), rtol=0.02)


if __name__ == '__main__':
    test_static_plate_quad_point_load(plot=True, refinement=1)
