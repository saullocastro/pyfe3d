import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse.linalg import cg
from scipy.sparse import coo_matrix

from pyfe3d.shellprop_utils import isotropic_plate
from pyfe3d import Quad4, Quad4Data, Quad4Probe, INT, DOUBLE, DOF


def test_quad4_probe_in_plane(plot=False):
    data = Quad4Data()
    probe = Quad4Probe()
    nx = 7
    ny = 11

    a = 3
    b = 7
    h = 0.005 # m

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

    num_elements = len(n1s)

    KC0r = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(data.KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    N = DOF*nx*ny

    prop = isotropic_plate(thickness=h, E=E, nu=nu, calc_scf=True)

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
        quad = Quad4(probe)
        quad.n1 = n1
        quad.n2 = n2
        quad.n3 = n3
        quad.n4 = n4
        quad.c1 = DOF*nid_pos[n1]
        quad.c2 = DOF*nid_pos[n2]
        quad.c3 = DOF*nid_pos[n3]
        quad.c4 = DOF*nid_pos[n4]
        quad.init_k_KC0 = init_k_KC0
        quad.update_rotation_matrix(ncoords_flatten)
        quad.update_probe_xe(ncoords_flatten)
        quad.update_KC0(KC0r, KC0c, KC0v, prop)
        quads.append(quad)
        init_k_KC0 += data.KC0_SPARSE_SIZE

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

    ufield = u[0::DOF].reshape(nx, ny).T

    quad.update_probe_ue(u)
    quad.update_probe_xe(ncoords_flatten)
    quad.probe.update_BL(xi=0., eta=0.)
    BLexx = np.asarray(quad.probe.BLexx)
    ue = np.asarray(quad.probe.ue)
    exx = BLexx @ ue
    print('exx', exx)
    print('delta_u = exx*a', exx*a)
    assert np.isclose(exx*a, ufield.min() - ufield.max())


    if plot:
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(ufield.min(), ufield.max(), 10)
        plt.contourf(xmesh, ymesh, ufield, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_quad4_probe_in_plane(plot=True)
